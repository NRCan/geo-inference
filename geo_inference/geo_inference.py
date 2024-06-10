import logging
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torchgeo.datasets import stack_samples
from tqdm import tqdm

from .config.logging_config import logger
from .geo_blocks import InferenceMerge, InferenceSampler, RasterDataset
from .utils.helpers import cmd_interface, get_device, get_directory, get_model
from .utils.polygon import gdf_to_yolo, mask_to_poly_geojson, geojson2coco

logger = logging.getLogger(__name__)


class GeoInference:
    """
    A class for performing geo inference on geospatial imagery using a pre-trained model.

    Args:
        model (str): The path or url to the model file
        work_dir (str): The directory where the model and output files will be saved.
        batch_size (int): The batch size to use for inference.
        mask_to_vec (bool): Whether to convert the output mask to vector format.
        device (str): The device to use for inference (either "cpu" or "gpu").
        gpu_id (int): The ID of the GPU to use for inference (if device is "gpu").

    Attributes:
        batch_size (int): The batch size to use for inference.
        work_dir (Path): The directory where the model and output files will be saved.
        device (torch.device): The device to use for inference.
        mask_to_vec (bool): Whether to convert the output mask to vector format.
        model (torch.jit.ScriptModule): The pre-trained model to use for inference.
        classes (int): The number of classes in the output of the model.

    """

    def __init__(self,
                 model: str = None,
                 work_dir: str = None,
                 batch_size: int = 1,
                 mask_to_vec: bool = False,
                 device: str = "gpu",
                 gpu_id: int = 0):
        self.gpu_id = int(gpu_id)
        self.batch_size = int(batch_size)
        self.work_dir: Path = get_directory(work_dir)
        self.device = get_device(device=device, 
                                 gpu_id=self.gpu_id)
        model_path: Path = get_model(model_path_or_url=model, 
                                     work_dir=self.work_dir)
        self.mask_to_vec = mask_to_vec
        self.model = torch.jit.load(model_path, map_location=self.device)
        dummy_input = torch.ones((1, 3, 32, 32), device=self.device)
        with torch.no_grad():
            self.classes = self.model(dummy_input).shape[1]
    
    @torch.no_grad() 
    def __call__(self, tiff_image: str, bbox: str = None, patch_size: int = 512, stride_size: str = None) -> None:
        """
        Perform geo inference on geospatial imagery.

        Args:
            tiff_image (str): The path to the geospatial image to perform inference on.
            bbox (str): The bbox or extent of the image in this format "minx, miny, maxx, maxy"
            patch_size (int): The size of the patches to use for inference.
            stride_size (int): The stride to use between patches.

        Returns:
            None

        """
        mask_path = self.work_dir.joinpath(Path(tiff_image).stem + "_mask.tif")
        polygons_path = self.work_dir.joinpath(Path(tiff_image).stem + "_polygons.geojson")
        yolo_csv_path = self.work_dir.joinpath(Path(tiff_image).stem + "_yolo.csv")
        coco_json_path = self.work_dir.joinpath(Path(tiff_image).stem + "_coco.json")
        
        dataset = RasterDataset(tiff_image, bbox=bbox)
        sampler = InferenceSampler(dataset, size=patch_size, stride=patch_size >> 1 if stride_size is None else stride_size, roi=dataset.bbox)
        roi_height = sampler.im_height 
        roi_width = sampler.im_width
        h_padded, w_padded = roi_height + patch_size, roi_width + patch_size
        output_meta = dataset.src.meta
        merge_patches = InferenceMerge(height=h_padded, width=w_padded, classes=self.classes, device=self.device)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, sampler=sampler, collate_fn=stack_samples)
        
        start_time = time.time()
        
        for batch in tqdm(dataloader, desc='extracting features', unit='batch', total=len(dataloader)):
            image_tensor = batch["image"].to(self.device)
            window_tensor = batch["window"].unsqueeze(1).to(self.device)
            pixel_xy = batch["pixel_coords"]
            output = self.model(image_tensor) 
            merge_patches.merge_on_cpu(batch=output, windows=window_tensor, pixel_coords=pixel_xy)
        merge_patches.save_as_tiff(height=dataset.image_height, 
                                   width=dataset.image_width, 
                                   output_meta=output_meta, 
                                   output_path=mask_path)
        
        if self.mask_to_vec:
            mask_to_poly_geojson(mask_path, polygons_path)
            gdf_to_yolo(polygons_path, mask_path, yolo_csv_path)
            geojson2coco(mask_path, polygons_path, coco_json_path)
            
        end_time = time.time() - start_time
        
        logger.info('Extraction Completed in {:.0f}m {:.0f}s'.format(end_time // 60, end_time % 60))

def main() -> None:
    arguments = cmd_interface()
    geo_inference = GeoInference(model=arguments["model"],
                                 work_dir=arguments["work_dir"],
                                 batch_size=arguments["batch_size"],
                                 mask_to_vec=arguments["vec"],
                                 device=arguments["device"],
                                 gpu_id=arguments["gpu_id"])
    geo_inference(tiff_image=arguments["image"], bbox=arguments["bbox"])
               
if __name__ == "__main__":
    main()