import os
import time
import torch  # type: ignore
import logging
import pystac  # type: ignore
import numpy as np
import dask.array as da
import asyncio
import gc
from dask import config
from typing import Dict  # type: ignore
from pathlib import Path
import rasterio  # type: ignore
from rasterio.windows import from_bounds  # type: ignore
from dask_image.imread import imread as dask_imread  # type: ignore
from typing import Union, Sequence, List
from omegaconf import ListConfig  # type: ignore
import threading
import xarray as xr
from dask.diagnostics import ResourceProfiler, ProgressBar
from multiprocessing.pool import ThreadPool

from .utils.helpers import (
    cmd_interface,
    get_directory,
    get_model,
    xarray_profile_info,
    select_model_device,
    asset_by_common_name,
)
from .geo_dask import (
    runModel,
    sum_overlapped_chunks,
)

from .utils.polygon import gdf_to_yolo, mask_to_poly_geojson, geojson2coco

logger = logging.getLogger(__name__)


class GeoInference:
    
    """
    A class for performing geo inference on geospatial imagery using a pre-trained model.

    Args:
        model (str): The path or url to the model file
        work_dir (str): The directory where the model and output files will be saved.
        mask_to_vec (bool): Whether to convert the output mask to vector format.
        mask_to_coco (bool): Whether to convert the output mask to coco format.
        mask_to_yolo (bool): Whether to convert the output mask to yolo format.
        device (str): The device to use for inference (either "cpu" or "gpu").
        multi_gpu (bool): Whether to run the inference on multi-gpu or not.
        gpu_id (int): The ID of the GPU to use for inference (if device is "gpu").
        num_classes (int) : The number of classes in the output of the model.

    Attributes:
        work_dir (Path): The directory where the model and output files will be saved.
        device (str): The device to use for inference (either "cpu" or "gpu").
        model (str): The path or url to the model file.
        mask_to_vec (bool): Whether to convert the output mask to vector format.
        mask_to_coco (bool): Whether to convert the output mask to coco format.
        mask_to_yolo (bool): Whether to convert the output mask to yolo format.
        classes (int): The number of classes in the output of the model.
        raster_meta : The metadata of the input raster.

    """

    def __init__(
        self,
        model: str = None,
        work_dir: str = None,
        mask_to_vec: bool = False,
        mask_to_coco: bool = False,
        mask_to_yolo: bool = False,
        device: str = None,
        multi_gpu: bool = False,
        gpu_id: int = 0,
        num_classes: int = 5,
    ):
        self.work_dir: Path = get_directory(work_dir)
        self.device = (
            device if device == "cpu" else select_model_device(gpu_id, multi_gpu)
        )
        self.model = torch.jit.load(
            get_model(
                model_path_or_url=model,
                work_dir=self.work_dir,
            ),
            map_location=self.device,
        )
        self.mask_to_vec = mask_to_vec
        self.mask_to_coco = mask_to_coco
        self.mask_to_yolo = mask_to_yolo
        self.classes = num_classes
        self.raster_meta = None

    @torch.no_grad()
    def __call__(
        self,
        inference_input: Union[Path, str],
        bands_requested: List[str] = [],
        patch_size: int = 1024,
        bbox: str = None,
    ) -> None:
        
        async def run_async():
            
            # Start the periodic garbage collection task
            self.gc_task = asyncio.create_task(self.constant_gc(5))  # Calls gc.collect() every 5 seconds
            # Run the main computation asynchronously
            await self.async_run_inference(
                inference_input=inference_input,
                bands_requested=bands_requested,
                patch_size=patch_size,
                bbox=bbox
            )
            self.gc_task.cancel()
            
            try:
                await self.gc_task
            except asyncio.CancelledError:
                logger.info("The End of Inference")
        
        asyncio.run(run_async())

    async def async_run_inference(self,
        inference_input: Union[Path, str],
        bands_requested: List[str] = [],
        patch_size: int = 1024,
        bbox: str = None,
    ) -> None:
        
        """
        Perform geo inference on geospatial imagery using dask array.

        Args:
            inference_input Union[Path, str]: The path/url to the geospatial image to perform inference on.
            bands_requested List[str]: The requested bands to consider for the inference.
            patch_size (int): The size of the patches to use for inference.
            bbox (str): The bbox or extent of the image in this format "minx, miny, maxx, maxy".

        Returns:
            None

        """
        
        # configuring dask 
        try:
            config.set(scheduler='threads', num_workers=int(os.getenv('SLURM_CPUS_PER_TASK', 'Not available')) - 1)
            config.set(pool=ThreadPool(int(os.getenv('SLURM_CPUS_PER_TASK', 'Not available')) - 1))
        except ValueError:
            config.set(scheduler='threads', num_workers = os.cpu_count() - 1)
            config.set(pool=ThreadPool(os.cpu_count() -1))
        
        if not isinstance(inference_input, (str, Path)):
            raise TypeError(
                f"Invalid raster type.\nGot {inference_input} of type {type(inference_input)}"
            )
        if not isinstance(bands_requested, (Sequence, ListConfig)):
            raise ValueError(
                f"Requested bands should be a list."
                f"\nGot {bands_requested} of type {type(bands_requested)}"
            )
        if not isinstance(patch_size, int):
            raise TypeError(
                f"Invalid patch size. Patch size should be an integer..\nGot {patch_size}"
            )

        base_name = os.path.basename(
            Path(inference_input)
            if isinstance(inference_input, str)
            else inference_input
        )
        # it takes care of urls
        prefix_base_name = (
            base_name if not base_name.endswith(".tif") else base_name[:-4]
        )
        mask_path = self.work_dir.joinpath(prefix_base_name + "_mask.tif")
        polygons_path = self.work_dir.joinpath(prefix_base_name + "_polygons.geojson")
        yolo_csv_path = self.work_dir.joinpath(prefix_base_name + "_yolo.csv")
        coco_json_path = self.work_dir.joinpath(prefix_base_name + "_coco.json")
        stride_patch_size = int(patch_size / 2)

        """ Processing starts"""
        start_time = time.time()
        import rioxarray # type: ignore
        try:
            raster_stac_item = False
            if isinstance(inference_input, pystac.Item):
                raster_stac_item = True
            else:
                try:
                    pystac.Item.from_file(str(inference_input))
                    raster_stac_item = True
                except Exception:
                    raster_stac_item = False
            if not raster_stac_item:
                with rasterio.open(inference_input, "r") as src:
                    self.raster_meta = src.meta
                    self.raster = src
                aoi_dask_array = rioxarray.open_rasterio(inference_input, chunks=stride_patch_size)
                try:
                    if bands_requested:
                        raster_bands_request = [int(b) for b in bands_requested.split(",")]
                        if (
                            len(raster_bands_request) != 0
                            and len(raster_bands_request) != aoi_dask_array.shape[0]
                        ):
                            aoi_dask_array = xr.concat(
                                [aoi_dask_array[i - 1, :, :] for i in raster_bands_request],
                                dim="band"
                            )
                except Exception as e:
                    raise e
            else:
                assets = asset_by_common_name(inference_input)
                bands_requested = {
                    band: assets[band] for band in bands_requested.split(",")
                }
                rio_gdal_options = {
                    "GDAL_DISABLE_READDIR_ON_OPEN": "EMPTY_DIR",
                    "CPL_VSIL_CURL_ALLOWED_EXTENSIONS": ".tif",
                }
                all_bands_requested = []
                with rasterio.Env(**rio_gdal_options):
                    with rasterio.open(bands_requested[next(iter(bands_requested))]["meta"].href, "r") as src:
                        self.raster_meta = src.meta
                        self.raster = src
                    for key, value in bands_requested.items():
                        all_bands_requested.append(rioxarray.open_rasterio(value["meta"].href, chunks=stride_patch_size))
                aoi_dask_array = xr.concat(all_bands_requested, dim="band")
                del all_bands_requested

            if bbox is not None:
                bbox = tuple(map(float, bbox.split(", ")))
                roi_window = from_bounds(
                    left=bbox[0],
                    bottom=bbox[1],
                    right=bbox[2],
                    top=bbox[3],
                    transform=self.raster_meta["transform"],
                )
                col_off, row_off = roi_window.col_off, roi_window.row_off
                width, height = roi_window.width, roi_window.height
                aoi_dask_array = aoi_dask_array[
                    :, row_off : row_off + height, col_off : col_off + width
                ]
            self.original_shape = aoi_dask_array.shape
            # Pad the array to make dimensions multiples of the patch size
            pad_height = (
                stride_patch_size - aoi_dask_array.shape[1] % stride_patch_size
            ) % stride_patch_size
            pad_width = (
                stride_patch_size - aoi_dask_array.shape[2] % stride_patch_size
            ) % stride_patch_size
            print(aoi_dask_array.shape[0])
            aoi_dask_array = da.pad(
                aoi_dask_array.data,
                ((0, 0), (0, pad_height), (0, pad_width)),
                mode="constant",
            ).rechunk((aoi_dask_array.shape[0], stride_patch_size, stride_patch_size))


            # run the model
            aoi_dask_array = aoi_dask_array.map_overlap(
                runModel,
                model=self.model,
                patch_size=patch_size,
                device=self.device,
                chunks=(
                    self.classes + 1,
                    patch_size,
                    patch_size,
                ),
                depth={1: (0, stride_patch_size), 2: (0, stride_patch_size)},
                boundary="none",
                trim=False,
                dtype=np.float16,
            )
            aoi_dask_array = aoi_dask_array.map_overlap(
                sum_overlapped_chunks,
                chunk_size=patch_size,
                drop_axis=0,
                chunks=(
                    stride_patch_size,
                    stride_patch_size,
                ),
                depth={1: (stride_patch_size, 0), 2: (stride_patch_size, 0)},
                trim=False,
                boundary="none",
                dtype=np.uint8,
            )
            
        
            with ResourceProfiler(dt=1) as prof:
                with ProgressBar() as pbar:
                    pbar.register()
                    import rioxarray
                    logger.info("Inference is running:")
                    aoi_dask_array = xr.DataArray(aoi_dask_array[: self.original_shape[1], : self.original_shape[2]], dims=("y", "x"),attrs=xarray_profile_info(self.raster))
                    aoi_dask_array.rio.to_raster(mask_path, tiled=True, lock=threading.Lock())
                
            total_time = time.time() - start_time
            if self.mask_to_vec:
                mask_to_poly_geojson(mask_path, polygons_path)
                if self.mask_to_yolo:
                    gdf_to_yolo(polygons_path, mask_path, yolo_csv_path)
                if self.mask_to_coco:
                    geojson2coco(mask_path, polygons_path, coco_json_path)
            logger.info(
                "Extraction Completed in {:.0f}m {:.0f}s".format(
                    total_time // 60, total_time % 60
                )
            )
            torch.cuda.empty_cache()

        except Exception as e:
            print(f"Processing on the Dask cluster failed due to: {e}")
            raise e
    
    async def constant_gc(self,interval_seconds):
        while True:
            gc.collect()  # Call garbage collection
            await asyncio.sleep(interval_seconds)  # Wait for the specified interval

def main() -> None:
    arguments = cmd_interface()
    geo_inference = GeoInference(
        model=arguments["model"],
        work_dir=arguments["work_dir"],
        mask_to_vec=arguments["vec"],
        mask_to_coco=arguments["coco"],
        mask_to_yolo=arguments["yolo"],
        multi_gpu=arguments["multi_gpu"],
        device=arguments["device"],
        gpu_id=arguments["gpu_id"],
        num_classes=arguments["classes"],
    )
    geo_inference(
        inference_input=arguments["image"],
        bands_requested=arguments["bands_requested"],
        patch_size=arguments["patch_size"],
        bbox=arguments["bbox"],
    )


if __name__ == "__main__":
    main()
