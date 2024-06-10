import logging
import os
import sys
from typing import Any, Dict, Iterator, Optional, Tuple, Union, cast

import numpy as np
import rasterio as rio
import scipy.signal.windows as w
import torch
from rasterio.crs import CRS
from scipy.special import expit
from rasterio.windows import Window
from torch import Tensor
from torch.nn import functional as F
from torchgeo.datasets import GeoDataset
from torchgeo.datasets.utils import BoundingBox
from torchgeo.samplers import GeoSampler
from torchgeo.samplers.utils import _to_tuple, tile_to_chips

from .config.logging_config import logger
from .utils.helpers import validate_asset_type

logger = logging.getLogger(__name__)


class RasterDataset(GeoDataset):
    """
    A dataset class for raster data.
    
    Attributes:
        image_asset: The path to the image asset.
        src: The rasterio dataset object.
        cmap: The colormap of the image asset.
        crs: The coordinate reference system of the image asset.
        res: The resolution of the image asset.
        bands: The number of bands in the image asset.
        index: The rtree index of the image asset.
    """
    def __init__(self, image_asset: str, bbox: str = None) -> None:
        """Initializes a RasterDataset object.
        
        Args:
            image_asset (str): The path to the image asset.
            bounding_box (str): The bounding box of the image asset.
        """
        super().__init__()
        self.image_asset = validate_asset_type(image_asset)
        
        if self.image_asset is None:
            logger.error(f"image asset is neither a tiff image | tiff url")
            raise ValueError("Invalid image_asset type")
        
        with rio.open(self.image_asset) as src:
            self.src = src
            try:
                self.cmap = src.colormap(1)
            except ValueError:
                pass
            
            crs = src.crs
            res = src.res[0]
            bands = src.count
            image_height = src.height
            image_width = src.width
            minx, miny, maxx, maxy = src.bounds
        
            mint: float = 0
            maxt: float = sys.maxsize
            
            if bbox is None:
                bbox = BoundingBox(minx=minx, 
                                   miny=miny, 
                                   maxx=maxx, 
                                   maxy=maxy, 
                                   mint=mint, maxt=maxt)
            else:
                bbox = tuple(map(float, bbox.split(', ')))
                bbox = BoundingBox(minx=bbox[0], 
                                   miny=bbox[1], 
                                   maxx=bbox[2], 
                                   maxy=bbox[3], 
                                   mint=mint, maxt=maxt)
                
            coords = (minx, maxx, miny, maxy, mint, maxt)
            self.index.insert(0, coords, self.image_asset)
        
        self._crs = cast(CRS, crs)
        self.res = cast(float, res)
        self.bands = bands
        self.bbox = bbox
        self.image_height = image_height
        self.image_width = image_width
    
    def __getitem__(self, query: Dict[str, Any]) -> Dict[str, Any]:
        """
        Get a sample from the dataset.

        Args:
            query: A dictionary containing the query parameters.

        Returns:
            A dictionary containing the sample data.
        """
        filepath = query['path']
        window = query["window"]
        pixel_coords = query["pixel_coords"]
        patch_size = pixel_coords[-1]
        
        data = self._get_tensor(pixel_coords, patch_size)
        sample = {"image": data, 
                  "crs": self.crs,
                  "pixel_coords": pixel_coords, 
                  "window": window,
                  "path": filepath}

        return sample
    
    def _get_tensor(self, query, size):
        """
        Get a patch based on the given query (pixel coordinates).

        Args:
            query: The pixel coordinates of the patch.
            size: The desired patch size.

        Returns:
            A torch tensor patch.
        """
        (x_min, y_min, patch_width, patch_height) = query
        
        window = Window.from_slices(slice(y_min, y_min + patch_height),
                                    slice(x_min, x_min + patch_width))
        
        with rio.open(self.image_asset) as src:
            dest = src.read(window=window)
            if dest.dtype == np.uint16:
                dest = dest.astype(np.int32)
            elif dest.dtype == np.uint32:
                dest = dest.astype(np.int64)

            tensor = torch.tensor(dest)
            tensor = self.pad_patch(tensor, size)
        
        return tensor
    
    @staticmethod
    def pad_patch(x: Tensor, patch_size: int):
        """
        Pad the patch to desired patch_size.

        Args:
            x: The tensor patch to pad.
            patch_size: The desired patch size.

        Returns:
            The padded tensor patch.
        """
        h, w = x.shape[-2:]
        pad_h = patch_size - h
        pad_w = patch_size - w
        # pads are described starting from the last dimension and moving forward.
        x = F.pad(x, (0, pad_w, 0, pad_h))
        return x


class InferenceSampler(GeoSampler):
    """Class for creating an inference sampler.

    This class extends GeoSampler and is designed for generating patches
    for inference on a GeoDataset.

    Attributes:
        dataset (GeoDataset): The dataset to generate patches from.
        size (Union[Tuple[float, float], float]): Dimensions of each patch.
        stride (Union[Tuple[float, float], float]): Distance to skip between each patch.
        roi (Optional[BoundingBox]): Region of interest to sample from.
    """
    def __init__(self,
                 dataset: GeoDataset,
                 size: Union[Tuple[float, float], float],
                 stride: Union[Tuple[float, float], float],
                 roi: Optional[BoundingBox] = None,
                 ) -> None:
        """
        Initializes an InferenceSampler object.

        Args:
            dataset (GeoDataset): A GeoDataset object.
            size (Union[Tuple[float, float], float]): The size of the patch.
            stride (Union[Tuple[float, float], float]): The stride of the patch.
            roi (Optional[BoundingBox], optional): A BoundingBox object. Defaults to None.
        """
        super().__init__(dataset, roi)
        self.size = _to_tuple(size)
        self.patch_size = self.size
        self.stride = _to_tuple(stride)
        
        # Generates 9 2D signal windows of patch size that covers edge and corner coordinates
        self.windows = torch.tensor(self.generate_corner_windows(self.patch_size[0]), dtype=torch.float32)
        self.size_in_crs_units = (self.size[0] * self.res, self.size[1] * self.res)
        self.stride_in_crs_units = (self.stride[0] * self.res, self.stride[1] * self.res)
        self.hits = []
        self.hits_small = []
        for hit in self.index.intersection(tuple(self.roi), objects=True):
            bounds = BoundingBox(*hit.bounds)
            if (bounds.maxx - bounds.minx >= self.size_in_crs_units[1]
                and bounds.maxy - bounds.miny >= self.size_in_crs_units[0]):
                self.hits.append(hit)
            else:
                self.hits_small.append(hit)
        
        self.length = 0
        for hit in self.hits:
            bounds = BoundingBox(*hit.bounds)
            rows, cols = tile_to_chips(bounds=bounds, size=self.size_in_crs_units, stride=self.stride_in_crs_units)
            self.length += rows * cols
        
        for hit in self.hits_small:
            bounds = BoundingBox(*hit.bounds)
            self.length += 1
        
        for hit in self.hits + self.hits_small:
            if hit in self.hits:
                bounds = BoundingBox(*hit.bounds)
                self.im_height = round((bounds.maxy - bounds.miny) / self.res)
                self.im_width = round((bounds.maxx - bounds.minx) / self.res)
            else:
                bounds = BoundingBox(*hit.bounds) 
                self.im_height = round((bounds.maxy - bounds.miny) / self.res)
                self.im_width = round((bounds.maxx - bounds.minx) / self.res)
                
    def __iter__(self) -> Iterator[Dict[str, Any]]:
        """
        Yields a dictionary containing the pixel coordinates, path, and window.
        """
        for hit in self.hits + self.hits_small:
            if hit in self.hits:
                y_steps = int(np.ceil((self.im_height - self.size[0]) / self.stride[0]) + 1)
                x_steps = int(np.ceil((self.im_width - self.size[1]) / self.stride[1]) + 1)
                for y in range(y_steps):
                    if self.stride[0] * y + self.size[0] > self.im_height:
                        y_min = self.im_height - self.size[0]
                    else:
                        y_min = self.stride[0] * y
                    for x in range(x_steps):
                        if self.stride[1] * x + self.size[1] > self.im_width:
                            x_min = self.im_width - self.size[1]
                        else:
                            x_min = self.stride[1] * x
                        # get center, border and corner windows
                        border_x, border_y = 1, 1
                        if y == 0: border_x = 0
                        if x == 0: border_y = 0
                        if y == y_steps - 1: border_x = 2
                        if x == x_steps - 1: border_y = 2
                        # Select the right window
                        current_window = self.windows[border_x, border_y]
                        query = {"pixel_coords": (x_min, y_min, self.patch_size[1], self.patch_size[0]),
                                 "path": cast(str, hit.object),
                                 "window": current_window}
                        yield query
            else:
                x_min, y_min = (0, 0)
                current_window = torch.ones((self.patch_size[0], self.patch_size[1]))
                query = {"pixel_coords": (x_min, y_min, self.patch_size[1], self.patch_size[0]),
                         "path": cast(str, hit.object),
                         "window": current_window}
                yield query
    
    def __len__(self) -> int:
        """
        Returns the number of samples over the ROI.

        Returns:
            int: The number of patches that will be sampled.
        """
        return self.length
    
    @staticmethod
    def generate_corner_windows(window_size: int) -> np.ndarray:
        """
        Generates 9 2D signal windows that covers edge and corner coordinates
        
        Args:
            window_size (int): The size of the window.

        Returns:
            np.ndarray: 9 2D signal windows stacked in array (3, 3).
        """
        step = window_size >> 1
        window = np.matrix(w.hann(M=window_size, sym=False))
        window = window.T.dot(window)
        window_u = np.vstack([np.tile(window[step:step+1, :], (step, 1)), window[step:, :]])
        window_b = np.vstack([window[:step, :], np.tile(window[step:step+1, :], (step, 1))])
        window_l = np.hstack([np.tile(window[:, step:step+1], (1, step)), window[:, step:]])
        window_r = np.hstack([window[:, :step], np.tile(window[:, step:step+1], (1, step))])
        window_ul = np.block([[np.ones((step, step)), window_u[:step, step:]],
                              [window_l[step:, :step], window_l[step:, step:]]])
        window_ur = np.block([[window_u[:step, :step], np.ones((step, step))],
                              [window_r[step:, :step], window_r[step:, step:]]])
        window_bl = np.block([[window_l[:step, :step], window_l[:step, step:]],
                              [np.ones((step, step)), window_b[step:, step:]]])
        window_br = np.block([[window_r[:step, :step], window_r[:step, step:]],
                              [window_b[step:, :step], np.ones((step, step))]])
        return np.array([[window_ul, window_u, window_ur],
                         [window_l, window, window_r],
                         [window_bl, window_b, window_br]])
    

class InferenceMerge:
    """
    A class for merging inference results.
    
    Attributes:
        height (int): The padded height of roi.
        width (int): The padded width of roi.
        device (torch.device): The device to use for computation.
        image (np.ndarray): The merged image.
        norm_mask (np.ndarray): The normalization mask.
    
    """
    def __init__(self, 
                 height: int, 
                 width: int,
                 classes: int,  
                 device: torch.device) -> None:
        """
        Initializes a new instance of the InferenceMerge class.

        Args:
            height (int): The padded height of roi.
            width (int): The padded width of roi.
            device (torch.device): The device to use for computation.
        """
        self.height = height
        self.width = width
        self.classes = classes
        self.device = device
        self.image = np.zeros((self.classes, self.height, self.width), dtype=np.float16)
        self.norm_mask = np.ones((1, self.height, self.width), dtype=np.float16)
        # self.image = torch.zeros((self.classes, self.height, self.width), dtype=torch.float16, device=self.device)
        # self.norm_mask = torch.ones((1, self.height, self.width), dtype=torch.float16, device=self.device)
    
    @torch.no_grad()
    def merge_on_cpu(self, batch: torch.Tensor, windows: torch.Tensor, pixel_coords):
        """
        Merge the patches on CPU.

        Args:
            batch (torch.Tensor): The batch of inference results.
            windows (torch.Tensor): The windows used for inference.
            pixel_coords (list): The pixel coordinates of the patches.

        Returns:
            None
        """
        for output, window, (x, y, patch_width, patch_height) in zip(batch, windows, pixel_coords):
            # It is best to have these functions scripted in the model
            # if self.classes == 1:
            #     output = F.sigmoid(output) * window
            # else:
            #     output = F.softmax(output, dim=0) * window
            output = output * window
            self.image[:, y : y + patch_height, x : x + patch_width] += output.cpu().numpy()
            self.norm_mask[:, y : y + patch_height, x : x + patch_width] += window.cpu().numpy()
    
    @torch.no_grad()
    def merge_on_gpu(self,):
        """
        Merge the patches on GPU.
        """
        pass

    def save_as_tiff(self, 
                     height: int, 
                     width: int, 
                     output_meta: dict, 
                     output_path: os.PathLike) -> torch.Tensor:
        """
        Save mask to file. 

        Args:
            height (int): The height of the output mask.
            width (int): The width of the output mask.
            output_meta (dict): The meta data of the output mask.
            output_path (os.PathLike): The path to save the output mask.

        Returns:
            None
        """
        threshold = 0.5
        self.image /= self.norm_mask
        
        # Binary mask
        if self.image.shape[0] == 1:
            self.image = expit(self.image)
            self.image = np.where(self.image > threshold, 1, 0).squeeze(0).astype(np.uint8)
        else:
            self.image = np.argmax(self.image, axis=0).astype(np.uint8)
            # self.image = torch.argmax(self.image, dim=0).byte().cpu().numpy()
            
        self.image = self.image[np.newaxis, :height, :width]
        output_meta.update({"driver": "GTiff",
                            "height": self.image.shape[1],
                            "width": self.image.shape[2],
                            "count": self.image.shape[0],
                            "dtype": 'uint8',
                            "compress": 'lzw'})
        with rio.open(output_path, 'w+', **output_meta) as dest:
            dest.write(self.image)
        logger.info(f"Mask saved to {output_path}")