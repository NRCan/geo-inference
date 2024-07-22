import sys
import pims
import pystac
import torch
import logging
import pathlib
import rasterio

import numpy as np
import dask.array as da
import scipy.signal.windows as w
from rasterio.io import DatasetReader


from tqdm import tqdm
from pathlib import Path
from scipy.special import expit
from omegaconf import ListConfig
from pandas.io.common import is_url
from rasterio.windows import Window
from collections import OrderedDict
from pystac.extensions.eo import Band
from hydra.utils import to_absolute_path
from typing import Dict, Union, Sequence, Tuple, List
from dask_image.imread import _map_read_frame  # type: ignore

if str(Path(__file__).parents[0]) not in sys.path:
    sys.path.insert(0, str(Path(__file__).parents[0]))

from utils.helpers import get_device


logger = logging.getLogger(__name__)


class DaskAOI(object):
    """
    Object containing all data information about a single area of interest
    based on https://github.com/stac-extensions/ml-aoi
    """

    def __init__(
        self,
        raster: Union[Path, str],
        raster_bands_request: List[str],
        work_dir: str,
    ):
        """
        @param raster: pathlib.Path or str
            Path to source imagery
        @param raster_bands_request: list
            list of bands
        @param work_dir:
            root directory where dataset can be found or saved
        """
        """ ---------------------------Input Validation-------------------------------"""
        if not isinstance(raster, (str, Path)):
            raise TypeError(
                f"Raster path should be a string.\nGot {raster} of type {type(raster)}"
            )
        self.raster_raw_input = raster

        if not isinstance(raster_bands_request, (Sequence, ListConfig)):
            raise ValueError(
                f"Requested bands should be a list."
                f"\nGot {raster_bands_request} of type {type(raster_bands_request)}"
            )
        self.raster_bands_request = raster_bands_request

        if not isinstance(work_dir, str):
            raise TypeError(
                f"Work dir should be a string.\nGot {work_dir} of type {type(work_dir)}"
            )
        self.work_dir = Path(work_dir)

        """ -------------------------Processing---------------------------------"""
        if isinstance(self.raster_raw_input, pystac.Item):
            self.raster_stac_item = True
        else:
            try:
                pystac.Item.from_file(str(self.raster_raw_input))
                self.raster_stac_item = True
            except Exception:
                self.raster_stac_item = False

        # it constructs a list of URLs or file paths for raster bands: This function now returns a dict of {band:url}
        self.raster_parsed = self.parse_input_raster(
            csv_raster_str=self.raster_raw_input,
            raster_bands_requested=self.raster_bands_request,
        )
        logger.info(f"Successfully parsed Rasters: \n{self.raster_parsed}\n")

        self.raster_src_is_multiband = False
        if len(self.raster_parsed) == 1:
            raster_count = rasterio.open(next(iter(self.raster_parsed.values()))).count
            if raster_count > 1 and not self.raster_stac_item:
                self.raster_src_is_multiband = True
        else:
            if len(self.raster_parsed) == 3:
                desired_bands = (
                    ["R", "G", "B"]
                    if not self.raster_stac_item
                    else ["red", "green", "blue"]
                )
            elif len(self.raster_parsed) == 4:
                desired_bands = (
                    ["N", "R", "G", "B"]
                    if not self.raster_stac_item
                    else ["nir", "red", "green", "blue"]
                )
            self.raster_parsed = {
                key: self.raster_parsed[key]
                for key in desired_bands
                if key in self.raster_bands_request
            }

        for single_raster in self.raster_parsed.values():
            size = (
                Path(single_raster).stat().st_size
                if not is_url(single_raster)
                else None
            )
            logger.debug(
                f"Raster to validate: {raster}\n"
                f"Size: {size}\n"
                f"Extended check: {False}"
            )
        self.raster_parsed = self.raster_parsed

        rio_gdal_options = {
            "GDAL_DISABLE_READDIR_ON_OPEN": "EMPTY_DIR",
            "CPL_VSIL_CURL_ALLOWED_EXTENSIONS": ".tif",
        }
        with rasterio.Env(**rio_gdal_options):
            for tiff_band, raster in self.raster_parsed.items():
                with rasterio.open(raster, "r") as src:
                    bands_count = range(1, src.count + 1)
                    if self.raster_src_is_multiband and self.raster_bands_request:
                        bands_count = self.raster_bands_request
                    for band in bands_count:
                        self.raster = src.read(band)
                        self.high_or_low_contrast = self.is_low_contrast()
                self.raster = src

        self.num_bands = len(self.raster_parsed)
        logger.debug(self)

    def asset_by_common_name(self) -> Dict:
        """
        Get assets by common band name (only works for assets containing 1 band)
        Adapted from:
        https://github.com/sat-utils/sat-stac/blob/40e60f225ac3ed9d89b45fe564c8c5f33fdee7e8/satstac/item.py#L75
        @return:
        """
        _assets_by_common_name = OrderedDict()
        self.item = pystac.Item.from_file(self.raster_raw_input)
        for name, a_meta in self.item.assets.items():
            bands = []
            if "eo:bands" in a_meta.extra_fields.keys():
                bands = a_meta.extra_fields["eo:bands"]
            if len(bands) == 1:
                eo_band = bands[0]
                if "common_name" in eo_band.keys():
                    common_name = eo_band["common_name"]
                    if not Band.band_range(common_name):
                        raise ValueError(
                            f'Must be one of the accepted common names. Got "{common_name}".'
                        )
                    else:
                        _assets_by_common_name[common_name] = {
                            "meta": a_meta,
                            "name": name,
                        }
        if not _assets_by_common_name:
            raise ValueError("Common names for assets cannot be retrieved")
        return _assets_by_common_name

    @classmethod
    def from_dict(
        cls,
        aoi_dict,
        bands_requested: List = None,
        work_dir: str = None,
    ):
        """Instanciates an AOI object from an input-data dictionary as expected by geo-deep-learning"""
        if not isinstance(aoi_dict, dict):
            raise TypeError("Input data should be a dictionary.")
        if not {"tif", "gpkg", "split"}.issubset(set(aoi_dict.keys())):
            raise ValueError(
                "Input data should minimally contain the following keys: \n"
                "'tif', 'gpkg', 'split'."
            )
        new_aoi = cls(
            raster=aoi_dict["tif"],
            raster_bands_request=bands_requested,
            work_dir=work_dir,
        )
        return new_aoi

    def is_low_contrast(self, fraction_threshold=0.3):
        """This function checks if a raster is low contrast
        https://scikit-image.org/docs/stable/api/skimage.exposure.html#skimage.exposure.is_low_contrast
        Args:
            fraction_threshold (float, optional): low contrast fraction threshold. Defaults to 0.3.
        Returns:
            bool: False for high contrast image | True for low contrast image
        """
        data_type = self.raster.dtype
        grayscale = np.mean(self.raster, axis=0)
        grayscale = np.round(grayscale).astype(data_type)
        from skimage import exposure

        high_or_low_contrast = exposure.is_low_contrast(
            grayscale, fraction_threshold=fraction_threshold
        )
        return high_or_low_contrast

    def parse_input_raster(
        self,
        csv_raster_str: str,
        raster_bands_requested: Sequence,
    ) -> Union[Dict, Tuple]:
        """
        From input csv, determine if imagery is
        1. A Stac Item with single-band assets (multi-band assets not implemented)
        2. Single-band imagery as path or url with hydra-like interpolation for band identification
        3. Multi-band path or url
        @param csv_raster_str:
            input imagery to parse
        @param raster_bands_requested:
            dataset configuration parameters
        @return:
        """
        raster = {}
        if self.raster_stac_item:
            assets = self.asset_by_common_name()
            bands_requested = {band: assets[band] for band in self.raster_bands_request}
            for key, value in bands_requested.items():
                raster[key] = value["meta"].href
            return raster
        elif "${dataset.bands}" in csv_raster_str:
            if (
                not raster_bands_requested
                or not isinstance(raster_bands_requested, (List, ListConfig, tuple))
                or len(raster_bands_requested) == 0
            ):
                raise TypeError(
                    f"\nRequested bands should be a list of bands. "
                    f"\nGot {raster_bands_requested} of type {type(raster_bands_requested)}"
                )
            for band in raster_bands_requested:
                raster[band] = csv_raster_str.replace("${dataset.bands}", str(band))
            return raster
        else:
            try:
                self.validate_raster(csv_raster_str)
                return {"all_counts": csv_raster_str}
            except (FileNotFoundError, rasterio.RasterioIOError, TypeError) as e:
                logger.critical(f"Couldn't parse input raster. Got {csv_raster_str}")
                raise e

    @staticmethod
    def validate_raster(raster: rasterio.DatasetReader, extended: bool = False) -> None:
        """
        Checks if raster is valid, i.e. not corrupted (based on metadata, or actual byte info if under size threshold)
        @param raster: Path to raster to be validated
        @param extended: if True, raster data will be entirely read to detect any problem
        @return: if raster is valid, returns True, else False (with logger.critical)
        """
        if not raster:
            raise FileNotFoundError(f"No raster provided. Got: {raster}")

        try:
            if raster.meta["dtype"] not in [
                "uint8",
                "uint16",
            ]:  # will trigger exception if invalid raster
                logger.warning(
                    f"Only uint8 and uint16 are supported in current version.\n"
                    f"Datatype {raster.meta['dtype']} for {raster} may cause problems."
                )
            if extended:
                logger.debug(
                    f"Will perform extended check.\nWill read first band: {raster}"
                )
                window = Window(
                    raster.width - 200, raster.height - 200, raster.width, raster.height
                )
                print("**************************")
                print(window)
                raster_np = raster.read(1, window=window)
                logger.debug(raster_np.shape)
                if not np.any(raster_np):
                    logger.critical(
                        f"Raster data filled with zero values.\nRaster path: {raster}"
                    )
        except FileNotFoundError as e:
            logger.critical(
                f"Could not locate raster file.\nRaster path: {raster}\n{e}"
            )
            raise e
        except (rasterio.errors.RasterioIOError, TypeError) as e:
            logger.critical(f"\nRasterio can't open the provided raster: {raster}\n{e}")
            raise e


def aoi(
    aoi_dict: dict,
    bands_requested: List = [],
    work_dir: str = None,
):
    """
    Creates a single AOI from the provided tiff path of the csv file referencing input data.
    @param tiff_path:
        path to tiff file containing data.
    Returns: a single AOU object
    """
    try:
        new_aoi = DaskAOI.from_dict(
            aoi_dict=aoi_dict,
            bands_requested=bands_requested,
            work_dir=work_dir,
        )
        logger.debug(new_aoi)
    except FileNotFoundError as e:
        logger.error(f"Failed to create AOI due to {e}:\n{aoi_dict}\n")
    return new_aoi


def dask_imread_modified(
    fname,
    nframes=1,
    *,
    arraytype="numpy",
):
    """
    This function is a modification to the dask_image.imread.imread to handle the shape of tiff files read from URLs.
    source: https://image.dask.org/en/latest/_modules/dask_image/imread.html#imread
    """
    sfname = str(fname)
    with pims.open(sfname) as imgs:
        shape = (1,) + imgs.frame_shape
        dtype = np.dtype(imgs.pixel_type)
    ar = da.from_array([sfname] * shape[0], chunks=(nframes,))
    a = ar.map_blocks(
        _map_read_frame,
        chunks=da.core.normalize_chunks((nframes,) + shape[1:], shape),
        multiple_files=False,
        new_axis=list(range(1, len(shape))),
        arrayfunc=np.asanyarray,
        meta=np.asanyarray([]).astype(dtype),  # meta overwrites `dtype` argument
    )
    return a


def equalize_adapthist_enhancement(
    aoi_chunk: np.ndarray,
    clip_limit: int,
):
    """This function applies equalize_adapthist on each chunk of dask array
    each chunk is [band, height, width] --> So, the contrast enhancement is applied on each hand separately"""
    from kornia.enhance import equalize_clahe

    if aoi_chunk.size > 0 and aoi_chunk is not None:
        ready_np_array = aoi_chunk[0, :, :]
        ready_np_array = minmax_scale(
            img=ready_np_array, scale_range=(0, 1), orig_range=(0, 255)
        )
        ready_np_array = torch.as_tensor(ready_np_array[np.newaxis, ...])
        ready_np_array = (
            equalize_clahe(
                ready_np_array.float(),
                clip_limit=float(clip_limit),
                grid_size=(2, 2),
            )
            .cpu()
            .numpy()[0]
        )
        ready_np_array = (ready_np_array * 255).astype(np.int32)
        torch.cuda.empty_cache()
        return ready_np_array[np.newaxis, ...]
    return aoi_chunk  # If the chunk size is 0, return the original chunk


def minmax_scale(
    img,
    scale_range=(0, 1),
    orig_range=(0, 255),
):
    """
    Scale data values from original range to specified range.

    Args:
        img (numpy array) : Image to be scaled.
        scale_range: Desired range of transformed data (0, 1) or (-1, 1).
        orig_range: Original range of input data.

    Returns:
        Scaled image (np.ndarray)

    """
    if img.min() < orig_range[0] or img.max() > orig_range[1]:
        raise ValueError(
            f"Actual original range exceeds expected original range.\n"
            f"Expected: {orig_range}\n"
            f"Actual: ({img.min()}, {img.max()})"
        )
    o_r = orig_range[1] - orig_range[0]
    s_r = scale_range[1] - scale_range[0]
    if isinstance(img, (np.ndarray, torch.Tensor)):
        scale_img = (s_r * (img - orig_range[0]) / o_r) + scale_range[0]
    else:
        raise TypeError(f"Expected a numpy array or torch tensor, got {type(img)}")
    return scale_img


def runModel_partial_neighbor(
    chunk_data: np.ndarray,
    chunk_size: int,
    model_path: str,
    multi_gpu: bool = False,
    gpu_id: int = 0,
    num_classes: int = 5,
    block_info=None,
):
    """
    This function is for running the model on partial neighbor (The right and bottom neighbors).
    After running the model, depending on the location of chuck, it multiplies the chunk with a window and adds the windows to another dimension of the chunk
    This window is used to deal with edge artifact
    Args:
        chunk_data (np.ndarray) : This is a chunk of data in dask array.
        chunk_size (int) : The size of chunk data that we want to feed the model with.
        model_path (str) : The path to the scripted model.
        multi_gpu (boo) : Whether we have multiple gpus available or not.
        gpu_id (int) : The id of gpu is we are running on a single gpu.
        num_classes (int) : The number of classes that model outputs.
        block_info (none) : This input is having all the info about the chunk relative to the whole data (dask array).
    Returns:
        (np.ndarray)

    """
    num_chunks = block_info[0]["num-chunks"]
    chunk_location = block_info[0]["chunk-location"]

    if chunk_data.size > 0 and chunk_data is not None:
        try:
            # Defining the base window for window creation later
            step = chunk_size >> 1
            window = w.hann(M=chunk_size, sym=False)
            window = window[:, np.newaxis] * window[np.newaxis, :]
            final_window = np.empty((1, 1))
            chunk_data_ = chunk_data

            # Getting a (chunk_size, chunk_size) chunk with its window
            if chunk_location[2] == num_chunks[2] - 1 and chunk_location[1] == 0:
                """ If chunk is top right corner"""
                chunk_data_ = chunk_data_[
                    :,
                    :chunk_size,
                    :chunk_size,
                ]
                """ Top right corner window"""
                window_u = np.vstack(
                    [
                        np.tile(window[step : step + 1, :], (step, 1)),
                        window[step:, :],
                    ]
                )
                window_r = np.hstack(
                    [
                        window[:, :step],
                        np.tile(window[:, step : step + 1], (1, step)),
                    ]
                )
                final_window = np.block(
                    [
                        [window_u[:step, :step], np.ones((step, step))],
                        [window_r[step:, :step], window_r[step:, step:]],
                    ]
                )
            elif chunk_location[2] == num_chunks[2] - 1 and (
                chunk_location[1] > 0 and chunk_location[1] < num_chunks[1] - 1
            ):
                """ If chunk is on the right egde but not corners"""
                chunk_data_ = chunk_data[
                    :,
                    int(chunk_size / 2) : int(chunk_size / 2) + chunk_size,
                    :chunk_size,
                ]
                """ left egde window """
                final_window = np.hstack(
                    [
                        window[:, :step],
                        np.tile(window[:, step : step + 1], (1, step)),
                    ]
                )
            elif chunk_location[2] == num_chunks[2] - 1 and (
                chunk_location[1] == num_chunks[1] - 1
            ):
                """ If chunk is in the bottom right corner"""
                chunk_data_ = chunk_data_[
                    :,
                    :chunk_size,
                    :chunk_size,
                ]
                """ bottom right window """
                window_r = np.hstack(
                    [
                        window[:, :step],
                        np.tile(window[:, step : step + 1], (1, step)),
                    ]
                )
                window_b = np.vstack(
                    [
                        window[:step, :],
                        np.tile(window[step : step + 1, :], (step, 1)),
                    ]
                )
                final_window = np.block(
                    [
                        [window_r[:step, :step], window_r[:step, step:]],
                        [window_b[step:, :step], np.ones((step, step))],
                    ]
                )
            elif chunk_location[1] == num_chunks[1] - 1 and (
                chunk_location[2] > 0 and chunk_location[2] < num_chunks[2] - 1
            ):
                """ If chunk is on the bottom egde but not corners"""
                chunk_data_ = chunk_data_[
                    :,
                    :chunk_size,
                    int(chunk_size / 2) : int(chunk_size / 2) + chunk_size,
                ]
                """ bottom egde window"""
                final_window = np.vstack(
                    [
                        window[:step, :],
                        np.tile(window[step : step + 1, :], (step, 1)),
                    ]
                )
            elif chunk_location[1] == num_chunks[1] - 1 and chunk_location[2] == 0:
                """ If chunk is on the bottom left corner"""
                chunk_data_ = chunk_data_[
                    :,
                    :chunk_size,
                    :chunk_size,
                ]
                """ bottom left window"""
                window_l = np.hstack(
                    [
                        np.tile(window[:, step : step + 1], (1, step)),
                        window[:, step:],
                    ]
                )
                window_b = np.vstack(
                    [
                        window[:step, :],
                        np.tile(window[step : step + 1, :], (step, 1)),
                    ]
                )
                final_window = np.block(
                    [
                        [window_l[:step, :step], window_l[:step, step:]],
                        [np.ones((step, step)), window_b[step:, step:]],
                    ]
                )
            elif chunk_location[1] == 0 and chunk_location[2] == 0:
                """ If chunk is on the top left corner"""
                chunk_data_ = chunk_data[:, :chunk_size, :chunk_size]
                """ Top left window"""
                window_u = np.vstack(
                    [
                        np.tile(window[step : step + 1, :], (step, 1)),
                        window[step:, :],
                    ]
                )
                window_l = np.hstack(
                    [
                        np.tile(window[:, step : step + 1], (1, step)),
                        window[:, step:],
                    ]
                )
                final_window = np.block(
                    [
                        [np.ones((step, step)), window_u[:step, step:]],
                        [window_l[step:, :step], window_l[step:, step:]],
                    ]
                )
            elif chunk_location[2] == 0 and (
                chunk_location[1] > 0 and chunk_location[1] < num_chunks[1]
            ):
                """ If chunk is on the left edge but not corners"""
                chunk_data_ = chunk_data[
                    :,
                    int(chunk_size / 2) : int(chunk_size / 2) + chunk_size,
                    :chunk_size,
                ]
                """ top edge window"""
                final_window = np.hstack(
                    [
                        np.tile(window[:, step : step + 1], (1, step)),
                        window[:, step:],
                    ]
                )
            elif (chunk_location[2] > 0 and chunk_location[2] < num_chunks[2] - 1) and (
                chunk_location[1] == 0
            ):
                """ If chunk is on the top edge but not corners"""
                chunk_data_ = chunk_data[
                    :,
                    :chunk_size,
                    int(chunk_size / 2) : int(chunk_size / 2) + chunk_size,
                ]
                """ top edge window"""
                final_window = np.vstack(
                    [
                        np.tile(window[step : step + 1, :], (step, 1)),
                        window[step:, :],
                    ]
                )
            elif (chunk_location[1] > 0 and chunk_location[1] < num_chunks[1] - 1) and (
                chunk_location[2] > 0 and chunk_location[2] < num_chunks[2] - 1
            ):
                """ if the chunk is in the middle"""
                chunk_data_ = chunk_data[
                    :,
                    int(chunk_size / 2) : int(chunk_size / 2) + chunk_size,
                    int(chunk_size / 2) : int(chunk_size / 2) + chunk_size,
                ]
                final_window = window

            device_id = None
            if torch.cuda.is_available():
                if multi_gpu:
                    num_devices = torch.cuda.device_count()
                    for i in range(num_devices):
                        res = {"gpu": torch.cuda.utilization(i)}
                        torch_cuda_mem = torch.cuda.mem_get_info(i)
                        mem = {
                            "used": torch_cuda_mem[-1] - torch_cuda_mem[0],
                            "total": torch_cuda_mem[-1],
                        }
                        used_ram = mem["used"] / (1024**2)
                        max_ram = mem["total"] / (1024**2)
                        used_ram_percentage = (used_ram / max_ram) * 100
                        if used_ram_percentage < 70 and res["gpu"] < 70:
                            device_id = i
                            break
                    else:
                        device_id = gpu_id
            """ 
            if device_id is not None:
                device = torch.device(f"cuda:{device_id}")
            else:
                device = torch.device("cpu")
            """
            device = torch.device("cuda:0")
            model = torch.jit.load(model_path, map_location=device).to(
                device
            )  # load the model
            # convert chunck to torch Tensor
            tensor = torch.as_tensor(chunk_data_[np.newaxis, ...]).to(device)
            out = np.empty(
                shape=(num_classes, chunk_data_.shape[1], chunk_data_.shape[2])
            )  # Create the output but empty
            # run the model
            with torch.no_grad():
                out = model(tensor.to(model.device)).cpu().numpy()[0]
            del tensor
            del model  # Delete the model if it's not needed anymore

            if out.shape[1:] != final_window.shape:
                logger.error(
                    f" In runModel the shape of window and the array do not match"
                    f" The shape of window is {final_window.shape}"
                    f" The shape of array is {out.shape[1:]}"
                )
            if out.shape[1:] != (chunk_size, chunk_size):
                logger.error(
                    f" In runModel the shape of the array is not the expected shape"
                    f" The shape of array is {out.shape[1:]}"
                )
            else:
                return np.concatenate(
                    (out * final_window, final_window[np.newaxis, :, :]), axis=0
                )
        except Exception as e:
            logger.error(f"Error occured in IrunModel: {e}")
        finally:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()  # Release unused memory


def sum_overlapped_chunks(
    aoi_chunk: np.ndarray,
    chunk_size: int,
    block_info=None,
):
    """
    This function is running on each chunk of dask array in parallel.
    The chunks are overlapped with their neighbors and the purpose of running this function is to sum up the overlapped neighbors to deal with edge artifact.

    Args:
        aoi_chunk (np.ndarray): The overlapped chunk of dask array.
        chunk_size (int): The size for dask chunked data.
        block_info (None) : This input carries the information of dask chunked data relative to the whole dask array.

    Returns:
        summed_aoi_chunk (np.ndarray)

    """
    num_chunks = block_info[0]["num-chunks"]
    chunk_location = block_info[0]["chunk-location"]
    full_array = np.empty((1, 1))

    if aoi_chunk.size > 0 and aoi_chunk is not None:
        if (chunk_location[1] == 0 or chunk_location[1] == num_chunks[1] - 1) and (
            chunk_location[2] == 0 or chunk_location[2] == num_chunks[2] - 1
        ):
            """ All 4 corners"""
            full_array = aoi_chunk[
                :,
                : int(chunk_size / 2),
                : int(chunk_size / 2),
            ]
        elif (chunk_location[1] == 0 or chunk_location[1] == num_chunks[1] - 1) and (
            chunk_location[2] > 0 and chunk_location[2] < num_chunks[2] - 1
        ):
            """ Top and bottom edges but not corners"""
            full_array = (
                aoi_chunk[
                    :,
                    : int(chunk_size / 2),
                    int(chunk_size / 2) : int(chunk_size / 2) * 2,
                ]
                + aoi_chunk[
                    :,
                    : int(chunk_size / 2),
                    : int(chunk_size / 2),
                ]
            )
        elif (chunk_location[2] == 0 or chunk_location[2] == num_chunks[2] - 1) and (
            chunk_location[1] > 0 and chunk_location[1] < num_chunks[1] - 1
        ):
            """ Left and right edges but not corners"""
            full_array = (
                aoi_chunk[
                    :,
                    int(chunk_size / 2) : int(chunk_size / 2) * 2,
                    : int(chunk_size / 2),
                ]
                + aoi_chunk[
                    :,
                    : int(chunk_size / 2),
                    : int(chunk_size / 2),
                ]
            )
        elif (chunk_location[2] > 0 and chunk_location[2] < num_chunks[2] - 1) and (
            chunk_location[1] > 0 and chunk_location[1] < num_chunks[1] - 1
        ):
            """ Middle chunks """
            full_array = (
                aoi_chunk[
                    :,
                    : int(chunk_size / 2),
                    : int(chunk_size / 2),
                ]
                + aoi_chunk[
                    :,
                    : int(chunk_size / 2),
                    int(chunk_size / 2) : int(chunk_size / 2) * 2,
                ]
                + aoi_chunk[
                    :,
                    int(chunk_size / 2) : int(chunk_size / 2) * 2,
                    : int(chunk_size / 2),
                ]
                + aoi_chunk[
                    :,
                    int(chunk_size / 2) : int(chunk_size / 2) * 2,
                    int(chunk_size / 2) : int(chunk_size / 2) * 2,
                ]
            )

        if full_array.shape != (
            aoi_chunk.shape[0],
            int(chunk_size / 2),
            int(chunk_size / 2),
        ):
            logger.error(
                f" In sum_overlapped_chunks the shape of full_array is not {(6, int(chunk_size / 2), int(chunk_size / 2))}"
                f" The size of it {full_array.shape}"
            )
        else:
            with np.errstate(divide="ignore", invalid="ignore"):
                summed_aoi_chunk = np.divide(
                    full_array[:-1, :, :],
                    full_array[-1, :, :][np.newaxis, :, :],
                    out=np.zeros_like(full_array[:-1, :, :], dtype=float),
                    where=full_array[-1, :, :] != 0,
                )
                if summed_aoi_chunk.shape[0] == 1:
                    summed_aoi_chunk = expit(summed_aoi_chunk)
                    summed_aoi_chunk = (
                        np.where(summed_aoi_chunk > 0.5, 1, 0)
                        .squeeze(0)
                        .astype(np.uint8)
                    )
                else:
                    summed_aoi_chunk = np.argmax(summed_aoi_chunk, axis=0).astype(
                        np.uint8
                    )
                return summed_aoi_chunk


def write_inference_to_tiff(
    raster_reader: DatasetReader,
    mask_image: np.ndarray,
    mask_path: pathlib.Path,
):
    """
    Save mask to file.

    Args:
        raster_reader (DatasetReader) : The rasterio object.
        mask_image (np.ndarray): The output mask.
        mask_path (pathlib.Path) : The path to save the mask.
    Returns:
        None
    """
    mask_image = mask_image[np.newaxis, : mask_image.shape[0], : mask_image.shape[1]]
    raster_reader.meta.update(
        {
            "driver": "GTiff",
            "height": mask_image.shape[1],
            "width": mask_image.shape[2],
            "count": mask_image.shape[0],
            "dtype": "uint8",
            "compress": "lzw",
        }
    )
    with rasterio.open(
        mask_path,
        "w+",
        **raster_reader.meta,
    ) as dest:
        dest.write(mask_image)
    logger.info(f"Mask saved to {mask_path}")
