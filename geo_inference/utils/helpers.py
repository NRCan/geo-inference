import argparse
import logging
import os
import re
import tarfile
import rasterio
from pathlib import Path
from urllib.parse import urlparse
import requests
import torch
import yaml
import numpy as np
import xarray as xr
import subprocess
import csv
from tqdm import tqdm
from typing import Dict, Union
from hydra.utils import to_absolute_path
from pandas.io.common import is_url
from collections import OrderedDict
import pystac
from pystac.extensions.eo import Band
from pathlib import Path


from ..config.logging_config import logger

logger = logging.getLogger(__name__)

USER_CACHE = Path.home().joinpath(".cache")
script_dir = Path(__file__).resolve().parent.parent
MODEL_CONFIG = script_dir / "config" / "models.yaml"


def is_tiff_path(path: str):
    # Check if the given path ends with .tiff or .tif (case insensitive)
    return re.match(r".*\.(tiff|tif)$", path, re.IGNORECASE) is not None


def is_tiff_url(url: str):
    # Check if the URL ends with .tiff or .tif (case insensitive)
    parsed_url = urlparse(url)
    return (
        re.match(r".*\.(tiff|tif)$", os.path.basename(parsed_url.path), re.IGNORECASE)
        is not None
    )


def read_yaml(yaml_file_path: str | Path):
    with open(yaml_file_path, "r") as f:
        config = yaml.safe_load(f.read())
    return config


def validate_asset_type(image_asset: str):
    """Validate image asset type

    Args:
        image_asset (str): File path, Rasterio Dataset or URL of image asset.

    Returns:
        rasterio.io.DatasetReader: rasterio.io.DatasetReader.
    """
    if isinstance(image_asset, rasterio.io.DatasetReader):
        return (
            image_asset if not image_asset.closed else rasterio.open(image_asset.name)
        )

    if isinstance(image_asset, str):
        if urlparse(image_asset).scheme in ("http", "https") and is_tiff_url(
            image_asset
        ):
            try:
                return rasterio.open(image_asset)
            except rasterio.errors.RasterioIOError as e:
                logger.error(f"Failed to open URL {image_asset}: {e}")
                raise ValueError(f"Invalid image_asset URL: {image_asset}")
        if os.path.isfile(image_asset) and is_tiff_path(image_asset):
            try:
                return rasterio.open(image_asset)
            except rasterio.errors.RasterioIOError as e:
                logger.error(f"Failed to open file {image_asset}: {e}")
                raise ValueError(f"Invalid image_asset file: {image_asset}")

    logger.error(
        "Image asset is neither a valid TIFF image, Rasterio dataset, nor a valid TIFF URL."
    )
    raise ValueError("Invalid image_asset type")


def calculate_gpu_stats(gpu_id: int = 0):
    """Calculate GPU stats

    Args:
        gpu_id (int, optional): GPU id. Defaults to 0.

    Returns:
        tuple(dict, dict): gpu stats.
    """
    res = {"gpu": torch.cuda.utilization(gpu_id)}
    torch_cuda_mem = torch.cuda.mem_get_info(gpu_id)
    mem = {"used": torch_cuda_mem[-1] - torch_cuda_mem[0], "total": torch_cuda_mem[-1]}
    return res, mem


def extract_tar_gz(tar_gz_file: str | Path, target_directory: str | Path):
    """Extracts a tar.gz file to a target directory
    Args:
        tar_gz_file (str or Path): Path to the tar.gz file.
        target_directory (str o rPath): Path to the target directory.
    """
    try:
        with tarfile.open(tar_gz_file, "r:gz") as tar:
            for member in tar.getmembers():
                if member.isreg():
                    member.name = os.path.basename(member.name)
                    tar.extract(member, target_directory)
            # tar.extractall(path=target_directory)
        logger.info(f"Successfully extracted {tar_gz_file} to {target_directory}")
        os.remove(tar_gz_file)
    except tarfile.TarError as e:
        logger.error(f"Error while extracting {tar_gz_file}: {e}")
    except Exception as e:
        logger.error(f"An error occurred: {e}")
        raise


def get_directory(work_directory: str) -> Path:
    """
    Returns a working directory
    Args:
        work_directory (str): User's specified path

    Returns:
        Path: working directory
    """

    if work_directory:
        work_directory = Path(work_directory)
        if not work_directory.is_dir():
            Path.mkdir(work_directory, parents=True)
    else:
        work_directory = USER_CACHE.joinpath("geo-inference")
        if not work_directory.is_dir():
            Path.mkdir(work_directory, parents=True)

    return work_directory


def download_file_from_url(url, save_path, access_token=None):
    """Download a file from a URL
    
    Args:
        url (str): URL to the file.
        save_path (str or Path): Path to save the file.
        access_token (str, optional): Access token. Defaults to None.
    """
    try:
        headers = {}
        headers["Authorization"] = f"Bearer {access_token}"
        response = requests.get(url, headers=headers, stream=True)
        if response.status_code == 200:
            with open(save_path, 'wb') as file:
                for chunk in response.iter_content(chunk_size=128):
                    file.write(chunk)
            logger.info(f"Downloaded {save_path}")
        else:
            logger.error(f"Failed to download the file from {url}. Status code: {response.status_code}")
    except Exception as e:
        logger.error(f"An error occurred: {e}")
        raise


def get_model(model_path_or_url: str, work_dir: Path) -> Path:
    """Download a model from the model zoo

    Args:
        model_path_or_url (str): Path or url of model file.
        work_dir (Path): Working directory.

    Returns:
        Path: Path to the model file.
    """
    parsed_string = urlparse(model_path_or_url)
    if parsed_string.scheme and not os.path.exists(model_path_or_url):
        model_url = model_path_or_url
        model_name = os.path.basename(parsed_string.path)
        cached_file = work_dir.joinpath(model_name)
        if not cached_file.is_file():
            download_file_from_url(model_url, cached_file)
        return cached_file
    else:
        model_path = Path(model_path_or_url)
        if model_path.is_file():
            return model_path
        else:
            logger.error(f"Model {model_path_or_url} not found")
            raise ValueError("Invalid model path")


def select_model_device(gpu_id: int, multi_gpu: bool, device: str="cpu"):
    """
    Selects an appropriate GPU device based on memory usage and GPU utilization.

    The function checks if a GPU is available using `torch.cuda.is_available()` 
    and then evaluates either a single GPU or multiple GPUs based on the provided `multi_gpu` flag. 
    It analyzes memory and utilization for each available GPU, and selects a device that has memory 
    and utilization usage below a specific threshold.

    Parameters:
    -----------
    multi_gpu : bool
        If True, checks multiple GPUs and selects one with suitable memory and utilization stats.
    gpu_id : int
        The index of the GPU to evaluate when not in multi-GPU mode.    
    device : str
        The device string representing the current device (e.g., "cpu" or "cuda:X").

    Returns:
    --------
    device : str
        The updated device string, specifying which GPU to use (e.g., "cuda:0", "cuda:1", etc.), 
        or retains the original device (e.g., "cpu") if no suitable GPU is found.

    Logic:
    ------
    1. If a GPU is available and the device is not set to "cpu":
        - **Single GPU Mode (multi_gpu=False)**: 
            - Checks the specified `gpu_id`'s memory and utilization.
            - If the memory usage is below 70% and GPU utilization is below 70%, sets the device to the appropriate GPU.
        - **Multi-GPU Mode (multi_gpu=True)**:
            - Iterates over all available GPUs.
            - For each GPU, checks memory usage and utilization.
            - Selects the first GPU that has memory and utilization below the 70% threshold.
    2. If no GPU meets the criteria, retains the current device (usually "cpu").

    Note:
    -----
    - Memory is calculated as the difference between total and available memory (via `torch.cuda.mem_get_info`).
    - Both memory and GPU utilization thresholds are set to 70%.
    - The GPU utilization is retrieved via `torch.cuda.utilization()`.

    Exceptions:
    -----------
    - Assumes that GPU-related PyTorch functions like `torch.cuda.utilization()` and `torch.cuda.mem_get_info()` are available and accessible.
    - GPU-related functions will fail if run in an environment without CUDA support.

    """

    if torch.cuda.is_available() and device != "cpu":
        if not multi_gpu:
            res = {"gpu": torch.cuda.utilization(gpu_id)}
            torch_cuda_mem = torch.cuda.mem_get_info(gpu_id)
            mem = {
                "used": torch_cuda_mem[-1] - torch_cuda_mem[0],
                "total": torch_cuda_mem[-1],
            }
            used_ram = mem["used"] / (1024**2)
            max_ram = mem["total"] / (1024**2)
            used_ram_percentage = (used_ram / max_ram) * 100
            if used_ram_percentage < 70 and res["gpu"] < 70:
                device = f"cuda:{gpu_id}"
        else:
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
                    device = f"cuda:{i}"
                    break
    return device


def normalize_with_mask(
    da: xr.DataArray,
    nodata_value: int | float = 0,
    target_dtype: np.dtype | str | None = None,
):
    """
    Normalize a raster DataArray with an internal mask so that nodata is
    explicitly encoded as a numeric value that does not overlap with valid data.

    This function assumes the input raster was opened with
    ``rioxarray.open_rasterio(..., masked=True)``, meaning invalid pixels
    (from an internal mask or alpha band) are represented as NaN values.

    The normalization guarantees:
    - All invalid pixels are set to ``nodata_value`` (default: 0)
    - All valid pixels are strictly non-zero
    - The original data type is preserved

    The operation is performed lazily and is compatible with Dask-backed
    DataArrays.

    Parameters
    ----------
    da : xr.DataArray
        Input raster opened with ``masked=True``. Masked pixels must be NaN.
        The DataArray may be single-band or multi-band (``band`` dimension).
    nodata_value : int or float, optional
        Value to assign to invalid pixels. Defaults to 0.
        Must not overlap with valid data after normalization.
    target_dtype : numpy.dtype or str, optional
        Target dtype of the output raster (e.g. ``np.uint16``, ``np.float32``).
        Required to correctly determine integer vs floating-point behavior.

    Returns
    -------
    xr.DataArray
        Normalized raster with:
        - nodata explicitly set to ``nodata_value``
        - valid data shifted so that ``nodata_value`` is not a valid value
        - dtype preserved
        - ``rio.nodata`` metadata written

    Notes
    -----
    - Integer rasters are shifted so the minimum valid value becomes ``1``.
    - Floating-point rasters are shifted by ``min(valid) + eps`` to avoid zeros.
    - This function does *not* rescale value ranges; it only shifts them.
    - Per-band normalization is applied for multi-band rasters.

    Raises
    ------
    ValueError
        If ``target_dtype`` is not provided.
    """
    if target_dtype is None:
        raise ValueError("target_dtype must be provided to normalize_with_mask")

    if "band" in da.dims:
        return xr.concat(
            [
                _normalize_single_band(da.sel(band=b), nodata_value, target_dtype)
                for b in da.band.values
            ],
            dim="band",
        )

    return _normalize_single_band(da, nodata_value, target_dtype)


def _normalize_single_band(
    da: xr.DataArray,
    nodata_value: int | float,
    target_dtype: np.dtype | str,
):
    """
    Normalize a single-band DataArray so that nodata is encoded explicitly and
    does not overlap with valid values.

    Assumes invalid pixels are represented as NaN.
    """


    valid_min = da.where(~np.isnan(da)).min().compute()
    if np.issubdtype(target_dtype, np.integer):
        da = da - valid_min + 1
    else:
        eps = np.finfo(target_dtype).eps
        da = da - valid_min + eps
        
    da = da.fillna(nodata_value)
    da = da.astype(target_dtype)
    da = da.rio.write_nodata(nodata_value, inplace=False)

    return da


def has_internal_mask(raster: str) -> bool:
    """
    Check if a raster file has an internal GDAL dataset mask.

    This method uses `gdalinfo` to inspect the raster's mask flags.
    It returns True only if the raster contains a PER_DATASET internal mask,
    which is a true GDAL internal mask applied to the entire dataset.

    Parameters
    ----------
    raster : str
        Path to the raster file to check.

    Returns
    -------
    bool
        True if the raster has a PER_DATASET internal mask, False otherwise.

    Notes
    -----
    - This does not detect alpha bands (Mask Flags: ALPHA).
    - Uses the `gdalinfo` command-line utility, so GDAL must be installed.
    """
    output = subprocess.check_output(["gdalinfo", f"{raster}"]).decode()
    has_mask = "Mask Flags: PER_DATASET" in output

    return has_mask


def xarray_profile_info(
    raster_meta
):
    """
    Save mask to file.
    Args:
        raster : The meta data of the input raster.
    Returns:
        None
    """
    raster_meta["driver"] = 'GTiff' if raster_meta["driver"] == 'VRT' else raster_meta["driver"]
    profile_kwargs = {
        'crs': raster_meta["crs"],  # Coordinate Reference System, using src.crs.to_string() to get a string representation
        'transform': raster_meta['transform'],  # Affine transformation matrix
        'count': 1,  # Number of bands
        'width': raster_meta['width'],  # Width of the raster
        'height': raster_meta['height'],  # Height of the raster
        'driver': raster_meta["driver"],  # Raster format driver
        'dtype': "uint8",  # Data type (use dtype directly if it's a valid format for xarray)
        'BIGTIFF': 'YES',  # BigTIFF option
        'compress': 'lzw'  # Compression type
    }
    return profile_kwargs


def get_tiff_paths_from_csv(
    csv_path: Union[str, Path],
):
    """
    Creates list of to-be-processed tiff files from a csv file referencing input data
    Args:
        csv_path (Union[str, Path]) : path to csv file containing list of input data. See README for details on expected structure of csv.
    Returns:
        A list of tiff path
    """
    aois_dictionary = []
    data_list = read_csv(csv_path)
    logger.info(
        f"\n\tSuccessfully read csv file: {Path(csv_path).name}\n"
        f"\tNumber of rows: {len(data_list)}\n"
        f"\tCopying first row:\n{data_list[0]}\n"
    )
    with tqdm(
        enumerate(data_list), desc="Creating A list of tiff paths", total=len(data_list)
    ) as _tqdm:
        for i, aoi_dict in _tqdm:
            _tqdm.set_postfix_str(f"Image: {Path(aoi_dict['tif']).stem}")
            try:
                aois_dictionary.append(aoi_dict)
            except FileNotFoundError as e:
                logger.error(
                    f"{e}" f"Failed to get the path of :\n{aoi_dict}\n" f"Index: {i}"
                )
    return aois_dictionary


def asset_by_common_name(raster_raw_input) -> Dict:
    """
    Get assets by common band name (only works for assets containing 1 band)
    Adapted from:
    https://github.com/sat-utils/sat-stac/blob/40e60f225ac3ed9d89b45fe564c8c5f33fdee7e8/satstac/item.py#L75
    @return:
    """
    _assets_by_common_name = OrderedDict()
    item = pystac.Item.from_file(raster_raw_input)
    for name, a_meta in item.assets.items():
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


def read_csv(csv_file_name: str) -> Dict:
    """
    Open csv file and parse it, returning a list of dictionaries with keys:
    - "tif": path to a single image
    - "gpkg": path to a single ground truth file
    - dataset: (str) "trn" or "tst"
    - aoi_id: (str) a string id for area of interest
    @param csv_file_name:
        path to csv file containing list of input data with expected columns
        expected columns (without header): imagery, ground truth, dataset[, aoi id]
    source : geo_deep_learning
    """
    list_values = []
    with open(csv_file_name, "r") as f:
        reader = csv.reader(f)
        row_lengths_set = set()
        for row in reader:
            row_lengths_set.update([len(row)])
            if ";" in row[0]:
                raise TypeError(
                    "Elements in rows should be delimited with comma, not semicolon."
                )
            if not len(row_lengths_set) == 1:
                raise ValueError(
                    f"Rows in csv should be of same length. Got rows with length: {row_lengths_set}"
                )
            row = [str(i) or None for i in row]  # replace empty strings to None.
            row.extend(
                [None] * (4 - len(row))
            )  # fill row with None values to obtain row of length == 5

            row[0] = (
                to_absolute_path(row[0]) if not is_url(row[0]) else row[0]
            )  # Convert relative paths to absolute with hydra's util to_absolute_path()
            try:
                row[1] = str(to_absolute_path(row[1]) if not is_url(row[1]) else row[1])
            except TypeError:
                row[1] = None
            # save all values
            list_values.append(
                {"tif": str(row[0]), "gpkg": row[1], "split": row[2], "aoi_id": row[3]}
            )
    try:
        # Try sorting according to dataset name (i.e. group "train", "val" and "test" rows together)
        list_values = sorted(list_values, key=lambda k: k["split"])
    except TypeError:
        logger.warning("Unable to sort csv rows")
    return list_values


def cmd_interface(argv=None):
    """
    Parse command line arguments for extracting features from high-resolution imagery using pre-trained models.

    Args:
        argv (list): List of arguments to parse. If None, the arguments are taken from sys.argv.

    Returns:
        dict: A dictionary containing the parsed arguments.

    Raises:
        SystemExit: If the arguments are not valid.

    Usage:
        Use the -h option to get supported arguments.
    """
    parser = argparse.ArgumentParser(
        usage="%(prog)s [-h HELP] use -h to get supported arguments.",
        description="Extract features from high-resolution imagery using pre-trained models.",
    )

    parser.add_argument(
        "-a",
        "--args",
        nargs=1,
        help="Path to arguments stored in yaml, consult ./config/sample_config.yaml",
    )

    parser.add_argument(
        "-bb", "--bbox", nargs=1, help="AOI bbox in this format'minx, miny, maxx, maxy'"
    )

    parser.add_argument(
        "-br",
        "--bands_requested",
        nargs="*",
        help="bands_requested in this format '-b Red Green Blue' or '-br 1 2 3'",
    )

    parser.add_argument(
        "-i", "--image", nargs=1, help="Path or URL to the input image"
    )

    parser.add_argument("-m", "--model", nargs=1, help="Path or URL to the model file")

    parser.add_argument("-wd", "--work_dir", nargs=1, help="Working Directory")

    parser.add_argument("-ps", "--patch_size", type=int, nargs=1, help="The Patch Size")

    parser.add_argument("-w", "--workers", type=int, nargs=1, default=0, help="Numbers of workers")

    parser.add_argument("-v", "--vec", nargs=1, help="Vector Conversion")

    parser.add_argument("-mg", "--mgpu", nargs=1, help="Multi GPU")

    parser.add_argument("-cls", "--classes", type=int, nargs=1, help="Inference Classes")

    parser.add_argument("-y", "--yolo", nargs=1, help="Yolo Conversion")

    parser.add_argument("-c", "--coco", nargs=1, help="Coco Conversion")

    parser.add_argument("-d", "--device", nargs=1, help="CPU or GPU Device")

    parser.add_argument("-id", "--gpu_id", nargs=1, help="GPU ID", default = 0)
    
    parser.add_argument("-pr", "--prediction_thr", type=float, nargs=1, help="Prediction Threshold")
    
    parser.add_argument("-tr", "--transformers", nargs=1, help="Transformers Addition")
    parser.add_argument("-tr_f", "--transformer_flip", nargs=1, help="Transformers Addition - Flip")
    parser.add_argument("-tr_e", "--transformer_rotate", nargs=1, help="Transformers Addition - Rotate")
    
    args = parser.parse_args()

    if args.args:
        config = read_yaml(args.args[0])
        image = config["arguments"]["image"]
        model = config["arguments"]["model"]
        bbox = None if config["arguments"]["bbox"] == "None" else config["arguments"]["bbox"]
        work_dir = config["arguments"]["work_dir"]
        bands_requested = config["arguments"]["bands_requested"]
        workers = config["arguments"]["workers"]
        vec = config["arguments"]["vec"]
        yolo = config["arguments"]["yolo"]
        coco = config["arguments"]["coco"]
        device = config["arguments"]["device"]
        gpu_id = config["arguments"]["gpu_id"]
        multi_gpu = config["arguments"]["mgpu"]
        classes = config["arguments"]["classes"]
        patch_size = config["arguments"]["patch_size"]
        prediction_threshold = config["arguments"]["prediction_thr"]
        transformers = config["arguments"]["transformers"]
        transformer_flip = config["arguments"]["transformer_flip"]
        transformer_rotate = config["arguments"]["transformer_rotate"]

    elif args.image:
        image =args.image[0]
        model = args.model[0] if args.model else None
        bbox = args.bbox[0] if args.bbox else None
        work_dir = args.work_dir[0] if args.work_dir else None
        bands_requested = args.bands_requested if args.bands_requested else []
        workers = args.workers[0] if args.workers else 0
        vec = args.vec[0] if args.vec else False
        yolo = args.yolo[0] if args.yolo else False
        coco = args.coco[0] if args.coco else False
        device = args.device[0] if args.device else "gpu"
        gpu_id = args.gpu_id[0] if args.gpu_id else 0
        multi_gpu = args.mgpu[0] if args.mgpu else False
        classes = args.classes[0] if args.classes else 5
        patch_size = args.patch_size[0] if args.patch_size else 1024 
        prediction_threshold = args.prediction_thr[0] if args.prediction_thr else 0.3
        transformers = args.transformers[0] if args.transformers else False
        transformer_flip = args.transformer_flip if args.transformer_flip else False
        transformer_rotate = args.transformer_rotate if args.transformer_rotate else False
    
    else:
        print("use the help [-h] option for correct usage")
        raise SystemExit
    arguments = {
        "model": model,
        "image": image,
        "bands_requested": bands_requested,
        "workers": workers,
        "work_dir": work_dir,
        "classes": classes,
        "bbox": bbox,
        "multi_gpu": multi_gpu,
        "vec": vec,
        "yolo": yolo,
        "coco": coco,
        "device": device,
        "gpu_id": gpu_id,
        "patch_size": patch_size,
        "prediction_threshold": prediction_threshold,
        "transformers": transformers,
        "transformer_flip": transformer_flip,
        "transformer_rotate":transformer_rotate,
    }
    return arguments


if __name__ == "__main__":
    pass