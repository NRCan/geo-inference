import os
import re
import yaml
import torch
import logging
import requests
import tarfile

from pathlib import Path
from urllib.parse import urlparse

from config.logging_config import logger
logger = logging.getLogger(__name__)

USER_CACHE = Path.home().joinpath(".cache")
M0DEL_CONFIG = Path("geo_inference/config/models.yaml")

def is_tiff_path(path):
    # Check if the given path ends with .tiff or .tif (case insensitive)
    return re.match(r'.*\.(tiff|tif)$', path, re.IGNORECASE) is not None

def is_tiff_url(url):
    # Check if the URL ends with .tiff or .tif (case insensitive)
    parsed_url = urlparse(url)
    return re.match(r'.*\.(tiff|tif)$', os.path.basename(parsed_url.path), re.IGNORECASE) is not None


def validate_asset_type(image_asset: str):
    """Validate image asset type

    Args:
        image_asset (str): File path or URL to the image asset.

    Returns:
        str: file path or url.
    """
    if os.path.isfile(image_asset) and is_tiff_path(image_asset):
        return image_asset
    elif urlparse(image_asset).scheme in ('http', 'https') and is_tiff_url(image_asset):
        return image_asset
    return None

def calculate_gpu_stats(gpu_id: int = 0):
    """Calculate GPU stats

    Args:
        gpu_id (int, optional): GPU id. Defaults to 0.

    Returns:
        tuple(dict, dict): gpu stats.
    """
    res = {'gpu': torch.cuda.utilization(gpu_id)}
    torch_cuda_mem = torch.cuda.mem_get_info(gpu_id)
    mem = {
        'used': torch_cuda_mem[-1] - torch_cuda_mem[0],
        'total': torch_cuda_mem[-1]
    }    
    return res, mem

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

def extract_tar_gz(tar_gz_file: str or Path, target_directory: str or Path):
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

def get_device(device: str = "gpu",
               gpu_id: int = 0,
               gpu_max_ram_usage: int = 25,
               gpu_max_utilization: int = 15):
    """Returns a torch device

    Args:
        device (str): Accepts "cpu" or "gpu". Defaults to "gpu".
        gpu_id (int): GPU id. Defaults to 0.
        gpu_max_ram_usage (int): GPU max ram usage. Defaults to 25.
        gpu_max_utilization (int): GPU max utilization. Defaults to 15.

    Returns:
        torch.device: torch device.
    """
    if device == "cpu":
        return torch.device('cpu')
    elif device == "gpu":
        res, mem = calculate_gpu_stats(gpu_id=gpu_id)
        used_ram = mem['used'] / (1024 ** 2)
        max_ram = mem['total'] / (1024 ** 2)
        used_ram_percentage = (used_ram / max_ram) * 100
        logger.info(f"\nGPU RAM used: {round(used_ram_percentage)}%" 
                    f"[used_ram: {used_ram:.0f}MiB] [max_ram: {max_ram:.0f}MiB]\n"
                    f"GPU Utilization: {res['gpu']}%")
        if used_ram_percentage < gpu_max_ram_usage:
            if res["gpu"] < gpu_max_utilization:
                return torch.device(f"cuda:{gpu_id}")
            else:
                logger.warning(f"Reverting to CPU!\n"
                              f"Current GPU:{gpu_id} utilization: {res['gpu']}%\n"
                              f"Max GPU utilization allowed: {gpu_max_utilization}%")
                return torch.device('cpu')
        else:
            logger.warning(f"Reverting to CPU!\n"
                           f"Current GPU:{gpu_id} RAM usage: {used_ram_percentage}%\n"
                           f"Max used RAM allowed: {gpu_max_ram_usage}%")
            return torch.device('cpu')
    else:
        logger.error("Invalid device type requested: {device}")
        raise ValueError("Invalid device type")

def get_directory(work_directory: str)-> Path:
    """Returns a working directory

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

def get_model(model_name: str, work_dir: Path)-> Path:
    """Download a model from the model zoo

    Args:
        model_name (str): Model name.
        work_dir (Path): Working directory.

    Returns:
        Path: Path to the model file.
    """
    with open(M0DEL_CONFIG, "r") as f:
        model_config = yaml.safe_load(f.read())
        if model_name is not None and model_name in model_config.keys():
            model_dir = work_dir.joinpath(model_name)
            if not model_dir.is_dir():
                Path.mkdir(model_dir, parents=True)
            cached_file = model_dir.joinpath(model_name + ".pt")
            if not cached_file.is_file():
                access_token = os.environ["GEOSYS_TOKEN"]
                url = model_config[model_name]["url"]
                tar_asset = model_dir.joinpath(url.split('/')[-1])
                logger.info(f"Downloading model asset {tar_asset}")
                download_file_from_url(url, tar_asset, access_token=access_token)
                extract_tar_gz(tar_asset, model_dir)
    
    return cached_file

if __name__ == '__main__':
    pass
    