import argparse
import logging
import os
import re
import tarfile
from pathlib import Path
from urllib.parse import urlparse

import requests
import torch
import yaml

from ..config.logging_config import logger

logger = logging.getLogger(__name__)

USER_CACHE = Path.home().joinpath(".cache")
script_dir = Path(__file__).resolve().parent.parent
MODEL_CONFIG = script_dir / "config" /  "models.yaml"

def is_tiff_path(path: str):
    # Check if the given path ends with .tiff or .tif (case insensitive)
    return re.match(r'.*\.(tiff|tif)$', path, re.IGNORECASE) is not None

def is_tiff_url(url: str):
    # Check if the URL ends with .tiff or .tif (case insensitive)
    parsed_url = urlparse(url)
    return re.match(r'.*\.(tiff|tif)$', os.path.basename(parsed_url.path), re.IGNORECASE) is not None

def read_yaml(yaml_file_path: str or Path):
    with open(yaml_file_path, "r") as f:
        config = yaml.safe_load(f.read())
    return config

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

def get_model(model_path_or_url: str, work_dir: Path) -> Path:
    """Download a model from the model zoo

    Args:
        model_path_or_url (str): Path or url of model file.
        work_dir (Path): Working directory.

    Returns:
        Path: Path to the model file.
    """
    parsed_string = urlparse(model_path_or_url)
    if parsed_string.scheme:
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
    parser = argparse.ArgumentParser(usage="%(prog)s [-h HELP] use -h to get supported arguments.",
                                     description='Extract features from high-resolution imagery using pre-trained models.')
    
    parser.add_argument("-a", "--args", nargs=1, help="Path to arguments stored in yaml, consult ./config/sample_config.yaml")
    
    parser.add_argument("-i", "--image", nargs=1, help="Path to Geotiff")
    
    parser.add_argument("-bb", "--bbox", nargs=1, help="AOI bbox in this format'minx, miny, maxx, maxy'")
    
    parser.add_argument("-m", "--model", nargs=1, help="Path or URL to the model file")
    
    parser.add_argument("-wd", "--work_dir", nargs=1, help="Working Directory")
    
    parser.add_argument("-bs", "--batch_size", nargs=1, help="The Batch Size")
    
    parser.add_argument("-v", "--vec", nargs=1, help="Vector Conversion")
    
    parser.add_argument("-d", "--device", nargs=1, help="CPU or GPU Device")
    
    parser.add_argument("-id", "--gpu_id", nargs=1, help="GPU ID, Default = 0")
    
    args = parser.parse_args()
    
    if args.args:
        config = read_yaml(args.args[0])
        image = config["arguments"]["image"]
        bbox = config["arguments"]["bbox"]
        model = config["arguments"]["model"]
        work_dir = config["arguments"]["work_dir"]
        batch_size = config["arguments"]["batch_size"]
        vec = config["arguments"]["vec"]
        device = config["arguments"]["device"]
        gpu_id = config["arguments"]["gpu_id"]
    elif args.image:
        image = args.image[0]
        bbox = args.bbox[0] if args.bbox else None
        model = args.model[0] if args.model else None
        work_dir = args.work_dir[0] if args.work_dir else None
        batch_size = args.batch_size[0] if args.batch_size else 1
        vec = args.vec[0] if args.vec else False
        device = args.device[0] if args.device else "gpu"
        gpu_id = args.gpu_id[0] if args.gpu_id else 0
    else:
        print('use the help [-h] option for correct usage')
        raise SystemExit
    
    arguments= {"image": image,
                "bbox": bbox,
                "model": model,
                "work_dir": work_dir,
                "batch_size": batch_size,
                "vec": vec,
                "device": device,
                "gpu_id": gpu_id
                }
    return arguments
    
if __name__ == '__main__':
    pass
    