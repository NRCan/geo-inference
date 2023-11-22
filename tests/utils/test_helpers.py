import io
import os
import torch
import pytest
import tarfile 

from unittest.mock import patch, MagicMock, call
from pathlib import Path
from geo_inference.utils.helpers import is_tiff_path, is_tiff_url, read_yaml, validate_asset_type, calculate_gpu_stats, download_file_from_url, extract_tar_gz, get_device, get_directory, get_model, cmd_interface

@pytest.fixture
def temp_tar_gz_file():
    tmp_path = Path(__file__).parent.parent / "data"
    tar_gz_content = b'test_content'
    tar_gz_file_path = tmp_path / 'test.tar.gz'
    with tarfile.open(tar_gz_file_path, 'w:gz') as tar:
        tarinfo = tarfile.TarInfo('test.tiff')
        tarinfo.size = len(tar_gz_content)
        tar.addfile(tarinfo, fileobj=io.BytesIO(tar_gz_content))
    return tar_gz_file_path

def test_is_tiff_path():
    assert is_tiff_path('test.tiff') == True
    assert is_tiff_path('test.tif') == True
    assert is_tiff_path('test.jpg') == False

def test_is_tiff_url():
    assert is_tiff_url('http://example.com/test.tiff') == True
    assert is_tiff_url('http://example.com/test.tif') == True
    assert is_tiff_url('http://example.com/test.jpg') == False

def test_read_yaml():
    with patch('builtins.open', new_callable=MagicMock) as mock_open:
        mock_open.return_value.__enter__.return_value.read.return_value = "key: value"
        result = read_yaml('test.yaml')
        assert result == {"key": "value"}
        mock_open.assert_called_once_with('test.yaml', 'r')

def test_validate_asset_type():
    with patch('geo_inference.utils.helpers.is_tiff_path', return_value=True), patch('geo_inference.utils.helpers.is_tiff_url', return_value=True):
        assert validate_asset_type('test.tiff') == 'test.tiff'
        assert validate_asset_type('http://example.com/test.tiff') == 'http://example.com/test.tiff'

def test_calculate_gpu_stats():
    with patch('torch.cuda.utilization', return_value=50), patch('torch.cuda.mem_get_info', return_value=(500, 1000)):
        assert calculate_gpu_stats() == ({'gpu': 50}, {'used': 500, 'total': 1000})

def test_download_file_from_url():
    with patch('requests.get') as mocked_get:
        mocked_get.return_value.status_code = 200
        mocked_get.return_value.iter_content.return_value = [b'test']
        download_file_from_url('http://example.com/test.tiff', 'test.tiff')

def test_extract_tar_gz(temp_tar_gz_file):
    tmp_path = Path(__file__).parent.parent / "data"
    target_directory = tmp_path
    extract_tar_gz(temp_tar_gz_file, target_directory)
    extracted_file_path = tmp_path / 'test.tiff'
    assert extracted_file_path.is_file()
    assert not temp_tar_gz_file.is_file()

def test_get_device():
    with patch('geo_inference.utils.helpers.calculate_gpu_stats') as mock_calculate_gpu_stats:
        mock_calculate_gpu_stats.return_value = ({"gpu": 10}, {"used": 100, "total": 1024})
        device = get_device(device="gpu", gpu_id=1, gpu_max_ram_usage=20, gpu_max_utilization=10)
        assert device == torch.device('cpu')

def test_get_directory():
    with patch('pathlib.Path.is_dir', return_value=False), patch('pathlib.Path.mkdir'):
        assert get_directory('test') == Path('test')

def test_get_model_with_invalid_model():
    with patch('geo_inference.utils.helpers.read_yaml') as mock_read_yaml:
        mock_read_yaml.return_value = {"model1": {"url": "https://example.com/model1.tar.gz"}}
        with pytest.raises(ValueError, match="Invalid model name"):
            get_model("invalid_model", Path("work_dir"))

def test_get_model_with_missing_geosys_token():
    with patch('geo_inference.utils.helpers.read_yaml') as mock_read_yaml, \
         patch('geo_inference.utils.helpers.Path') as mock_path:
        mock_read_yaml.return_value = {"model1": {"url": "https://example.com/model1.tar.gz"}}
        mock_path.return_value.joinpath.return_value.is_dir.return_value = False
        mock_path.return_value.joinpath.return_value.is_file.return_value = False
        # Mocking os.environ to simulate missing GEOSYS_TOKEN
        with patch.dict(os.environ, clear=True):
            with pytest.raises(KeyError):
                get_model("model1", Path("work_dir"))