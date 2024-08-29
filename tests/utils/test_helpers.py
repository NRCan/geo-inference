import io
import os
import tarfile
import rasterio
from pathlib import Path
from unittest.mock import MagicMock, patch
import pytest
from geo_inference.utils.helpers import (calculate_gpu_stats,
                                         extract_tar_gz,
                                         get_directory, get_model,
                                         download_file_from_url,
                                         select_model_device,
                                         is_tiff_path, is_tiff_url, read_yaml,
                                         validate_asset_type,
                                         cmd_interface)

@pytest.fixture
def test_data_dir():
    return Path(__file__).parent.parent / "data"

@pytest.fixture
def temp_tar_gz_file(test_data_dir):
    tar_gz_content = b'test_content'
    tar_gz_file_path = test_data_dir / 'test.tar.gz'
    tif = str(test_data_dir / 'dummy.pt')
    with tarfile.open(tar_gz_file_path, 'w:gz') as tar:
        tarinfo = tarfile.TarInfo(tif)
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

def test_read_yaml(test_data_dir):
    config_path = str(test_data_dir / 'sample.yaml')
    result = read_yaml(config_path)
    assert result["arguments"] == {"image": "./data/areial.tiff",
                                   "bbox": "None",
                                   "model": "rgb-4class-segformer",
                                   "work_dir": "None",
                                   "patch_size": 1024,
                                   "workers": 0,
                                   "vec": False,
                                   "yolo": False,
                                   "coco": False,
                                   "device": "gpu",
                                   "gpu_id": 0,
                                   "bands_requested": '1,2,3',
                                    "mgpu": False,
                                    "classes": 5,
                                    "n_workers": 20
                                   }

def test_validate_asset_type(test_data_dir):
    local_tiff_path = str(test_data_dir / '0.tif')
    url_image = "http://example.com/test.tiff"
    invalid_tiff = str(test_data_dir / '0_corrupt.tif')
    with pytest.raises(ValueError, match="Invalid image_asset type"):
        validate_asset_type(123)
    with pytest.raises(ValueError, match="Invalid image_asset URL"):
        validate_asset_type(url_image)
    with pytest.raises(ValueError, match="Invalid image_asset file"):
        validate_asset_type(invalid_tiff)
    with rasterio.open(local_tiff_path) as dataset:
        assert validate_asset_type(dataset) == dataset
    with rasterio.open(local_tiff_path) as dataset:
        dataset.close()
        reopened_dataset = validate_asset_type(dataset)
        assert reopened_dataset.name == dataset.name
        assert not reopened_dataset.closed
    assert validate_asset_type(local_tiff_path).name == local_tiff_path
        
def test_calculate_gpu_stats():
    with patch('torch.cuda.utilization', return_value=50), patch('torch.cuda.mem_get_info', return_value=(500, 1000)):
        assert calculate_gpu_stats() == ({'gpu': 50}, {'used': 500, 'total': 1000})

def test_download_file_from_url(test_data_dir):
    with patch('requests.get') as mocked_get:
        mocked_get.return_value.status_code = 200
        mocked_get.return_value.iter_content.return_value = [b'test']
        download_path = str(test_data_dir / 'dummy.tif')
        download_file_from_url('http://example.com/dummy.tiff', download_path)
        os.remove(download_path)

def test_extract_tar_gz(temp_tar_gz_file, test_data_dir):
    extract_tar_gz(temp_tar_gz_file, test_data_dir)
    extracted_file_path = test_data_dir / 'dummy.pt'
    assert extracted_file_path.is_file()
    assert not temp_tar_gz_file.is_file()
    os.remove(extracted_file_path)

def test_get_device():
    with patch('geo_inference.utils.helpers.calculate_gpu_stats') as mock_calculate_gpu_stats:
        mock_calculate_gpu_stats.return_value = ({"gpu": 10}, {"used": 100, "total": 1024})
        device = select_model_device(gpu_id=1, multi_gpu=False)
        assert device == 'cpu'

def test_get_directory():
    with patch('pathlib.Path.is_dir', return_value=False), patch('pathlib.Path.mkdir'):
        assert get_directory('test') == Path('test')

def test_get_model_local_file(test_data_dir):
    model_file = test_data_dir / "inference" / "test_model" / "test_model.pt"
    model_path = get_model(str(model_file), test_data_dir)
    assert model_path == model_file

@patch('geo_inference.utils.helpers.download_file_from_url')
def test_get_model_url(mock_download_file_from_url, test_data_dir):
    mock_download_file_from_url.return_value = None
    model_path = get_model("https://example.com/test_model.pt", test_data_dir)
    assert model_path == test_data_dir / "test_model.pt"
    
def test_get_model_file_not_exists(test_data_dir):
    with pytest.raises(ValueError):
        get_model('nonexistent_model.pth', test_data_dir)

def test_cmd_interface_with_args(monkeypatch, test_data_dir):
    config_path = str(test_data_dir / 'sample.yaml')
    # Mock the command line arguments
    monkeypatch.setattr('sys.argv', ['prog', '-a', config_path])

    # Call the function
    result = cmd_interface()

    assert result == {"image": "./data/areial.tiff",
                      "bbox": None,
                      "bands_requested" : "1,2,3",
                      "model": "rgb-4class-segformer",
                      "work_dir": "None",
                      "workers": 0,
                      "vec": False,
                      "yolo": False,
                      "coco": False,
                      "device": "gpu",
                      "gpu_id": 0,
                      "classes": 5,
                      "multi_gpu": False,
                      "prediction_threshold": 0.3,
                      "patch_size": 1024
                      }

def test_cmd_interface_with_image(monkeypatch):
    # Mock the command line arguments
    monkeypatch.setattr('sys.argv', ['prog', '-i', 'image.tif'])
    # Call the function
    result = cmd_interface()
    # Assert the result
    assert result == {
        "image": "image.tif",
        "bbox": None,
        "bands_requested" : [],
        "workers": 0,
        "model": None,
        "work_dir": None,
        "patch_size": 1024,
        "vec": False,
        "yolo": False,
        "coco": False,
        "device": "gpu",
        "gpu_id": 0,
        "classes": 5,
        "prediction_threshold": 0.3,
        "multi_gpu": False,
    }

def test_cmd_interface_no_args(monkeypatch):
    # Mock the command line arguments
    monkeypatch.setattr('sys.argv', ['prog'])
    # Call the function and assert that it raises SystemExit
    with pytest.raises(SystemExit):
        cmd_interface()