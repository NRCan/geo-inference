import io
import os
import tarfile
import rasterio
from pathlib import Path
from unittest.mock import MagicMock, patch
import pytest
import numpy as np
import xarray as xr
import rioxarray  # noqa: F401 (needed for .rio accessor)
from rasterio.transform import from_origin
from geo_inference.utils.helpers import (calculate_gpu_stats,
                                         extract_tar_gz,
                                         get_directory, get_model,
                                         download_file_from_url,
                                         select_model_device,
                                         is_tiff_path, is_tiff_url, read_yaml,
                                         validate_asset_type,
                                         cmd_interface, normalize_with_mask, has_internal_mask)

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
    assert result["arguments"] == {"image": "./data/0.tif",
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
                                   "bands_requested": [1,2,3],
                                    "mgpu": False,
                                    "classes": 5,
                                    "prediction_thr": 0.3,
                                    "transformers": False,
                                    "transformer_flip" : False,
                                    "transformer_rotate" : False
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
    assert Path(validate_asset_type(local_tiff_path).name) == Path(local_tiff_path)
        
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
        assert device == "cpu"

def test_get_directory():
    with patch('pathlib.Path.is_dir', return_value=False), patch('pathlib.Path.mkdir'):
        assert get_directory('test') == Path('test')

def test_get_model_local_file(test_data_dir):
    model_file = test_data_dir / "inference" / "test_model" / "cpu_scripted.pt"
    model_path = get_model(str(model_file), test_data_dir)
    assert model_path == model_file

@patch('geo_inference.utils.helpers.download_file_from_url')
def test_get_model_url(mock_download_file_from_url, test_data_dir):
    mock_download_file_from_url.return_value = None
    model_path = get_model("https://example.com/cpu_scripted.pt", test_data_dir)
    assert model_path == test_data_dir / "cpu_scripted.pt"
    
def test_get_model_file_not_exists(test_data_dir):
    with pytest.raises(ValueError):
        get_model('nonexistent_model.pth', test_data_dir)

def test_cmd_interface_with_args(monkeypatch, test_data_dir):
    config_path = str(test_data_dir / 'sample.yaml')
    # Mock the command line arguments
    monkeypatch.setattr('sys.argv', ['prog', '-a', config_path])

    # Call the function
    result = cmd_interface()

    assert result == {"image": "./data/0.tif",
                      "bbox": None,
                      "bands_requested" : [1,2,3],
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
                      "transformers": False,
                      "transformer_flip" : False,
                      "transformer_rotate" : False,
                      "patch_size": 1024
                      }

def test_cmd_interface_with_image(monkeypatch):
    # Mock the command line arguments
    monkeypatch.setattr('sys.argv', ['prog', '-i', 'image.tif', '-br', '1', '2', '3'])
    # Call the function
    result = cmd_interface()
    # Assert the result
    assert result == {
        "image": "image.tif",
        "bbox": None,
        "bands_requested" : ['1', '2', '3'],
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
        "transformers": False,
        "transformer_flip" : False,
        "transformer_rotate" : False,
        "multi_gpu": False,
    }

def test_cmd_interface_no_args(monkeypatch):
    # Mock the command line arguments
    monkeypatch.setattr('sys.argv', ['prog'])
    # Call the function and assert that it raises SystemExit
    with pytest.raises(SystemExit):
        cmd_interface()

def test_normalize_with_mask_integer_single_band():
    data = np.array([[np.nan, 5], [10, np.nan]], dtype=float)

    da = xr.DataArray(
        data,
        dims=("y", "x"),
    )

    out = normalize_with_mask(
        da,
        nodata_value=0,
        target_dtype=np.uint16,
    )

    out_vals = out.values

    # nodata replaced with 0
    assert out_vals[0, 0] == 0
    assert out_vals[1, 1] == 0

    # valid values shifted so min becomes 1
    assert out_vals[0, 1] == 1
    assert out_vals[1, 0] == 6

    # dtype preserved
    assert out.dtype == np.uint16

    # nodata metadata written
    assert out.rio.nodata == 0

def test_normalize_with_mask_float_single_band():
    data = np.array([[np.nan, 2.0], [4.0, np.nan]], dtype=float)

    da = xr.DataArray(
        data,
        dims=("y", "x"),
    )

    out = normalize_with_mask(
        da,
        nodata_value=0.0,
        target_dtype=np.float32,
    )

    out_vals = out.values
    eps = np.finfo(np.float32).eps

    # nodata replaced
    assert out_vals[0, 0] == 0.0
    assert out_vals[1, 1] == 0.0

    # valid values > 0 (shifted by min + eps)
    assert out_vals[0, 1] == pytest.approx(eps)
    assert out_vals[1, 0] == pytest.approx(2.0 + eps)

    assert out.dtype == np.float32
    assert out.rio.nodata == 0.0

def test_normalize_with_mask_multi_band():
    data = np.array(
        [
            [[np.nan, 1], [2, np.nan]],   # band 1
            [[np.nan, 10], [20, np.nan]], # band 2
        ],
        dtype=float,
    )

    da = xr.DataArray(
        data,
        dims=("band", "y", "x"),
        coords={"band": [1, 2]},
    )

    out = normalize_with_mask(
        da,
        nodata_value=0,
        target_dtype=np.uint16,
    )

    assert "band" in out.dims
    assert out.shape == da.shape

    # per-band normalization
    band1 = out.sel(band=1).values
    band2 = out.sel(band=2).values

    assert band1[0, 1] == 1
    assert band1[1, 0] == 2

    assert band2[0, 1] == 1
    assert band2[1, 0] == 11

def test_normalize_with_mask_requires_target_dtype():
    da = xr.DataArray(
        np.array([[1, 2], [3, 4]], dtype=float),
        dims=("y", "x"),
    )

    with pytest.raises(ValueError, match="target_dtype must be provided"):
        normalize_with_mask(da)


def test_has_internal_mask_true(tmp_path):
    tif = tmp_path / "with_dataset_mask.tif"

    data = np.ones((10, 10), dtype=np.uint8)

    with rasterio.open(
        tif,
        "w",
        driver="GTiff",
        height=10,
        width=10,
        count=1,
        dtype=data.dtype,
        transform=from_origin(0, 0, 1, 1),
    ) as ds:
        ds.write(data, 1)

        # Create a dataset-wide internal mask
        mask = np.ones((10, 10), dtype=np.uint8) * 255
        ds.write_mask(mask)

    assert has_internal_mask(str(tif)) is True


def test_has_internal_mask_false(tmp_path):
    tif = tmp_path / "without_mask.tif"

    data = np.ones((10, 10), dtype=np.uint8)

    with rasterio.open(
        tif,
        "w",
        driver="GTiff",
        height=10,
        width=10,
        count=1,
        dtype=data.dtype,
        transform=from_origin(0, 0, 1, 1),
    ) as ds:
        ds.write(data, 1)

    assert has_internal_mask(str(tif)) is False