from pathlib import Path
from typing import Any, Dict

import os
import pytest
import numpy as np
import scipy.signal.windows as w
import rasterio as rio
import torch
from geo_inference.geo_blocks import RasterDataset, InferenceSampler, InferenceMerge


@pytest.fixture
def test_data_dir():
    return Path(__file__).parent / "data"

@pytest.fixture
def raster_dataset(test_data_dir):
    image_asset = str(test_data_dir / "0.tif")
    return RasterDataset(image_asset)

class TestRasterDataset:
    def test_init(self, raster_dataset):
        assert isinstance(raster_dataset.src, rio.DatasetReader)
        assert raster_dataset.bands > 0
        assert raster_dataset.res > 0
        assert raster_dataset._crs is not None

    def test_getitem(self, raster_dataset):
        query: Dict[str, Any] = {
            'path': raster_dataset.image_asset,
            'window': (0, 0, 10, 10),  # replace with actual window
            'pixel_coords': (0, 0, 10, 10)  # replace with actual pixel_coords
        }
        sample = raster_dataset.__getitem__(query)
        assert isinstance(sample, dict)
        assert 'image' in sample
        assert 'crs' in sample
        assert 'pixel_coords' in sample
        assert 'window' in sample
        assert 'path' in sample

    def test_get_tensor(self, raster_dataset):
        query = (0, 0, 10, 10)  # replace with actual query
        size = 10  # replace with actual size
        tensor = raster_dataset._get_tensor(query, size)
        assert isinstance(tensor, torch.Tensor)
        assert tensor.shape[-2:] == (size, size)

    def test_pad_patch(self):
        x = torch.rand((3, 5, 5))
        patch_size = 10
        padded = RasterDataset.pad_patch(x, patch_size)
        assert isinstance(padded, torch.Tensor)
        assert padded.shape[-2:] == (patch_size, patch_size)


class TestInferenceSampler:
    @pytest.fixture
    def inference_sampler(self, raster_dataset):
        size = (10, 10)
        stride = (5, 5)
        return InferenceSampler(raster_dataset, size, stride)

    def test_init(self, inference_sampler):
        assert inference_sampler.size == (10, 10)
        assert inference_sampler.stride == (5, 5)
        assert inference_sampler.length > 0

    def test_iter(self, inference_sampler):
        for sample in inference_sampler:
            assert isinstance(sample, dict)
            assert 'pixel_coords' in sample
            assert 'path' in sample
            assert 'window' in sample

    def test_len(self, inference_sampler):
        assert len(inference_sampler) == inference_sampler.length

    def test_generate_corner_windows(self):
        window_size = 10
        step = window_size >> 1
        windows = InferenceSampler.generate_corner_windows(window_size)
        center_window = np.matrix(w.hann(M=window_size, sym=False))
        center_window = center_window.T.dot(center_window)
        window_top = np.vstack([np.tile(center_window[step:step + 1, :], (step, 1)), center_window[step:, :]])
        window_bottom = np.vstack([center_window[:step, :], np.tile(center_window[step:step + 1, :], (step, 1))])
        window_left = np.hstack([np.tile(center_window[:, step:step + 1], (1, step)), center_window[:, step:]])
        window_right = np.hstack([center_window[:, :step], np.tile(center_window[:, step:step + 1], (1, step))])
        window_top_left = np.block([[np.ones((step, step)), window_top[:step, step:]],
                                    [window_left[step:, :step], window_left[step:, step:]]])
        window_top_right = np.block([[window_top[:step, :step], np.ones((step, step))],
                                     [window_right[step:, :step], window_right[step:, step:]]])
        window_bottom_left = np.block([[window_left[:step, :step], window_left[:step, step:]],
                                       [np.ones((step, step)), window_bottom[step:, step:]]])
        window_bottom_right = np.block([[window_right[:step, :step], window_right[:step, step:]],
                                        [window_bottom[step:, :step], np.ones((step, step))]])
        assert isinstance(windows, np.ndarray)
        assert windows.shape == (3, 3, window_size, window_size)
        assert np.all(windows >= 0) and np.all(windows <= 1)
        assert np.allclose(windows[1, 1], center_window)
        assert np.allclose(windows[0, 1], window_top)
        assert np.allclose(windows[2, 1], window_bottom)
        assert np.allclose(windows[1, 0], window_left)
        assert np.allclose(windows[1, 2], window_right)
        assert np.allclose(windows[0, 0], window_top_left)
        assert np.allclose(windows[0, 2], window_top_right)
        assert np.allclose(windows[2, 0], window_bottom_left)
        assert np.allclose(windows[2, 2], window_bottom_right)

class TestInferenceMerge:
    @pytest.fixture
    def inference_merge(self):
        height = 100
        width = 100
        classes = 3
        device = torch.device('cpu')
        return InferenceMerge(height, width, classes, device)

    def test_init(self, inference_merge):
        assert inference_merge.height == 100
        assert inference_merge.width == 100
        assert inference_merge.classes == 3
        assert inference_merge.device == torch.device('cpu')
        assert isinstance(inference_merge.image, np.ndarray)
        assert isinstance(inference_merge.norm_mask, np.ndarray)

    def test_merge_on_cpu(self, inference_merge):
        batch = torch.rand((3, 3, 10, 10))
        windows = torch.rand((3, 1, 10, 10))
        pixel_coords = [(0, 0, 10, 10), (10, 10, 10, 10), (20, 20, 10, 10)]
        inference_merge.merge_on_cpu(batch, windows, pixel_coords)
        assert np.all(inference_merge.image >= 0)
        assert np.all(inference_merge.norm_mask >= 1)

    def test_save_as_tiff(self, inference_merge, test_data_dir):
        height = 100
        width = 100
        output_meta = {
            "crs": "+proj=latlong",
            "transform": rio.Affine(1.0, 0, 0, 0, 1.0, 0)
        }
        output_path = test_data_dir / "test_1.tiff"
        inference_merge.save_as_tiff(height, width, output_meta, output_path)
        assert output_path.exists()
        os.remove(output_path)