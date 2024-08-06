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
            "path": raster_dataset.src.name,
            "window": (0, 0, 10, 10),  # replace with actual window
            "pixel_coords": (0, 0, 10, 10),  # replace with actual pixel_coords
        }
        sample = raster_dataset.__getitem__(query)
        assert isinstance(sample, dict)
        assert "image" in sample
        assert "crs" in sample
        assert "pixel_coords" in sample
        assert "window" in sample
        assert "path" in sample

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
            assert "pixel_coords" in sample
            assert "path" in sample
            assert "window" in sample

    def test_len(self, inference_sampler):
        assert len(inference_sampler) == inference_sampler.length

    def test_generate_corner_windows(self):
        window_size = 10
        step = window_size >> 1
        windows = InferenceSampler.generate_corner_windows(window_size)
        center_window = np.matrix(w.hann(M=window_size, sym=False))
        center_window = center_window.T.dot(center_window)
        window_top = np.vstack(
            [
                np.tile(center_window[step : step + 1, :], (step, 1)),
                center_window[step:, :],
            ]
        )
        window_bottom = np.vstack(
            [
                center_window[:step, :],
                np.tile(center_window[step : step + 1, :], (step, 1)),
            ]
        )
        window_left = np.hstack(
            [
                np.tile(center_window[:, step : step + 1], (1, step)),
                center_window[:, step:],
            ]
        )
        window_right = np.hstack(
            [
                center_window[:, :step],
                np.tile(center_window[:, step : step + 1], (1, step)),
            ]
        )
        window_top_left = np.block(
            [
                [np.ones((step, step)), window_top[:step, step:]],
                [window_left[step:, :step], window_left[step:, step:]],
            ]
        )
        window_top_right = np.block(
            [
                [window_top[:step, :step], np.ones((step, step))],
                [window_right[step:, :step], window_right[step:, step:]],
            ]
        )
        window_bottom_left = np.block(
            [
                [window_left[:step, :step], window_left[:step, step:]],
                [np.ones((step, step)), window_bottom[step:, step:]],
            ]
        )
        window_bottom_right = np.block(
            [
                [window_right[:step, :step], window_right[:step, step:]],
                [window_bottom[step:, :step], np.ones((step, step))],
            ]
        )
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


@pytest.fixture
def mock_block_info_top_right_corner():
    # Mocking block_info with predefined values
    block_info = [{"num-chunks": [1, 3, 3], "chunk-location": [0, 0, 2]}]
    return block_info


@pytest.fixture
def mock_block_info_top_left_corner():
    # Mocking block_info with predefined values
    block_info = [{"num-chunks": [1, 3, 3], "chunk-location": [0, 0, 0]}]
    return block_info


@pytest.fixture
def mock_block_info_bottom_right_corner():
    # Mocking block_info with predefined values
    block_info = [{"num-chunks": [1, 3, 3], "chunk-location": [0, 2, 2]}]
    return block_info


@pytest.fixture
def mock_block_info_bottom_left_corner():
    # Mocking block_info with predefined values
    block_info = [{"num-chunks": [1, 3, 3], "chunk-location": [0, 2, 0]}]
    return block_info


@pytest.fixture
def mock_block_info_bottom_edge():
    # Mocking block_info with predefined values
    block_info = [{"num-chunks": [1, 3, 3], "chunk-location": [0, 2, 1]}]
    return block_info


@pytest.fixture
def mock_block_info_top_edge():
    # Mocking block_info with predefined values
    block_info = [{"num-chunks": [1, 3, 3], "chunk-location": [0, 0, 1]}]
    return block_info


@pytest.fixture
def mock_block_info_left_edge():
    # Mocking block_info with predefined values
    block_info = [{"num-chunks": [1, 3, 3], "chunk-location": [0, 1, 0]}]
    return block_info


@pytest.fixture
def mock_block_info_right_edge():
    # Mocking block_info with predefined values
    block_info = [{"num-chunks": [1, 3, 3], "chunk-location": [0, 1, 2]}]
    return block_info


class TestSumOverlappedChunks:
    def test_sum_overlapped_chunks_top_edge(
        self,
        mock_block_info_top_edge,
    ):
        from geo_inference import geo_dask as code

        arr = np.zeros((3, 6, 8))
        arr[0, :, :] = np.random.randint(low=1, high=5, size=(1, 6, 8))
        arr[1, :, :] = np.random.randint(low=1, high=5, size=(1, 6, 8))

        expected_result = arr[:, :2, :2] + arr[:, :2, 2:4]
        produced_result = code.sum_overlapped_chunks(arr, 4, mock_block_info_top_edge)
        np.testing.assert_array_almost_equal(
            np.argmax(expected_result, axis=0), produced_result
        )

    def test_sum_overlapped_chunks_top_right_corner(
        self,
        mock_block_info_top_right_corner,
    ):
        from geo_inference import geo_dask as code

        arr = np.zeros((3, 6, 6))
        arr[0, :, :] = np.random.randint(low=1, high=5, size=(1, 6, 6))
        arr[1, :, :] = np.random.randint(low=1, high=5, size=(1, 6, 6))

        expected_result = arr[:, :2, :2]
        produced_result = code.sum_overlapped_chunks(
            arr, 4, mock_block_info_top_right_corner
        )
        np.testing.assert_array_almost_equal(
            np.argmax(expected_result, axis=0), produced_result
        )

    def test_sum_overlapped_chunks_top_left_corner(
        self, mock_block_info_top_left_corner
    ):
        from geo_inference import geo_dask as code

        arr = np.zeros((3, 6, 6))
        arr[0, :, :] = np.random.randint(low=1, high=5, size=(1, 6, 6))
        arr[1, :, :] = np.random.randint(low=1, high=5, size=(1, 6, 6))

        expected_result = arr[:, :2, :2]
        produced_result = code.sum_overlapped_chunks(
            arr, 4, mock_block_info_top_left_corner
        )
        np.testing.assert_array_almost_equal(
            np.argmax(expected_result, axis=0), produced_result
        )

    def test_sum_overlapped_chunks_bottom_right_corner(
        self,
        mock_block_info_bottom_right_corner,
    ):
        from geo_inference import geo_dask as code

        arr = np.zeros((3, 6, 6))
        arr[0, :, :] = np.random.randint(low=1, high=5, size=(1, 6, 6))
        arr[1, :, :] = np.random.randint(low=1, high=5, size=(1, 6, 6))

        expected_result = arr[:, :2, :2]
        produced_result = code.sum_overlapped_chunks(
            arr, 4, mock_block_info_bottom_right_corner
        )
        np.testing.assert_array_almost_equal(
            np.argmax(expected_result, axis=0), produced_result
        )

    def test_sum_overlapped_chunks_bottom_left_corner(
        self, mock_block_info_bottom_left_corner
    ):
        from geo_inference import geo_dask as code

        arr = np.zeros((3, 6, 6))
        arr[0, :, :] = np.random.randint(low=1, high=5, size=(1, 6, 6))
        arr[1, :, :] = np.random.randint(low=1, high=5, size=(1, 6, 6))

        expected_result = arr[:, :2, :2]
        produced_result = code.sum_overlapped_chunks(
            arr, 4, mock_block_info_bottom_left_corner
        )
        np.testing.assert_array_almost_equal(
            np.argmax(expected_result, axis=0), produced_result
        )

    def test_sum_overlapped_chunks_bottom_edge(
        self,
        mock_block_info_bottom_edge,
    ):
        from geo_inference import geo_dask as code

        arr = np.zeros((3, 6, 8))
        arr[0, :, :] = np.random.randint(low=1, high=5, size=(1, 6, 8))
        arr[1, :, :] = np.random.randint(low=1, high=5, size=(1, 6, 8))

        expected_result = arr[:, :2, :2] + arr[:, :2, 2:4]
        produced_result = code.sum_overlapped_chunks(
            arr, 4, mock_block_info_bottom_edge
        )
        np.testing.assert_array_almost_equal(
            np.argmax(expected_result, axis=0), produced_result
        )

    def test_sum_overlapped_chunks_right_edge(
        self,
        mock_block_info_right_edge,
    ):
        from geo_inference import geo_dask as code

        arr = np.zeros((3, 8, 6))
        arr[0, :, :] = np.random.randint(low=1, high=5, size=(1, 8, 6))
        arr[1, :, :] = np.random.randint(low=1, high=5, size=(1, 8, 6))

        expected_result = arr[:, :2, :2] + arr[:, 2:4, :2]
        produced_result = code.sum_overlapped_chunks(arr, 4, mock_block_info_right_edge)
        np.testing.assert_array_almost_equal(
            np.argmax(expected_result, axis=0), produced_result
        )

    def test_sum_overlapped_chunks_left_edge(
        self,
        mock_block_info_left_edge,
    ):
        from geo_inference import geo_dask as code

        arr = np.zeros((3, 8, 6))
        arr[0, :, :] = np.random.randint(low=1, high=5, size=(1, 8, 6))
        arr[1, :, :] = np.random.randint(low=1, high=5, size=(1, 8, 6))

        expected_result = arr[:, :2, :2] + arr[:, 2:4, :2]
        produced_result = code.sum_overlapped_chunks(arr, 4, mock_block_info_left_edge)
        np.testing.assert_array_almost_equal(
            np.argmax(expected_result, axis=0), produced_result
        )


class TestModelInference:
    from unittest.mock import patch

    @patch("torch.jit.load")
    def test_run_model_inference(self, mock_load, mock_block_info_left_edge):
        from unittest.mock import MagicMock
        from geo_inference import geo_dask as code

        # Mocking parameters
        model_path = "dummy_model.pt"
        chunk_data = np.zeros((3, 12, 12))

        # Mock TorchScript model and its methods
        mock_model = MagicMock(spec=torch.jit.ScriptModule)
        mock_model.to.return_value = mock_model  # Mock the `to` method
        mock_model.device = torch.device("cuda:0")  # Mock the `device` attribute

        # Mock model's `cpu` method
        mock_model.cpu.return_value = mock_model

        # Mock model's `__call__` method (for inference)
        mock_model.__call__.return_value = MagicMock()
        mock_model.__call__.return_value.cpu.return_value.numpy.return_value = np.full(
            (3, 12, 12), fill_value=1
        )

        # Patch torch.jit.load to return the mock_model
        # Call the function under test
        output = code.runModel(
            chunk_data, 4, model_path, False, 0, 3, mock_block_info_left_edge
        )
        print(output)
