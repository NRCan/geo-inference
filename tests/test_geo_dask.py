

import pytest
import numpy as np
import scipy.signal.windows as w
import torch



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

@pytest.fixture
def generate_corner_windows() -> np.ndarray:
    """
    Generates 9 2D signal windows that covers edge and corner coordinates

    Args:
        window_size (int): The size of the window.

    Returns:
        np.ndarray: 9 2D signal windows stacked in array (3, 3).
    """
    step = 4 >> 1
    window = np.matrix(w.hann(M=4, sym=False))
    window = window.T.dot(window)
    window_u = np.vstack(
        [np.tile(window[step : step + 1, :], (step, 1)), window[step:, :]]
    )
    window_b = np.vstack(
        [window[:step, :], np.tile(window[step : step + 1, :], (step, 1))]
    )
    window_l = np.hstack(
        [np.tile(window[:, step : step + 1], (1, step)), window[:, step:]]
    )
    window_r = np.hstack(
        [window[:, :step], np.tile(window[:, step : step + 1], (1, step))]
    )
    window_ul = np.block(
        [
            [np.ones((step, step)), window_u[:step, step:]],
            [window_l[step:, :step], window_l[step:, step:]],
        ]
    )
    window_ur = np.block(
        [
            [window_u[:step, :step], np.ones((step, step))],
            [window_r[step:, :step], window_r[step:, step:]],
        ]
    )
    window_bl = np.block(
        [
            [window_l[:step, :step], window_l[:step, step:]],
            [np.ones((step, step)), window_b[step:, step:]],
        ]
    )
    window_br = np.block(
        [
            [window_r[:step, :step], window_r[:step, step:]],
            [window_b[step:, :step], np.ones((step, step))],
        ]
    )
    return np.array(
        [
            [window_ul, window_u, window_ur],
            [window_l, window, window_r],
            [window_bl, window_b, window_br],
        ]
    )


class TestSumOverlappedChunks:
    
    def test_sum_overlapped_chunks_top_edge(
        self,
        mock_block_info_top_edge,
    ):
        from geo_inference import geo_dask as code

        arr = np.zeros((3, 6, 8))
        arr[0, :, :] = np.random.randint(low=1, high=5, size=(1, 6, 8))
        arr[1, :, :] = np.random.randint(low=1, high=5, size=(1, 6, 8))
        arr[2, :, :] = np.random.randint(low=1, high=5, size=(1, 6, 8))

        expected_result = np.divide(
            arr[:-1, :2, :2] + arr[:-1, :2, 2:4],
            arr[-1, :2, :2],
            out=np.zeros_like(arr[:-1, :2, :2], dtype=float),
            where=arr[:-1, :2, :2] != 0,
        )
        
        produced_result = code.sum_overlapped_chunks(arr, 4, 0.3, mock_block_info_top_edge)
        np.testing.assert_array_almost_equal(
            np.argmax(expected_result, axis=0), produced_result,decimal=2)

    def test_sum_overlapped_chunks_top_right_corner(
        self,
        mock_block_info_top_right_corner,
    ):
        from geo_inference import geo_dask as code

        arr = np.zeros((3, 6, 6))
        arr[0, :, :] = np.random.randint(low=1, high=5, size=(1, 6, 6))
        arr[1, :, :] = np.random.randint(low=1, high=5, size=(1, 6, 6))
        arr[2, :, :] = np.random.randint(low=1, high=5, size=(1, 6, 6))
        expected_result = np.divide(
            arr[:-1, :2, :2],
            arr[-1, :2, :2],
            out=np.zeros_like(arr[:-1, :2, :2], dtype=float),
            where=arr[:-1, :2, :2] != 0,
        )
        produced_result = code.sum_overlapped_chunks(
            arr, 4, 0.3, mock_block_info_top_right_corner
        )
        np.testing.assert_array_almost_equal(
            np.argmax(expected_result, axis=0), produced_result,decimal=2)

    def test_sum_overlapped_chunks_top_left_corner(
        self, mock_block_info_top_left_corner
    ):
        from geo_inference import geo_dask as code

        arr = np.zeros((3, 6, 6))
        arr[0, :, :] = np.random.randint(low=1, high=5, size=(1, 6, 6))
        arr[1, :, :] = np.random.randint(low=1, high=5, size=(1, 6, 6))
        arr[2, :, :] = np.random.randint(low=1, high=5, size=(1, 6, 6))
        expected_result = np.divide(
            arr[:-1, :2, :2],
            arr[-1, :2, :2],
            out=np.zeros_like(arr[:-1, :2, :2], dtype=float),
            where=arr[:-1, :2, :2] != 0,
        )
        produced_result = code.sum_overlapped_chunks(
            arr, 4, 0.3, mock_block_info_top_left_corner
        )
        np.testing.assert_array_almost_equal(
            np.argmax(expected_result, axis=0), produced_result,decimal=2)

    def test_sum_overlapped_chunks_bottom_right_corner(
        self,
        mock_block_info_bottom_right_corner,
    ):
        from geo_inference import geo_dask as code

        arr = np.zeros((3, 6, 6))
        arr[0, :, :] = np.random.randint(low=1, high=5, size=(1, 6, 6))
        arr[1, :, :] = np.random.randint(low=1, high=5, size=(1, 6, 6))

        arr[2, :, :] = np.random.randint(low=1, high=5, size=(1, 6, 6))
        expected_result = np.divide(
            arr[:-1, :2, :2],
            arr[-1, :2, :2],
            out=np.zeros_like(arr[:-1, :2, :2], dtype=float),
            where=arr[:-1, :2, :2] != 0,
        )
        produced_result = code.sum_overlapped_chunks(
            arr, 4, 0.3, mock_block_info_bottom_right_corner
        )
        np.testing.assert_array_almost_equal(
            np.argmax(expected_result, axis=0), produced_result,decimal=2)

    def test_sum_overlapped_chunks_bottom_left_corner(
        self, mock_block_info_bottom_left_corner
    ):
        from geo_inference import geo_dask as code

        arr = np.zeros((3, 6, 6))
        arr[0, :, :] = np.random.randint(low=1, high=5, size=(1, 6, 6))
        arr[1, :, :] = np.random.randint(low=1, high=5, size=(1, 6, 6))
        arr[2, :, :] = np.random.randint(low=1, high=5, size=(1, 6, 6))
        expected_result = np.divide(
            arr[:-1, :2, :2],
            arr[-1, :2, :2],
            out=np.zeros_like(arr[:-1, :2, :2], dtype=float),
            where=arr[:-1, :2, :2] != 0,
        )
        produced_result = code.sum_overlapped_chunks(
            arr, 4, 0.3, mock_block_info_bottom_left_corner
        )
        np.testing.assert_array_almost_equal(
            np.argmax(expected_result, axis=0), produced_result,decimal=2)

    def test_sum_overlapped_chunks_bottom_edge(
        self,
        mock_block_info_bottom_edge,
    ):
        from geo_inference import geo_dask as code

        arr = np.zeros((3, 6, 8))
        arr[0, :, :] = np.random.randint(low=1, high=5, size=(1, 6, 8))
        arr[1, :, :] = np.random.randint(low=1, high=5, size=(1, 6, 8))
        arr[2, :, :] = np.random.randint(low=1, high=5, size=(1, 6, 8))

        expected_result = np.divide(
            arr[:-1, :2, :2] + arr[:-1, :2, 2:4],
            arr[-1, :2, :2],
            out=np.zeros_like(arr[:-1, :2, :2], dtype=float),
            where=arr[:-1, :2, :2] != 0,
        )
        
        produced_result = code.sum_overlapped_chunks(
            arr, 4, 0.3, mock_block_info_bottom_edge
        )
        np.testing.assert_array_almost_equal(
            np.argmax(expected_result, axis=0), produced_result,decimal=2)

    def test_sum_overlapped_chunks_right_edge(
        self,
        mock_block_info_right_edge,
    ):
        from geo_inference import geo_dask as code

        arr = np.zeros((3, 6, 8))
        arr[0, :, :] = np.random.randint(low=1, high=5, size=(1, 6, 8))
        arr[1, :, :] = np.random.randint(low=1, high=5, size=(1, 6, 8))
        arr[2, :, :] = np.random.randint(low=1, high=5, size=(1, 6, 8))

        expected_result = np.divide(
            arr[:-1, :2, :2] + arr[:-1, 2:4, :2],
            arr[-1, :2, :2],
            out=np.zeros_like(arr[:-1, :2, :2], dtype=float),
            where=arr[:-1, :2, :2] != 0,
        )
        
        produced_result = code.sum_overlapped_chunks(arr, 4, 0.3, mock_block_info_right_edge)
        np.testing.assert_array_almost_equal(
            np.argmax(expected_result, axis=0), produced_result,decimal=2)

    def test_sum_overlapped_chunks_left_edge(
        self,
        mock_block_info_left_edge,
    ):
        from geo_inference import geo_dask as code

        arr = np.zeros((3, 6, 8))
        arr[0, :, :] = np.random.randint(low=1, high=5, size=(1, 6, 8))
        arr[1, :, :] = np.random.randint(low=1, high=5, size=(1, 6, 8))
        arr[2, :, :] = np.random.randint(low=1, high=5, size=(1, 6, 8))

        expected_result = np.divide(
            arr[:-1, :2, :2] + arr[:-1, 2:4, :2],
            arr[-1, :2, :2],
            out=np.zeros_like(arr[:-1, :2, :2], dtype=float),
            where=arr[:-1, :2, :2] != 0,
        )
        
        produced_result = code.sum_overlapped_chunks(arr, 4, 0.3, mock_block_info_left_edge)
        np.testing.assert_array_almost_equal(
            np.argmax(expected_result, axis=0), produced_result,decimal=2)


class TestModelInference:
    
    from unittest.mock import patch

    @patch("torch.jit.load")
    def test_run_model_inference_left_edge(self, mock_load, mock_block_info_left_edge, generate_corner_windows):
        from unittest.mock import MagicMock
        from geo_inference import geo_dask as code

        # Mocking parameters
        chunk_data = np.ones((3, 12, 12))
        mock_call_result = MagicMock()
        mock_cpu_result = MagicMock()
        mock_chunk_size = 4
        mock_num_classes = 3
        mock_numpy_result = np.full((1, 3, 4, 4), fill_value=1)  # Example NumPy array

        mock_cpu_result.numpy.return_value = mock_numpy_result
        mock_cpu_result.shape = mock_numpy_result.shape
        mock_call_result.cpu.return_value = mock_cpu_result

        # Mock TorchScript model and its methods
        mock_model = MagicMock(
            spec=torch.jit.ScriptModule, return_value=mock_call_result
        )
        mock_model.to.return_value = mock_model
        # Call the function under test
        output = code.runModel(
            chunk_data,
            mock_model,
            mock_chunk_size,
            "cpu",
            mock_num_classes,
            mock_block_info_left_edge,
        )
        assert np.array_equal(
            output[0, :, :], generate_corner_windows[2, 0, :, :]
        )
        assert output.shape[0] == mock_num_classes + 1
        assert output.shape[1:] == (mock_chunk_size, mock_chunk_size)
    
    @patch("torch.jit.load")
    def test_run_model_inference_left_edge_nodata(self, mock_load, mock_block_info_left_edge, generate_corner_windows):
        from unittest.mock import MagicMock
        from geo_inference import geo_dask as code

        # Mocking parameters
        chunk_data = np.zeros((3, 12, 12))
        mock_call_result = MagicMock()
        mock_cpu_result = MagicMock()
        mock_chunk_size = 4
        mock_num_classes = 3
        mock_numpy_result = np.full((1, 3, 4, 4), fill_value=1)  # Example NumPy array

        mock_cpu_result.numpy.return_value = mock_numpy_result
        mock_cpu_result.shape = mock_numpy_result.shape
        mock_call_result.cpu.return_value = mock_cpu_result

        # Mock TorchScript model and its methods
        mock_model = MagicMock(
            spec=torch.jit.ScriptModule, return_value=mock_call_result
        )
        mock_model.to.return_value = mock_model
        # Call the function under test
        output = code.runModel(
            chunk_data,
            mock_model,
            mock_chunk_size,
            "cpu",
            mock_num_classes,
            mock_block_info_left_edge,
        )

        assert np.array_equal(
            output[0, :, :], np.zeros((4,4))
        )
        assert output.shape[0] == mock_num_classes + 1
        assert output.shape[1:] == (mock_chunk_size, mock_chunk_size)
    
    @patch("torch.jit.load")
    def test_run_model_inference_right_edge(
        self, mock_load, mock_block_info_right_edge, generate_corner_windows
    ):
        from unittest.mock import MagicMock
        from geo_inference import geo_dask as code

        # Mocking parameters
        chunk_data = np.ones((3, 12, 12))
        mock_call_result = MagicMock()
        mock_cpu_result = MagicMock()
        mock_chunk_size = 4
        mock_num_classes = 3
        mock_numpy_result = np.full((1, 3, 4, 4), fill_value=1)  # Example NumPy array

        mock_cpu_result.numpy.return_value = mock_numpy_result
        mock_cpu_result.shape = mock_numpy_result.shape
        mock_call_result.cpu.return_value = mock_cpu_result

        # Mock TorchScript model and its methods
        mock_model = MagicMock(
            spec=torch.jit.ScriptModule, return_value=mock_call_result
        )
        mock_model.to.return_value = mock_model
        # Call the function under test
        output = code.runModel(
            chunk_data,
            mock_model,
            mock_chunk_size,
            "cpu",
            mock_num_classes,
            mock_block_info_right_edge,
        )
        assert np.array_equal(
            output[0, :, :], generate_corner_windows[2, 2, :, :]
        )
        assert output.shape[0] == mock_num_classes + 1
        assert output.shape[1:] == (mock_chunk_size, mock_chunk_size)

    @patch("torch.jit.load")
    def test_run_model_inference_bottom_edge(
        self, mock_load, mock_block_info_bottom_edge, generate_corner_windows
    ):
        from unittest.mock import MagicMock
        from geo_inference import geo_dask as code

        # Mocking parameters
        chunk_data = np.ones((3, 12, 12))
        mock_call_result = MagicMock()
        mock_cpu_result = MagicMock()
        mock_chunk_size = 4
        mock_num_classes = 3
        mock_numpy_result = np.full((1, 3, 4, 4), fill_value=1)  # Example NumPy array

        mock_cpu_result.numpy.return_value = mock_numpy_result
        mock_cpu_result.shape = mock_numpy_result.shape
        mock_call_result.cpu.return_value = mock_cpu_result

        # Mock TorchScript model and its methods
        mock_model = MagicMock(
            spec=torch.jit.ScriptModule, return_value=mock_call_result
        )
        mock_model.to.return_value = mock_model
        # Call the function under test
        output = code.runModel(
            chunk_data,
            mock_model,
            mock_chunk_size,
            "cpu",
            mock_num_classes,
            mock_block_info_bottom_edge,
        )
        assert np.array_equal(
            output[0, :, :], generate_corner_windows[2, 2, :, :]
        )
        assert output.shape[0] == mock_num_classes + 1
        assert output.shape[1:] == (mock_chunk_size, mock_chunk_size)
    
    @patch("torch.jit.load")
    def test_run_model_inference_bottom_left_corner(
        self, mock_load, mock_block_info_bottom_left_corner, generate_corner_windows
    ):
        from unittest.mock import MagicMock
        from geo_inference import geo_dask as code

        # Mocking parameters
        chunk_data = np.ones((3, 12, 12))
        mock_call_result = MagicMock()
        mock_cpu_result = MagicMock()
        mock_chunk_size = 4
        mock_num_classes = 3
        mock_numpy_result = np.full((1, 3, 4, 4), fill_value=2)  # Example NumPy array

        mock_cpu_result.numpy.return_value = mock_numpy_result
        mock_cpu_result.shape = mock_numpy_result.shape
        mock_call_result.cpu.return_value = mock_cpu_result

        # Mock TorchScript model and its methods
        mock_model = MagicMock(
            spec=torch.jit.ScriptModule, return_value=mock_call_result
        )
        mock_model.to.return_value = mock_model
        # Call the function under test
        output = code.runModel(
            chunk_data,
            mock_model,
            mock_chunk_size,
            "cpu",
            mock_num_classes,
            mock_block_info_bottom_left_corner,
        )
        assert np.array_equal(
            output[0, :, :], generate_corner_windows[2,0, :, :] * mock_numpy_result[0, 0, :, :]
        )
        assert np.array_equal(
            output[3, :, :], generate_corner_windows[2,0, :, :]
        )
        assert output.shape[0] == mock_num_classes + 1
        assert output.shape[1:] == (mock_chunk_size, mock_chunk_size)
    
    @patch("torch.jit.load")
    def test_run_model_inference_bottom_right_corner(
        self, mock_load, mock_block_info_bottom_right_corner, generate_corner_windows
    ):
        from unittest.mock import MagicMock
        from geo_inference import geo_dask as code

        # Mocking parameters
        chunk_data = np.ones((3, 12, 12))
        mock_call_result = MagicMock()
        mock_cpu_result = MagicMock()
        mock_chunk_size = 4
        mock_num_classes = 3
        mock_numpy_result = np.full((1, 3, 4, 4), fill_value=2)  # Example NumPy array

        mock_cpu_result.numpy.return_value = mock_numpy_result
        mock_cpu_result.shape = mock_numpy_result.shape
        mock_call_result.cpu.return_value = mock_cpu_result

        # Mock TorchScript model and its methods
        mock_model = MagicMock(
            spec=torch.jit.ScriptModule, return_value=mock_call_result
        )
        mock_model.to.return_value = mock_model
        # Call the function under test
        output = code.runModel(
            chunk_data,
            mock_model,
            mock_chunk_size,
            "cpu",
            mock_num_classes,
            mock_block_info_bottom_right_corner,
        )
        assert np.array_equal(
            output[0, :, :], generate_corner_windows[2,2, :, :] * mock_numpy_result[0, 0, :, :]
        )
        assert np.array_equal(
            output[3, :, :], generate_corner_windows[2,2, :, :]
        )
        assert output.shape[0] == mock_num_classes + 1
        assert output.shape[1:] == (mock_chunk_size, mock_chunk_size)

    @patch("torch.jit.load")
    def test_run_model_inference_top_left_corner(
        self, mock_load, mock_block_info_top_left_corner, generate_corner_windows
    ):
        from unittest.mock import MagicMock
        from geo_inference import geo_dask as code

        # Mocking parameters
        chunk_data = np.ones((3, 12, 12))
        mock_call_result = MagicMock()
        mock_cpu_result = MagicMock()
        mock_chunk_size = 4
        mock_num_classes = 3
        mock_numpy_result = np.full((1, 3, 4, 4), fill_value=2)  # Example NumPy array

        mock_cpu_result.numpy.return_value = mock_numpy_result
        mock_cpu_result.shape = mock_numpy_result.shape
        mock_call_result.cpu.return_value = mock_cpu_result

        # Mock TorchScript model and its methods
        mock_model = MagicMock(
            spec=torch.jit.ScriptModule, return_value=mock_call_result
        )
        mock_model.to.return_value = mock_model
        # Call the function under test
        output = code.runModel(
            chunk_data,
            mock_model,
            mock_chunk_size,
            "cpu",
            mock_num_classes,
            mock_block_info_top_left_corner,
        )
        assert np.array_equal(
            output[0, :, :], generate_corner_windows[0,0, :, :] * mock_numpy_result[0, 0, :, :]
        )
        assert np.array_equal(
            output[3, :, :], generate_corner_windows[0,0, :, :]
        )
        assert output.shape[0] == mock_num_classes + 1
        assert output.shape[1:] == (mock_chunk_size, mock_chunk_size)
    
    @patch("torch.jit.load")
    def test_run_model_inference_top_right_corner(
        self, mock_load, mock_block_info_top_right_corner, generate_corner_windows
    ):
        from unittest.mock import MagicMock
        from geo_inference import geo_dask as code

        # Mocking parameters
        chunk_data = np.ones((3, 12, 12))
        mock_call_result = MagicMock()
        mock_cpu_result = MagicMock()
        mock_chunk_size = 4
        mock_num_classes = 3
        mock_numpy_result = np.full((1, 3, 4, 4), fill_value=2)  # Example NumPy array

        mock_cpu_result.numpy.return_value = mock_numpy_result
        mock_cpu_result.shape = mock_numpy_result.shape
        mock_call_result.cpu.return_value = mock_cpu_result

        # Mock TorchScript model and its methods
        mock_model = MagicMock(
            spec=torch.jit.ScriptModule, return_value=mock_call_result
        )
        mock_model.to.return_value = mock_model
        # Call the function under test
        output = code.runModel(
            chunk_data,
            mock_model,
            mock_chunk_size,
            "cpu",
            mock_num_classes,
            mock_block_info_top_right_corner,
        )
        assert np.array_equal(
            output[0, :, :], generate_corner_windows[0,2, :, :] * mock_numpy_result[0, 0, :, :]
        )
        assert np.array_equal(
            output[3, :, :], generate_corner_windows[0,2, :, :]
        )
        assert output.shape[0] == mock_num_classes + 1
        assert output.shape[1:] == (mock_chunk_size, mock_chunk_size)

    @patch("torch.jit.load")
    def test_run_model_inference_top_edge(
        self, mock_load, mock_block_info_top_edge, generate_corner_windows
    ):
        from unittest.mock import MagicMock
        from geo_inference import geo_dask as code

        # Mocking parameters
        chunk_data = np.ones((3, 12, 12))
        mock_call_result = MagicMock()
        mock_cpu_result = MagicMock()
        mock_chunk_size = 4
        mock_num_classes = 3
        mock_numpy_result = np.full((1, 3, 4, 4), fill_value=2)  # Example NumPy array

        mock_cpu_result.numpy.return_value = mock_numpy_result
        mock_cpu_result.shape = mock_numpy_result.shape
        mock_call_result.cpu.return_value = mock_cpu_result

        # Mock TorchScript model and its methods
        mock_model = MagicMock(
            spec=torch.jit.ScriptModule, return_value=mock_call_result
        )
        mock_model.to.return_value = mock_model
        # Call the function under test
        output = code.runModel(
            chunk_data,
            mock_model,
            mock_chunk_size,
            "cpu",
            mock_num_classes,
            mock_block_info_top_edge,
        )
        assert np.array_equal(
            output[0, :, :], generate_corner_windows[0,2, :, :] * mock_numpy_result[0, 0, :, :]
        )
        assert np.array_equal(
            output[3, :, :], generate_corner_windows[0,2, :, :]
        )
        assert output.shape[0] == mock_num_classes + 1
        assert output.shape[1:] == (mock_chunk_size, mock_chunk_size)
    