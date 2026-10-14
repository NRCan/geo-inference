import dask.array as da
import numpy as np
import pytest
import scipy.signal.windows as w
import torch

from geo_inference import geo_dask as code


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


class ConstantModel(torch.nn.Module):
    """Returns per-class logits that are constant over the patch and record call batch sizes."""

    def __init__(self, logits):
        super().__init__()
        self.logits = torch.as_tensor(logits, dtype=torch.float32)
        self.batch_sizes = []

    def forward(self, x):
        self.batch_sizes.append(x.shape[0])
        b, _, h, wd = x.shape
        return self.logits.view(1, -1, 1, 1).expand(b, -1, h, wd).clone()


def block_info(num_chunks, location):
    return [{"num-chunks": [1, *num_chunks], "chunk-location": [0, *location]}]


class TestPatchWindow:
    @pytest.mark.parametrize("row", range(3))
    @pytest.mark.parametrize("col", range(3))
    def test_matches_legacy_edge_windows(self, generate_corner_windows, row, col):
        window = code.patch_window(4, row == 0, row == 2, col == 0, col == 2)
        np.testing.assert_allclose(window, generate_corner_windows[row, col], atol=1e-6)

    def test_single_patch_is_flat(self):
        np.testing.assert_allclose(code.patch_window(8, True, True, True, True), 1.0)

    def test_overlapping_windows_sum_to_one(self):
        # 50% overlapping Hann tapers form a partition of unity, so interior weights are even.
        win = code.patch_window(8, False, False, False, False)
        total = np.zeros((12, 12))
        for r in (0, 4):
            for c in (0, 4):
                total[r : r + 8, c : c + 8] += win
        np.testing.assert_allclose(total[4:8, 4:8], 1.0, atol=1e-6)


class TestRunModel:
    def run(self, chunk, model, k=2, batch_size=4, num_chunks=(3, 3), location=(1, 1), no_data=0):
        return code.runModel(
            chunk,
            model,
            ordered_input=[],
            patch_size=4,
            device="cpu",
            no_data=no_data,
            num_classes=3,
            block_info=block_info(num_chunks, location),
            patches_per_chunk=k,
            batch_size=batch_size,
        )

    def test_shape_and_dtype(self):
        out = self.run(np.ones((3, 6, 6)), ConstantModel([0.0, 1.0, 2.0]))
        assert out.shape == (4, 6, 6)
        assert out.dtype == np.float16

    def test_batches_are_capped(self):
        model = ConstantModel([0.0, 1.0, 2.0])
        self.run(np.ones((3, 6, 6)), model, batch_size=3)
        assert model.batch_sizes == [3, 1]

    def test_batch_size_does_not_change_result(self):
        rng = np.random.default_rng(0)
        chunk = rng.random((3, 6, 6))
        model = ConstantModel([0.5, 1.0, 2.0])
        np.testing.assert_array_equal(
            self.run(chunk, model, batch_size=1), self.run(chunk, model, batch_size=4)
        )

    def test_weights_are_overlap_added_windows(self):
        # Uniform logits give maximal entropy, so the confidence factor is the floor.
        out = self.run(np.ones((3, 6, 6)), ConstantModel([1.0, 1.0, 1.0]))
        win = code.patch_window(4, False, False, False, False) * code.ENTROPY_WEIGHT_FLOOR
        expected = np.zeros((6, 6))
        for r in (0, 2):
            for c in (0, 2):
                expected[r : r + 4, c : c + 4] += win
        np.testing.assert_allclose(out[-1], expected, rtol=1e-3, atol=1e-6)
        np.testing.assert_allclose(out[0], expected, rtol=1e-3, atol=1e-6)

    def test_edge_patches_use_flat_windows(self):
        # Top-left chunk: its first patch row and column touch the raster edge.
        out = self.run(np.ones((3, 6, 6)), ConstantModel([1.0, 1.0, 1.0]), location=(0, 0))
        assert out[-1, 0, 0] > 0

    def test_last_chunk_has_one_fewer_patch(self):
        model = ConstantModel([0.0, 1.0, 2.0])
        out = self.run(np.ones((3, 4, 4)), model, location=(2, 2))
        assert sum(model.batch_sizes) == 1
        assert out.shape == (4, 6, 6)
        assert not out[:, 4:, :].any() and not out[:, :, 4:].any()

    def test_nodata_patches_are_skipped(self):
        model = ConstantModel([0.0, 1.0, 2.0])
        chunk = np.zeros((3, 6, 6))
        chunk[:, 4:, 4:] = 1  # only the bottom-right patch has data
        out = self.run(chunk, model)
        assert sum(model.batch_sizes) == 1
        assert not out[-1, :2, :2].any()

    def test_all_nodata_returns_zeros_without_calling_model(self):
        model = ConstantModel([0.0, 1.0, 2.0])
        out = self.run(np.full((3, 6, 6), np.nan), model, no_data=np.nan)
        assert model.batch_sizes == []
        assert not out.any()

    def test_wrong_class_count_returns_zeros(self):
        out = self.run(np.ones((3, 6, 6)), ConstantModel([0.0, 1.0]))
        assert not out.any()


class TestSumOverlappedChunks:
    def chunk(self, top, left, k=2, step=2, channels=3, fill=None):
        size = (k + 1) * step
        rng = np.random.default_rng(1)
        arr = rng.random((channels, top + size, left + size)) if fill is None else fill
        arr[-1] = 1.0
        return arr

    def test_interior_chunk_adds_top_left_spill(self):
        arr = self.chunk(2, 2)
        result = code.sum_overlapped_chunks(arr, 4, patches_per_chunk=2)
        own = arr[:, 2:6, 2:6].copy()
        own[:, :2, :] += arr[:, :2, 2:6]
        own[:, :, :2] += arr[:, 2:6, :2]
        own[:, :2, :2] += arr[:, :2, :2]
        np.testing.assert_array_equal(result, np.argmax(own[:-1] / own[-1], axis=0))

    def test_first_chunk_has_no_spill(self):
        arr = self.chunk(0, 0)
        result = code.sum_overlapped_chunks(arr, 4, patches_per_chunk=2)
        np.testing.assert_array_equal(result, np.argmax(arr[:-1, :4, :4], axis=0))

    def test_first_row_takes_left_spill_only(self):
        arr = self.chunk(0, 2)
        result = code.sum_overlapped_chunks(arr, 4, patches_per_chunk=2)
        own = arr[:, :4, 2:6].copy()
        own[:, :, :2] += arr[:, :4, :2]
        np.testing.assert_array_equal(result, np.argmax(own[:-1] / own[-1], axis=0))

    def test_binary_threshold(self):
        arr = np.zeros((2, 6, 6))
        arr[0, :, :3] = 0.9
        arr[0, :, 3:] = 0.1
        arr[-1] = 1.0
        result = code.sum_overlapped_chunks(arr, 4, 0.3, patches_per_chunk=2)
        np.testing.assert_array_equal(result[:, :3], 1)
        np.testing.assert_array_equal(result[:, 3:], 0)

    def test_zero_weight_gives_class_zero(self):
        arr = np.ones((3, 6, 6))
        arr[-1] = 0.0
        result = code.sum_overlapped_chunks(arr, 4, patches_per_chunk=2)
        assert not result.any()

    def test_unexpected_shape_returns_zeros(self):
        result = code.sum_overlapped_chunks(np.ones((3, 5, 7)), 4, patches_per_chunk=2)
        assert result.shape == (4, 4) and not result.any()


@pytest.mark.parametrize("patches_per_chunk", [1, 2, 3])
def test_pipeline_matches_unchunked_reference(patches_per_chunk):
    """The two map_overlap stages equal a plain overlap-add over the whole raster."""
    patch, step, k, classes = 4, 2, patches_per_chunk, 3
    rng = np.random.default_rng(2)
    cells = -(-(3 * k + 1) // k) * k  # whole chunks, with at least one spare stride
    image = rng.random((3, cells * step, cells * step)).astype(np.float32)
    model = ConstantModel([0.2, 0.5, 0.1])

    def logits(tile):
        # Data-dependent but deterministic per-pixel logits.
        return np.stack([tile[0] * 3, tile[1] * 3, tile[2] * 3])

    model.forward = lambda x: torch.as_tensor(
        np.stack([logits(t) for t in x.numpy()])
    )

    arr = da.from_array(image, chunks=(3, k * step, k * step))
    summed = arr.map_overlap(
        code.runModel,
        model=model, ordered_input=[], patch_size=patch, device="cpu", no_data=None,
        num_classes=classes, patches_per_chunk=k, batch_size=k * k,
        chunks=(classes + 1, (k + 1) * step, (k + 1) * step),
        depth={1: (0, step), 2: (0, step)}, boundary="none", trim=False, dtype=np.float16,
    )
    labels = summed.map_overlap(
        code.sum_overlapped_chunks,
        chunk_size=patch, patches_per_chunk=k, drop_axis=0,
        chunks=(k * step, k * step),
        depth={1: (step, 0), 2: (step, 0)}, boundary="none", trim=False, dtype=np.uint8,
    ).compute(scheduler="sync")

    n = cells - 1
    acc = np.zeros((classes + 1, cells * step, cells * step))
    for r in range(n):
        for c in range(n):
            tile = image[:, r * step : r * step + patch, c * step : c * step + patch]
            lg = torch.as_tensor(logits(tile))
            lp = torch.log_softmax(lg, 0)
            ent = -(lp.exp() * lp).sum(0) / np.log(classes)
            blend = (1 - ent.clamp(0, 1)).numpy() + code.ENTROPY_WEIGHT_FLOOR
            wgt = code.patch_window(patch, r == 0, r == n - 1, c == 0, c == n - 1) * blend
            acc[:-1, r * step : r * step + patch, c * step : c * step + patch] += lg.numpy() * wgt
            acc[-1, r * step : r * step + patch, c * step : c * step + patch] += wgt
    expected = np.argmax(acc[:-1] / acc[-1], axis=0)

    # float16 intermediates may flip near-ties.
    assert (labels != expected).mean() < 0.01
