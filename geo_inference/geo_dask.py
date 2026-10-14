import json
import logging
import threading
from contextlib import nullcontext
from functools import cache, lru_cache
from pathlib import Path

import numpy as np
import scipy.signal.windows as w
import torch
from rasterio.transform import Affine

logger = logging.getLogger(__name__)

ENTROPY_WEIGHT_FLOOR: float = 1e-2
_CUDA_LOCK = threading.Lock()
_BATCH_CAP: dict = {}
# Copied to the device as-is and cast to float32 there.
_DEVICE_CAST_DTYPES = (np.uint8, np.int16, np.float16, np.float32)


@cache
def _hann_1d(patch_size: int, flat_start: bool, flat_end: bool) -> np.ndarray:
    """1D Hann taper. A border half stays at the peak so edge patches don't fade out."""
    step = patch_size >> 1
    win = w.hann(M=patch_size, sym=False).astype(np.float32)
    if flat_start:
        win[:step] = win[step]
    if flat_end:
        win[step:] = win[step]
    win.flags.writeable = False
    return win


@lru_cache(maxsize=64)
def patch_window(
    patch_size: int, first_row: bool, last_row: bool, first_col: bool, last_col: bool
) -> np.ndarray:
    win = np.outer(
        _hann_1d(patch_size, first_row, last_row),
        _hann_1d(patch_size, first_col, last_col),
    )
    win.flags.writeable = False
    return win


@lru_cache(maxsize=64)
def _device_window(
    patch_size: int,
    first_row: bool,
    last_row: bool,
    first_col: bool,
    last_col: bool,
    device: str,
) -> torch.Tensor:
    return torch.from_numpy(
        patch_window(patch_size, first_row, last_row, first_col, last_col).copy()
    ).to(device)


def _is_nodata(tile: np.ndarray, no_data: float | None) -> bool:
    if no_data is None or np.isnan(no_data):
        return not np.isfinite(tile).any()
    return bool(np.all(tile == no_data))


def _forward(
    model,
    batch: np.ndarray,
    windows: torch.Tensor,
    extra_inputs: list,
    device: str,
    acc: torch.Tensor,
    batch_offsets: list,
    step: int,
) -> None:
    """Weight one batch by window x entropy and overlap-add it into `acc` on device."""
    is_cuda = device.startswith("cuda")
    tensor = torch.from_numpy(batch)
    if is_cuda:
        tensor = tensor.pin_memory()
    with _CUDA_LOCK if is_cuda else nullcontext(), torch.inference_mode():
        tensor = tensor.to(device, non_blocking=True).float()
        y = model(tensor, *extra_inputs)
        if isinstance(y, (tuple, list)):
            y = y[0]
        logits = y.float()
        if logits.shape[2:] != windows.shape[1:]:
            raise ValueError(
                f"Model output shape {tuple(logits.shape[1:])} does not match "
                f"patch {windows.shape[1:]}"
            )
        num_classes = logits.shape[1]
        if num_classes != acc.shape[0] - 1:
            raise ValueError(
                f"Model returned {num_classes} classes, expected {acc.shape[0] - 1}"
            )

        log_probs = torch.log_softmax(logits, dim=1)
        entropy = -(log_probs.exp() * log_probs).sum(dim=1)
        if num_classes > 1:
            normalized_entropy = torch.clamp(
                entropy / float(np.log(num_classes)), 0.0, 1.0
            )
        else:
            normalized_entropy = torch.zeros_like(entropy)
        blend_factor = (1.0 - normalized_entropy) + ENTROPY_WEIGHT_FLOOR

        weights = windows * blend_factor
        weighted = logits * weights.unsqueeze(1)
        patch_size = windows.shape[1]
        for k, (i, j) in enumerate(batch_offsets):
            rows = slice(i * step, i * step + patch_size)
            cols = slice(j * step, j * step + patch_size)
            acc[:-1, rows, cols] += weighted[k]
            acc[-1, rows, cols] += weights[k]


def runModel(
    chunk_data: np.ndarray,
    model,
    ordered_input: list,
    patch_size: int,
    device: str,
    no_data: float | None,
    num_classes: int = 5,
    block_info=None,
    patches_per_chunk: int = 1,
    batch_size: int = 1,
):
    """Overlap-add windowed logits for every valid patch in a chunk.

    The chunk is `patches_per_chunk` stride cells per axis, plus the neighbour cell
    added by map_overlap. The extra stride of the return value spills into the next
    chunk and is combined by `sum_overlapped_chunks`.
    """
    step = patch_size >> 1
    out_size = (patches_per_chunk + 1) * step
    empty = np.zeros((num_classes + 1, out_size, out_size), dtype=np.float16)
    if chunk_data is None or chunk_data.size == 0:
        return empty

    num_chunks = block_info[0]["num-chunks"]
    chunk_location = block_info[0]["chunk-location"]
    # Patch p covers stride cells p and p + 1, so the raster has (cells - 1) patches.
    total_rows = num_chunks[1] * patches_per_chunk - 1
    total_cols = num_chunks[2] * patches_per_chunk - 1
    row0 = chunk_location[1] * patches_per_chunk
    col0 = chunk_location[2] * patches_per_chunk
    n_rows = min(chunk_data.shape[1] // step - 1, patches_per_chunk)
    n_cols = min(chunk_data.shape[2] // step - 1, patches_per_chunk)

    offsets = []
    for i in range(n_rows):
        for j in range(n_cols):
            tile = chunk_data[
                :, i * step : i * step + patch_size, j * step : j * step + patch_size
            ]
            if not _is_nodata(tile, no_data):
                offsets.append((i, j))
    if not offsets:
        return empty

    try:
        extra_inputs = [
            torch.as_tensor(
                extra_input, dtype=torch.float32, device=torch.device(device)
            )
            for extra_input in (ordered_input or [])
        ]
        acc = torch.zeros(
            (num_classes + 1, out_size, out_size), dtype=torch.float32, device=device
        )
        start = 0
        while start < len(offsets):
            size = max(1, min(batch_size, _BATCH_CAP.get(device, batch_size)))
            batch_offsets = offsets[start : start + size]
            batch = np.stack(
                [
                    chunk_data[
                        :,
                        i * step : i * step + patch_size,
                        j * step : j * step + patch_size,
                    ]
                    for i, j in batch_offsets
                ]
            )
            if batch.dtype not in _DEVICE_CAST_DTYPES:
                batch = batch.astype(np.float32)
            windows = torch.stack(
                [
                    _device_window(
                        patch_size,
                        row0 + i == 0,
                        row0 + i == total_rows - 1,
                        col0 + j == 0,
                        col0 + j == total_cols - 1,
                        device,
                    )
                    for i, j in batch_offsets
                ]
            )
            try:
                _forward(
                    model,
                    batch,
                    windows,
                    extra_inputs,
                    device,
                    acc,
                    batch_offsets,
                    step,
                )
            except torch.cuda.OutOfMemoryError:
                if size == 1:
                    raise
                with _CUDA_LOCK:
                    _BATCH_CAP[device] = size // 2
                torch.cuda.empty_cache()
                logger.warning(
                    "CUDA OOM at batch %s; retrying with %s", size, size // 2
                )
                continue
            start += len(batch_offsets)
        return acc.half().cpu().numpy()

    except Exception:
        logger.exception("runModel failed")
        return empty


def sum_overlapped_chunks(
    aoi_chunk: np.ndarray,
    chunk_size: int,
    prediction_threshold: float = 0.3,
    block_info=None,
    patches_per_chunk: int = 1,
):
    """Add the top/left spill, normalize by the weight channel, and label pixels."""
    step = chunk_size // 2
    own = patches_per_chunk * step
    if aoi_chunk is None or aoi_chunk.size == 0:
        return np.zeros((own, own), dtype=np.uint8)

    top = aoi_chunk.shape[1] - (patches_per_chunk + 1) * step
    left = aoi_chunk.shape[2] - (patches_per_chunk + 1) * step
    if top not in (0, step) or left not in (0, step):
        logger.error(
            "sum_overlapped_chunks shape %s does not match patch %s and %s patches",
            aoi_chunk.shape,
            chunk_size,
            patches_per_chunk,
        )
        return np.zeros((own, own), dtype=np.uint8)

    full_array = aoi_chunk[:, top : top + own, left : left + own].astype(np.float32)
    if top:
        full_array[:, :step, :] += aoi_chunk[:, :step, left : left + own]
    if left:
        full_array[:, :, :step] += aoi_chunk[:, top : top + own, :step]
    if top and left:
        full_array[:, :step, :step] += aoi_chunk[:, :step, :step]

    final_result = np.divide(
        full_array[:-1, :, :],
        full_array[-1, :, :][np.newaxis, :, :],
        out=np.zeros_like(full_array[:-1, :, :]),
        where=full_array[-1, :, :] != 0,
    )
    if final_result.shape[0] == 1:
        labels = np.where(final_result > prediction_threshold, 1, 0)
        return labels.squeeze(0).astype(np.uint8)
    return np.argmax(final_result, axis=0).astype(np.uint8)


def read_zarr_metadata(metadata_json: Path | str):
    try:
        with open(metadata_json, "r") as metadat_json:
            metadata = json.load(metadat_json)
            lines = metadata["transform"].strip().split("\n")
            matrix_values = []
            for line in lines:
                values = line.strip("|").split(",")
                matrix_values.extend(float(val.strip()) for val in values)
            # Create and return the Affine object
            trs = Affine(
                matrix_values[0],
                matrix_values[1],
                matrix_values[2],
                matrix_values[3],
                matrix_values[4],
                matrix_values[5],
            )
            metadata.update(
                {
                    "crs": metadata["crs"],
                    "transform": trs,
                    "count": metadata["count"],
                    "width": metadata["width"],
                    "height": metadata["height"],
                    "driver": metadata["driver"],
                    "dtype": metadata["dtype"],
                    "BIGTIFF": metadata.get("BIGTIFF", "Unknown"),
                    "compress": metadata.get("compress", "Unknown"),
                }
            )
            return metadata
    except FileNotFoundError:
        logger.exception("Error: The file '%s' was not found.", metadata_json)
    except json.JSONDecodeError:
        logger.exception("Error: Failed to decode JSON from the file.")
