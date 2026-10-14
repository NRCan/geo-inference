import numpy as np
import pytest
import rasterio
import torch
from rasterio.transform import from_origin

from geo_inference.geo_inference import GeoInference
from geo_inference.utils.post_inference import Config, _clean_block, clean_mask

PROFILE = dict(
    driver="GTiff", count=3, dtype="uint8", crs="EPSG:3978",
    transform=from_origin(1000, 5000, 1, 1), tiled=True, blockxsize=256, blockysize=256,
)


class TinyNet(torch.nn.Module):
    def __init__(self):
        super().__init__()
        torch.manual_seed(0)
        self.conv = torch.nn.Conv2d(3, 5, 5, padding=2)

    def forward(self, x):
        return self.conv(x / 255.0)


def export(path, dynamic_batch, max_batch=64):
    example = (torch.randn(2 if dynamic_batch else 1, 3, 64, 64),)
    shapes = {"x": {0: torch.export.Dim("b", min=1, max=max_batch)}} if dynamic_batch else None
    program = torch.export.export(TinyNet().eval(), example, dynamic_shapes=shapes)
    torch.export.save(program, path, extra_files={"metadata.json": ""})
    return str(path)


@pytest.fixture(scope="module")
def image(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("img")
    rng = np.random.default_rng(0)
    data = rng.integers(1, 255, (3, 301, 437)).astype(np.uint8)
    path = tmp / "img.tif"
    with rasterio.open(path, "w", height=301, width=437, **PROFILE) as dst:
        dst.write(data)
    masked = tmp / "img_masked.tif"
    with rasterio.Env(GDAL_TIFF_INTERNAL_MASK=True):
        with rasterio.open(masked, "w", height=301, width=437, **PROFILE) as dst:
            dst.write(data)
            mask = np.full((301, 437), 255, np.uint8)
            mask[100:180, 150:300] = 0
            dst.write_mask(mask)
    return path, masked


def infer(model_path, image_path, tmp_path, **call):
    gi = GeoInference(model=model_path, work_dir=str(tmp_path), device="cpu", num_classes=5)
    name = gi(inference_input=str(image_path), patch_size=64, workers=2, **call)
    with rasterio.open(tmp_path / name) as src:
        return gi, src.read(1)


def test_export_batch_bound_is_detected(tmp_path):
    gi = GeoInference(model=export(tmp_path / "s.pt2", False), work_dir=str(tmp_path), device="cpu")
    assert gi.max_batch_size == 1
    gi = GeoInference(model=export(tmp_path / "d.pt2", True, 32), work_dir=str(tmp_path), device="cpu")
    assert gi.max_batch_size == 32


@pytest.mark.parametrize("dynamic_batch", [True, False])
def test_chunking_does_not_change_result(tmp_path, image, dynamic_batch):
    model = export(tmp_path / "m.pt2", dynamic_batch)
    _, single = infer(model, image[0], tmp_path, patches_per_chunk=1)
    _, batched = infer(model, image[0], tmp_path, patches_per_chunk=4)
    assert single.shape == (301, 437)
    # Only float16 rounding of near-ties may differ.
    assert (single != batched).mean() < 1e-3


def test_internal_mask_is_nodata(tmp_path, image):
    model = export(tmp_path / "m.pt2", True)
    _, plain = infer(model, image[0], tmp_path, patches_per_chunk=2)
    _, masked = infer(model, image[1], tmp_path, patches_per_chunk=2)
    with rasterio.open(image[1]) as src:
        valid = src.dataset_mask() > 0
    np.testing.assert_array_equal(masked == 255, ~valid)
    # Away from the hole (more than a patch), predictions are unchanged.
    assert (masked[:30] != plain[:30]).mean() < 1e-3


def test_windowed_clean_mask_matches_whole_raster(tmp_path):
    rng = np.random.default_rng(3)
    labels = np.zeros((900, 1000), np.uint8)
    for _ in range(1500):
        r, c = rng.integers(0, 890), rng.integers(0, 990)
        labels[r : r + rng.integers(1, 12), c : c + rng.integers(1, 12)] = rng.integers(1, 5)
    labels[400:403, :] = 2  # road crossing every block column, with a gap to close
    labels[400:403, 500:502] = 0
    path = tmp_path / "labels.tif"
    with rasterio.open(path, "w", height=900, width=1000, **dict(PROFILE, count=1)) as dst:
        dst.write(labels, 1)

    cfg = Config(road_class_index=2, gsd=0.5, min_area_m2={1: 5.0, 2: 10.0, 3: 2.0, 4: 8.0})
    with rasterio.open(clean_mask(path, cfg, chunk_size=128)) as src:
        windowed = src.read(1)

    min_px = {k: max(1, int(v / cfg.gsd**2)) for k, v in cfg.min_area_m2.items()}
    whole = _clean_block(labels, min_px, {2: round(1.5 / cfg.gsd)}, cfg.background_index)
    assert (whole != labels).any()
    np.testing.assert_array_equal(windowed, whole)
