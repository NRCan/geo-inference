import os
import pytest
import torch
import rasterio

from geo_inference.geo_inference import GeoInference
from pathlib import Path


@pytest.fixture
def test_data_dir():
    return Path(__file__).parent / "data"


class TestGeoInference:

    @pytest.fixture
    def geo_inference(self, test_data_dir, tmp_path):
        model = str(test_data_dir / "inference" / "test_model" / "cpu_scripted.pt")
        work_dir = str(tmp_path / "inference")
        mask_to_vec = True
        mask_to_yolo = True
        mask_to_coco = True
        device = "cpu"
        gpu_id = 0
        num_classes = 5
        prediction_threshold = 0.3
        transformer = True
        transform_flip = True
        transform_rotate = True
        return GeoInference(
            model=model,
            work_dir=work_dir,
            mask_to_vec=mask_to_vec,
            mask_to_yolo=mask_to_yolo,
            mask_to_coco=mask_to_coco,
            device=device,
            gpu_id=gpu_id,
            multi_gpu=False,
            num_classes=num_classes,
            prediction_threshold=prediction_threshold,
            transformers=transformer,
            transformer_flip=transform_flip,
            transformer_rotate=transform_rotate,
        )

    def test_init(self, geo_inference, tmp_path):

        assert geo_inference.work_dir == tmp_path / "inference"
        assert geo_inference.device == "cpu"
        assert geo_inference.mask_to_vec == True
        assert geo_inference.mask_to_yolo == True
        assert geo_inference.mask_to_coco == True
        assert isinstance(geo_inference.model.model, torch.jit.ScriptModule)
        assert geo_inference.classes > 0

    def test_call_local_tif(self, geo_inference: GeoInference, test_data_dir: Path):
        tiff_image = test_data_dir / "0.tif"
        bbox = [600500.0, 6097000.0, 601012.0, 6097300.0]
        patch_size = 512
        bands_requested = [1,2,3]
        workers = 0
        # call with bbox.
        mask_name = geo_inference(
            inference_input=str(tiff_image),
            bands_requested=bands_requested,
            patch_size=patch_size,
            workers=workers,
            bbox=bbox,
        )
        mask_path = geo_inference.work_dir / mask_name
        assert mask_path.exists()

        with rasterio.open(mask_path) as img:
            xmin, ymin, xmax, ymax = img.bounds
            assert round(xmin) == round(bbox[0])
            assert round(ymin) == round(bbox[1])
            assert round(xmax) == round(bbox[2])
            assert round(ymax) == round(bbox[3])
        u_id = mask_name.rsplit("_", 1)[-1]
        u_id_no_ext = os.path.splitext(u_id)[0]
        polygons_path = geo_inference.work_dir / f"0_polygons_{u_id_no_ext}.geojson"
        assert polygons_path.exists()
        os.remove(polygons_path)
        os.remove(mask_path)

        # call without bbox.
        bbox = None
        mask_name = geo_inference(
            inference_input=str(tiff_image),
            bands_requested=bands_requested,
            patch_size=patch_size,
            workers=workers,
            bbox=bbox,
        )
        u_id = mask_name.rsplit("_", 1)[-1]
        u_id_no_ext = os.path.splitext(u_id)[0]
        mask_path = geo_inference.work_dir / mask_name
        assert mask_path.exists()
        polygons_path = geo_inference.work_dir / f"0_polygons_{u_id_no_ext}.geojson"
        assert polygons_path.exists()
        os.remove(polygons_path)
        os.remove(mask_path)
    
    def test_call_stac(self, geo_inference: GeoInference, test_data_dir: Path):
        tiff_image = test_data_dir / "stac" / "SpaceNet_AOI_2_Las_Vegas.json"
        bbox = None
        patch_size = 512
        bands_requested = ["Red","Green","Blue"]
        workers = 10
        mask_name = geo_inference(
            inference_input=str(tiff_image.resolve()),
            bands_requested=bands_requested,
            patch_size=patch_size,
            workers=workers,
            bbox=bbox,
        )
        mask_path = geo_inference.work_dir / mask_name
        assert mask_path.exists()
        os.remove(mask_path)
