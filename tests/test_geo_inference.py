import os
import pytest
import torch

from geo_inference.geo_inference import GeoInference
from pathlib import Path


@pytest.fixture
def test_data_dir():
    return Path(__file__).parent / "data"


class TestGeoInference:

    @pytest.fixture
    def geo_inference(self, test_data_dir):
        model = str(test_data_dir / "inference" / "test_model" / "cpu_scripted.pt")
        work_dir = str(test_data_dir / "inference")
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

    def test_init(self, geo_inference, test_data_dir):

        assert geo_inference.work_dir == test_data_dir / "inference"
        assert geo_inference.device == "cpu"
        assert geo_inference.mask_to_vec == True
        assert geo_inference.mask_to_yolo == True
        assert geo_inference.mask_to_coco == True
        assert isinstance(geo_inference.model.model, torch.jit.ScriptModule)
        assert geo_inference.classes > 0

    def test_call(self, geo_inference: GeoInference, test_data_dir: Path):
        tiff_image = test_data_dir / "0.tif"
        # bbox = '0,0,100,100'
        bbox = None
        patch_size = 512
        bands_requested = "1,2,3"
        workers = 10
        mask_name = geo_inference(
            inference_input=str(tiff_image),
            bands_requested=bands_requested,
            patch_size=patch_size,
            workers=workers,
            bbox=bbox,
        )
        mask_path = geo_inference.work_dir / mask_name
        assert mask_path.exists()
        polygons_path = geo_inference.work_dir / "0_polygons.geojson"
        assert polygons_path.exists()
        os.remove(polygons_path)
        os.remove(mask_path)
