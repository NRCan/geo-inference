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
        model = str(test_data_dir / "inference"/ "test_model" / "test_model.pt")
        work_dir = str(test_data_dir / "inference")
        mask_to_vec = True
        mask_to_yolo = True
        mask_to_coco = True
        device = 'cpu'
        gpu_id = 0
        return GeoInference(model, work_dir, mask_to_vec, mask_to_yolo, mask_to_coco, device, gpu_id)

    def test_init(self, geo_inference, test_data_dir):

        assert geo_inference.work_dir == test_data_dir / "inference"
        assert geo_inference.device == 'cpu'
        assert geo_inference.mask_to_vec == True
        assert geo_inference.mask_to_yolo == True
        assert geo_inference.mask_to_coco == True
        assert isinstance(geo_inference.model, torch.jit.ScriptModule)
        assert geo_inference.classes >0

    def test_call(self, geo_inference, test_data_dir):
        tiff_image = test_data_dir / '0.tif'
        # bbox = '0,0,100,100'
        patch_size = 512
        bands_requested="1,2,3"
        geo_inference(str(tiff_image), bands_requested, patch_size, None) 
        mask_path = geo_inference.work_dir / "0_mask.tif"
        assert mask_path.exists()
        if geo_inference.mask_to_vec:
            polygons_path = geo_inference.work_dir / "0_polygons.geojson"
            yolo_csv_path = geo_inference.work_dir / "0_yolo.csv"
            coco_path = geo_inference.work_dir / "0_coco.json"
            assert polygons_path.exists()
            assert yolo_csv_path.exists()
            assert coco_path.exists()
            os.remove(polygons_path)
            os.remove(yolo_csv_path)
            os.remove(coco_path)
        os.remove(mask_path)