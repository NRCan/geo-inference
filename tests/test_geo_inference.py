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
        model_name = 'test_model'
        work_dir = str(test_data_dir / "inference")
        batch_size = 1
        mask_to_vec = True
        device = 'cpu'
        gpu_id = 0
        return GeoInference(model_name, work_dir, batch_size, mask_to_vec, device, gpu_id)

    def test_init(self, geo_inference, test_data_dir):
        assert geo_inference.gpu_id == 0
        assert geo_inference.batch_size == 1
        assert geo_inference.work_dir == test_data_dir / "inference"
        assert geo_inference.device == torch.device('cpu')
        assert geo_inference.mask_to_vec == True
        assert isinstance(geo_inference.model, torch.jit.ScriptModule)
        assert geo_inference.classes > 0

    def test_call(self, geo_inference, test_data_dir):
        tiff_image = test_data_dir / '0.tif'
        # bbox = '0,0,100,100'
        # patch_size = 512
        # stride_size = 256
        geo_inference(str(tiff_image))
        mask_path = geo_inference.work_dir / "0_mask.tif"
        assert mask_path.exists()
        if geo_inference.mask_to_vec:
            polygons_path = geo_inference.work_dir / "0_polygons.geojson"
            yolo_csv_path = geo_inference.work_dir / "0_yolo.csv"
            assert polygons_path.exists()
            assert yolo_csv_path.exists()