import os
import pytest
import math
import torch
import rasterio
import numpy as np
from unittest.mock import patch
from geo_inference.geo_inference import GeoInference, logger 
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
            assert math.isclose(xmin, bbox[0], abs_tol=1)
            assert math.isclose(ymin, bbox[1], abs_tol=1)
            assert math.isclose(xmax, bbox[2], abs_tol=1)
            assert math.isclose(ymax, bbox[3], abs_tol=1)
        polygons_path = geo_inference.work_dir / "0_polygons.geojson"
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
        mask_path = geo_inference.work_dir / mask_name
        assert mask_path.exists()
        polygons_path = geo_inference.work_dir / "0_polygons.geojson"
        assert polygons_path.exists()
        os.remove(polygons_path)
        os.remove(mask_path)
    
    def test_call_stac(self, geo_inference: GeoInference, test_data_dir: Path):
        tiff_image = r"./tests/data/stac/SpaceNet_AOI_2_Las_Vegas.json"
        bbox = None
        patch_size = 512
        bands_requested = ["Red","Green","Blue"]
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
        os.remove(mask_path)
    
    def test_band_reordering_logic(self, geo_inference, test_data_dir):
        """Test the specific band reordering logic with xr.concat and da.stack."""
        tiff_image = test_data_dir / "0.tif"
        # Track what gets logged
        logged_messages = []
        
        def capture_log(message):
            logged_messages.append(message)
            print(f"LOG: {message}")

        # Mock logger to capture the "Bands are reordeing" message
        with patch.object(logger, 'info', side_effect=capture_log), \
             patch("geo_inference.geo_inference.runModel", return_value=np.ones((4, 4), dtype=np.uint8)), \
             patch("geo_inference.geo_inference.sum_overlapped_chunks", side_effect=lambda arr, **kwargs: arr), \
             patch("xarray.DataArray.rio.to_raster"), \
             patch("os.path.exists", return_value=True):

            geo_inference.mask_to_vec = False
            
            bands_requested = ["3", "1", "2"]
            
            mask_name = geo_inference(
                inference_input=str(tiff_image), 
                bands_requested=bands_requested,
                patch_size=4,
                workers=0,
                bbox=None
            )
            
            # Verify the band reordering message was logged
            reorder_messages = [msg for msg in logged_messages if "Bands are reordeing to bands_requested" in msg]
            assert len(reorder_messages) > 0, "Band reordering log message should appear"

            
