from pathlib import Path
import os
import geopandas as gpd
import pandas as pd
import pytest
import json

from geo_inference.utils.polygon import gdf_to_yolo, mask_to_poly_geojson, geojson2coco


@pytest.fixture
def test_data_dir():
    return Path(__file__).parent.parent / "data"

class TestPolygon:
    
    def test_mask_to_poly_geojson(self, test_data_dir):
        mask_path = test_data_dir / '0_mask.tif'
        poly_path = test_data_dir / '0_polygons.geojson'
        output_path = test_data_dir / 'results.geojson'
        mask_to_poly_geojson(mask_path, output_path, min_area=40, simplify=False, tolerance=0.5)
        
        expected_gdf = gpd.read_file(poly_path)
        result_gdf = gpd.read_file(output_path)
        assert result_gdf.iloc[0].equals(expected_gdf.iloc[0])
        assert result_gdf.iloc[-1].equals(expected_gdf.iloc[-1])
        os.remove(output_path)
    
    def test_gdf_to_yolo(self, test_data_dir):
        poly_path = test_data_dir / '0_polygons.geojson' 
        mask_path = test_data_dir / '0_mask.tif'
        yolo_path = test_data_dir / '0_yolo.csv'
        yolo_output_path = test_data_dir / 'results_yolo.csv' 
        gdf_to_yolo(poly_path, mask_path, yolo_output_path)
        
        expected_result_df = pd.read_csv(yolo_path, sep=' ', header=None, names=['class', 'x', 'y', 'w', 'h'])
        result_df = pd.read_csv(yolo_output_path, sep=' ', header=None, names=['class', 'x', 'y', 'w', 'h'])
        assert result_df.iloc[0].equals(expected_result_df.iloc[0])
        assert result_df.iloc[-1].equals(expected_result_df.iloc[-1])
        os.remove(yolo_output_path)
    
    def test_geojson2coco(self, test_data_dir):
        poly_path = test_data_dir / '0_polygons.geojson'
        mask_path = test_data_dir / '0_mask.tif'
        coco_path = test_data_dir / '0_coco.json'
        coco_output_path = test_data_dir / 'results_coco.json'
        geojson2coco(mask_path, poly_path, coco_output_path)
        
        with open(coco_path, 'r') as f:
            expected_result = json.load(f)
        
        with open(coco_output_path, 'r') as f:
            result = json.load(f)
        
        assert result['annotations'][0]['bbox'] == expected_result['annotations'][0]['bbox']
        assert result['annotations'][-1]['bbox'] == expected_result['annotations'][-1]['bbox']
        os.remove(coco_output_path)
        
        
         
        

    