from pathlib import Path

import geopandas as gpd
import pandas as pd
import pytest

from geo_inference.utils.polygon import gdf_to_yolo, mask_to_poly_geojson


@pytest.fixture
def test_data_dir():
    return Path(__file__).parent.parent / "data"

class TestPolygon:
    def test_gdf_to_yolo(self, test_data_dir):
        poly_path = test_data_dir / 'polygons.geojson' 
        mask_path = test_data_dir / '0_mask.tif'
        yolo_path = test_data_dir / 'yolo_0.csv' 
        gdf_to_yolo(poly_path, mask_path, yolo_path, column="value")
        result_df = pd.read_csv(yolo_path, sep=' ', header=None, names=['class', 'x', 'y', 'w', 'h'])
        assert not result_df.empty
        assert 'class' in result_df.columns
        assert 'x' in result_df.columns

    def test_mask_to_poly_geojson(self, test_data_dir):
        mask_path = test_data_dir / '0_mask.tif'
        output_path = test_data_dir / '0_polygons.geojson'
        mask_to_poly_geojson(mask_path, output_path, min_area=40, simplify=False, tolerance=0.5)
        result_gdf = gpd.read_file(output_path)
        assert not result_gdf.empty
        assert 'geometry' in result_gdf.columns
        assert 'value' in result_gdf.columns