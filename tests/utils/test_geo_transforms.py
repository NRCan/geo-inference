import pytest
import pandas as pd
import geopandas as gpd
from pathlib import Path
from geo_inference.utils.geo import gdf_load
from geo_inference.utils.geo_transforms import convert_poly_coords, affine_transform_gdf, geojson_to_px_gdf, get_overlapping_subset
from shapely.wkt import dumps
from shapely.geometry import Polygon
from affine import Affine

@pytest.fixture
def test_data_dir():
    return Path(__file__).parent.parent / "data"

class TestGeoTransforms:
    def test_convert_poly_coords(self):
        # Create a simple polygon
        geom = Polygon([(0, 0), (1, 1), (1, 0)])
        # Create an affine transformation
        affine_obj = Affine(1, 2, 3, 4, 5, 6)

        # Test with affine_obj
        result = convert_poly_coords(geom, affine_obj)
        assert isinstance(result, Polygon)
        assert result.bounds != geom.bounds

        # Test with inverse
        result = convert_poly_coords(geom, affine_obj, inverse=True)
        assert isinstance(result, Polygon)
        assert result.bounds != geom.bounds

        # Test with WKT
        geom = "POLYGON ((0 0, 1 1, 1 0, 0 0))"
        result = convert_poly_coords(geom, affine_obj)
        assert isinstance(result, str)
        assert result != geom

        # Test without affine_obj
        with pytest.raises(ValueError):
            convert_poly_coords(geom)

        # Test with invalid geom
        geom = 12345
        with pytest.raises(TypeError):
            convert_poly_coords(geom, affine_obj)
    
    def test_affine_transform_gdf(self, test_data_dir):
        # Create an affine transformation
        affine_obj = Affine(0.5, 0.0, 733601.0, 0.0, -0.5, 3725139.0)
        
        # Load a label_csv
        label_gdf_csv = test_data_dir / "aff_gdf_result.csv"
        label_gdf = pd.read_csv(label_gdf_csv)
        
        # Load with DataFrame and GeoDataFrame
        df_path = str(test_data_dir / "sample.csv")
        gdf_path = str(test_data_dir / "sample.geojson")
        
        # Test with DataFrame
        result = affine_transform_gdf(df_path, affine_obj, geom_col="PolygonWKT_Pix", precision=0)
        result['geometry'] = result['geometry'].apply(dumps, trim=True)
        assert result.equals(label_gdf)
        
        # Test with GeoDataFrame
        result = affine_transform_gdf(gdf_path, affine_obj)
        assert isinstance(result, gpd.GeoDataFrame)
    
    def test_geojson_to_px_gdf(self, test_data_dir):
         
        image_path = str(test_data_dir / "0.tif")
        geojson_path = str(test_data_dir / "0_polygons.geojson")
        gj_to_px_result = str(test_data_dir / "gj_to_px_result.geojson")
        result_gdf = gpd.read_file(gj_to_px_result)

        # Test with GeoJSON and image path
        result = geojson_to_px_gdf(geojson_path, image_path)
        
        truth_subset = result_gdf[['geometry']]
        output_subset = result[['geometry']].reset_index(drop=True)
        
        assert isinstance(result, gpd.GeoDataFrame)
        assert truth_subset.equals(output_subset)
    
    def test_get_overlapping_subset(self, test_data_dir):
        # Load a GeoDataFrame
        gdf_path = test_data_dir / "sample.geojson"
        gdf = gdf_load(gdf_path)
        
        # Create a bounding box
        bbox = (733601.0, 3725139.0, 733602.0, 3725140.0)
        
        # Test with GeoDataFrame
        result = get_overlapping_subset(gdf, bbox=bbox)
        assert isinstance(result, gpd.GeoDataFrame)