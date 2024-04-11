import pytest
import rasterio
import pandas as pd
import geopandas as gpd
from pathlib import Path
from pyproj import CRS
from shapely.wkt import loads
from shapely.geometry import Point
from shapely.geometry.base import BaseGeometry
from geo_inference.utils.geo import rasterio_load, gdf_load, df_load, check_geom, check_crs


@pytest.fixture
def test_data_dir():
    return Path(__file__).parent.parent / "data"

class TestGeo:
    def test_rasterio_load(self, test_data_dir):
        # Test loading a rasterio image from a path
        im_path = test_data_dir / "0.tif"
        with rasterio_load(im_path) as dataset:
            assert isinstance(dataset, rasterio.DatasetReader)

        # Test loading a rasterio image from a rasterio object
        with rasterio.open(im_path) as src:
            with rasterio_load(src) as dataset:
                assert isinstance(dataset, rasterio.DatasetReader)
        
        # Test Invalid image format
        with pytest.raises(ValueError):
            rasterio_load(1)

    def test_gdf_load(self, test_data_dir):
        # Test loading a GeoDataFrame from a path
        gdf_path = test_data_dir / "0_polygons.geojson"
        gdf = gdf_load(gdf_path)
        assert isinstance(gdf, gpd.GeoDataFrame)

        # Test loading a GeoDataFrame from a GeoDataFrame object
        gdf2 = gdf_load(gdf)
        assert isinstance(gdf2, gpd.GeoDataFrame)
        
        # Test Invalid path
        gdf3 = gdf_load("invalid_path")
        assert gdf3.empty
        
        # Test Invalid GeoDataFrame
        with pytest.raises(ValueError):
            gdf_load(1)
            
    def test_df_load(self, test_data_dir):
        csv_path = test_data_dir / "0_yolo.csv"
        df_1 = df_load(str(csv_path))
        assert isinstance(df_1, pd.DataFrame)
        
        json_path = test_data_dir / "0_coco.json"
        gdf = gdf_load(json_path)
        assert isinstance(gdf, gpd.GeoDataFrame)
        
        df_3 = pd.DataFrame({'a': [1, 2], 'b': [3, 4]})
        loaded_df = df_load(df_3)
        assert isinstance(loaded_df, pd.DataFrame)
        
        with pytest.raises(ValueError):
            df_load(1)
        
    def test_check_geom(self):
        # Test checking a shapely geometry object
        geom = Point(0, 0)
        geom2 = check_geom(geom)
        assert isinstance(geom2, BaseGeometry)

        # Test checking a wkt string
        geom3 = check_geom("POINT (0 0)")
        assert isinstance(geom3, BaseGeometry)

        # Test checking a list of coordinates
        geom4 = check_geom([0, 0])
        assert isinstance(geom4, BaseGeometry)

    def test_check_crs(self):
        # Test checking a CRS string
        crs_str = "EPSG:4326"
        crs = check_crs(crs_str)
        assert isinstance(crs, CRS)

        # Test checking a CRS object
        crs2 = CRS.from_epsg(4326)
        crs3 = check_crs(crs2)
        assert isinstance(crs3, CRS)

        # Test returning a rasterio CRS object
        crs4 = check_crs(crs_str, return_rasterio=True)
        assert isinstance(crs4, rasterio.crs.CRS)