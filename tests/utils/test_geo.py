import pytest
import rasterio
import geopandas as gpd
from pathlib import Path
from pyproj import CRS

from geo_inference.utils.geo import rasterio_load, gdf_load, check_crs


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

    def test_gdf_load(self, test_data_dir):
        # Test loading a GeoDataFrame from a path
        gdf_path = test_data_dir / "0_polygons.geojson"
        gdf = gdf_load(gdf_path)
        assert isinstance(gdf, gpd.GeoDataFrame)

        # Test loading a GeoDataFrame from a GeoDataFrame object
        gdf2 = gdf_load(gdf)
        assert isinstance(gdf2, gpd.GeoDataFrame)

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

if __name__ == '__main__':
    pytest.main()