import logging
from distutils.version import LooseVersion
from pathlib import Path

import geopandas as gpd
import pyproj
import rasterio
from fiona._err import CPLE_OpenFailedError
from fiona.errors import DriverError

from ..config.logging_config import logger

logger = logging.getLogger(__name__)

"""Utility funtions adapted from Solaris: https://github.com/CosmiQ/solaris/blob/main/solaris/utils/core.py"""

def rasterio_load(im):
    """Load a rasterio image from a path or a rasterio object.
    Args:
        im (str or Path or rasterio.DatasetReader): Path to the image or a rasterio object. 
        
    Returns:
        rasterio.DatasetReader: rasterio dataset.
    """
    if isinstance(im, (str, Path)):
        return rasterio.open(im)
    elif isinstance(im, rasterio.DatasetReader):
        return im
    else:
        raise ValueError("{} is not an accepted image format for rasterio.".format(im))

def gdf_load(gdf):
    """Load a GeoDataFrame from a path or a GeoDataFrame object.

    Args:
        gdf (str or Path or GeoDataFrame): Path to the GeoDataFrame or a GeoDataFrame object.

    Returns:
        GeoDataFrame: GeoDataFrame.
    """
    if isinstance(gdf, (str, Path)):
        # as of geopandas 0.6.2, using the OGR CSV driver requires some add'nal
        # kwargs to create a valid geodataframe with a geometry column. see
        # https://github.com/geopandas/geopandas/issues/1234
        if str(gdf).lower().endswith('csv'):
            return gpd.read_file(gdf, GEOM_POSSIBLE_NAMES="geometry",
                                 KEEP_GEOM_COLUMNS="NO")
        try:
            return gpd.read_file(gdf)
        except (DriverError, CPLE_OpenFailedError):
            logger.warning(f"GeoDataFrame couldn't be loaded: either {gdf} isn't a valid"
                           f" path or it isn't a valid vector file. Returning an empty"
                           f" GeoDataFrame.")
            return gpd.GeoDataFrame()
    elif isinstance(gdf, gpd.GeoDataFrame):
        return gdf
    else:
        raise ValueError(f"{gdf} is not an accepted GeoDataFrame format.")
    
def check_crs(input_crs, return_rasterio=False):
    """Check the input CRS and return a pyproj CRS object.

    Args:
        input_crs (str or pyproj.CRS): Input CRS.
        return_rasterio (bool, optional): Returns rasterio crs object. Defaults to False.

    Returns:
        rasterio or pyproj crs: CRS object.
    """
    if not isinstance(input_crs, pyproj.CRS) and input_crs is not None:
        out_crs = pyproj.CRS(input_crs)
    else:
        out_crs = input_crs

    if return_rasterio:
        if LooseVersion(rasterio.__gdal_version__) >= LooseVersion("3.0.0"):
            out_crs = rasterio.crs.CRS.from_wkt(out_crs.to_wkt())
        else:
            out_crs = rasterio.crs.CRS.from_wkt(out_crs.to_wkt("WKT1_GDAL"))

    return out_crs