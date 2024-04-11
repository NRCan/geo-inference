import logging
from distutils.version import LooseVersion
from pathlib import Path

import pandas as pd
import geopandas as gpd
import pyproj
import rasterio
from shapely.wkt import loads
from shapely.geometry import Point
from shapely.geometry.base import BaseGeometry
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

def df_load(df):
    """Check if `df` is already loaded in, if not, load from file."""
    if isinstance(df, str):
        if df.lower().endswith('json'):
            return gdf_load(df)
        else:
            return pd.read_csv(df)
    elif isinstance(df, pd.DataFrame):
        return df
    else:
        raise ValueError(f"{df} is not an accepted DataFrame format.")

def check_geom(geom):
    """Check if a geometry is loaded in.

    Returns the geometry if it's a shapely geometry object. If it's a wkt
    string or a list of coordinates, convert to a shapely geometry.
    """
    if isinstance(geom, BaseGeometry):
        return geom
    elif isinstance(geom, str):  # assume it's a wkt
        return loads(geom)
    elif isinstance(geom, list) and len(geom) == 2:  # coordinates
        return Point(geom)

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