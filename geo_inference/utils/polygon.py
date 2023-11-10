"""Adapted from Solaris: https://github.com/CosmiQ/solaris/tree/main/solaris """

import os
import shapely
import rasterio
import logging
import pandas as pd
import geopandas as gpd
from affine import Affine
from rasterio import features
from rasterio.warp import transform_bounds
from shapely.geometry import shape, Polygon
from rtree.core import RTreeError
from utils.geo import rasterio_load, gdf_load, check_crs

from config.logging_config import logger
logger = logging.getLogger(__name__)

def mask_to_poly_geojson(mask_path,
                         output_path="", 
                         min_area=40,
                         simplify=False,
                         tolerance=0.5):
    """Get polygons from an image mask.

    Args:
        mask_path : str
            Generated mask path.
        
        output_path : str, optional
            Path to save the output file to. If not provided, no file is saved.
        min_area : int, optional
            The minimum area of a polygon to retain. Filtering is done AFTER
            any coordinate transformation, and therefore will be in destination
            units.
        simplify : bool, optional
            If ``True``, will use the Douglas-Peucker algorithm to simplify edges,
            saving memory and processing time later. Defaults to ``False``.
        tolerance : float, optional
            The tolerance value to use for simplification with the Douglas-Peucker
            algorithm. Defaults to ``0.5``. Only has an effect if
            ``simplify=True``.

    Returns:
        gdf : :class:`geopandas.GeoDataFrame`
            A GeoDataFrame of polygons.

    """
    
    with rasterio.open(mask_path, 'r') as src:
        raster = src.read()
        transform = src.transform
        crs = src.crs
    mask = raster > 0
    polygon_generator = features.shapes(raster, transform=transform, mask=mask)
    polygons = []
    values = []  # pixel values for the polygon in mask_arr
    for polygon, value in polygon_generator:
        p = shape(polygon).buffer(0.0)
        if p.area >= min_area:
            polygons.append(shape(polygon).buffer(0.0))
            values.append(value)

    polygon_gdf = gpd.GeoDataFrame({"geometry": polygons, "value": values}, 
                                   crs=crs.to_wkt())
    if simplify:
        polygon_gdf['geometry'] = polygon_gdf['geometry'].apply(lambda x: x.simplify(tolerance=tolerance))
    
    polygon_gdf.to_file(output_path, driver='GeoJSON')
    logger.info(f"GeoJSON saved to {output_path}")

def convert_poly_coords(geom, affine_obj=None, inverse=False):
    """Georegister geometry objects currently in pixel coords or vice versa.

    Args:
        geom : :class:`shapely.geometry.shape` or str
            A :class:`shapely.geometry.shape`, or WKT string-formatted geometry
            object currently in pixel coordinates.
        
        affine_obj: list or :class:`affine.Affine`
            An affine transformation to apply to `geom` in the form of an
            ``[a, b, d, e, xoff, yoff]`` list or an :class:`affine.Affine` object.
            Required if not using `raster_src`.
        inverse : bool, optional
            If true, will perform the inverse affine transformation, going from
            geospatial coordinates to pixel coordinates.

    Returns:
        out_geom
            A geometry in the same format as the input with its coordinate system
            transformed to match the destination object.
    """

    if not affine_obj:
        raise ValueError("affine_obj must be provided.")
    
    if isinstance(affine_obj, Affine):
        affine_xform = affine_obj
    else:
        # assume it's a list in either gdal or "standard" order
        # (list_to_affine checks which it is)
        if len(affine_obj) == 9:  # if it's straight from rasterio
            affine_obj = affine_obj[0:6]
            affine_xform = Affine(*affine_obj)

    if inverse:  # geo->px transform
        affine_xform = ~affine_xform

    if isinstance(geom, str):
        # get the polygon out of the wkt string
        g = shapely.wkt.loads(geom)
    elif isinstance(geom, shapely.geometry.base.BaseGeometry):
        g = geom
    else:
        raise TypeError('The provided geometry is not an accepted format. '
                        'This function can only accept WKT strings and '
                        'shapely geometries.')

    xformed_g = shapely.affinity.affine_transform(g, [affine_xform.a,
                                                      affine_xform.b,
                                                      affine_xform.d,
                                                      affine_xform.e,
                                                      affine_xform.xoff,
                                                      affine_xform.yoff])
    if isinstance(geom, str):
        # restore to wkt string format
        xformed_g = shapely.wkt.dumps(xformed_g)

    return xformed_g

def affine_transform_gdf(gdf, affine_obj, inverse=False, geom_col="geometry"):
    """Perform an affine transformation on a GeoDataFrame.

    Args:
        gdf : :class:`geopandas.GeoDataFrame`, :class:`pandas.DataFrame`, or `str`
            A GeoDataFrame, pandas DataFrame with a ``"geometry"`` column (or a
            different column containing geometries, identified by `geom_col` -
            note that this column will be renamed ``"geometry"`` for ease of use
            with geopandas), or the path to a saved file in .geojson or .csv
            format.
        affine_obj : list or :class:`affine.Affine`
            An affine transformation to apply to `geom` in the form of an
            ``[a, b, d, e, xoff, yoff]`` list or an :class:`affine.Affine` object.
        inverse : bool, optional
            Use this argument to perform the inverse transformation.
        geom_col : str, optional
            The column in `gdf` corresponding to the geometry. Defaults to
            ``'geometry'``.
        precision : int, optional
            Decimal precision to round the geometries to. If not provided, no
            rounding is performed.
    Returns:
        gdf : :class:`geopandas.GeoDataFrame`
    """
    if isinstance(gdf, str):  # assume it's a geojson
        if gdf.lower().endswith('json'):
            gdf = gpd.read_file(gdf)
        elif gdf.lower().endswith('csv'):
            gdf = pd.read_csv(gdf)
        else:
            raise ValueError(
                "The file format is incompatible with this function.")
    if 'geometry' not in gdf.columns:
        gdf = gdf.rename(columns={geom_col: 'geometry'})
    if not isinstance(gdf['geometry'][0], Polygon):
        gdf['geometry'] = gdf['geometry'].apply(shapely.wkt.loads)
    gdf["geometry"] = gdf["geometry"].apply(convert_poly_coords,
                                            affine_obj=affine_obj,
                                            inverse=inverse)
    # the CRS is no longer valid - remove it
    gdf.crs = None

    return gdf

def geojson_to_px_gdf(geojson, im_path, geom_col='geometry',
                      output_path=None, override_crs=False):
    """Convert a geojson or set of geojsons from geo coords to px coords.

    Args:
        geojson : str
            Path to a geojson. This function will also accept a
            :class:`pandas.DataFrame` or :class:`geopandas.GeoDataFrame` with a
            column named ``'geometry'`` in this argument.
        im_path : str
            Path to a georeferenced image (ie a GeoTIFF) that geolocates to the
            same geography as the `geojson`(s). This function will also accept a
            :class:`osgeo.gdal.Dataset` or :class:`rasterio.DatasetReader` with
            georeferencing information in this argument.
        geom_col : str, optional
            The column containing geometry in `geojson`. If not provided, defaults
            to ``"geometry"``.
        output_path : str, optional
            Path to save the resulting output to. If not provided, the object
            won't be saved to disk.
        override_crs: bool, optional
            Useful if the geojsons generated by the vector tiler or otherwise were saved
            out with a non EPSG code projection. True sets the gdf crs to that of the 
            image, the inputs should have the same underlying projection for this to work. 
            If False, and the gdf does not have an EPSG code, this function will fail.
    Returns:
        output_df : :class:`pandas.DataFrame`
            A :class:`pandas.DataFrame` with all geometries in `geojson` that
            overlapped with the image at `im_path` converted to pixel coordinates.
            Additional columns are included with the filename of the source
            geojson (if available) and images for reference.

    """
    # get the bbox and affine transforms for the image
    im = rasterio_load(im_path)
    if isinstance(im_path, rasterio.DatasetReader):
        im_path = im_path.name
    # make sure the geo vector data is loaded in as geodataframe(s)
    gdf = gdf_load(geojson)

    if len(gdf):  # if there's at least one geometry
        if override_crs:
            gdf.crs = im.crs 
        overlap_gdf = get_overlapping_subset(gdf, im)
    else:
        overlap_gdf = gdf

    affine_obj = im.transform
    transformed_gdf = affine_transform_gdf(overlap_gdf, affine_obj=affine_obj,
                                           inverse=True, geom_col=geom_col)
    transformed_gdf['image_fname'] = os.path.split(im_path)[1]

    if output_path is not None:
        if output_path.lower().endswith('json'):
            transformed_gdf.to_file(output_path, driver='GeoJSON')
        else:
            transformed_gdf.to_csv(output_path, index=False)
    return transformed_gdf

def get_overlapping_subset(gdf, im=None, bbox=None, bbox_crs=None):
    """Extract a subset of geometries in a GeoDataFrame that overlap with `im`.

    Notes
    -----
    This function uses RTree's spatialindex, which is much faster (but slightly
    less accurate) than direct comparison of each object for overlap.

    Args:
        gdf : :class:`geopandas.GeoDataFrame`
            A :class:`geopandas.GeoDataFrame` instance or a path to a geojson.
        im : :class:`rasterio.DatasetReader` or `str`, optional
            An image object loaded with `rasterio` or a path to a georeferenced
            image (i.e. a GeoTIFF).
        bbox : `list` or :class:`shapely.geometry.Polygon`, optional
            A bounding box (either a :class:`shapely.geometry.Polygon` or a
            ``[bottom, left, top, right]`` `list`) from an image. Has no effect
            if `im` is provided (`bbox` is inferred from the image instead.) If
            `bbox` is passed and `im` is not, a `bbox_crs` should be provided to
            ensure correct geolocation - if it isn't, it will be assumed to have
            the same crs as `gdf`.
        bbox_crs : int, optional
            The coordinate reference system that the bounding box is in as an EPSG
            int. If not provided, it's assumed that the CRS is the same as `im`
            (if provided) or `gdf` (if not).

    Returns:
        output_gdf : :class:`geopandas.GeoDataFrame`
            A :class:`geopandas.GeoDataFrame` with all geometries in `gdf` that
            overlapped with the image at `im`.
            Coordinates are kept in the CRS of `gdf`.

    """
    if im is None and bbox is None:
        raise ValueError('Either `im` or `bbox` must be provided.')
    gdf = gdf_load(gdf)
    sindex = gdf.sindex
    if im is not None:
        im = rasterio_load(im)
        # currently, convert CRSs to WKT strings here to accommodate rasterio.
        bbox = transform_bounds(im.crs, check_crs(gdf.crs, return_rasterio=True),
                                *im.bounds)
        bbox_crs = im.crs
    # use transform_bounds in case the crs is different - no effect if not
    if isinstance(bbox, Polygon):
        bbox = bbox.bounds
    if bbox_crs is None:
        try:
            bbox_crs = check_crs(gdf.crs, return_rasterio=True)
        except AttributeError:
            raise ValueError('If `im` and `bbox_crs` are not provided, `gdf`'
                             'must provide a coordinate reference system.')
    else:
        bbox_crs = check_crs(bbox_crs, return_rasterio=True)
    # currently, convert CRSs to WKT strings here to accommodate rasterio.
    bbox = transform_bounds(bbox_crs, check_crs(gdf.crs, return_rasterio=True), *bbox)
    try:
        intersectors = list(sindex.intersection(bbox))
    except RTreeError:
        intersectors = []

    return gdf.iloc[intersectors, :]

def gdf_to_yolo(geojson_path="", mask_path="", output_path="", column='value',
                im_size=(0, 0), min_overlap=0.66):
    """Convert a geodataframe containing polygons to yolo/yolt format.

    Args:
        geodataframe : str
            Path to a :class:`geopandas.GeoDataFrame` with a column named
            ``'geometry'``.  Can be created from a geojson with labels for unique
            objects. Can be converted to this format with
            ``geodataframe=gpd.read_file("./xView_30.geojson")``.
        im_path : str
            Path to a georeferenced image (ie a GeoTIFF or png created with GDAL)
            that geolocates to the same geography as the `geojson`(s). If a
            directory, the bounds of each GeoTIFF will be loaded in and all
            overlapping geometries will be transformed. This function will also
            accept a :class:`osgeo.gdal.Dataset` or :class:`rasterio.DatasetReader`
            with georeferencing information in this argument.
        output_dir : str
            Path to an output directory where all of the yolo readable text files
            will be placed.
        column : str, optional
            The column name that contians an unique integer id for each of object
            class.
        im_size : tuple, optional
            A tuple specifying the x and y heighth of a an image.  If specified as
            ``(0,0)`` (the default,) then the size is determined automatically.
        min_overlap : float, optional
            A float value ranging from 0 to 1.  This is a percantage.  If a polygon
            does not overlap the image by at least min_overlap, the polygon is
            discarded.  i.e. 0.66 = 66%. Default value of 0.66.

    Returns:
        gdf : :class:`geopandas.GeoDataFrame`.
            The csv file will be written to the output_dir.
    """

    if im_size == (0, 0):
        src = rasterio_load(mask_path)
        im_size = (src.width, src.height)
            
    gdf = gpd.read_file(geojson_path)

    [x0, y0, x1, y1] = [0, 0, im_size[0], im_size[1]]
    out_coords = [[x0, y0], [x0, y1], [x1, y1], [x1, y0]]
    points = [shapely.geometry.Point(coord) for coord in out_coords]
    pix_poly = shapely.geometry.Polygon([[p.x, p.y] for p in points])
    dw = 1. / im_size[0]
    dh = 1. / im_size[1]
    header = [column, "x", "y", "w", "h"]
    output = output_path
    gdf = geojson_to_px_gdf(gdf, mask_path)
    gdf['area'] = gdf['geometry'].area
    gdf['intersection'] = (
        gdf['geometry'].intersection(pix_poly).area / gdf['area'])
    gdf = gdf[gdf['area'] != 0]
    gdf = gdf[gdf['intersection'] >= min_overlap]
    if not gdf.empty:
        boxy = gdf['geometry'].bounds
        boxy['xmid'] = (boxy['minx'] + boxy['maxx']) / 2.0
        boxy['ymid'] = (boxy['miny'] + boxy['maxy']) / 2.0
        boxy['w0'] = (boxy['maxx'] - boxy['minx'])
        boxy['h0'] = (boxy['maxy'] - boxy['miny'])
        boxy['x'] = boxy['xmid'] * dw
        boxy['y'] = boxy['ymid'] * dh
        boxy['w'] = boxy['w0'] * dw
        boxy['h'] = boxy['h0'] * dh
        if not boxy.empty:
            gdf = gdf.join(boxy)
            gdf.to_csv(path_or_buf=output, sep=' ', columns=header, index=False, header=False)
            logger.info(f"Yolo file saved to {output}")

if __name__ == "__main__":
    pass


