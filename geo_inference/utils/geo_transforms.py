"""Adapted from Solaris: https://github.com/CosmiQ/solaris/tree/main/solaris """

import logging
import os

import numpy as np
import geopandas as gpd
import pandas as pd
import rasterio
import shapely
import json

from shapely.wkt import loads
from affine import Affine
from rasterio import features
from rasterio.warp import transform_bounds
from rtree.core import RTreeError
from shapely.geometry import Polygon, mapping, shape

from ..config.logging_config import logger
from .geo import check_crs, check_geom, df_load, gdf_load, rasterio_load

logger = logging.getLogger(__name__)


def _reduce_geom_precision(geom, precision=2):
    geojson = mapping(geom)
    geojson['coordinates'] = np.round(np.array(geojson['coordinates']),
                                      precision)
    return shape(geojson)

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

def affine_transform_gdf(gdf, affine_obj, inverse=False, geom_col="geometry", precision=None):
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
    if precision is not None:
        gdf['geometry'] = gdf['geometry'].apply(
            _reduce_geom_precision, precision=precision)
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

def bbox_corners_to_coco(bbox):
    """Converts bbox from [minx, miny, maxx, maxy] to COCO format.

    Args:
        bbox (list-like of numerics): A 4-element list of the form [minx, miny, maxx, maxy].

    Returns:
        list: A list of the form [minx, miny, width, height].

    """
    return [bbox[0], bbox[1], bbox[2]-bbox[0], bbox[3]-bbox[1]]

def polygon_to_coco(polygon):
    """Converts a geometry to COCO polygon format.

    Args:
        polygon: A shapely geometry object or a Well-Known Text (WKT) string representing a polygon.

    Returns:
        A list of coordinates in COCO polygon format.

    Raises:
        ValueError: If the polygon is not a shapely geometry object or a WKT string.

    """
    if isinstance(polygon, Polygon):
        coords = polygon.exterior.coords.xy
    elif isinstance(polygon, str):  # assume it's WKT
        coords = loads(polygon).exterior.coords.xy
    else:
        raise ValueError('polygon must be a shapely geometry or WKT.')
    # zip together x,y pairs
    coords = list(zip(coords[0], coords[1]))
    coords = [item for coordinate in coords for item in coordinate]

    return coords

def coco_categories_dict_from_df(df, 
                                 category_id_col, 
                                 category_name_col, 
                                 supercategory_col=None):
    """Extracts category IDs, category names, and supercategory names from a DataFrame.

    Args:
        df (pandas.DataFrame): A DataFrame of records to filter for category info.
        category_id_col (str): The name for the column in `df` that contains category IDs.
        category_name_col (str): The name for the column in `df` that contains category names.
        supercategory_col (str, optional): The name for the column in `df` that contains supercategory names,
            if one exists. If not provided, supercategory will be left out of the output.

    Returns:
        list of dict: A list of dictionaries that contain category records per the COCO dataset specification.

    Raises:
        None

    Examples:
        >>> df = pd.DataFrame({'category_id': [1, 2, 3],
        ...                    'category_name': ['cat', 'dog', 'bird'],
        ...                    'supercategory': ['animal', 'animal', 'animal']})
        >>> coco_categories_dict_from_df(df, 'category_id', 'category_name', 'supercategory')
        [{'id': 1, 'name': 'cat', 'supercategory': 'animal'},
         {'id': 2, 'name': 'dog', 'supercategory': 'animal'},
         {'id': 3, 'name': 'bird', 'supercategory': 'animal'}]
    """
    cols_to_keep = [category_id_col, category_name_col]
    rename_dict = {category_id_col: 'id',
                   category_name_col: 'name'}
    if supercategory_col is not None:
        cols_to_keep.append(supercategory_col)
        rename_dict[supercategory_col] = 'supercategory'
    coco_cat_df = df[cols_to_keep]
    coco_cat_df = coco_cat_df.rename(columns=rename_dict)
    coco_cat_df = coco_cat_df.drop_duplicates()

    return coco_cat_df.to_dict(orient='records')

def _coco_category_name_id_dict_from_list(category_list):
    """Extracts a dictionary mapping category names to category IDs from a list.

    Args:
        category_list (list): A list of category dictionaries.

    Returns:
        dict: A dictionary mapping category names to category IDs.

    """
    # check if this is a full annotation json or just the categories
    category_dict = {category['name']: category['id']
                     for category in category_list}
    return category_dict

def make_coco_image_dict(image_ref, license_id=None):
    """Creates a COCO-formatted image record list from a dictionary of image filenames and IDs.

    Args:
        image_ref (dict): A dictionary of image filenames and their corresponding IDs.
        license_id (int, optional): The license ID number for the relevant license. Defaults to None.

    Returns:
        list: A list of COCO-formatted image records ready for export to JSON.
    """
    image_records = []
    for image_fname, image_id in image_ref.items():
        with rasterio.open(image_fname) as f:
            width = f.width
            height = f.height
        im_record = {'id': image_id,
                     'file_name': os.path.split(image_fname)[1],
                     'width': width,
                     'height': height}
        if license_id is not None:
            im_record['license'] = license_id
        image_records.append(im_record)

    return image_records

def df_to_coco_annos(df, output_path=None, geom_col='geometry',
                     image_id_col=None, category_col=None, score_col=None,
                     preset_categories=None, supercategory_col=None,
                     include_other=True, starting_id=1, verbose=0):
    """Extract COCO-formatted annotations from a pandas DataFrame.

    This function assumes that annotations are already in pixel coordinates.
    If this is not the case, you can transform them using
    solaris.vector.polygon.geojson_to_px_gdf.

    Note that this function generates annotations formatted per the COCO object
    detection specification. For additional information, see
    the COCO dataset specification.

    Args:
        df (pandas.DataFrame): A DataFrame containing geometries to store as annos.
        output_path (str, optional): The output file path to save the COCO-formatted annotations.
        geom_col (str, optional): The name of the column in df that contains geometries.
            The geometries should either be shapely.geometry.Polygon or WKT strings.
            Defaults to "geometry".
        image_id_col (str, optional): The column containing image IDs.
            If not provided, it's assumed that all are in the same image, which will be assigned the ID of 1.
        category_col (str, optional): The name of the column that specifies categories for each object.
            If not provided, all objects will be placed in a single category named "other".
        score_col (str, optional): The name of the column that specifies the output confidence of a model.
            If not provided, will not be output.
        preset_categories (list of dict, optional): A pre-set list of categories to use for labels.
            These categories should be formatted per the COCO category specification.
        supercategory_col (str, optional): The name of the column that specifies the supercategory for each category.
        include_other (bool, optional): Whether to include objects not contained in preset_categories.
            Defaults to True.
        starting_id (int, optional): The number to start numbering annotation IDs at. Defaults to 1.
        verbose (int, optional): Verbose text output. By default, none is provided;
            if True or 1, information-level outputs are provided; if 2, extremely verbose text is output.

    Returns:
        dict: A dictionary containing COCO-formatted annotation and category entries
            per the COCO dataset specification.
    """
    df = df_load(df)
    temp_df = df.copy()  # for manipulation
    if preset_categories is not None and category_col is None:
        logger.debug('preset_categories has a value, category_col is None.')
        raise ValueError('category_col must be specified if using'
                         ' preset_categories.')
    elif preset_categories is not None and category_col is not None:
        logger.debug('Both preset_categories and category_col have values.')
        logger.debug('Getting list of category names.')
        category_dict = _coco_category_name_id_dict_from_list(preset_categories)
        category_names = list(category_dict.keys())
        if not include_other:
            logger.debug('Filtering out objects not contained in '
                        ' preset_categories')
            temp_df = temp_df.loc[temp_df[category_col].isin(category_names),:]
        else:
            logger.debug('Setting category to "other" for objects outside of '
                        'preset category list.')
            temp_df.loc[~temp_df[category_col].isin(category_names),
                        category_col] = 'other'
            if 'other' not in category_dict.keys():
                logger.debug('Adding "other" to category_dict.')
                other_id = np.array(list(category_dict.values())).max() + 1
                category_dict['other'] = other_id
                preset_categories.append({'id': other_id,
                                          'name': 'other',
                                          'supercategory': 'other'})
    elif preset_categories is None and category_col is not None:
        logger.debug('No preset_categories, have category_col.')
        logger.debug(f'Collecting unique category names from {category_col}.')
        category_names = list(temp_df[category_col].unique())
        logger.debug('Generating category ID numbers arbitrarily.')
        category_dict = {k: v for k, v in zip(category_names,
                                              range(1, len(category_names)+1))}
    else:
        logger.debug('No category column or preset categories.')
        logger.debug('Setting category to "other" for all objects.')
        category_col = 'category_col'
        temp_df[category_col] = 'other'
        category_names = ['other']
        category_dict = {'other': 1}

    if image_id_col is None:
        temp_df['image_id'] = 1
    else:
        temp_df.rename(columns={image_id_col: 'image_id'})
    logger.debug('Checking geometries.')
    temp_df[geom_col] = temp_df[geom_col].apply(check_geom)
    logger.debug('Getting area of geometries.')
    temp_df['area'] = temp_df[geom_col].apply(lambda x: x.area)
    logger.debug('Getting geometry bounding boxes.')
    temp_df['bbox'] = temp_df[geom_col].apply(lambda x: bbox_corners_to_coco(x.bounds))
    temp_df['category_id'] = temp_df[category_col].map(category_dict)
    temp_df['annotation_id'] = list(range(starting_id,
                                          starting_id + len(temp_df)))
    if score_col is not None:
        temp_df['score'] = df[score_col]

    def _row_to_coco(row, geom_col, category_id_col, image_id_col, score_col):
        "get a single annotation record from a row of temp_df."
        if score_col is None:

            return {'id': row['annotation_id'],
                    'image_id': int(row[image_id_col]),
                    'category_id': int(row[category_id_col]),
                    'segmentation': [polygon_to_coco(row[geom_col])],
                    'area': row['area'],
                    'bbox': row['bbox'],
                    'iscrowd': 0}
        else:
            return {'id': row['annotation_id'],
                    'image_id': int(row[image_id_col]),
                    'category_id': int(row[category_id_col]),
                    'segmentation': [polygon_to_coco(row[geom_col])],
                    'score': float(row[score_col]),
                    'area': row['area'],
                    'bbox': row['bbox'],
                    'iscrowd': 0}

    coco_annotations = temp_df.apply(_row_to_coco, axis=1, geom_col=geom_col,
                                     category_id_col='category_id',
                                     image_id_col=image_id_col,
                                     score_col=score_col).tolist()
    coco_categories = coco_categories_dict_from_df(
        temp_df, category_id_col='category_id',
        category_name_col=category_col,
        supercategory_col=supercategory_col)

    output_dict = {'annotations': coco_annotations,
                   'categories': coco_categories}

    if output_path is not None:
        with open(output_path, 'w') as outfile:
            json.dump(output_dict, outfile)

    return output_dict
