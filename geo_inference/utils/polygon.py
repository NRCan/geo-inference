"""Adapted from Solaris: https://github.com/CosmiQ/solaris/tree/main/solaris """

import logging
import json
import pandas as pd
import geopandas as gpd
import rasterio
import shapely
from rasterio import features
from shapely.geometry import shape

from ..config.logging_config import logger
from .geo import rasterio_load
from .geo_transforms import geojson_to_px_gdf, df_to_coco_annos, make_coco_image_dict

logger = logging.getLogger(__name__)

def mask_to_poly_geojson(mask_path,
                         output_path="", 
                         min_area=40,
                         simplify=False,
                         tolerance=1.):
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
            The column name that contains a unique integer id for each of object
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

def geojson2coco(image_src, label_src, output_path=None, category_attribute="value", score_attribute=None,
                 preset_categories=None, include_other=True, info_dict=None,
                 license_dict=None, verbose=0):
    """Generate COCO-formatted labels from mask and polygon geojson.

    Args:
        image_src (str): A string path to an image.
        label_src (str): A string path to a geojson.
        output_path (str, optional): The path to save the JSON-formatted COCO records to. If not provided,
            the records will only be returned as a dict, and not saved to file.
        category_attribute (str, optional): The name of an attribute in the geojson that specifies which category
            a given instance corresponds to. If not provided, it's assumed that only one class of object is present
            in the dataset, which will be termed "other" in the output json.
        score_attribute (str, optional): The name of an attribute in the geojson that specifies the prediction
            confidence of a model.
        preset_categories (list of dicts, optional): A pre-set list of categories to use for labels. These categories
            should be formatted per the COCO category specification. Example:
            [{'id': 1, 'name': 'Fighter Jet', 'supercategory': 'plane'},
            {'id': 2, 'name': 'Military Bomber', 'supercategory': 'plane'}, ... ]
        include_other (bool, optional): If set to True, and preset_categories is provided, objects that don't fall
            into the specified categories will not be removed from the dataset. They will instead be passed into a
            category named "other" with its own associated category id. If False, objects whose categories don't match
            a category from preset_categories will be dropped.
        info_dict (dict, optional): A dictionary with the following key-value pairs:
            - "year": int year of creation
            - "version": str version of the dataset
            - "description": str string description of the dataset
            - "contributor": str who contributed the dataset
            - "url": str URL where the dataset can be found
            - "date_created": datetime.datetime when the dataset was created
        license_dict (dict, optional): A dictionary containing the licensing information for the dataset, with the
            following key-value pairs:
            - "name": str the name of the license.
            - "url": str a link to the dataset's license.
            Note: This implementation assumes that all of the data uses one license. If multiple licenses are provided,
            the image records will not be assigned a license ID.
        verbose (int, optional): Verbose text output. By default, none is provided; if True or 1, information-level
            outputs are provided; if 2, extremely verbose text is output.

    Returns:
        dict: A dictionary following the COCO dataset specification. Depending on arguments provided, it may or may
        not include license and info metadata.
    """
    logger.debug('Loading labels.')
    label_df = pd.DataFrame({'label_fname': [],
                             'category_str': [],
                             'geometry': []})
    curr_gdf = gpd.read_file(label_src)

    curr_gdf['label_fname'] = label_src
    curr_gdf['image_fname'] = ''
    curr_gdf['image_id'] = 1
    if category_attribute is None:
        logger.debug('No category attribute provided. Creating a default "other" category.')
        curr_gdf['category_str'] = 'other'  # add arbitrary value
        tmp_category_attribute = 'category_str'
    else:
        tmp_category_attribute = category_attribute
   
    logger.debug('Converting to pixel coordinates.')
    curr_gdf = geojson_to_px_gdf(curr_gdf, image_src)
    curr_gdf = curr_gdf.rename(columns={tmp_category_attribute: 'category_str'})
    if score_attribute is not None:
        curr_gdf = curr_gdf[['image_id', 'label_fname', 'category_str', score_attribute, 'geometry']]
    else:
        curr_gdf = curr_gdf[['image_id', 'label_fname', 'category_str', 'geometry']]
    label_df = pd.concat([label_df, curr_gdf], axis='index', ignore_index=True, sort=False)

    logger.info('Generating COCO-formatted annotations.')
    coco_dataset = df_to_coco_annos(label_df,
                                    geom_col='geometry',
                                    image_id_col='image_id',
                                    category_col='category_str',
                                    score_col=score_attribute,
                                    preset_categories=preset_categories,
                                    include_other=include_other,
                                    verbose=verbose)

    logger.debug('Generating COCO-formatted image and license records.')
    if license_dict is not None:
        logger.debug('Getting license ID.')
        if len(license_dict) == 1:
            logger.debug('Only one license present; assuming it applies to all images.')
            license_id = 1
        else:
            logger.debug('Zero or multiple licenses present. Not trying to match to images.')
            license_id = None
        logger.debug('Adding licenses to dataset.')
        coco_licenses = []
        license_idx = 1
        for license_name, license_url in license_dict.items():
            coco_licenses.append({'name': license_name,
                                  'url': license_url,
                                  'id': license_idx})
            license_idx += 1
        coco_dataset['licenses'] = coco_licenses
    else:
        logger.debug('No license information provided, skipping for image COCO records.')
        license_id = None
    coco_image_records = make_coco_image_dict({image_src: 1}, license_id)
    coco_dataset['images'] = coco_image_records

    if info_dict is not None:
        coco_dataset['info'] = info_dict

    
    with open(output_path, 'w') as outfile:
        json.dump(coco_dataset, outfile)
    logger.info(f"CocoJson file saved to {output_path}")


