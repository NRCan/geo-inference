import os
import re
import sys
import time
import torch
import psutil
import logging
import tracemalloc

import numpy as np
import dask.array as da

from typing import Dict
from pathlib import Path
from rasterio.io import DatasetReader
from rasterio.windows import from_bounds
from dask.distributed import LocalCluster
from dask.distributed import Client as daskclient
from dask_image.imread import imread as dask_imread  # type: ignore

if str(Path(__file__).parents[0]) not in sys.path:
    sys.path.insert(0, str(Path(__file__).parents[0]))

from utils.helpers import (
    cmd_interface,
    get_directory,
    get_model,
    display_top_memory,
    get_tiff_paths_from_csv,
)
from geo_dask import (
    dask_imread_modified,
    equalize_adapthist_enhancement,
    runModel_partial_neighbor,
    sum_overlapped_chunks,
    write_inference_to_tiff,
    aoi,
)
from utils.polygon import gdf_to_yolo, mask_to_poly_geojson, geojson2coco


logger = logging.getLogger(__name__)


class GeoInferenceDask:
    """
    A class for performing geo inference on geospatial imagery using a pre-trained model.

    Args:
        model (str): The path or url to the model file
        work_dir (str): The directory where the model and output files will be saved.
        mask_to_vec (bool): Whether to convert the output mask to vector format.
        device (str): The device to use for inference (either "cpu" or "gpu").
        gpu_id (int): The ID of the GPU to use for inference (if device is "gpu").
        num_classes (int) : The number of classes in the output of the model.

    Attributes:
        work_dir (Path): The directory where the model and output files will be saved.
        mask_to_vec (bool): Whether to convert the output mask to vector format.
        model_path (path): The path to the pre-trained model to use for inference.
        classes (int): The number of classes in the output of the model.

    """

    def __init__(
        self,
        model: str = None,
        work_dir: str = None,
        mask_to_vec: bool = False,
        multi_gpu: bool = False,
        gpu_id: int = 0,
        num_classes: int = 5,
    ):
        self.gpu_id = int(gpu_id)
        self.multi_gpu = multi_gpu
        self.work_dir: Path = get_directory(work_dir)
        self.model_path: Path = get_model(
            model_path_or_url=model, work_dir=self.work_dir
        )
        self.mask_to_vec = mask_to_vec
        self.classes = num_classes

    @torch.no_grad()
    def __call__(
        self,
        parsed_bands: Dict[str, str],
        raster_reader: DatasetReader,
        chunk_size: int = 1024,
        n_workers: int = 8,
        memory_limit: str = "100GB",
        bbox: str = None,
        enhance_clip_limit: int = 25,
        high_or_low_contrast: bool = False,
        stride_size: int = 2,
    ) -> None:
        """
        Perform geo inference on geospatial imagery using dask array.

        Args:
            parsed_bands (Dict[str, str]): The dictionary of {band: path/url} to the geospatial image to perform inference on.
            chunk_size (int): The size for dask chunked data.
            n_workers (int) : The number of available cores for creating dask cluster.
            memory_limit (str) : The available memory for each core in dask cluster. Note: if the core reaches 70% of this memory it spills its data to disk.
            bbox (str): The bbox or extent of the image in this format "minx, miny, maxx, maxy".
            enhance_clip_limit (int): The clahe limit for image enhancement.
            stride_size (int): The stride to use between patches.

        Returns:
            None

        """

        if len(parsed_bands) == 0:
            logger.error("The given list of bands is empty")
        base_name = os.path.basename(Path(list(parsed_bands.values())[0]))
        prefix_base_name = re.match(r"^(.*?)-[A-Za-z]\.tif$", base_name).group(0)

        mask_path = self.work_dir.joinpath(prefix_base_name + "_mask.tif")
        polygons_path = self.work_dir.joinpath(prefix_base_name + "_polygons.geojson")
        yolo_csv_path = self.work_dir.joinpath(prefix_base_name + "_yolo.csv")
        coco_json_path = self.work_dir.joinpath(prefix_base_name + "_coco.json")

        """ Processing starts"""
        tracemalloc.start()  # For memory tracking
        cluster = LocalCluster(
            n_workers=n_workers,
            memory_limit=memory_limit,
            local_directory=self.work_dir.joinpath("scratch"),
        )
        process = psutil.Process()  # For cpu tracking
        start_time = time.time()
        base_shape = 5000
        with daskclient(cluster, timeout="60s") as client:
            try:
                if len(parsed_bands) == 1:
                    aoi_dask_array = dask_imread(list(parsed_bands.values())[0])
                    aoi_dask_array = da.transpose(da.squeeze(aoi_dask_array), (2, 0, 1))
                else:
                    aoi_dask_array = [
                        dask_imread_modified(url) for band, url in parsed_bands.items()
                    ]
                    aoi_dask_array = da.stack(aoi_dask_array, axis=0)
                    aoi_dask_array = da.squeeze(
                        da.transpose(aoi_dask_array, (1, 0, 2, 3))
                    )

                x_chunk = aoi.chunk_size
                y_chunk = aoi.chunk_size
                if aoi_dask_array.shape[1] <= base_shape:
                    y_chunk = int(aoi_dask_array.shape[1] / 2)
                elif (
                    aoi_dask_array.shape[1] <= base_shape * 3
                    and aoi_dask_array.shape[1] > base_shape
                ):
                    y_chunk = int(aoi_dask_array.shape[1] / 4)
                elif (
                    aoi_dask_array.shape[1] <= base_shape * 6
                    and aoi_dask_array.shape[1] > base_shape * 3
                ):
                    y_chunk = int(aoi_dask_array.shape[1] / 6)
                elif (
                    aoi_dask_array.shape[1] < base_shape * 8
                    and aoi_dask_array.shape[1] > base_shape * 6
                ):
                    y_chunk = int(aoi_dask_array.shape[1] / 15)

                if aoi_dask_array.shape[2] <= base_shape:
                    x_chunk = int(aoi_dask_array.shape[1] / 2)
                elif (
                    aoi_dask_array.shape[2] <= base_shape * 3
                    and aoi_dask_array.shape[2] > base_shape
                ):
                    x_chunk = int(aoi_dask_array.shape[2] / 4)
                elif (
                    aoi_dask_array.shape[2] <= base_shape * 6
                    and aoi_dask_array.shape[2] > base_shape * 3
                ):
                    x_chunk = int(aoi_dask_array.shape[2] / 12)
                elif (
                    aoi_dask_array.shape[2] < base_shape * 8
                    and aoi_dask_array.shape[2] > base_shape * 6
                ):
                    x_chunk = int(aoi_dask_array.shape[2] / 20)

                aoi_dask_array = aoi_dask_array.rechunk(
                    (
                        1,
                        y_chunk,
                        x_chunk,
                    )
                )
                logger.info(
                    "The dashboard link for dask cluster is at "
                    f"{client.dashboard_link}"
                )
                if bbox is not None:
                    bbox = tuple(map(float, bbox.split(", ")))
                    bbox = from_bounds(
                        left=bbox[0],
                        bottom=bbox[1],
                        right=bbox[2],
                        top=bbox[3],
                        transform=raster_reader.transform,
                    )
                    col_off, row_off = (
                        bbox.col_off,
                        bbox.row_off,
                    )
                    width, height = bbox.width, bbox.height
                    aoi_dask_array = aoi_dask_array[
                        :, row_off : row_off + height, col_off : col_off + width
                    ]
                if enhance_clip_limit > 0 and high_or_low_contrast:
                    logger.info("Performing image enhancement on the dask array ")
                    aoi_dask_array = da.map_overlap(
                        equalize_adapthist_enhancement,
                        aoi_dask_array,
                        clip_limit=enhance_clip_limit,
                        depth={
                            1: y_chunk,
                            2: x_chunk,
                        },
                        trim=True,
                        boundary="reflect",
                        dtype=np.int32,
                    )
                pad_height = (
                    int(chunk_size / stride_size)
                    - aoi_dask_array.shape[1] % int(chunk_size / stride_size)
                ) % int(chunk_size / stride_size)
                pad_width = (
                    int(chunk_size / stride_size)
                    - aoi_dask_array.shape[2] % int(chunk_size / stride_size)
                ) % int(chunk_size / stride_size)
                # Pad the array to make dimensions multiples of the chunk size
                aoi_dask_array = da.pad(
                    aoi_dask_array,
                    ((0, 0), (0, pad_height), (0, pad_width)),
                    mode="constant",
                ).rechunk(
                    (
                        3,  # assuming that alwasy we have RGB
                        int(chunk_size / stride_size),
                        int(chunk_size / stride_size),
                    )
                )  # now we rechunk data so that each chunk has 3 bands
                logger.info(
                    "The dask array to be fed to Inference model is \n"
                    f"{aoi_dask_array}"
                )
                # Run the model and gather results
                aoi_dask_array = da.map_overlap(
                    runModel_partial_neighbor,
                    aoi_dask_array,
                    chunk_size=chunk_size,
                    model_path=self.model_path,
                    multi_gpu=self.multi_gpu,
                    gpu_id=self.gpu_id,
                    num_classes=self.classes,
                    chunks=(
                        self.classes + 1,
                        chunk_size,
                        chunk_size,
                    ),
                    depth={
                        1: int(chunk_size / stride_size),
                        2: int(chunk_size / stride_size),
                    },
                    trim=False,
                    boundary="none",
                    dtype=np.float16,
                )
                logger.info(
                    "The dask array to be fed to sum_overlapped_chunks is \n"
                    f"{aoi_dask_array}"
                )
                aoi_dask_array = da.map_overlap(
                    sum_overlapped_chunks,
                    aoi_dask_array,
                    drop_axis=0,
                    chunk_size=chunk_size,
                    chunks=(
                        int(chunk_size / stride_size),
                        int(chunk_size / stride_size),
                    ),
                    depth={
                        1: int(chunk_size / stride_size),
                        2: int(chunk_size / stride_size),
                    },
                    allow_rechunk=True,
                    trim=False,
                    boundary="none",
                    dtype=np.uint8,
                )
                logger.info(
                    f"The Inference output dask arrray is \n" f"{aoi_dask_array}"
                )

                all_processed_chunks = client.gather(aoi_dask_array)
                model_mask = all_processed_chunks.compute()
                write_inference_to_tiff(raster_reader, model_mask, mask_path)
                if self.mask_to_vec:
                    mask_to_poly_geojson(mask_path, polygons_path)
                    gdf_to_yolo(polygons_path, mask_path, yolo_csv_path)
                    geojson2coco(mask_path, polygons_path, coco_json_path)

                total_time = time.time() - start_time
                hours, remainder = divmod(total_time, 3600)
                minutes, seconds = divmod(remainder, 60)
                logger.info(
                    f"The total time of running Inference is {int(hours)}:{int(minutes)}:{seconds:.2f} \n"
                )

                # Print Performance results
                torch.cuda.synchronize()
                logger.info(
                    f"Memory usage for the Inference is: {process.memory_info().rss / 1024 ** 2} MB"
                )
                logger.info(
                    f"CPU usage for the Inference is: {psutil.cpu_percent(interval=1)}%"
                )
                snapshot = tracemalloc.take_snapshot()  # memory track
                display_top_memory(snapshot)
                torch.cuda.empty_cache()
            except Exception as e:
                print(f"Processing on the Dask cluster failed due to: {e}")
                client.close()
                raise e


def main() -> None:
    arguments = cmd_interface()
    geo_inference = GeoInferenceDask(
        model=arguments["model"],
        work_dir=arguments["work_dir"],
        mask_to_vec=arguments["vec"],
        multi_gpu=arguments["multi_gpu"],
        gpu_id=arguments["gpu_id"],
        num_classes=arguments["classes"],
    )
    inference_data = get_tiff_paths_from_csv(arguments["data_dir"])
    for aoi_dict in inference_data:
        _aoi = aoi(
            aoi_dict=aoi_dict,
            bands_requested=list(arguments["bands_requested"].split(",")),
            work_dir=arguments["work_dir"],
        )
        geo_inference(
            parsed_bands=_aoi.raster_parsed,
            raster_reader=_aoi.raster,
            chunk_size=arguments["chunk_size"],
            n_workers=arguments["n_workers"],
            memory_limit=arguments["memory_limit"],
            high_or_low_contrast=_aoi.high_or_low_contrast,
        )

    """
    How to run this script?
        python /gpfs/fs5/nrcan/nrcan_geobase/work/dev/datacube/parallel/geo_inference/geo-inference-dask/geo_inference/geo_inference_dask.py --args /gpfs/fs5/nrcan/nrcan_geobase/work/dev/datacube/parallel/geo_inference/geo-inference-dask/geo_inference/config/sample.yaml 

    """


if __name__ == "__main__":
    main()
