# Geo Inference

[![PyPI - Version](https://img.shields.io/pypi/v/geo-inference)](https://pypi.org/project/geo-inference/)
[![Codecov](https://img.shields.io/codecov/c/github/valhassan/geo-inference)](https://app.codecov.io/github/valhassan/geo-inference)
[![tests](https://github.com/valhassan/geo-inference/actions/workflows/test.yml/badge.svg)](https://github.com/valhassan/geo-inference/actions/workflows/test.yml)





geo-inference is a Python package designed for feature extraction from geospatial imagery using compatible deep learning models. It provides a convenient way to extract features from large TIFF images and save the output mask as a TIFF file. It also supports converting the output mask to vector format (*file_name.geojson*), YOLO format (*file_name.csv*), and COCO format (*file_name.json*). This package is particularly useful for applications in remote sensing, environmental monitoring, and urban planning.

## Installation

Geo-inference requires Python 3.11. To install the package, use:

```
pip install geo-inference
```

## Usage

**Input:** GeoTiffs with compatible TorchScript model. For example: A pytorch model trained on high resolution geospatial imagery with the following features:

- pixel size (0.1m to 3m)
- data type (uint8)

expects an input image with the same features. An example notebook for how the package is used is provided in this repo. 


*Here's an example of how to use Geo Inference (Command line and Script):*

**Command line**
```bash
python geo_inference -a <args>
```
- `-a`, `--args`: Path to arguments stored in yaml, consult ./config/sample_config.yaml
```bash
python geo_inference -i <image> -br <bands_requested> -m <model> -wd <work_dir> -ps <patch_size> -v <vec> -d <device> -id <gpu_id> -cls <classes> -mg <mgpu>
```
- `-i`, `--image`: Path to Geotiff
- `-bb`, `--bbox`: AOI bbox in this format "minx, miny, maxx, maxy" (Optional)
- `-br`, `--bands_requested`: The requested bands from provided Geotiff (if not provided, it uses all bands)
- `-m`, `--model`: Path or URL to the model file
- `-wd`, `--work_dir`: Working Directory
- `-ps`, `--patch_size`: The patch Size, the size of dask chunks, Default = 1024
- `-v`, `--vec`: Vector Conversion
- `-y`, `--yolo`: Yolo Conversion
- `-c`, `--coco`: Coco Conversion
- `-d`, `--device`: CPU or GPU Device
- `-id`, `--gpu_id`: GPU ID, Default = 0
- `-cls`, `--classes`: The number of classes that model outputs, Default = 5
- `-mg`, `--mgpu`: Whether to use multi-gpu processing or not, Default = False


You can also use the `-h` option to get a list of supported arguments:

```bash
python geo_inference -h
```

**Import script**
```python
from geo_inference.geo_inference import GeoInference

# Initialize the GeoInference object
geo_inference = GeoInference(
    model="/path/to/segformer_B5.pt",
    work_dir="/path/to/work/dir",
    patch_size=1024,
    mask_to_vec=False,
    mask_to_yolo=False,
    mask_to_coco=False, 
    device="gpu",
    multi_gpu=False,
    gpu_id=0, 
    num_classes=5
)

# Perform feature extraction on a TIFF image
image_path = "/path/to/image.tif"
patch_size = 512
geo_inference(tiff_image = image_path,  bands_requested = bands_requested, patch_size = patch_size,)
```

## Parameters

The `GeoInference` class takes the following parameters:

- `model`: The path or URL to the model file (.pt for PyTorch models) to use for feature extraction.
- `work_dir`: The path to the working directory. Default is `"~/.cache"`.
- `patch_size`: The patch size to use for feature extraction. Default is `4`.
- `mask_to_vec`: If set to `"True"`, vector data will be created from mask. Default is `"False"`
- `mask_to_yolo`: If set to `"True"`, vector data will be converted to YOLO format. Default is `"False"`
- `mask_to_coco`: If set to `"True"`, vector data will be converted to COCO format. Default is `"False"`
- `device`: The device to use for feature extraction. Can be `"cpu"` or `"gpu"`. Default is `"gpu"`.
- `multi_gpu`: If set to `"True"`, uses multi-gpu for running the inference. Default is `"False"`
- `gpu_id`: The ID of the GPU to use for feature extraction. Default is `0`.
- `num_classes`: The number of classes that the TorchScript model outputs. Default is `5`.
## Output

The `GeoInference` class outputs the following files:

- `mask.tif`: The output mask file in TIFF format.
- `polygons.geojson`: The output polygon file in GeoJSON format. This file is only generated if the `mask_to_vec` parameter is set to `True`.
- `yolo.csv`: The output YOLO file in CSV format. This file is only generated if the `mask_to_vec`, `vec_to_yolo` parameters are set to `True`.
- `coco.json`: The output COCO file in JSON format. This file is only generated if the `mask_to_vec`, `vec_to_coco` parameters are set to `True`.

Each file contains the extracted features from the input geospatial imagery.

## License

Geo Inference is released under the MIT License. See [`LICENSE`](https://github.com/NRCan/geo-inference/blob/main/LICENSE) for more information.

## Contact

For any questions or concerns, please open an issue on GitHub.