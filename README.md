# Geo Inference

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
python geo_inference.py -a <args>
```
- `-a`, `--args`: Path to arguments stored in yaml, consult ./config/sample_config.yaml
```bash
python geo_inference.py -i <image> -m <model> -wd <work_dir> -bs <batch_size> -v <vec> -d <device> -id <gpu_id>
```
- `-i`, `--image`: Path to Geotiff
- `-bb`, `--bbox`: AOI bbox in this format "minx, miny, maxx, maxy" (Optional)
- `-m`, `--model`: Path or URL to the model file
- `-wd`, `--work_dir`: Working Directory
- `-bs`, `--batch_size`: The Batch Size
- `-v`, `--vec`: Vector Conversion
- `-d`, `--device`: CPU or GPU Device
- `-id`, `--gpu_id`: GPU ID, Default = 0

You can also use the `-h` option to get a list of supported arguments:

```bash
python geo_inference.py -h
```

**Import script**
```python
from geo_inference.geo_inference import GeoInference

# Initialize the GeoInference object
geo_inference = GeoInference(
    model="/path/to/segformer_B5.pt",
    work_dir="/path/to/work/dir",
    batch_size=4,
    mask_to_vec=True,
    device="gpu",
    gpu_id=0
)

# Perform feature extraction on a TIFF image
image_path = "/path/to/image.tif"
patch_size = 512
stride_size = 256
geo_inference(image_path, patch_size, stride_size)
```

## Parameters

The `GeoInference` class takes the following parameters:

- `model`: The path or URL to the model file (.pt for PyTorch models) to use for feature extraction.
- `work_dir`: The path to the working directory. Default is `"~/.cache"`.
- `batch_size`: The batch size to use for feature extraction. Default is `4`.
- `mask_to_vec`: If set to `"True"`, vector files will be created. Default is `"False"`
- `device`: The device to use for feature extraction. Can be `"cpu"` or `"gpu"`. Default is `"gpu"`.
- `gpu_id`: The ID of the GPU to use for feature extraction. Default is `0`.

## Output

The `GeoInference` class outputs the following files:

- `mask.tif`: The output mask file in TIFF format.
- `polygons.geojson`: The output polygon file in GeoJSON format. This file is only generated if the `mask_to_vec` parameter is set to `True`.
- `yolo.csv`: The output YOLO file in CSV format. This file is only generated if the `mask_to_vec` parameter is set to `True`.

Each file contains the extracted features from the input geospatial imagery.

## License

Geo Inference is released under the MIT License. See `LICENSE` for more information.

## Contact

For any questions or concerns, please open an issue on GitHub.