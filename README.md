# Geo Inference

Geo Inference is a light-weight Python package for performing feature extraction from high-resolution imagery using custom pre-trained foundation models. It provides a convenient way to extract features from large TIFF images and save the output as a TIFF file. It also supports converting the output mask to a polygon GeoJSON file and a YOLO CSV file.

## Installation

To install Geo Inference, you can use pip:

```
pip install geo-inference
```

## Usage

Here's an example of how to use Geo Inference (Command line and Script):

**Command line**
```bash
python geo_inference.py -a <args>
```
- `-a`, `--args`: Path to arguments stored in yaml, consult ./config/sample_config.yaml
```bash
python geo_inference.py -i <image> -m <model> -wd <work_dir> -bs <batch_size> -v <vec> -d <device> -id <gpu_id>
```
- `-i`, `--image`: Path to Geotiff
- `-m`, `--model`: Name of Extraction Model
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
from geo_inference import GeoInference

# Initialize the GeoInference object
geo_inference = GeoInference(
    model_name="segformer_B5",
    work_dir="/path/to/work/dir",
    batch_size=4,
    device_type="gpu",
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

- `model_name`: The name of the model to use for feature extraction.
- `work_dir`: The path to the working directory. Default is `"~/.cache"`.
- `batch_size`: The batch size to use for feature extraction. Default is `4`.
- `mask_to_vec`: The bool value to create vector files. Default is `"False"`
- `device`: The device to use for feature extraction. Can be `"cpu"` or `"gpu"`. Default is `"gpu"`.
- `gpu_id`: The ID of the GPU to use for feature extraction. Default is `0`.

## Output

The `GeoInference` class outputs the following files:

- `mask.tif`: The output mask file in TIFF format.
- `polygons.geojson`: The output polygon file in GeoJSON format. This file is only generated if the `mask_to_vec` parameter is set to `True`.
- `yolo.csv`: The output YOLO file in CSV format. This file is only generated if the `mask_to_vec` parameter is set to `True`.

## Available Models
- `RGB_4class_Segformer_B5`: Not released to the public.
- `RBG_4class_HRNet_W48`: Not released to the public.

## License

Geo Inference is released under the MIT License. See `LICENSE` for more information.

<!--  
## Acknowledgments

This project was inspired by the [SpaceNet Challenge](https://spacenet.ai/).

-->