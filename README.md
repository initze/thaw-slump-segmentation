# Thaw Slump Segmentation

## Installation

### Environment setup

#### Python / conda

We recommend using a new conda environment from scratch

```bash
conda create -n thaw_slump_segmentation python=3.10 mamba -c conda-forge
conda activate thaw_slump_segmentation
```

#### gdal

gdal incl. gdal-utilities (preferably version >=3.6) need to be installed in your environment, e.g. with conda/mamba

```bash
mamba install gdal>=3.6 -c conda-forge
```

### Package Installation

* Latest development version: `pip install git+https://github.com/initze/thaw-slump-segmentation`
* Latest release: `pip install https://github.com/initze/thaw-slump-segmentation/releases/download/untagged-f6739f56e0ee4c2c64fe/thaw_slump_segmentation-0.10.0-py3-none-any.whl`

This will pull the CUDA 12 version of pytorch. If you are running CUDA 11, you need to manually switch to the corresponding Pytorch package afterwards by running `pip3 install torch==2.2.0+cu118 torchvision==0.17.0+cu118 --index-url https://download.pytorch.org/whl/cu118`

### Additional packages

#### cucim

You can install cucim to speed up the postprocessing process. cucim will use the gpu to perform binary erosion of edge artifacts, which runs alot faster than the standard CPU implementation of scikit-learn.

`pip install --extra-index-url=https://pypi.nvidia.com cucim-cu11==24.4.*`

Installation for other cuda versions see here:

<https://docs.rapids.ai/install>

## System and Data Setup

### Option 1 - Singularity container

<https://cloud.sylabs.io/library/initze/aicore/thaw_slump_segmentation>

The container contains all requirements to run the processing code, singularity must be installed

```
singularity pull library://initze/aicore/thaw_slump_segmentation
singularity shell --nv --bind <your bind path> thaw_slump_segmentation.sif
```

### Option 2 - anaconda

### Environment setup

We recommend using a new conda environment from the provided environment.yml file

```bash
conda env create -n aicore -f environment.yml
```

### Data Preparation

1. copy/move data into <DATA_DIR>/input
2. copy/move data into <DATA_DIR>/auxiliary (e.g. prepared ArcticDEM data)

### Set gdal paths in system.yml file

#### Linux

```yaml
gdal_path: '$CONDA_PREFIX/bin' # must be single quote
gdal_bin: '$CONDA_PREFIX/bin' # must be single quote
```

#### Windows version

```yaml
gdal_path: '%CONDA_PREFIX%\Scripts' # must be single quote
gdal_bin: '%CONDA_PREFIX%\Library\bin' # must be single quote
```

## CLI

Run in dev:

```sh
$ rye run thaw-slump-segmentation hello tobi
Hello tobi
```

or run as python module:

```sh
$ rye run python -m thaw_slump_segmentation hello tobi
Hello tobi
```

With activated env, e.g. after installation, just remove the `rye run`:

```sh
$ source .venv/bin/activate
$ thaw-slump-segmentation hello tobi
Hello tobi
```

or

```sh
$ source .venv/bin/activate
$ python -m thaw_slump_segmentation hello tobi
Hello tobi
```

## Data Processing

### Data Preprocessing for Planet data

#### Setting up all required files for training and/or inference

```bash
python setup_raw_data.py --data_dir <DATA_DIR>
```

#### Setting up required files for training

```bash
python prepare_data.py --data_dir <DATA_DIR>
```

### Data Preprocessing for Sentinel 2 data to match planet format

```bash
python download_s2_4band_planet_format.py --s2id <IMAGE_ID> --data_dir <DATA_DIR>
```

### Training a model

```bash
python train.py --data_dir <DATA_DIR> -n <MODEL_NAME>
```

### Running Inference

```bash
python setup_raw_data.py --data_dir <DATA_DIR> --nolabel
python inference.py --data_dir <DATA_DIR> --model_dir <MODEL_NAME> 20190727_160426_104e 20190709_042959_08_1057
```

## Configuration

### Run Configuration

Configuration is done via the `config.yml` file. Example config:

```yaml
# Model Specification
model:
  # Model Architecture. Available:
  # Unet, UnetPlusPlus, MAnet, Linknet, FPN, PSPNet, DeepLabV3, DeepLabV3Plus, PAN]
  architecture: Unet
  # Model Encoder. Examples:
  # resnet18, resnet34, resnet50, resnet101, resnet152
  # Check https://github.com/qubvel/segmentation_models.pytorch#encoders for the full list of available encoders
  encoder: resnet34
  # Encoder weights to use (if transfer learning is desired)
  # `imagenet` is available for all encoders, some of them have more options available
  # `random` initializes the weights randomly
  # Check https://github.com/qubvel/segmentation_models.pytorch#encoders for the
  # full list of weights available for each encoder
  encoder_weights: imagenet
# Loss Function to use. Available:
# JaccardLoss, DiceLoss, FocalLoss, LovaszLoss, SoftBCEWithLogitsLoss
loss_function: FocalLoss
# Data Configuration
data_threads: 4  # Number of threads for data loading, must be 0 on Windows
data_sources:  # Enabled input features
  - planet
  - ndvi
  - tcvis
  - relative_elevation
  - slope
datasets:
  train:
    augment: true
    augment_types:
    - HorizontalFlip
    - VerticalFlip
    - Blur
    - RandomRotate90
    - RandomBrightnessContrast
    - MultiplicativeNoise
    shuffle: true
    scenes:
      - "20190618_201847_1035"
      - "20190618_201848_1035"
      - "20190623_200555_0e19"
  val:
    augment: false
    shuffle: false
    scenes:
      - "20190727_160426_104e"
  test:
    augment: false
    shuffle: false
    scenes:
      - "20190709_042959_08_1057"
# Training Parameters
batch_size: 4
learning_rate: 0.01
# Learning rate scheduler. Available:
# ExponentialLR, StepLR (https://pytorch.org/docs/stable/optim.html)
# if no lr_step_size given then lr_step_size=10, gamma=0.1 for StepLR and gamma=0.9 for ExponentialLR
learning_rate_scheduler: StepLR
lr_step_size: 10
lr_gamma: 0.1

# Training Schedule
schedule:
  - phase: Training
    epochs: 30
    steps:
      - train_on: train
      - validate_on: val
      - log_images
# Visualization Configuration
visualization_tiles:
  20190727_160426_104e: [5, 52, 75, 87, 113, 139, 239, 270, 277, 291, 305]

```
