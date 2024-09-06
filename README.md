# Thaw Slump Segmentation

## Installation

### Environment Setup

Prereq:

- [Rye](https://rye.astral.sh/): `curl -sSf https://rye.astral.sh/get | bash`
- [GDAL](https://gdal.org/en/latest/index.html): `sudo apt update && sudo apt install libpq-dev gdal-bin libgdal-dev` or for HPC: `conda install conda-forge::gdal`
- Clang: `sudo apt update && sudo apt install clang` or for HPC: `conda install conda-forge::clang_linux-64`

> If you install GDAL via apt for linux you can view the supported versions here: <https://pkgs.org/search/?q=libgdal-dev>. For a finer controll over the versions please use conda.

Now first check your gdal-version:

```sh
$ gdal-config --version
3.9.2
```

And your CUDA version (if you want to use CUDA):

```sh
$ nvidia-smi
# Now look on the top right of the table
```

> The GDAL version is relevant, since the version of the python bindings needs to match the installed GDAL version  
> The CUDA feature is used for `cucim` to accelerate certain image transformations.

If your version is one of: `3.9.2`, `3.8.5`, `3.7.3` or `3.6.4` you can sync with the respecting command of these:

```sh
rye sync -f --features gdal39,cuda12 # For CUDA 12 and GDAL 3.9.2
rye sync -f --features gdal38,cuda12 # For CUDA 12 and GDAL 3.8.5
rye sync -f --features gdal37,cuda12 # For CUDA 12 and GDAL 3.7.3
rye sync -f --features gdal36,cuda12 # For CUDA 12 and GDAL 3.6.4

rye sync -f --features gdal39,cuda11 # For CUDA 11 and GDAL 3.9.2
rye sync -f --features gdal38,cuda11 # For CUDA 11 and GDAL 3.8.5
rye sync -f --features gdal37,cuda11 # For CUDA 11 and GDAL 3.7.3
rye sync -f --features gdal36,cuda11 # For CUDA 11 and GDAL 3.6.4

rye sync -f --features gdal39,cpu # For CPU only and GDAL 3.9.2
rye sync -f --features gdal38,cpu # For CPU only and GDAL 3.8.5
rye sync -f --features gdal37,cpu # For CPU only and GDAL 3.7.3
rye sync -f --features gdal36,cpu # For CPU only and GDAL 3.6.4
```

If your GDAL version is not supported (yet) please sync without GDAL and then install GDAL to an new optional group. For example, if your GDAL version is 3.8.4:

```sh
rye sync -f
rye add --optional=gdal384 "gdal==3.8.4"
```

> IMPORTANT! If you installed any of clang or gdal with conda, please ensure that while installing dependencies and working on the project to have the conda environment activated in which you installed clang and or gdal.

#### Troubleshoot: Rye can't find the right versions

Because the `pyproject.toml` specifies additional sources, e.g. `https://download.pytorch.org/whl/cpu`, it can happen that the a package with an older version is found in these package-indexes.
If such a version is found, `uv` (the installer behind `Rye`) currently stops searching other sources for the right version and stops with an `Version not found` error.
This can look something like this:

```sh
No solution found when resolving dependencies:
  ╰─▶ Because only torchmetrics==1.0.3 is available and you require torchmetrics>=1.4.1, we can conclude that your requirements are unsatisfiable.
```

To fix this you can set an environment variable to tell `uv` to search all package-indicies:

```sh
UV_INDEX_STRATEGY="unsafe-best-match" rye sync ...
```

Please see these issues:

- [Rye: Can't specify per-dependency package index / can't specify uv behavior in config file](https://github.com/astral-sh/rye/issues/1210#issuecomment-2263761535)
- [UV: Add support for pinning a package to a specific index](https://github.com/astral-sh/uv/issues/171)

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
Hello, tobi!
```

or run as python module:

```sh
$ rye run python -m thaw_slump_segmentation hello tobi
Hello, tobi!
```

With activated env, e.g. after installation, just remove the `rye run`:

```sh
$ source .venv/bin/activate
$ thaw-slump-segmentation hello tobi
Hello, tobi!
```

or

```sh
$ source .venv/bin/activate
$ python -m thaw_slump_segmentation hello tobi
Hello, tobi!
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

### Tests

We use [Pytest](https://pytest.org/) as testing library for unit tests.

Run tests (via rye, for other environments activate the environment
with `.venv/bin/activate` or similar and then simply remove the `rye run` before the command):

```sh
rye run pytest
```

Run all tests from a specific file:

```sh
rye run pytest tests/test_hello.py
```

Run a specific test from a specific file:

```sh
rye run pytest 
```

#### Test Parameters

Several arguments can be passed to pytest to configure the environment for the tests, but not all test parameters are needed by all tests. Tests needing a parameter are skipped if the parameter is missing. In case the reason for skipping a test is of interest, [the argument `-rs`](https://docs.pytest.org/en/6.2.x/usage.html#detailed-summary-report) can be passed to pytest.

##### `--data_dir`

Most tests need input data which will be imported from a data directory as root. The tests expect a specific folder structure below the root. For now this is:
* a folder `raw_data_dir` where the folders `scenes` and `tiles` exist for the planet source data
* a folder `auxiliary` with the subfolder `ArcticDEM` where the virtual raster files reside, which are pointing to the slope and relative elevation data.

No files are changed within those directories. The data is copied to a temporary directory and the tests are run there.

##### `--gdal_bin`, `--gdal_path`

Paths to the folders where the gdal executables (like `gdaltransform`, `--gdal_bin`) and gdal scripts like `gdal_retile.py` (`--gdal_path`) reside.

##### `--proj_data_env`

The proj library is needed for some processes by the gdal binaries and scripts. In case it cannot find or open its data folder, the PROJ_DATA environment variable needs to be set. Passing the parameter `--proj_data_env` to pytest will set this environment variable during the tests. 

### Todos testing

* [ ] Add GitHub Workflow for [Pytest](https://docs.github.com/en/actions/automating-builds-and-tests/building-and-testing-python#testing-your-code) and [Ruff](https://github.com/chartboost/ruff-action)
* [ ] Add tests for data-related scripts (single and multiple files (for multiprocessing) each):
  * [ ] Setup Raw Data
  * [x] Prepare data
  * [ ] Inference
  * [ ] (S2 related scripts)
* [ ] Utilities
  * [ ] GDAL init
  * [ ] EE auth + init
  * [ ] TBD
