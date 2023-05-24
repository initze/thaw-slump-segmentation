# AICore - 

## System and Data Setup

### Option 1 - Singularity container
https://cloud.sylabs.io/library/initze/aicore/thaw_slump_segmentation

The container contains all requirements to run the processing code, singularity must be installed

```
singularity pull --arch amd64 library://initze/aicore/thaw_slump_segmentation:0.8.0
singularity shell --nv --bind <your bind path> thaw_slump_segmentation.sif
```
### Option 2 - anaconda
### Environment setup
We recommend using a new conda environment from the provided environment.yml file
```bash
conda env create -n aicore -f environment.yml
```

### Data Preparation

* copy/move data into <DATA_DIR>/input

## Data Processing

### Data Preprocessing

Supported modes: `planet`, `s2`

```bash
python build_datacubes.py --mode=<MODE> --n_jobs=<N_THREADS> --data_dir <DATA_DIR>
```

### Training a model

```bash
python train.py --data_dir <DATA_DIR>/<MODE> -n <MODEL_NAME>
```

### Running Inference

TODO

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
  - PlanetScope
  - TCVIS
  - RelativeElevation
  - Slope
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
      - 20180702_025400_0f31_3B_AnalyticMS_SR
      - 20180702_025401_0f31_3B_AnalyticMS_SR
      - 20180703_192559_1006_3B_AnalyticMS_SR
      - 20180704_075857_1044_3B_AnalyticMS_SR
      - 20180708_075756_1011_3B_AnalyticMS_SR
  val:
    augment: false
    shuffle: false
    scenes:
      - 20180921_203252_101f_3B_AnalyticMS_SR
      - 20190607_204203_1044_3B_AnalyticMS_SR
  test:
    augment: false
    shuffle: false
    scenes:
      - 20190607_204204_1044_3B_AnalyticMS_SR
      - 20190607_204205_1044_3B_AnalyticMS_SR
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
```
