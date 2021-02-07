# AICore - Usecase 2

## Data Preparation

1. copy/move data into data_input
2. copy/move data into data_aux (e.g. prepared ArcticDEM data)

## Data Preprocessing

```bash
python setup_raw_data.py
```
```bash
python prepare_data.py
```

## Training a model

```bash
python train.py
```

## Running Inference

```bash
python inference.py logs/TrainedModel 20190727_160426_104e 20190709_042959_08_1057
```

## Configuration

Configuration is done via the `config.yml` file. Example config:

```yaml
# Model Specification
model: UNet
model_args:
  # Number of classes in the training data (e.g. 2 for background & thaw slump)
  output_channels: 2
  # LayerNorm to use (Available: none, BatchNorm, InstanceNorm, SqueezeExcitation)
  norm: none
  # Other model parameters, e.g. number of down-/upsampling blocks for a UNet
  stack_height: 4
# Data Configuration
data_threads: 4  # Number of threads for data loading
data_sources:  # Enabled input features
  - planet
  - ndvi
  - tcvis
  # - relative_elevation
  # - slope
datasets:
  train:
    augment: true
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
learning_rate: 0.001
loss_function:
  type: AutoCE
# Training Schedule
schedule:
  - phase: Training
    epochs: 30
    steps:
      - train_on: slump_tiles(train)
      - validate_on: val
      - log_images
# Visualization Configuration
visualization_tiles:
  20190727_160426_104e: [5, 52, 75, 87, 113, 139, 239, 270, 277, 291, 305]

```

