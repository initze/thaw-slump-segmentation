# Changelog

## [0.10.2] - 2024-05-08

### Added

- re-added weights and biases to training module

### Changed

- bugfixes of postprocessing scripts
- code and syntax cleanup

## [0.10.1] - 2024-04-25

### Changed

- added gdal as pre-requirement to readme
- fixed path issue for postprocessing scripts

## [0.10.0] - 2024-04-24

### Added

- pip-installable packaging

### Changed

- entire project structure (to make it pip-installable)

## [0.8.0] - 2022-09-09

### Added

- added Unet3+ model
- added deep supervision loss plot

## [0.7.1] - 2022-08-02

### Added

- added learning rate scheduler (StepLR, ExponentialLR)
- added example configuration file

## [0.7.0] - 2022-03-29

### Added

- added singularity container definition

## [0.6.1] - 2022-03-22

### Changed

- fixed bug (crashed prepare_data.py for Planet Scenes)

## [0.6.0] - 2021-12-21

### Added

- data path selection for all scripts
- inference path selection for inference.py

### Changed

- overhauled data structure: disentangled from processing/code directory
- fixed masking bug (did not recognize udm values > 1) for PSOrthoTile data

## [0.5.5] - 2021-11-25

### Changed

- added support for Multi- and single-GPU trained models

## [0.5.4] - 2021-11-16

### Changed

- added Multi-GPU support for Training

## [0.5.3] - 2021-11-10

### Changed

- added support for PSOrthotile input (flexible input data resolution)

## [0.5.2] - 2021-08-18

### Changed

- fixed inference masking bug

## [0.5.1] - 2021-06-10

### Changed

- bugs from 0.5.0 fixed
- logfiles moved to logs directory
- completely moved to argparse (from docopt) for argument parsing
- (false) nodata removed from inference vector output

## [0.5.0] - 2021-05-19 - Broken Version

### Added

- Training/Inference/Data Preparation logs are now written to corresponding log files
- gdal configuration file
- minor code cleanup

## [0.4.2] - 2021-03-19

### Added

- shapefiles as inference output

### Changed

- improved inference control figures

## [0.4.1] - 2021-03-11

### Changed

- fixed broken inference

## [0.4.0] - 2021-03-01

### Added

- selection of different DL models and backbones
- implementation of augmentation types based on the albumentations package

## [0.3.3] - 2021-02-22

### Added

- added robust futureproof datasource sorting

## [0.3.2] - 2021-02-15

### Added

- added parallel preprocessing (setup_raw_data.py, prepare_data.py)

### Changed

- changed option parsing from docopt to argparse for preprocessing scripts

## [0.3.1] - 2021-02-08

### Changed

- fixed bug causing crash running inference.py

## [0.3.0] - 2021-01-26

### Added

- Named training runs

### Changed

- Loading Arctic DEM elevations and slopes locally
  - fixed too low resolution bug (loading from Google Earthengine)
- minor code style fixes

## [0.2.1] - 2020-12-07

### Changed

- Metrics are now logged into CSV files: `train.csv`, `val.csv`
- xlabels of metrics plots in epochs
- fixed cropped xlabel in precision recall plot

## [0.2.0] - 2020-11-19

### Changed

- fixed wrong accuracy metrics

## [0.1.1] - 2020-10-30

### Added

- Slope visualization in tile-predictions

### Changed

- Color scheme in tile-predictions
- fixed printout of "label" to "skipped" during preprocessing stage of inference  


## [0.1.0] - 2020-10-19

first version with full processing pipeline
- data preprocessing
- training
- inference
