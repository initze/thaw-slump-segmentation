# Changelog

## [0.5.0] - 2021-05-19
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
