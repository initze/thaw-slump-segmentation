#!/usr/bin/env python
# Copyright (c) Ingmar Nitze and Konrad Heidler

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
Usecase 2 Data Preprocessing Script
"""

import argparse
import os
from datetime import datetime
from pathlib import Path

import ee
import typer
from joblib import Parallel, delayed
from typing_extensions import Annotated

from thaw_slump_segmentation import data_pre_processing
from thaw_slump_segmentation.data_pre_processing import (
    aux_data_to_tiles,
    check_input_data,
    gdal,
    get_tcvis_from_gee,
    has_projection,
    make_ndvi_file,
    mask_input_data,
    move_files,
    pre_cleanup,
    rename_clip_to_standard,
    vector_to_raster_mask,
)

from thaw_slump_segmentation.utils import get_logger, init_logging

is_ee_initialized = False  # Module-global flag to avoid calling ee.Initialize multiple times


STATUS = {0: 'failed', 1: 'success', 2: 'skipped'}
SUCCESS_STATES = ['rename', 'label', 'ndvi', 'tcvis', 'rel_dem', 'slope', 'mask', 'move']


def preprocess_directory(image_dir, data_dir, aux_dir, backup_dir, log_path, gdal_bin, gdal_path, label_required=True):
    # Mock old args object
    gdal.initialize(bin=gdal_bin, path=gdal_path)

    init_logging(log_path)
    image_name = os.path.basename(image_dir)
    thread_logger = get_logger(f'setup_raw_data.{image_name}')
    data_pre_processing.earthengine._logger = get_logger(f'setup_raw_data.{image_name}.ee')
    data_pre_processing.utils._logger = get_logger(f'setup_raw_data.{image_name}.utils')

    global is_ee_initialized
    if not is_ee_initialized:
        try:
            thread_logger.debug('Initializing Earth Engine')
            ee.Initialize()
        except Exception:
            thread_logger.warn('Initializing Earth Engine failed, trying to authenticate')
            ee.Authenticate()
            ee.Initialize()
        is_ee_initialized = True
    success_state = dict(rename=0, label=0, ndvi=0, tcvis=0, rel_dem=0, slope=0, mask=0, move=0)
    thread_logger.info(f'Starting preprocessing {image_name}')

    pre_cleanup(image_dir)

    success_state['rename'] = rename_clip_to_standard(image_dir)

    if not has_projection(image_dir):
        thread_logger.error('Input File has no valid Projection!')
        return

    if label_required:
        success_state['label'] = vector_to_raster_mask(image_dir)
    else:
        success_state['label'] = 2

    success_state['ndvi'] = make_ndvi_file(image_dir)

    ee_image_tcvis = ee.ImageCollection('users/ingmarnitze/TCTrend_SR_2000-2019_TCVIS').mosaic()
    success_state['tcvis'] = get_tcvis_from_gee(image_dir, ee_image_tcvis, out_filename='tcvis.tif', resolution=3)

    success_state['rel_dem'] = aux_data_to_tiles(
        image_dir, aux_dir / 'ArcticDEM' / 'elevation.vrt', 'relative_elevation.tif'
    )

    success_state['slope'] = aux_data_to_tiles(image_dir, aux_dir / 'ArcticDEM' / 'slope.vrt', 'slope.tif')

    success_state['mask'] = mask_input_data(image_dir, data_dir)

    # backup_dir_full = os.path.join(backup_dir, os.path.basename(image_dir))
    backup_dir_full = backup_dir / image_dir.name
    success_state['move'] = move_files(image_dir, backup_dir_full)

    for status in SUCCESS_STATES:
        thread_logger.info(status + ': ' + STATUS[success_state[status]])
    return success_state


def setup_raw_data(
    gdal_bin: Annotated[str, typer.Option('--gdal_bin', help='Path to gdal binaries', envvar='GDAL_BIN')] = '/usr/bin',
    gdal_path: Annotated[
        str, typer.Option('--gdal_path', help='Path to gdal scripts', envvar='GDAL_PATH')
    ] = '/usr/bin',
    n_jobs: Annotated[int, typer.Option('--n_jobs', help='number of parallel joblib jobs, pass 0 to disable joblib')] = -1,
    label: Annotated[
        bool, typer.Option('--label/--nolabel', help='Set flag to do preprocessing with label file')
    ] = False,
    data_dir: Annotated[Path, typer.Option('--data_dir', help='Path to data processing dir')] = Path('data'),
    log_dir: Annotated[Path, typer.Option('--log_dir', help='Path to log dir')] = Path('logs'),
):
    INPUT_DATA_DIR = data_dir / 'input'
    BACKUP_DIR = data_dir / 'backup'
    DATA_DIR = data_dir / 'tiles'
    AUX_DIR = data_dir / 'auxiliary'

    timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    log_path = Path(log_dir) / f'setup_raw_data-{timestamp}.log'
    if not Path(log_dir).exists():
        os.mkdir(Path(log_dir))
    init_logging(log_path)
    logger = get_logger('setup_raw_data')
    logger.info('###########################')
    logger.info('# Starting Raw Data Setup #')
    logger.info('###########################')

    logger.info(f"images to preprocess in {INPUT_DATA_DIR}")
    dir_list = check_input_data(INPUT_DATA_DIR)
    [logger.info(f" * '{f.relative_to(INPUT_DATA_DIR)}'") for f in dir_list ]
    if len(dir_list) > 0:
        if n_jobs == 0:
            for image_dir in dir_list:
                preprocess_directory(image_dir, DATA_DIR, AUX_DIR, BACKUP_DIR, log_path, gdal_bin, gdal_path, label)
        else:
            Parallel(n_jobs=n_jobs)(
                delayed(preprocess_directory)(
                    image_dir, DATA_DIR, AUX_DIR, BACKUP_DIR, log_path, gdal_bin, gdal_path, label
                )
                for image_dir in dir_list
        )
    else:
        logger.error('Empty Input Data Directory! No Data available to process!')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gdal_bin', default=None, help='Path to gdal binaries (ignored if --skip_gdal is passed)')
    parser.add_argument('--gdal_path', default=None, help='Path to gdal scripts (ignored if --skip_gdal is passed)')
    parser.add_argument('--n_jobs', default=-1, type=int, help='number of parallel joblib jobs')
    parser.add_argument('--nolabel', action='store_false', help='Set flag to do preprocessing without label file')
    parser.add_argument('--data_dir', default='data', type=Path, help='Path to data processing dir')
    parser.add_argument('--log_dir', default='logs', type=Path, help='Path to log dir')

    args = parser.parse_args()

    setup_raw_data(
        gdal_bin=args.gdal_bin,
        gdal_path=args.gdal_path,
        n_jobs=args.n_jobs,
        label=args.nolabel,
        data_dir=args.data_dir,
        log_dir=args.log_dir,
    )

# ! Moving legacy argparse cli to main to maintain compatibility with the original script
if __name__ == '__main__':
    main()