#!/usr/bin/env python
"""
Usecase 2 Data Preprocessing Script
"""
import argparse
from datetime import datetime
from lib import data_pre_processing
from lib.data_pre_processing import *
from lib.utils import init_logging, get_logger
from joblib import Parallel, delayed
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("--gdal_bin", default='', help="Path to gdal binaries (ignored if --skip_gdal is passed)")
parser.add_argument("--gdal_path", default='', help="Path to gdal scripts (ignored if --skip_gdal is passed)")
parser.add_argument("--n_jobs", default=-1, type=int, help="number of parallel joblib jobs")

is_ee_initialized = False  # Module-global flag to avoid calling ee.Initialize multiple times

BASEDIR = os.path.abspath('.')
INPUT_DATA_DIR = os.path.join(BASEDIR, 'data_input')
BACKUP_DIR = os.path.join(BASEDIR, 'data_backup')
DATA_DIR = os.path.join(BASEDIR, 'data')

STATUS = {0: 'failed', 1: 'success', 2: 'skipped'}
SUCCESS_STATES = ['rename', 'label', 'ndvi', 'tcvis', 'rel_dem', 'slope', 'mask', 'move']


def preprocess_directory(image_dir, args, log_path, label_required=True):
    gdal.initialize(args)
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
        except:
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

    ee_image_tcvis = ee.ImageCollection("users/ingmarnitze/TCTrend_SR_2000-2019_TCVIS").mosaic()
    success_state['tcvis'] = get_tcvis_from_gee(image_dir,
                                                ee_image_tcvis,
                                                out_filename='tcvis.tif',
                                                resolution=3)

    success_state['rel_dem'] = aux_data_to_tiles(image_dir,
                                                 'data_aux/ArcticDEM/elevation.vrt',
                                                 'relative_elevation.tif')
                                                 

    success_state['slope'] = aux_data_to_tiles(image_dir,
                                               'data_aux/ArcticDEM/slope.vrt',
                                               'slope.tif')
                                               

    success_state['mask'] = mask_input_data(image_dir, DATA_DIR)

    backup_dir = os.path.join(BACKUP_DIR, os.path.basename(image_dir))
    success_state['move'] = move_files(image_dir, backup_dir)

    for status in SUCCESS_STATES:
        thread_logger.info(status + ': ' + STATUS[success_state[status]])
    return success_state


if __name__ == "__main__":
    args = parser.parse_args()

    timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    log_path = Path('logs') / f'setup_raw_data-{timestamp}.log'
    init_logging(log_path)
    logger = get_logger('setup_raw_data')
    logger.info('###########################')
    logger.info('# Starting Raw Data Setup #')
    logger.info('###########################')

    dir_list = check_input_data(INPUT_DATA_DIR)
    if len(dir_list) > 0:
        Parallel(n_jobs=args.n_jobs)(delayed(preprocess_directory)(image_dir, args, log_path) for image_dir in dir_list)
    else:
        logger.error("Empty Input Data Directory! No Data available to process!")
