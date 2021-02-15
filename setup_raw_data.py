#!/usr/bin/env python
"""
Usecase 2 Data Preprocessing Script

Usage:
    prepare_data.py [options]

Options:
    -h --help               Show this screen
    --skip_gdal             Skip the Gdal conversion stage (if it has already been done)
    --gdal_path=PATH        Path to gdal scripts (ignored if --skip_gdal is passed) [default: ]
    --gdal_bin=PATH         Path to gdal binaries (ignored if --skip_gdal is passed) [default: ]
"""

from deep_learning.data_pre_processing import *
from docopt import docopt
from deep_learning.data_pre_processing import get_tcvis_from_gee
#from parsl.app.app import python_app
#import parsl
#parsl.load()
from joblib import Parallel, delayed

is_ee_initialized = False  # Module-global flag to avoid calling ee.Initialize multiple times

BASEDIR = os.path.abspath('.')
INPUT_DATA_DIR = os.path.join(BASEDIR, 'data_input')
BACKUP_DIR = os.path.join(BASEDIR, 'data_backup')
DATA_DIR = os.path.join(BASEDIR, 'data')

STATUS = {0: 'failed', 1: 'success', 2: 'skipped'}
SUCCESS_STATES = ['rename', 'label', 'ndvi', 'tcvis', 'rel_dem', 'slope', 'mask', 'move']


#@python_app
def preprocess_directory(image_dir, gdal_bin, gdal_path, label_required=True):
    global is_ee_initialized
    if not is_ee_initialized:
        try:
            ee.Initialize()
        except:
            ee.Authenticate()
            ee.Initialize()
        is_ee_initialized = True
    success_state = dict(rename=0, label=0, ndvi=0, tcvis=0, rel_dem=0, slope=0, mask=0, move=0)
    print(f'\nStarting preprocessing: {os.path.basename(image_dir)}')

    pre_cleanup(image_dir)

    success_state['rename'] = rename_clip_to_standard(image_dir)

    if not has_projection(image_dir):
        print('Input File has no valid Projection!')
        return

    if label_required:
        success_state['label'] = vector_to_raster_mask(image_dir,
                                                       gdal_bin=gdal_bin,
                                                       gdal_path=gdal_path)
    else:
        success_state['label'] = 2

    success_state['ndvi'] = make_ndvi_file(image_dir)

    # TODO: resample to 3m
    ee_image_tcvis = ee.ImageCollection("users/ingmarnitze/TCTrend_SR_2000-2019_TCVIS").mosaic()
    success_state['tcvis'] = get_tcvis_from_gee(image_dir,
                                                ee_image_tcvis,
                                                out_filename='tcvis.tif',
                                                resolution=3)

    success_state['rel_dem'] = aux_data_to_tiles(image_dir,
                                                 'data_aux/ArcticDEM/elevation.vrt',
                                                 'relative_elevation.tif',
                                                 gdal_bin=gdal_bin, gdal_path=gdal_path)

    success_state['slope'] = aux_data_to_tiles(image_dir,
                                               'data_aux/ArcticDEM/slope.vrt',
                                               'slope.tif',
                                               gdal_bin=gdal_bin, gdal_path=gdal_path)

    success_state['mask'] = mask_input_data(image_dir, DATA_DIR)

    backup_dir = os.path.join(BACKUP_DIR, os.path.basename(image_dir))
    success_state['move'] = move_files(image_dir, backup_dir)

    for status in SUCCESS_STATES:
        print(status + ':', STATUS[success_state[status]])
    return success_state


if __name__ == "__main__":
    args = docopt(__doc__, version="Usecase 2 Data Preprocessing Script 1.0")

    dir_list = check_input_data(INPUT_DATA_DIR)
    if len(dir_list) > 0:
        # TODO: Joblib
        Parallel(n_jobs=-1)(delayed(preprocess_directory)(image_dir, gdal_bin=args['--gdal_bin'], gdal_path=args['--gdal_path']) for image_dir in dir_list)
        #_ = preprocess_directory(image_dir, gdal_bin=args['--gdal_bin'], gdal_path=args['--gdal_path'])
        #for image_dir in dir_list:
        #    preprocess_directory(image_dir, gdal_bin=args['--gdal_bin'], gdal_path=args['--gdal_path'])

    else:
        print("Empty Input Data Directory! No Data available to process!")
