#!/usr/bin/env python

import os

import ee

from deep_learning.data_pre_processing import *


if __name__ == "__main__":

    BASEDIR = os.path.abspath('.')
    INPUT_DATA_DIR = os.path.join(BASEDIR, 'data_input')
    BACKUP_DIR = os.path.join(BASEDIR, 'data_backup')
    DATA_DIR = os.path.join(BASEDIR, 'data')
    STATUS = {0: 'failed', 1: 'success', 2: 'skipped'}

    gdalwarp = 'gdalwarp'

    dir_list = check_input_data(INPUT_DATA_DIR)
    if len(dir_list) > 0:

        try:
            ee.Initialize()
        except:
            ee.Authenticate()
            ee.Initialize()
        for image_dir in dir_list:
            success_state = dict(rename=0, label=0, tcvis=0, mask=0, move=0)
            print(f'\nStarting preprocessing: {os.path.basename(image_dir)}')

            success_state['rename'] = rename_clip_to_standard(image_dir)

            if not has_projection(image_dir):
                print('Input File has no valid Projection!')
                continue

            success_state['label'] = vector_to_raster_mask(image_dir)
            success_state['tcvis'] = get_tcvis_from_gee(image_dir,
                                                        ee.ImageCollection("users/ingmarnitze/TCTrend_SR_2000-2019_TCVIS").mosaic(),
                                                        buffer=1000,
                                                        resolution=3,
                                                        remove_files=True)

            success_state['mask'] = mask_input_data(image_dir, DATA_DIR)

            target_dir = os.path.join(DATA_DIR, os.path.basename(image_dir))
            backup_dir = os.path.join(BACKUP_DIR, os.path.basename(image_dir))
            success_state['move'] = move_files(image_dir, target_dir, backup_dir)

            for status in ['rename', 'label', 'tcvis', 'mask', 'move']:
                print(status + ':', STATUS[success_state[status]])
    else:
        print("Empty Input Data Directory! No Data available to process!")