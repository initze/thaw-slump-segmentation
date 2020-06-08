#!/usr/bin/env python

import os
import glob

import ee

from deep_learning.data_pre_processing import *


def vector_to_raster_mask(image_directory):
    basename = os.path.basename(image_directory)
    vectorfile = glob.glob(os.path.join(image_directory, '*.shp'))[0]
    rasterfile = glob.glob(os.path.join(image_directory, r'*3B_AnalyticMS_SR.tif'))[0]
    maskfile = os.path.join(image_directory, 'mask.tif')
    maskfile2 = os.path.join(image_directory, f'{basename}_mask.tif')

    gdal_path = r'C:\Users\initze\AppData\Local\Continuum\anaconda3\envs\aicore\Scripts'
    gdal_bin = r'C:\Users\initze\AppData\Local\Continuum\anaconda3\envs\aicore\Library\bin'
    gdal_merge = os.path.join(gdal_path, 'gdal_merge.py')
    gdal_translate = os.path.join(gdal_bin, 'gdal_translate')
    gdal_rasterize = os.path.join(gdal_bin, 'gdal_rasterize')

    try:
        s_merge = f'python {gdal_merge} -createonly -init 0 -o {maskfile} -ot Byte -co COMPRESS=DEFLATE {rasterfile}'
        os.system(s_merge)
        # Add empty band to mask
        s_translate = f'{gdal_translate} -of GTiff -ot Byte -co COMPRESS=DEFLATE -b 1 {maskfile} {maskfile2}'
        os.system(s_translate)
        # Burn digitized polygons into mask
        s_rasterize = f'{gdal_rasterize} -l {basename} -a label {vectorfile} {maskfile2}'
        os.system(s_rasterize)
    except:
        return 2
    return 1

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