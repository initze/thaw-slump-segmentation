#!/usr/bin/env python

import glob
import os

import ee
import rasterio as rio

from pyproj import Transformer
import requests
import shutil
import zipfile


def check_input_data(input_directory):
    directory_list = [f for f in glob.glob(os.path.join(input_directory, '*')) if os.path.isdir(f)]
    return directory_list


def is_clip(input_dir):
    pass


def has_projection(image_directory):
    image_directory = os.path.abspath(image_directory)
    assert os.path.isdir(image_directory)
    image_list = glob.glob(os.path.join(image_directory, r'*3B_AnalyticMS_SR.tif'))
    impath = image_list[0]

    with rio.open(impath) as src:
        try:
            src.crs.to_epsg()
            return True
        except AttributeError:
            return False

def get_tcvis_from_gee(image_directory, ee_imagecollection, buffer=1000, resolution=3, remove_files=True):

    image_directory = os.path.abspath(image_directory)
    assert os.path.isdir(image_directory)
    outfile_tcvis = os.path.join(image_directory, 'tcvis.tif')
    if os.path.exists(outfile_tcvis):
        print('"tcvis.tif" already exists. Skipping download!')
        return 2
    else:
        print("Starting download Dataset from Google Earthengine")
    image_list = glob.glob(os.path.join(image_directory, r'*3B_AnalyticMS_SR.tif'))
    impath = image_list[0]
    basepath = os.path.basename(image_directory)
    basename_tcvis = basepath + '_TCVIS'

    with rio.open(impath) as src:
        epsg = 'EPSG:{}'.format(src.crs.to_epsg())
        xmin, xmax, ymin, ymax = [src.bounds.left, src.bounds.right, src.bounds.bottom, src.bounds.top]
    region_rect = [[xmin - buffer, ymin - buffer], [xmax + buffer, ymax + buffer]]

    # make polygon
    transformer = Transformer.from_crs(epsg, "epsg:4326")
    region_transformed = [transformer.transform(*vertex)[::-1] for vertex in region_rect]
    geom_4326 = ee.Geometry.Rectangle(coords=region_transformed)

    export_props = {'scale': 30,
                    'name': '{basename}'.format(basename=basename_tcvis),
                    'region': geom_4326,
                    'filePerBand': False}

    url = ee_imagecollection.getDownloadURL(export_props)

    zippath = os.path.join(image_directory, 'out.zip')
    myfile = requests.get(url, allow_redirects=True)
    open(zippath, 'wb').write(myfile.content)

    with zipfile.ZipFile(zippath, 'r') as zip_ref:
        zip_ref.extractall(image_directory)

    infile = os.path.join(image_directory, basename_tcvis + '.tif')

    s_warp = f'{gdalwarp} -t_srs {epsg} -tr {resolution} {resolution} -te {xmin} {ymin} {xmax} {ymax} {infile} \
             {outfile_tcvis} '

    os.system(s_warp)

    if remove_files:
        os.remove(zippath)
        os.remove(infile)

    return 1


def rename_clip_to_standard(image_directory):
    image_directory = os.path.abspath(image_directory)
    imlist = glob.glob(os.path.join(image_directory, r'*_clip*'))
    if len(imlist) > 0:
        for p in imlist:
            p_out = os.path.join(image_directory, os.path.basename(p).replace('_clip', ''))
            if not os.path.exists(p_out):
                os.rename(p, p_out)
        return 1
    else:
        print('No "_clip" naming found. Assume renaming not necessary')
        return 2


def burn_mask(file_src, file_dst, file_udm, mask_value=0):
    with rio.Env():
        with rio.open(file_src) as ds_src:
            with rio.open(file_udm) as ds_udm:
                clear_mask = ds_udm.read()[0] == mask_value
                data = ds_src.read()*clear_mask
                profile = ds_src.profile
                with rio.open(file_dst, 'w', **profile) as ds_dst:
                    ds_dst.write(data)
    return 1


def get_mask_images(image_directory, udm='udm.tif', images=['_SR.tif', 'tcvis.tif']):
    flist = glob.glob(os.path.join(image_directory, '*'))
    image_files = []
    for im in images:
        image_files.extend([f for f in flist if im in f])
    udm_file = [f for f in flist if udm in f][0]
    remaining_files = [f for f in flist if f not in [udm_file, *image_files]]

    return dict(udm=udm_file, images=image_files, others=remaining_files)


def move_files(image_directory, target_dir, backup_dir):
    mask_image_paths = get_mask_images(image_directory)
    # good files
    infiles = mask_image_paths['others'] + [(mask_image_paths['udm'])]
    [shutil.copy(infile, target_dir) for infile in infiles]
    # all files
    try:
        shutil.move(image_directory, backup_dir)
        return 1
    except:
        return 2


def mask_input_data(image_directory):
    mask_image_paths = get_mask_images(image_directory)
    for image in mask_image_paths['images']:
        dir_out = os.path.join(DATA_DIR, os.path.basename(image_directory))
        image_out = os.path.join(dir_out, os.path.basename(image))
        os.makedirs(dir_out, exist_ok=True)
        burn_mask(image, image_out, mask_image_paths['udm'])
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
            success_state = dict(rename=0, tcvis=0, mask=0, move=0)
            print(f'\nStarting preprocessing: {os.path.basename(image_dir)}')

            success_state['rename'] = rename_clip_to_standard(image_dir)

            if not has_projection(image_dir):
                print('Input File has no valid Projection!')
                continue

            success_state['tcvis'] = get_tcvis_from_gee(image_dir,
                                                         ee.ImageCollection("users/ingmarnitze/TCTrend_SR_2000-2019_TCVIS").mosaic(),
                                                         buffer=1000,
                                                         resolution=3,
                                                         remove_files=True)
            success_state['mask'] = mask_input_data(image_dir)
            success_state['move'] = move_files(image_dir, os.path.join(DATA_DIR, os.path.basename(image_dir)),
                       os.path.join(BACKUP_DIR, os.path.basename(image_dir)))
            for status in ['rename', 'tcvis', 'mask', 'move']:
                print(status + ':', STATUS[success_state[status]])
    else:
        print("Empty Input Data Directory! No Data available to process!")