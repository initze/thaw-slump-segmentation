#!/usr/bin/env python

import glob
import os
import rasterio as rio
import ee
from pyproj import Transformer
import requests
import shutil
import zipfile


def check_input_data(input_dir):
    dir_list = [f for f in glob.glob(os.path.join(input_dir, '*')) if os.path.isdir(f)]
    return dir_list

def is_clip(input_dir):
    pass

def get_tcvis_from_gee(image_dir, ee_imagecollection, buffer=1000, resolution=3, remove_files=True):

    image_dir = os.path.abspath(image_dir)
    assert os.path.isdir(image_dir)
    outfile_tcvis = os.path.join(image_dir, 'tcvis.tif')
    if os.path.exists(outfile_tcvis):
        print('"tcvis.tif" already exists. Skipping download!')
    else:
        print("Starting download Dataset from Google Earthengine")
    imlist = glob.glob(os.path.join(image_dir, r'*3B_AnalyticMS_SR.tif'))
    impath = imlist[0]
    basepath = os.path.basename(image_dir)
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

    zippath = os.path.join(image_dir, 'out.zip')
    myfile = requests.get(url, allow_redirects=True)
    open(zippath, 'wb').write(myfile.content)

    with zipfile.ZipFile(zippath, 'r') as zip_ref:
        zip_ref.extractall(image_dir)

    infile = os.path.join(image_dir, basename_tcvis + '.tif')

    s_warp = f'{gdalwarp} -t_srs {epsg} -tr {resolution} {resolution} -te {xmin} {ymin} {xmax} {ymax} {infile} {outfile_tcvis}'

    os.system(s_warp)

    if remove_files:
        os.remove(zippath)
        os.remove(infile)


def rename_clip_to_standard(image_dir):
    image_dir = os.path.abspath(image_dir)
    imlist = glob.glob(os.path.join(image_dir, r'*_clip*'))
    if len(imlist) > 0:
        for p in imlist:
            p_out = os.path.join(image_dir, os.path.basename(p).replace('_clip', ''))
            if not os.path.exists(p_out):
                os.rename(p, p_out)
    else:
        print('No "_clip" naming found. Assume renaming not necessary')


def burn_mask(file_src, file_dst, file_udm, mask_value=0):
    with rio.Env():
        # the new file's profile, we start with the profile of the source
        with rio.open(file_src) as ds_src:
            with rio.open(file_udm) as ds_udm:
                clear_mask = ds_udm.read()[0]==mask_value
                data = ds_src.read()*clear_mask
                profile = ds_src.profile #check projection here
                with rio.open(file_dst, 'w', **profile) as ds_dst:
                    ds_dst.write(data)


def get_mask_images(image_dir, udm='*udm.tif', images=['_SR.tif', 'tcvis.tif']):
    flist = glob.glob(os.path.join(image_dir, '*'))
    image_files = []
    for im in images:
        image_files.extend([f for f in flist if im in f])
    udm_file = [f for f in flist if 'udm.tif' in f][0]
    remaining_files = [f for f in flist if f not in [udm_file, *image_files]]

    return dict(udm=udm_file, images=image_files, others=remaining_files)


def move_files(image_dir, target_dir, backup_dir):
    mask_image_paths = get_mask_images(image_dir)
    # good files
    infiles = mask_image_paths['others'] + [(mask_image_paths['udm'])]
    [shutil.copy(infile, target_dir) for infile in infiles]
    # all files
    shutil.move(image_dir, backup_dir)


def mask_input_data(image_dir):
    mask_image_paths = get_mask_images(image_dir)
    for image in mask_image_paths['images']:
        dir_out = os.path.join(DATA_DIR, os.path.basename(image_dir))
        image_out = os.path.join(dir_out, os.path.basename(image))
        os.makedirs(dir_out, exist_ok=True)
        burn_mask(image, image_out, mask_image_paths['udm'])


if __name__ == "__main__":

    BASEDIR = os.path.abspath('.')
    INPUT_DATA_DIR = os.path.join(BASEDIR, 'input_data')
    BACKUP_DIR = os.path.join(BASEDIR, 'backup')
    DATA_DIR = os.path.join(BASEDIR, 'data')

    gdalwarp = 'gdalwarp'

    dir_list = check_input_data(INPUT_DATA_DIR)
    if len(dir_list) > 0:
        try:
            ee.Initialize()
        except:
            ee.Authenticate()
            ee.Initialize()
        for image_dir in dir_list:
            print(f'\nStarting preprocessing: {os.path.basename(image_dir)}')
            rename_clip_to_standard(image_dir)
            get_tcvis_from_gee(image_dir, ee.ImageCollection("users/ingmarnitze/TCTrend_SR_2000-2019_TCVIS").mosaic(),
                               buffer=1000, resolution=3, remove_files=True)
            mask_input_data(image_dir)
            move_files(image_dir, os.path.join(DATA_DIR, os.path.basename(image_dir)), os.path.join(BACKUP_DIR, os.path.basename(image_dir)))
    else:
        print("Empty Input Data Directory! No Data available to process!")