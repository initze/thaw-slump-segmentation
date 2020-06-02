#!/usr/bin/env python

import glob
import os
import rasterio as rio
import ee
from pyproj import Transformer
import requests
import zipfile


def check_input_data(input_dir):
    dir_list = [f for f in glob.glob(os.path.join(input_dir, '*')) if os.path.isdir(f)]
    return dir_list

def is_clip(input_dir):
    pass

def get_tcvis_from_gee(image_dir, ee_imagecollection, buffer=1000, resolution=3, remove_files=True):
    """
    Download and extract image data from Earthengine and fit to image
    :param image_dir:
    :param ee_imagecollection:
    :param buffer:
    :param resolution:
    :param remove_files:
    :return:
    """
    image_dir = os.path.abspath(image_dir)
    assert os.path.isdir(image_dir)
    imlist = glob.glob(os.path.join(image_dir, r'*3B_AnalyticMS_SR*.tif'))
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
    outfile = os.path.join(image_dir, 'tcvis.tif')
    s_warp = f'{gdalwarp} -t_srs {epsg} -tr {resolution} {resolution} -te {xmin} {ymin} {xmax} {ymax} {infile} {outfile}'

    os.system(s_warp)

    if remove_files:
        os.remove(zippath)
        os.remove(infile)

    return 0


def rename_clip_to_standard(image_dir):
    image_dir = os.path.abspath(image_dir)
    imlist = glob.glob(os.path.join(image_dir, r'*_clip*'))
    if len(imlist) > 0:
        for p in imlist:
            p_out = os.path.join(image_dir, os.path.basename(p).replace('_clip', ''))
            os.rename(p, p_out)
        pass


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

    return dict(udm=udm_file, images=image_files, remaining_files=remaining_files)


def mask_input_data(image_dir):
    mask_image_paths = get_mask_images(image_dir)
    for image in mask_image_paths['images']:
        image_out = os.path.join(IMAGE_DIR, os.path.basename(image_dir), os.path.basename(image))
        os.makedirs(os.path.dirname(image_out), exist_ok=True)
        burn_mask(image, image_out, mask_image_paths['udm'])


def move_files():
    pass


if __name__ == "__main__":

    BASEDIR = '.'
    INPUT_DATA_DIR = os.path.join(BASEDIR, 'input_data')
    TRASH_DIR = os.path.join(BASEDIR, 'preprocessed')
    IMAGE_DIR = os.path.join(BASEDIR, 'data')

    gdalwarp = 'gdalwarp'

    dir_list = check_input_data(INPUT_DATA_DIR)
    if len(dir_list) > 0:
        try:
            ee.Initialize()
        except:
            ee.Authenticate()
            ee.Initialize()
        for image_dir in dir_list:
            get_tcvis_from_gee(image_dir, ee.ImageCollection("users/ingmarnitze/TCTrend_SR_2000-2019_TCVIS").mosaic(),
                               buffer=1000, resolution=3, remove_files=True)
            rename_clip_to_standard(image_dir)
            mask_input_data(image_dir)
    else:
        print("Empty Input Data Directory!")