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


def mask_input_data():
    pass


def move_files():
    pass


if __name__ == "__main__":
    # TODO: remove hardcoded part
    gdalwarp = 'gdalwarp'

    dir_list = check_input_data('./input_data')
    if len(dir_list) > 0:
        try:
            ee.Initialize()
        except:
            ee.Authenticate()
            ee.Initialize()
        for image_dir in dir_list:
            get_tcvis_from_gee(image_dir, ee.ImageCollection("users/ingmarnitze/TCTrend_SR_2000-2019_TCVIS").mosaic(),
                               buffer=1000, resolution=3, remove_files=True)