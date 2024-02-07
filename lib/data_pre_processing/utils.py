# Copyright (c) Ingmar Nitze and Konrad Heidler

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import glob
import os
import shutil
import numpy as np
import rasterio as rio

from .udm import burn_mask
from ..utils import get_logger, log_run
from . import gdal

_logger = get_logger('preprocessing.data')


def check_input_data(input_directory):
    directory_list = [f for f in input_directory.glob('*') if f.is_dir()]
    return directory_list


def pre_cleanup(input_directory):
    flist_dirty = glob.glob(os.path.join(input_directory, '*.aux.xml'))
    if len(flist_dirty) > 0:
        for f in flist_dirty:
            os.remove(f)
            _logger.info(f'Removed File {f}')


def has_projection(image_directory):
    image_directory = os.path.abspath(image_directory)
    assert os.path.isdir(image_directory)
    image_list = glob.glob(os.path.join(image_directory, r'*_SR.tif'))
    impath = image_list[0]

    # TODO: crs not detected
    with rio.open(impath) as src:
        try:
            src.crs.to_epsg()
            return True
        except AttributeError:
            return False


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
        _logger.info('No "_clip" naming found. Assume renaming not necessary')
        return 2


def make_ndvi_file(image_directory, nir_band=3, red_band=2):
    images = get_mask_images(image_directory, images=['_SR.tif'])
    file_src = images['images'][0]
    file_dst = os.path.join(os.path.dirname(file_src), 'ndvi.tif')
    with rio.Env():
        with rio.open(images['images'][0]) as ds_src:
            data = ds_src.read().astype(np.float)
            mask = ds_src.read_masks()[0] != 0
            ndvi = np.zeros_like(data[0])
            upper = (data[nir_band][mask] - data[red_band][mask])
            lower = (data[nir_band][mask] + data[red_band][mask])
            ndvi[mask] = np.around((np.divide(upper, lower) + 1) * 1e4).astype(np.uint16)
            profile = ds_src.profile
            profile['count'] = 1

            with rio.open(file_dst, 'w', **profile) as ds_dst:
                ds_dst.write(ndvi.astype(rio.uint16), 1)
    return 1


def get_mask_images(image_directory, udm='udm.tif', udm2='udm2.tif', images=['_SR.tif', 'tcvis.tif', '_mask.tif', 'relative_elevation.tif', 'slope.tif', 'ndvi.tif']):
    flist = glob.glob(os.path.join(image_directory, '*'))
    image_files = []
    for im in images:
        image_files.extend([f for f in flist if im in f])
    # check which udms are available, if not then set to None
    try:
        udm_file = [f for f in flist if udm in f][0]
    except:
        udm_file = None
    try:
        udm2_file = [f for f in flist if udm2 in f][0]
    except:
        udm2_file = None
    # raise error if no udms available
    if (udm_file == None) & (udm2_file == None):
        raise ValueError(f'There are no udm or udm2 files for image {image_directory.name}!')

    remaining_files = [f for f in flist if f not in [udm_file, *image_files]]

    return dict(udm=udm_file, udm2=udm2_file, images=image_files, others=remaining_files)


def move_files(image_directory, backup_dir):
    try:
        shutil.move(image_directory, backup_dir)
        return 1
    except:
        return 2


def mask_input_data(image_directory, output_directory):
    mask_image_paths = get_mask_images(image_directory)
    for image in mask_image_paths['images']:
        dir_out = os.path.join(output_directory, os.path.basename(image_directory))
        image_out = os.path.join(dir_out, os.path.basename(image))
        os.makedirs(dir_out, exist_ok=True)
        burn_mask(image, image_out, file_udm=mask_image_paths['udm'], file_udm2=mask_image_paths['udm2'])
    return 1


def vector_to_raster_mask(image_directory, delete_intermediate_files=True):
    basename = os.path.basename(image_directory)
    vectorfile = glob.glob(os.path.join(image_directory, '*.shp'))[0]
    rasterfile = glob.glob(os.path.join(image_directory, r'*_SR.tif'))[0]
    maskfile = os.path.join(image_directory, 'mask.tif')
    maskfile2 = os.path.join(image_directory, f'{basename}_mask.tif')

    try:
        #s_merge = f'python {gdal.merge} -createonly -init 0 -o {maskfile} -ot Byte -co COMPRESS=DEFLATE {rasterfile}'
        s_merge = f'{gdal.merge} -createonly -init 0 -o {maskfile} -ot Byte -co COMPRESS=DEFLATE {rasterfile}'
        log_run(s_merge, _logger)
        # Add empty band to mask
        s_translate = f'{gdal.translate} -of GTiff -ot Byte -co COMPRESS=DEFLATE -b 1 {maskfile} {maskfile2}'
        log_run(s_translate, _logger)
        # Burn digitized polygons into mask
        s_rasterize = f'{gdal.rasterize} -l {basename} -a label {vectorfile} {maskfile2}'
        log_run(s_rasterize, _logger)
    except:
        return 2
    if delete_intermediate_files:
        os.remove(maskfile)
    return 1

def geom_from_image_bounds(image_path):
    with rio.open(image_path) as src:
        epsg = 'EPSG:{}'.format(src.crs.to_epsg())
        return [src.bounds.left, src.bounds.right, src.bounds.bottom, src.bounds.top]

def crs_from_image(image_path):
    with rio.open(image_path) as src:
        return 'EPSG:{}'.format(src.crs.to_epsg())
    
def resolution_from_image(image_path):
    with rio.open(image_path) as src:
        return src.res

def aux_data_to_tiles(image_directory, aux_data, outfile):
    # load template and get props
    images = get_mask_images(image_directory, udm='udm.tif', udm2='udm2.tif', images=['_SR.tif'])
    image = images['images'][0]
    # prepare gdalwarp call
    xmin, xmax, ymin, ymax = geom_from_image_bounds(image)
    crs = crs_from_image(image)
    xres, yres = resolution_from_image(image)
    # run gdalwarp call
    outfile = f'{image_directory}/{outfile}'#os.path.join(image_directory,outfile)
    s_run = f'{gdal.warp} -te {xmin} {ymin} {xmax} {ymax} -tr {xres} {yres} -r cubic -t_srs {crs} -co COMPRESS=DEFLATE {aux_data} {outfile}'
    #s_run = f'{gdal.warp} -te {xmin} {ymin} {xmax} {ymax} -tr 3 3 -r cubic -t_srs {crs} -co COMPRESS=DEFLATE {aux_data} {outfile}'
    log_run(s_run, _logger)
    return 1
