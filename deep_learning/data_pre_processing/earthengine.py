import glob
import os
import zipfile

import ee
import rasterio as rio
import requests
from pyproj import Transformer
from ..utils import get_logger, log_run

_logger = get_logger('preprocessing.ee')


def get_ArcticDEM_rel_el(kernel_size=300, offset=30, factor=300):
    dem = ee.Image("UMN/PGC/ArcticDEM/V3/2m_mosaic")
    conv = dem.convolve(ee.Kernel.circle(kernel_size, 'meters'))
    diff = (dem.subtract(conv).add(ee.Image.constant(offset)).multiply(ee.Image.constant(factor)).toInt16())
    return diff


def get_ArcticDEM_slope():
    dem = ee.Image("UMN/PGC/ArcticDEM/V3/2m_mosaic")
    slope = ee.Terrain.slope(dem)
    return slope


def ee_geom_from_image_bounds(image_path, buffer=1000):
    with rio.open(image_path) as src:
        epsg = 'EPSG:{}'.format(src.crs.to_epsg())
        xmin, xmax, ymin, ymax = [src.bounds.left, src.bounds.right, src.bounds.bottom, src.bounds.top]
    region_rect = [[xmin - buffer, ymin - buffer], [xmax + buffer, ymax + buffer]]

    # make polygon
    transformer = Transformer.from_crs(epsg, "epsg:4326")
    region_transformed = [transformer.transform(*vertex)[::-1] for vertex in region_rect]
    return ee.Geometry.Rectangle(coords=region_transformed), [xmin, xmax, ymin, ymax], epsg


def get_tcvis_from_gee(image_directory, ee_image, out_filename, buffer=200, resolution=3, remove_files=True):
    image_directory = os.path.abspath(image_directory)
    assert os.path.isdir(image_directory)
    outfile_path = os.path.join(image_directory, out_filename)
    if os.path.exists(outfile_path):
        _logger.info(f'{out_filename} already exists. Skipping download!')
        return 2
    else:
        _logger.info("Starting download Dataset from Google Earthengine")
    image_list = glob.glob(os.path.join(image_directory, r'*3B_AnalyticMS_SR.tif'))
    impath = image_list[0]
    basepath = os.path.basename(image_directory)
    basename_tmpimage = basepath + '_ee_tmp'

    geom_4326, coords, epsg = ee_geom_from_image_bounds(impath, buffer=buffer)

    export_props = {'scale': 30,
                    'name': '{basename}'.format(basename=basename_tmpimage),
                    'region': geom_4326,
                    'filePerBand': False,
                    'crs': epsg}

    url = ee_image.getDownloadURL(export_props)

    zippath = os.path.join(image_directory, 'out.zip')
    myfile = requests.get(url, allow_redirects=True)
    # SUper slow large download
    open(zippath, 'wb').write(myfile.content)

    with zipfile.ZipFile(zippath, 'r') as zip_ref:
        zip_ref.extractall(image_directory)

    infile = os.path.join(image_directory, basename_tmpimage + '.tif')

    xmin, xmax, ymin, ymax = coords
    s_warp = f'gdalwarp -t_srs {epsg} -tr {resolution} {resolution} \
               -srcnodata None -te {xmin} {ymin} {xmax} {ymax} {infile} {outfile_path}'
    log_run(s_warp, _logger)

    if remove_files:
        os.remove(zippath)
        os.remove(infile)

    return 1
