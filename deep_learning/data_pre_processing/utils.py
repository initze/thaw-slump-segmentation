import glob
import os
import shutil
import zipfile

import ee
import rasterio as rio
import requests
from pyproj import Transformer


def check_input_data(input_directory):
    directory_list = [f for f in glob.glob(os.path.join(input_directory, '*')) if os.path.isdir(f)]
    return directory_list


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

    s_warp = f'gdalwarp -t_srs {epsg} -tr {resolution} {resolution} -te {xmin} {ymin} {xmax} {ymax} {infile} \
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


# TODO: add rastermask image
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


def get_mask_images(image_directory, udm='udm.tif', images=['_SR.tif', 'tcvis.tif', '_mask.tif']):
    flist = glob.glob(os.path.join(image_directory, '*'))
    image_files = []
    for im in images:
        image_files.extend([f for f in flist if im in f])
    udm_file = [f for f in flist if udm in f][0]
    remaining_files = [f for f in flist if f not in [udm_file, *image_files]]

    return dict(udm=udm_file, images=image_files, others=remaining_files)


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
        burn_mask(image, image_out, mask_image_paths['udm'])
    return 1


def vector_to_raster_mask(image_directory, gdal_bin='', gdal_path='', delete_intermediate_files=True):
    basename = os.path.basename(image_directory)
    vectorfile = glob.glob(os.path.join(image_directory, '*.shp'))[0]
    rasterfile = glob.glob(os.path.join(image_directory, r'*3B_AnalyticMS_SR.tif'))[0]
    maskfile = os.path.join(image_directory, 'mask.tif')
    maskfile2 = os.path.join(image_directory, f'{basename}_mask.tif')

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
    if delete_intermediate_files:
        os.remove(maskfile)
    return 1