import rasterio
import os
from pathlib import Path
import numpy as np
import ee
import geemap
from joblib import Parallel, delayed
import argparse
from lib.data_pre_processing.earthengine import ee_geom_from_image_bounds
ee.Initialize(opt_url='https://earthengine-highvolume.googleapis.com')

def get_ndvi_from_4bandS2(image_path):
    image_path = Path(image_path)
    with rasterio.open(image_path) as src:
        #read data
        data = src.read().astype(float)
        # calc ndvi
        ndvi = (data[3]-data[2]) / (data[3]+data[2])
        # factor to correct output
        ndvi_out = ((np.clip(ndvi, -1, 1) + 1) * 1e4).astype(np.uint16)
        # get and adapt profile to match output
        profile = src.profile
        profile.update({'dtype':'uint16', 'count':1})
    # save ndvi
    ndvi_path = image_path.parent / 'ndvi.tif'
    with rasterio.open(ndvi_path, 'w', **profile) as target:
        target.write(np.expand_dims(ndvi_out, 0))

def get_elevation_and_slope(image_path, rel_el_vrt, slope_vrt, parallel=True):
    # get image_metadata
    with rasterio.open(image_path) as src:
        epsg = src.crs.to_string()
        bounds = src.bounds
        xres, yres = src.res
    
    # setup gdal runs
    target_el = image_path.parent / 'relative_elevation.tif'
    s_el = f'gdalwarp -multi -tap -te {bounds.left} {bounds.top} {bounds.right} {bounds.bottom} -t_srs {epsg} -tr {xres} {yres} -r cubic -co COMPRESS=DEFLATE {rel_el_vrt} {target_el}'
    target_slope = image_path.parent / 'slope.tif'
    s_slope = f'gdalwarp -multi -tap -te {bounds.left} {bounds.top} {bounds.right} {bounds.bottom} -t_srs {epsg} -tr {xres} {yres} -r cubic -co COMPRESS=DEFLATE {slope_vrt} {target_slope}'
    
    # run in console
    if parallel:
        def execute_command(cmd):
            os.system(cmd)
        # Parallel execution
        Parallel(n_jobs=2)(delayed(execute_command)(cmd) for cmd in [s_el, s_slope])
    else:
        os.system(s_el)
        os.system(s_slope)

def process_local_data(image_path, elevation, slope):
    get_ndvi_from_4bandS2(image_path)
    get_elevation_and_slope(image_path, elevation, slope)

"""
get_elevation_and_slope(image_path, elevation, slope)

with rasterio.open(image_path) as src:
    epsg = src.crs.to_string()
    bounds = src.bounds
    xres, yres = src.res

geom = ee_geom_from_image_bounds(image_path, buffer=0)
ee_image_tcvis = ee.ImageCollection("users/ingmarnitze/TCTrend_SR_2000-2019_TCVIS").mosaic()
tcvis_outfile = outdir/'tcvis.tif'

geemap.download_ee_image(ee_image_tcvis, filename=tcvis_outfile, region=geom[0], scale=xres, crs=epsg)
"""
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Prepare aux data for downloaded S2 images.')
    parser.add_argument('--data_dir', type=str, help='data directory (parent of download dir)', required=True)
    parser.add_argument('--image_regex', type=str, default='*/*SR.tif', help='')
    parser.add_argument('--aux_dir', 
                        default='/isipd/projects/p_aicore_pf/initze/processing/auxiliary/', 
                        type=str, 
                        help='parent directory of auxilliary data')
    args = parser.parse_args()
    
    input_dir = Path(args.data_dir)
    infiles = list(input_dir.glob(args.image_regex))
    
    base_dir = Path('/isipd/projects/p_aicore_pf/initze/processing/auxiliary/')
    # make absolute paths
    elevation = base_dir / 'elevation.vrt'
    slope = base_dir / 'slope.vrt'
    
    #for image_path in infiles:
    #    process_local_data(image_path, elevation, slope)
    
    Parallel(n_jobs=6)(delayed(process_local_data)(image_path, elevation, slope) for image_path in infiles)