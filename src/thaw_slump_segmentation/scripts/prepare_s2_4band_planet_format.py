import argparse
import os
from pathlib import Path

import ee
import geemap
import numpy as np
import rasterio
import typer
from joblib import Parallel, delayed

# from ..data_pre_processing.earthengine import ee_geom_from_image_bounds
from rasterio.coords import BoundingBox
from typing_extensions import Annotated

ee.Initialize(opt_url='https://earthengine-highvolume.googleapis.com')


def get_ndvi_from_4bandS2(image_path):
    image_path = Path(image_path)
    ndvi_path = image_path.parent / 'ndvi.tif'
    if not ndvi_path.exists():
        with rasterio.open(image_path) as src:
            # read data
            data = src.read().astype(float)
            # calc ndvi
            ndvi = (data[3] - data[2]) / (data[3] + data[2])
            # factor to correct output
            ndvi_out = ((np.clip(ndvi, -1, 1) + 1) * 1e4).astype(np.uint16)
            # get and adapt profile to match output
            profile = src.profile
            profile.update({'dtype': 'uint16', 'count': 1})
        # save ndvi

        with rasterio.open(ndvi_path, 'w', **profile) as target:
            target.write(np.expand_dims(ndvi_out, 0))
    else:
        print('NDVI file already exists!')


def get_elevation_and_slope(image_path, rel_el_vrt, slope_vrt, parallel=True):
    # get image_metadata
    with rasterio.open(image_path) as src:
        epsg = src.crs.to_string()
        bounds = src.bounds
        xres, yres = src.res

    # setup gdal runs
    target_el = image_path.parent / 'relative_elevation.tif'
    if not target_el.exists():
        s_el = f'gdalwarp -multi -tap -te {bounds.left} {bounds.top} {bounds.right} {bounds.bottom} -t_srs {epsg} -tr {xres} {yres} -r cubic -co COMPRESS=DEFLATE {rel_el_vrt} {target_el}'
    else:
        print('Elevation file already exists!')
        s_el = ''
    target_slope = image_path.parent / 'slope.tif'
    if not target_slope.exists():
        s_slope = f'gdalwarp -multi -tap -te {bounds.left} {bounds.top} {bounds.right} {bounds.bottom} -t_srs {epsg} -tr {xres} {yres} -r cubic -co COMPRESS=DEFLATE {slope_vrt} {target_slope}'
    else:
        print('Slope file already exists!')
        s_slope = ''
    # run in console
    if parallel:

        def execute_command(cmd):
            os.system(cmd)

        # Parallel execution
        Parallel(n_jobs=2)(delayed(execute_command)(cmd) for cmd in [s_el, s_slope])
    else:
        os.system(s_el)
        os.system(s_slope)


def replace_tcvis_zeronodata(infile):
    """
    This function replaces zero values in the input raster file with 1, and writes the result to a new file.

    Parameters:
    infile (Path): The path to the input raster file.

    The function performs the following steps:
    1. Reads the data from the input file.
    2. Creates a mask of zero values.
    3. Identifies values with single bands at 0.
    4. Replaces zero values with 1.
    5. Writes the modified data to a new file with the same profile as the input file.
    6. Deletes the original file.
    7. Renames the new file to have the same name as the original file.

    The function does not return any value.
    """
    # setup outfile name
    tcvis_replace = infile.parent / 'tcvis_tmpfix.tif'

    with rasterio.open(infile, 'r') as src:
        # read data
        ds = src.read()
        # get mask
        mask = ds == 0
        # get values with single bands at 0
        mask_all = mask.all(axis=0)
        mask_any = mask.any(axis=0)
        replace_mask = np.logical_and(mask_any, ~mask_all)
        # replace zero values with 1a
        for i in [0, 1, 2]:
            ds[i][(ds[i] == 0) & replace_mask] = 1
        # get profile for new output
        profile = src.profile

    with rasterio.open(tcvis_replace, 'w', **profile) as dst:
        dst.write(ds)

    # delete_original
    infile.unlink()
    # rename
    tcvis_replace.rename(infile)


def process_local_data(image_path, elevation, slope):
    get_ndvi_from_4bandS2(image_path)
    get_elevation_and_slope(image_path, elevation, slope)


def download_tcvis(image_path):
    """
    Downloads a TCVIS, processes it, and saves the result as a GeoTIFF file.

    Args:
        image_path (str): Path to the input Sentinel-2 image (JP2 format).

    Returns:
        None: The function saves the processed TCVIS GeoTIFF file locally.

    Example:
        download_tcvis('path/to/your/input_image.tif')
    """

    with rasterio.open(image_path) as src:
        epsg = src.crs.to_string()
        crs = src.crs
        bounds = src.bounds
        xres, yres = src.res

    # needs to cut one pixel on top
    fixed_bounds = BoundingBox(left=bounds.left, bottom=bounds.bottom, right=bounds.right, top=bounds.top - yres)
    # create ee geom
    gdf = geemap.bbox_to_gdf(fixed_bounds)
    gdf.crs = crs
    geom = geemap.gdf_to_ee(gdf).first().geometry()

    # download result
    ee_image_tcvis = ee.ImageCollection('users/ingmarnitze/TCTrend_SR_2000-2019_TCVIS').mosaic()
    tcvis_outfile = image_path.parent / 'tcvis.tif'

    if not tcvis_outfile.exists():
        geemap.download_ee_image(ee_image_tcvis, filename=tcvis_outfile, region=geom, scale=xres, crs=epsg)
    else:
        print(f'TCVIS file for {image_path.parent.name} already exists!')
        return 0

    # check outfile props
    with rasterio.open(tcvis_outfile) as src:
        bounds_tcvis = src.bounds
        xres_tcvis, yres_tcvis = src.res
    # show diff
    if not bounds == bounds_tcvis:
        print(f'Downloaded TCVIS Image for dataset {image_path.parent.name} has wrong dimensions')
        print('Input image:', bounds, xres, yres)
        print('TCVIS Image:', bounds_tcvis, xres_tcvis, yres_tcvis)

        # fix output in case of size mismatch
        tcvis_outfile_tmp = tcvis_outfile.parent / 'tcvis_temp.tif'
        tcvis_outfile.rename(tcvis_outfile_tmp)
        s_fix_tcvis = f'gdalwarp -te {bounds.left} {bounds.bottom} {bounds.right} {bounds.top} -tr {xres} {yres} -co COMPRESS=DEFLATE {tcvis_outfile_tmp} {tcvis_outfile}'
        os.system(s_fix_tcvis)
        tcvis_outfile_tmp.unlink()

    print('Write mask corrected TCVIS')
    replace_tcvis_zeronodata(tcvis_outfile)


def prepare_s2_4band_planet_format(
    data_dir: Annotated[Path, typer.Option('--data_dir', help='data directory (parent of download dir)')],
    image_regex: Annotated[str, typer.Option('--image_regex', help='regex term to find image file')] = '*/*SR.tif',
    n_jobs: Annotated[int, typer.Option('--n_jobs', help='Number of parallel- images to prepare data for')] = 6,
    aux_dir: Annotated[
        Path, typer.Option('--aux_dir', help='parent directory of auxilliary data')
    ] = '/isipd/projects/p_aicore_pf/initze/processing/auxiliary/',
):
    input_dir = Path(data_dir)
    infiles = list(input_dir.glob(image_regex))

    # make absolute paths
    elevation = aux_dir / 'elevation.vrt'
    slope = aux_dir / 'slope.vrt'

    # for image_path in infiles:
    Parallel(n_jobs=n_jobs)(delayed(process_local_data)(image_path, elevation, slope) for image_path in infiles)

    for image_path in infiles:
        download_tcvis(image_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Prepare aux data for downloaded S2 images.', formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('--data_dir', type=str, help='data directory (parent of download dir)', required=True)
    parser.add_argument('--image_regex', type=str, default='*/*SR.tif', help='regex term to find image file')
    parser.add_argument('--n_jobs', type=int, default=6, help='Number of parallel- images to prepare data for')
    parser.add_argument(
        '--aux_dir',
        default='/isipd/projects/p_aicore_pf/initze/processing/auxiliary/',
        type=str,
        help='parent directory of auxilliary data',
    )
    args = parser.parse_args()

    prepare_s2_4band_planet_format(
        data_dir=args.data_dir, image_regex=args.image_regex, n_jobs=args.n_jobs, aux_dir=args.aux_dir
    )
