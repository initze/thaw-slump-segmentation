#!/usr/bin/env python
# flake8: noqa: E501
# Copyright (c) Ingmar Nitze and Konrad Heidler

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
Usecase 2 Data Preprocessing Script
"""

import argparse
import shutil
import sys
from datetime import datetime
from pathlib import Path

import h5py
import numpy as np
import rasterio as rio
import typer
from joblib import Parallel, delayed
from skimage.io import imsave
from typing_extensions import Annotated

from thaw_slump_segmentation.data_pre_processing import gdal
from thaw_slump_segmentation.utils import get_logger, init_logging, log_run

# Paths setup
RASTERFILTER = '*_SR*.tif'
VECTORFILTER = '*.shp'


def read_and_assert_imagedata(image_path):
    with rio.open(image_path) as raster:
        if raster.count <= 3:
            data = raster.read()
        else:
            data = raster.read()[:3]
        # Assert data can safely be coerced to int16
        assert data.max() < 2**15
        return data


def get_planet_product_type(img_path):
    """
    return if file is scene or OrthoTile"""
    split = img_path.stem.split('_')
    # check if 4th last imagename segment is BGRN
    if split[-4] == 'BGRN':
        pl_type = 'OrthoTile'
    elif len(split) == 6 and len(split[2]) == 6:
        pl_type = "Sentinel2"
    else:
        pl_type = 'Scene'

    return pl_type


def mask_from_img(img_path):
    """
    Given an image path, return path for the mask
    """
    # change for
    product_type = get_planet_product_type(img_path)
    base = img_path.parent.parent

    if product_type == 'Scene':
        date, time, *block, platform, _, sr, row, col = img_path.stem.split('_')
        block = '_'.join(block)
        mask_path = base / 'mask' / f'{date}_{time}_{block}_mask_{row}_{col}.tif'
    elif product_type == "Sentinel2":
        date, processing_date, tile_id, _, row, col = img_path.stem.split('_')
        mask_path = base / 'mask' / f'{date}_{processing_date}_{tile_id}_mask_{row}_{col}.tif'
    else:
        block, tile, date, sensor, bgrn, sr, row, col = img_path.stem.split('_')
        mask_path = base / 'mask' / f'{block}_{tile}_{date}_{sensor}_mask_{row}_{col}.tif'

    assert mask_path.exists()

    return mask_path


def other_from_img(img_path, other):
    """
    Given an image path, return paths for mask and tcvis
    """
    product_type = get_planet_product_type(img_path)
    if product_type == 'Scene':
        date, time, *block, platform, _, sr, row, col = img_path.stem.split('_')
    elif product_type == "Sentinel2":
        date, processing_date, tile_id, _, row, col = img_path.stem.split('_')
    else:
        block, tile, date, sensor, bgrn, sr, row, col = img_path.stem.split('_')

    base = img_path.parent.parent

    path = base / other / f'{other}_{row}_{col}.tif'
    assert path.exists()

    return path


def glob_file(DATASET, filter_string, logger=None):
    candidates = list(DATASET.glob(f'{filter_string}'))
    if len(candidates) == 1:
        logger.debug(f'Found file: {candidates[0]}')
        return candidates[0]
    else:
        raise ValueError(f'Found {len(candidates)} candidates.' 'Please make selection more specific!')


def do_gdal_calls(
    DATASET,
    xsize: int,
    ysize: int,
    overlap: int,
    aux_data=['ndvi', 'tcvis', 'slope', 'relative_elevation'],
    logger=None,
):
    rasterfile = glob_file(DATASET, RASTERFILTER, logger)
    image_name = rasterfile.name[0:-7]

    maskfile = DATASET / f'{image_name}_mask.tif'

    tile_dir_data = DATASET / 'tiles' / 'data'
    tile_dir_mask = DATASET / 'tiles' / 'mask'

    # Create parents on the first data folder
    tile_dir_data.mkdir(exist_ok=True, parents=True)
    tile_dir_mask.mkdir(exist_ok=True)


    # Retile data, mask
    # log_run(f'python {gdal.retile} -ps {XSIZE} {YSIZE} -overlap {OVERLAP} -targetDir {tile_dir_data} {rasterfile}', logger)
    # log_run(f'python {gdal.retile} -ps {XSIZE} {YSIZE} -overlap {OVERLAP} -targetDir {tile_dir_mask} {maskfile}', logger)
    log_run(f'{gdal.retile} -ps {xsize} {ysize} -overlap {overlap} -targetDir {tile_dir_data} {rasterfile}', logger)
    log_run(f'{gdal.retile} -ps {xsize} {ysize} -overlap {overlap} -targetDir {tile_dir_mask} {maskfile}', logger)

    # Retile additional data
    for aux in aux_data:
        auxfile = DATASET / f'{aux}.tif'
        tile_dir_aux = DATASET / 'tiles' / aux
        tile_dir_aux.mkdir(exist_ok=True)
        # log_run(f'python {gdal.retile} -ps {XSIZE} {YSIZE} -overlap {OVERLAP} -targetDir {tile_dir_aux} {auxfile}', logger)
        log_run(f'{gdal.retile} -ps {xsize} {ysize} -overlap {overlap} -targetDir {tile_dir_aux} {auxfile}', logger)


def make_info_picture(tile, filename):
    "Make overview picture"
    rgbn = np.clip(tile['planet'][:4, :, :].transpose(1, 2, 0) / 3000 * 255, 0, 255).astype(np.uint8)
    tcvis = np.clip(tile['tcvis'].transpose(1, 2, 0), 0, 255).astype(np.uint8)

    rgb = rgbn[:, :, :3]
    nir = rgbn[:, :, [3, 2, 1]]
    mask = (tile['mask'][[0, 0, 0]].transpose(1, 2, 0) * 255).astype(np.uint8)

    img = np.concatenate(
        [
            np.concatenate([rgb, nir], axis=1),
            np.concatenate([tcvis, mask], axis=1),
        ]
    )
    imsave(filename, img)


def main_function(
    dataset, log_path, h5dir: Path, xsize: int, ysize: int, overlap: int, threshold: float, skip_gdal: bool, gdal_bin:str, gdal_path:str
):
    init_logging(log_path)
    thread_logger = get_logger(f'prepare_data.{dataset.name}')
    thread_logger.info(f'Starting preparation on dataset {dataset}')
    if not skip_gdal:
        gdal.initialize(bin=gdal_bin, path=gdal_bin)
        thread_logger.info('Doing GDAL Calls')
        do_gdal_calls(dataset, xsize, ysize, overlap, logger=thread_logger)
    else:
        thread_logger.info('Skipping GDAL Calls')

    tifs = list(sorted(dataset.glob('tiles/data/*.tif')))
    if len(tifs) == 0:
        thread_logger.warning(f'No tiles found for {dataset}, skipping this directory.')
        return

    h5_path = h5dir / f'{dataset.name}.h5'
    info_dir = h5dir / dataset.name
    info_dir.mkdir(parents=True)

    thread_logger.info(f'Creating H5File at {h5_path}')
    h5 = h5py.File(
        h5_path,
        'w',
        rdcc_nbytes=2 * (1 << 30),  # 2 GiB
        rdcc_nslots=200003,
    )
    channel_numbers = dict(planet=4, ndvi=1, tcvis=3, relative_elevation=1, slope=1)

    datasets = dict()
    for dataset_name, nchannels in channel_numbers.items():
        ds = h5.create_dataset(
            dataset_name,
            dtype=np.float32,
            shape=(len(tifs), nchannels, xsize, ysize),
            maxshape=(len(tifs), nchannels, xsize, ysize),
            chunks=(1, nchannels, xsize, ysize),
            compression='lzf',
            scaleoffset=3,
        )
        datasets[dataset_name] = ds

    datasets['mask'] = h5.create_dataset(
        'mask',
        dtype=np.uint8,
        shape=(len(tifs), 1, xsize, ysize),
        maxshape=(len(tifs), 1, xsize, ysize),
        chunks=(1, 1, xsize, ysize),
        compression='lzf',
    )

    # Convert data to HDF5 storage for efficient data loading
    i = 0
    bad_tiles = 0
    for img in tifs:
        tile = {}
        with rio.open(img) as raster:
            tile['planet'] = raster.read()

        if (tile['planet'] == 0).all(axis=0).mean() > threshold:
            bad_tiles += 1
            continue

        with rio.open(mask_from_img(img)) as raster:
            tile['mask'] = raster.read()
        assert tile['mask'].max() <= 1, "Mask can't contain values > 1"

        for other in channel_numbers:
            if other == 'planet':
                continue  # We already did this!
            with rio.open(other_from_img(img, other)) as raster:
                data = raster.read()
            if data.shape[0] > channel_numbers[other]:
                # This is for tcvis mostly
                data = data[: channel_numbers[other]]
            tile[other] = data

        # gdal_retile leaves narrow stripes at the right and bottom border,
        # which are filtered out here:
        is_narrow = False
        for tensor in tile.values():
            if tensor.shape[-2:] != (xsize, ysize):
                is_narrow = True
                break
        if is_narrow:
            bad_tiles += 1
            continue

        for t in tile:
            datasets[t][i] = tile[t]

        make_info_picture(tile, info_dir / f'{i}.jpg')
        i += 1

    for t in datasets:
        datasets[t].resize(i, axis=0)


def tile_size_callback(value: str):
    try:
        x, y = map(int, value.split('x'))
    except ValueError:
        raise typer.BadParameter('Tile size must be in the format "256x256"')
    return x, y


def prepare_data(
    data_dir: Annotated[Path, typer.Option('--data_dir', help='Path to data processing dir')] = Path('data'),
    log_dir: Annotated[Path, typer.Option('--log_dir', help='Path to log dir')] = Path('logs'),
    skip_gdal: Annotated[
        bool,
        typer.Option('--skip_gdal/--use_gdal', help='Skip the Gdal conversion stage (if it has already been done)'),
    ] = False,
    gdal_bin: Annotated[
        str,
        typer.Option('--gdal_bin', help='Path to gdal binaries (ignored if --skip_gdal is passed)', envvar='GDAL_BIN'),
    ] = '/usr/bin',
    gdal_path: Annotated[
        str,
        typer.Option('--gdal_path', help='Path to gdal scripts (ignored if --skip_gdal is passed)', envvar='GDAL_PATH'),
    ] = '/usr/bin',
    n_jobs: Annotated[int, typer.Option('--n_jobs', help='number of parallel joblib jobs')] = -1,
    nodata_threshold: Annotated[
        float, typer.Option('--nodata_threshold', help='Throw away data with at least this % of nodata pixels')
    ] = 50,
    tile_size: Annotated[
        str, typer.Option('--tile_size', help="Tiling size in pixels e.g. '256x256'", callback=tile_size_callback)
    ] = '256x256',
    tile_overlap: Annotated[int, typer.Option('--tile_overlap', help='Overlap of the tiles in pixels')] = 25,
):
    """Make data ready for training"""

    # Tiling Settings
    if isinstance(tile_size, tuple):
        xsize, ysize = tile_size
    else:
        xsize, ysize = map(int, tile_size.split('x'))
    overlap = tile_overlap

    timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    log_path = log_dir / f'prepare_data-{timestamp}.log'
    log_dir.mkdir(exist_ok=True)
    init_logging(log_path)
    logger = get_logger('prepare_data')
    logger.info('#############################')
    logger.info('# Starting Data Preparation #')
    logger.info('#############################')

    threshold = nodata_threshold / 100

    DATA_ROOT = data_dir
    DATA_DIR = DATA_ROOT / 'tiles'
    h5dir = DATA_ROOT / 'h5'
    h5dir.mkdir(exist_ok=True)

    # All folders that contain the big raster (...AnalyticsML_SR.tif) are assumed to be a dataset
    datasets = [raster.parent for raster in DATA_DIR.glob('*/' + RASTERFILTER)]

    overwrite_conflicts = []
    for dataset in datasets:
        check_dir = h5dir / dataset.name
        if check_dir.exists():
            overwrite_conflicts.append(check_dir)

    if overwrite_conflicts:
        logger.warning(f"Found old data directories: {', '.join(dir.name for dir in overwrite_conflicts)}.")
        decision = input('Delete and recreate them [d], skip them [s] or abort [a]? ').lower()
        if decision == 'd':
            logger.info('User chose to delete old directories.')
            for old_dir in overwrite_conflicts:
                shutil.rmtree(old_dir)
        elif decision == 's':
            logger.info('User chose to skip old directories.')
            already_done = [d.name for d in overwrite_conflicts]
            datasets = [d for d in datasets if d.name not in already_done]
        else:
            # When in doubt, don't overwrite/change anything to be safe
            logger.error('Aborting due to conflicts with existing data directories.')
            sys.exit(1)

    Parallel(n_jobs=n_jobs)(
        delayed(main_function)(dataset, log_path, h5dir, xsize, ysize, overlap, threshold, skip_gdal, gdal_bin, gdal_path)
        for dataset in datasets
    )


# ! Moving legacy argparse cli to main to maintain compatibility with the original script
if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Make data ready for training', formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('--data_dir', default='data', type=Path, help='Path to data processing dir')
    parser.add_argument('--log_dir', default='logs', type=Path, help='Path to log dir')
    parser.add_argument(
        '--skip_gdal', action='store_true', help='Skip the Gdal conversion stage (if it has already been ' 'done)'
    )
    parser.add_argument('--gdal_bin', default=None, help='Path to gdal binaries (ignored if --skip_gdal is passed)')
    parser.add_argument('--gdal_path', default=None, help='Path to gdal scripts (ignored if --skip_gdal is passed)')
    parser.add_argument('--n_jobs', default=-1, type=int, help='number of parallel joblib jobs')
    parser.add_argument(
        '--nodata_threshold', default=50, type=float, help='Throw away data with at least this % of ' 'nodata pixels'
    )
    parser.add_argument('--tile_size', default='256x256', type=str, help="Tiling size in pixels e.g. '256x256'")
    parser.add_argument('--tile_overlap', default=25, type=int, help='Overlap of the tiles in pixels')

    args = parser.parse_args()

    prepare_data(
        data_dir=args.data_dir,
        log_dir=args.log_dir,
        skip_gdal=args.skip_gdal,
        gdal_bin=args.gdal_bin,
        gdal_path=args.gdal_path,
        n_jobs=args.n_jobs,
        nodata_threshold=args.nodata_threshold,
        tile_size=args.tile_size,
        tile_overlap=args.tile_overlap,
    )
