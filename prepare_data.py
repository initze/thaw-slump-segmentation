#!/usr/bin/env python
# flake8: noqa: E501
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
from joblib import Parallel, delayed
from skimage.io import imsave

from lib.data_pre_processing import gdal
from lib.utils import init_logging, get_logger, log_run

parser = argparse.ArgumentParser()
parser.add_argument("--skip_gdal", action='store_true', help="Skip the Gdal conversion stage (if it has already been "
                                                             "done)")
parser.add_argument("--gdal_bin", default='', help="Path to gdal binaries (ignored if --skip_gdal is passed)")
parser.add_argument("--gdal_path", default='', help="Path to gdal scripts (ignored if --skip_gdal is passed)")
parser.add_argument("--n_jobs", default=-1, type=int, help="number of parallel joblib jobs")
parser.add_argument("--nodata_threshold", default=50, type=float, help="Throw away data with at least this % of "
                                                                       "nodata pixels")
parser.add_argument("--tile_size", default='256x256', type=str, help="Tiling size in pixels e.g. '256x256'")
parser.add_argument("--tile_overlap", default=25, type=int, help="Overlap of the tiles in pixels")

#Label names ToDo: Move to config and use from there
class_names = ["other", "slope", "lake", "island", "feature4", "feature5", "feature6", "feature7"]
class_count = np.zeros(len(class_names))


def read_and_assert_imagedata(image_path):
    with rio.open(image_path) as raster:
        if raster.count <= 3:
            data = raster.read()
        else:
            data = raster.read()[:3]
        # Assert data can safely be coerced to int16
        assert data.max() < 2 ** 15
        return data


def mask_from_img(img_path):
    """
    Given an image path, return path for the mask
    """
    date, time, *block, platform, _, sr, row, col = img_path.stem.split('_')
    block = '_'.join(block)
    base = img_path.parent.parent

    mask_path = base / 'mask' / f'{date}_{time}_{block}_mask_{row}_{col}.tif'
    assert mask_path.exists()

    return mask_path


def other_from_img(img_path, other):
    """
    Given an image path, return paths for mask and tcvis
    """
    date, time, *block, platform, _, sr, row, col = img_path.stem.split('_')
    block = '_'.join(block)
    base = img_path.parent.parent

    path = base / other / f'{other}_{row}_{col}.tif'
    assert path.exists()

    return path


def glob_file(DATASET, filter_string):
    candidates = list(DATASET.glob(f'{filter_string}'))
    if len(candidates) == 1:
        logger.debug(f'Found file: {candidates[0]}')
        return candidates[0]
    else:
        raise ValueError(f'Found {len(candidates)} candidates.'
                         'Please make selection more specific!')


def do_gdal_calls(DATASET, aux_data=['ndvi', 'tcvis', 'slope', 'relative_elevation'], logger=None):
    maskfile = DATASET / f'{DATASET.name}_mask.tif'

    tile_dir_data = DATASET / 'tiles' / 'data'
    tile_dir_mask = DATASET / 'tiles' / 'mask'

    # Create parents on the first data folder
    tile_dir_data.mkdir(exist_ok=True, parents=True)
    tile_dir_mask.mkdir(exist_ok=True)

    rasterfile = glob_file(DATASET, RASTERFILTER)

    # Retile data, mask
    log_run(f'python {gdal.retile} -ps {XSIZE} {YSIZE} -overlap {OVERLAP} -targetDir {tile_dir_data} {rasterfile}', logger)
    log_run(f'python {gdal.retile} -ps {XSIZE} {YSIZE} -overlap {OVERLAP} -targetDir {tile_dir_mask} {maskfile}', logger)

    # Retile additional data
    for aux in aux_data:
        auxfile = DATASET / f'{aux}.tif'
        tile_dir_aux = DATASET / 'tiles' / aux
        tile_dir_aux.mkdir(exist_ok=True)
        log_run(f'python {gdal.retile} -ps {XSIZE} {YSIZE} -overlap {OVERLAP} -targetDir {tile_dir_aux} {auxfile}', logger)


def make_info_picture(tile, filename):
    "Make overview picture"
    rgbn = np.clip(tile['planet'][:4,:,:].transpose(1, 2, 0) / 3000 * 255, 0, 255).astype(np.uint8)
    tcvis = np.clip(tile['tcvis'].transpose(1, 2, 0), 0, 255).astype(np.uint8)

    rgb = rgbn[:,:,:3]
    nir = rgbn[:,:,[3,2,1]]
    mask = (tile['mask'][[0,0,0]].transpose(1, 2, 0) * 255).astype(np.uint8)

    img = np.concatenate([
        np.concatenate([rgb, nir], axis=1),
        np.concatenate([tcvis, mask], axis=1),
    ])
    imsave(filename, img)


def main_function(dataset, args, log_path):
    if not args.skip_gdal:
        gdal.initialize(args)

    init_logging(log_path)
    thread_logger = get_logger(f'prepare_data.{dataset.name}')
    thread_logger.info(f'Starting preparation on dataset {dataset}')
    if not args.skip_gdal:
        thread_logger.info('Doing GDAL Calls')
        do_gdal_calls(dataset, logger=thread_logger)
    else:
        thread_logger.info('Skipping GDAL Calls')


    tifs = list(sorted(dataset.glob('tiles/data/*.tif')))
    if len(tifs) == 0:
        logger.warning(f'No tiles found for {dataset}, skipping this directory.')
        return

    h5_path = DEST / f'{dataset.name}.h5'
    info_dir = DEST / dataset.name
    info_dir.mkdir(parents=True)

    thread_logger.info(f'Creating H5File at {h5_path}')
    h5 = h5py.File(h5_path, 'w',
                   rdcc_nbytes=2 * (1 << 30),  # 2 GiB
                   rdcc_nslots=200003,
                   )
    channel_numbers = dict(planet=4, ndvi=1, tcvis=3, relative_elevation=1, slope=1)

    datasets = dict()
    for dataset_name, nchannels in channel_numbers.items():
        ds = h5.create_dataset(dataset_name,
                               dtype=np.float32,
                               shape=(len(tifs), nchannels, 256, 256),
                               maxshape=(len(tifs), nchannels, 256, 256),
                               chunks=(1, nchannels, 256, 256),
                               compression='lzf',
                               scaleoffset=3,
                               )
        datasets[dataset_name] = ds

    datasets['mask'] = h5.create_dataset("mask",
                                         dtype=np.uint8,
                                         shape=(min(1024, len(tifs)), 1, 256, 256),
                                         maxshape=(len(tifs), 1, 256, 256),
                                         chunks=(1, 1, 256, 256),
                                         compression='lzf',
                                         )

    # Convert data to HDF5 storage for efficient data loading
    i = 0
    bad_tiles = 0
    for img in tifs:
        tile = {}
        with rio.open(img) as raster:
            tile['planet'] = raster.read()

        if (tile['planet'] == 0).all(axis=0).mean() > THRESHOLD:
            bad_tiles += 1
            continue



        with rio.open(mask_from_img(img)) as raster:
            tile['mask'] = raster.read()

            #count labels
            features_unique, features_counts = np.unique(tile['mask'], return_counts=True)
            if features_unique is list:
                for i in enumerate(len(features_unique)):
                    class_count[features_unique[i]] += features_counts[i]
            else:
                class_count[features_unique] += features_counts

        for other in channel_numbers:
            if other == 'planet':
                continue  # We already did this!
            with rio.open(other_from_img(img, other)) as raster:
                data = raster.read()
            if data.shape[0] > channel_numbers[other]:
                # This is for tcvis mostly
                data = data[:channel_numbers[other]]
            tile[other] = data

        # gdal_retile leaves narrow stripes at the right and bottom border,
        # which are filtered out here:
        is_narrow = False
        for tensor in tile.values():
            if tensor.shape[-2:] != (256, 256):
                is_narrow = True
                break
        if is_narrow:
            bad_tiles += 1
            continue

        """
        if(datasets['planet'].shape[0] <= i):
            for ds in datasets.values():
                ds.resize(ds.shape[0] + 2048, axis=0)
        """
        for t in tile:
            datasets[t][i] = tile[t]

        make_info_picture(tile, info_dir / f'{i}.jpg')
        i += 1

    for t in datasets:
        datasets[t].resize(i, axis=0)

    class_count_percentage = (class_count/class_count.sum())*100
    class_print = 'Found '
    for i in range(1, len(class_names)):
        if class_count_percentage[i] > 0:
            class_print += f'{class_count_percentage[i]:0.2f} % of class {class_names[i]}, '
    class_print += f'and {class_count_percentage[0]:0.2f} % of class {class_names[0]}.' #undefined class named 'other'.
    thread_logger.info(class_print)

if __name__ == "__main__":
    args = parser.parse_args()
    # Tiling Settings
    XSIZE, YSIZE = map(int, args.tile_size.split('x'))
    OVERLAP = args.tile_overlap

    timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    log_path = Path('logs') / f'prepare_data-{timestamp}.log'
    init_logging(log_path)
    logger = get_logger('prepare_data')
    logger.info('#############################')
    logger.info('# Starting Data Preparation #')
    logger.info('#############################')

    # Paths setup
    RASTERFILTER = '*3B_AnalyticMS_SR*.tif'
    VECTORFILTER = '*.shp'
    THRESHOLD = args.nodata_threshold / 100

    if not args.skip_gdal:
        gdal.initialize(args)

    DATA = Path('data')
    DEST = Path('data_h5')
    DEST.mkdir(exist_ok=True)

    # All folders that contain the big raster (...AnalyticsML_SR.tif) are assumed to be a dataset
    datasets = [raster.parent for raster in DATA.glob('*/' + RASTERFILTER)]

    overwrite_conflicts = []
    for dataset in datasets:
        check_dir = DEST / dataset.name
        if check_dir.exists():
            overwrite_conflicts.append(check_dir)

    if overwrite_conflicts:
        logger.warning(f"Found old data directories: {', '.join(dir.name for dir in overwrite_conflicts)}.")
        decision = input("Delete and recreate them [d], skip them [s] or abort [a]? ").lower()
        if decision == 'd':
            logger.info(f"User chose to delete old directories.")
            for old_dir in overwrite_conflicts:
                shutil.rmtree(old_dir)
        elif decision == 's':
            logger.info(f"User chose to skip old directories.")
            already_done = [d.name for d in overwrite_conflicts]
            datasets = [d for d in datasets if d.name not in already_done]
        else:
            # When in doubt, don't overwrite/change anything to be safe
            logger.error("Aborting due to conflicts with existing data directories.")
            sys.exit(1)

    Parallel(n_jobs=args.n_jobs)(delayed(main_function)(dataset, args, log_path) for dataset in datasets)
