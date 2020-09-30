#!/usr/bin/env python
# flake8: noqa: E501
"""
Usecase 2 Data Preprocessing Script

Usage:
    prepare_data.py [options]

Options:
    -h --help               Show this screen
    --skip_gdal             Skip the Gdal conversion stage (if it has already been done)
    --gdal_path=PATH        Path to gdal scripts (ignored if --skip_gdal is passed) [default: ]
    --gdal_bin=PATH         Path to gdal binaries (ignored if --skip_gdal is passed) [default: ]
    --nodata_threshold=THR  Throw away data with at least this % of nodata pixels [default: 50]
    --tile_size=XxY         Tiling size in pixels [default: 256x256]
    --tile_overlap          Overlap of the tiles in pixels [default: 25]
    --make_overviews        Make additional overview images in a seperate 'info' folder
"""
import os
import shutil
import sys
from pathlib import Path

import numpy as np
import rasterio as rio
import h5py
from docopt import docopt
from skimage.io import imsave
from tqdm import tqdm


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
        print('Found file:', candidates[0])
        return candidates[0]
    else:
        raise ValueError(f'Found {len(candidates)} candidates.'
                         'Please make selection more specific!')


def do_gdal_calls(DATASET, aux_data=['ndvi', 'tcvis', 'slope', 'relative_elevation']):
    maskfile = DATASET / f'{DATASET.name}_mask.tif'

    tile_dir_data = DATASET / 'tiles' / 'data'
    tile_dir_mask = DATASET / 'tiles' / 'mask'

    # Create parents on the first data folder
    tile_dir_data.mkdir(exist_ok=True, parents=True)
    tile_dir_mask.mkdir(exist_ok=True)

    rasterfile = glob_file(DATASET, RASTERFILTER)

    # Retile data, mask
    os.system(f'python {gdal_retile} -ps {XSIZE} {YSIZE} -overlap {OVERLAP} -targetDir {tile_dir_data} {rasterfile}')
    os.system(f'python {gdal_retile} -ps {XSIZE} {YSIZE} -overlap {OVERLAP} -targetDir {tile_dir_mask} {maskfile}')

    # Retile additional data
    for aux in aux_data:
        auxfile = DATASET / f'{aux}.tif'
        tile_dir_aux = DATASET / 'tiles' / aux
        tile_dir_aux.mkdir(exist_ok=True)
        os.system(
            f'python {gdal_retile} -ps {XSIZE} {YSIZE} -overlap {OVERLAP} -targetDir {tile_dir_aux} {auxfile}')


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


if __name__ == "__main__":
    args = docopt(__doc__, version="Usecase 2 Data Preprocessing Script 1.0")
    # Tiling Settings
    XSIZE, YSIZE = map(int, args['--tile_size'].split('x'))
    OVERLAP = int(args['--tile_overlap'])

    # Paths setup
    RASTERFILTER = '*3B_AnalyticMS_SR.tif'
    VECTORFILTER = '*.shp'
    THRESHOLD = float(args['--nodata_threshold']) / 100

    if not args['--skip_gdal']:
        gdal_path = args['--gdal_path']
        gdal_bin = args['--gdal_bin']
        # TODO: we're only using gdal_retile here... is it safe to delete the others?
        gdal_merge = os.path.join(gdal_path, 'gdal_merge.py')
        gdal_retile = os.path.join(gdal_path, 'gdal_retile.py')
        gdal_rasterize = os.path.join(gdal_bin, 'gdal_rasterize')
        gdal_translate = os.path.join(gdal_bin, 'gdal_translate')

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
        print(f"Found old data directories: {', '.join(dir.name for dir in overwrite_conflicts)}.")
        decision = input("Delete and recreate them [d], skip them [s] or abort [a]? ").lower()
        if decision == 'd':
            for old_dir in overwrite_conflicts:
                shutil.rmtree(old_dir)
        elif decision == 's':
            already_done = [d.name for d in overwrite_conflicts]
            datasets = [d for d in datasets if d.name not in already_done]
        else:
            # When in doubt, don't overwrite/change anything to be safe
            print("Aborting due to conflicts with existing data directories.")
            sys.exit(1)

    for dataset in datasets:
        print(f'Doing {dataset}')
        h5_path = DEST / f'{dataset.name}.h5'
        info_dir = DEST / dataset.name
        info_dir.mkdir(parents=True)

        h5 = h5py.File(h5_path, 'w')
        channel_numbers = dict(planet=4, ndvi=1, tcvis=3, relative_elevation=1, slope=1)

        datasets = dict()
        for dataset_name, nchannels in channel_numbers.items():
            ds = h5.create_dataset(dataset_name,
                dtype = np.float32,
                shape = (512, nchannels, 256, 256),
                maxshape=(None, nchannels, 256, 256),
                chunks = (1, nchannels, 256, 256),
                compression = 'lzf',
                scaleoffset = 3,
            )
            datasets[dataset_name] = ds

        datasets['mask'] = h5.create_dataset("mask",
            dtype = np.uint8,
            shape = (512, 1, 256, 256),
            maxshape=(None, 1, 256, 256),
            chunks = (1, 1, 256, 256),
            compression = 'lzf',
        )

        if not args['--skip_gdal']:
            do_gdal_calls(dataset)

        # Convert data to HDF5 storage for efficient data loading
        i = 0
        for img in tqdm(list(dataset.glob('tiles/data/*.tif'))):
            tile = {}
            with rio.open(img) as raster:
                tile['planet'] = raster.read()

            if (tile['planet'] == 0).all(axis=0).mean() > THRESHOLD:
                continue

            with rio.open(mask_from_img(img)) as raster:
                tile['mask'] = raster.read()
            assert tile['mask'].max() <= 1, "Mask can't contain values > 1"

            for other in channel_numbers:
                if other == 'planet': continue  # We already did this!
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
                continue

            if(datasets['planet'].shape[0] <= i):
                for ds in datasets.values():
                    ds.resize(ds.shape[0] + 2048, axis=0)
            for t in tile:
                datasets[t][i] = tile[t]

            make_info_picture(tile, info_dir / f'{i}.jpg')
            i += 1

        for t in datasets:
            datasets[t].resize(i, axis=0)
