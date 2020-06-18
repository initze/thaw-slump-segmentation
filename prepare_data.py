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
import torch
from docopt import docopt
from skimage.io import imsave
from tqdm import tqdm


def read_and_asert_imagedata(image_path):
    with rio.open(image_path) as raster:
        if raster.count <= 3:
            data = raster.read()
        else:
            data = raster.read()[:3]
        # Assert data can safely be coerced to int16
        assert data.max() < 2 ** 15
        return data


def others_from_img(img_path, datasets=['ndvi', 'tcvis', 'relative_elevation', 'slope']):
    """
    Given an image path, return paths for mask and tcvis
    """
    date, time, *block, platform, _, sr, row, col = img_path.stem.split('_')
    block = '_'.join(block)
    base = img_path.parent.parent

    mask_path = base / 'mask' / f'{date}_{time}_{block}_mask_{row}_{col}.tif'
    assert mask_path.exists()
    other_paths = []
    for ds in datasets:
        path = base / ds / f'{ds}_{row}_{col}.tif'
        assert path.exists()
        other_paths.append(path)

    return mask_path, list(other_paths)


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


def make_info_picture(imgtensor, masktensor, filename):
    "Make overview picture"
    rgbn = np.clip(imgtensor[:4,:,:].numpy().transpose(1, 2, 0) / 3000 * 255, 0, 255).astype(np.uint8)
    tcvis = np.clip(imgtensor[5:8,:,:].numpy().transpose(1, 2, 0), 0, 255).astype(np.uint8)

    rgb = rgbn[:,:,:3]
    nir = rgbn[:,:,[3,2,1]]
    mask = (masktensor.numpy()[0][:,:,np.newaxis][:,:,[0,0,0]] * 255).astype(np.uint8)

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
        gdal_merge = os.path.join(gdal_path, 'gdal_merge.py')
        gdal_retile = os.path.join(gdal_path, 'gdal_retile.py')
        gdal_rasterize = os.path.join(gdal_bin, 'gdal_rasterize')
        gdal_translate = os.path.join(gdal_bin, 'gdal_translate')

    DATA = Path('data')
    DEST = Path('data_pytorch')
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
        decision = input("Delete? [y/N]")
        if decision.lower() == 'y':
            for old_dir in overwrite_conflicts:
                shutil.rmtree(old_dir)
        else:
            print("Won't overwrite old data directories.")
            sys.exit(1)

    for dataset in datasets:
        print(f'Doing {dataset}')
        data_dir = DEST / dataset.name / 'data'
        data_dir.mkdir(parents=True)
        mask_dir = DEST / dataset.name / 'mask'
        mask_dir.mkdir()
        info_dir = DEST / dataset.name / 'info'
        info_dir.mkdir()

        if not args['--skip_gdal']:
            do_gdal_calls(dataset)

        # Convert data to pytorch tensor files (pt) for efficient data loading
        for img in tqdm(list(dataset.glob('tiles/data/*.tif'))):
            mask, other_paths = others_from_img(img)

            with rio.open(img) as raster:
                imgdata = raster.read()
                # Skip nodata tiles
                if (imgdata == 0).all(axis=0).mean() > THRESHOLD:
                    continue
                # Assert data can safely be coerced to int16
                assert imgdata.max() < 2 ** 15

            datasets = [read_and_asert_imagedata(ds) for ds in other_paths]
            full_data = np.concatenate([imgdata] + datasets, axis=0)
            imgtensor = torch.from_numpy(full_data.astype(np.int16))

            with rio.open(mask) as raster:
                maskdata = raster.read()
                assert maskdata.max() <= 1, "Mask can't contain values > 1"
                masktensor = torch.from_numpy(maskdata.astype(np.bool))

            # gdal_retile leaves narrow stripes at the right and bottom border,
            # which are filtered out here:
            if imgtensor.shape != (10, 256, 256):
                continue
            if masktensor.shape != (1, 256, 256):
                continue

            # Write the tensor files
            filename = img.stem + '.pt'
            torch.save(imgtensor, data_dir / filename)
            torch.save(masktensor, mask_dir / filename)
            make_info_picture(imgtensor, masktensor,
                    (info_dir / filename).with_suffix('.jpg'))
