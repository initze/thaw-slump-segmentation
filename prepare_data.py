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
from skimage.io import imsave
from pathlib import Path
from tqdm import tqdm
import rasterio as rio
import numpy as np
import torch
import shutil
import os, sys

from docopt import docopt


def others_from_img(img_path):
    """
    Given an image path, return paths for mask and tcvis
    """
    date, time, *block, platform, _, sr, row, col = img_path.stem.split('_')
    block = '_'.join(block)
    base = img_path.parent.parent
    mask_path = base / 'mask' / f'{date}_{time}_{block}_mask_{row}_{col}.tif'
    tcvis_path = base / 'tcvis' / f'tcvis_{row}_{col}.tif'

    assert mask_path.exists()
    assert tcvis_path.exists()

    return mask_path, tcvis_path


def glob_file(DATASET, filter_string):
    candidates = list(DATASET.glob(f'{filter_string}'))
    if len(candidates) == 1:
        print('Found file:', candidates[0])
        return candidates[0]
    else:
        raise ValueError(f'Found {len(candidates)} candidates.'
                         'Please make selection more specific!')


def do_gdal_calls(DATASET):
    maskfile = DATASET / f'{DATASET.name}_mask.tif'
    tcvisfile = DATASET / 'tcvis.tif'

    tile_dir_data = DATASET / 'tiles' / 'data'
    tile_dir_tcvis = DATASET / 'tiles' / 'tcvis'
    tile_dir_mask = DATASET / 'tiles' / 'mask'

    # Create parents on the first data folder
    tile_dir_data.mkdir(exist_ok=True, parents=True)
    tile_dir_tcvis.mkdir(exist_ok=True)
    tile_dir_mask.mkdir(exist_ok=True)

    rasterfile = glob_file(DATASET, RASTERFILTER)

    # Retile data, mask and tcvis
    os.system(f'python {gdal_retile} -ps {XSIZE} {YSIZE} -overlap {OVERLAP} -targetDir {tile_dir_data} {rasterfile}')
    os.system(f'python {gdal_retile} -ps {XSIZE} {YSIZE} -overlap {OVERLAP} -targetDir {tile_dir_mask} {maskfile}')
    os.system(f'python {gdal_retile} -ps {XSIZE} {YSIZE} -overlap {OVERLAP} -targetDir {tile_dir_tcvis} {tcvisfile}')


def make_info_picture(imgtensor, masktensor, filename):
    "Make overview picture"
    rgbn = np.clip(imgtensor[:4,:,:].numpy().transpose(1, 2, 0) / 3000 * 255, 0, 255).astype(np.uint8)
    tcvis = np.clip(imgtensor[4:,:,:].numpy().transpose(1, 2, 0), 0, 255).astype(np.uint8)

    rgb = rgbn[:,:,:3]
    nir = rgbn[:,:,[3,3,3]]
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

    if not args['--skip_gdal']:
        gdal_path = args['--gdal_path']
        gdal_bin = args['--gdal_bin']
        gdal_merge = os.path.join(gdal_path, 'gdal_merge.py')
        gdal_retile = os.path.join(gdal_path, 'gdal_retile.py')
        gdal_rasterize = os.path.join(gdal_bin, 'gdal_rasterize')
        gdal_translate = os.path.join(gdal_bin, 'gdal_translate')

    DATA = Path('data')

    old_data_dirs = []
    for dest_dir in ('tiles_train', 'tiles_test', 'tiles_val', 'tiles_train_slump'):
        check_dir = DATA / dest_dir
        if check_dir.exists():
            old_data_dirs.append(check_dir)
    if old_data_dirs:
        print(f"Found old data directories: {', '.join(dir.name for dir in old_data_dirs)}.")
        decision = input("Delete? [y/N]")
        if decision.lower() == 'y':
            for old_dir in old_data_dirs:
                shutil.rmtree(old_dir)
        else:
            print("Won't overwrite old data directories.")
            sys.exit(1)

    # Train-val-test split
    setnames = list(sorted(DATA.glob('*')))
    setnames = [os.path.basename(x) for x in setnames if os.path.isdir(x)]
    setnames = [x for x in setnames if x not in ('tiles_train', 'tiles_test', 'tiles_val')]

    val_set = ['20190714_194400_1035']
    test_set = ['20190803_164436_1048']
    train_set = [t for t in setnames if t not in val_set + test_set]

    sets = dict(train=train_set, val=val_set, test=test_set)

    train_slump_dir = DATA / 'tiles_train_slump'
    train_slump_dir.mkdir(exist_ok=False)
    slump_data_dir = train_slump_dir / 'data'
    slump_data_dir.mkdir()
    slump_mask_dir = train_slump_dir / 'mask'
    slump_mask_dir.mkdir()
    slump_info_dir = train_slump_dir / 'info'
    slump_info_dir.mkdir()

    THRESHOLD = float(args['--nodata_threshold']) / 100

    for setname, dataset in sets.items():
        dest = DATA / f'tiles_{setname}'
        dest.mkdir(exist_ok=False)

        data_dir = dest / 'data'
        data_dir.mkdir()
        mask_dir = dest / 'mask'
        mask_dir.mkdir()
        info_dir = dest / 'info'
        info_dir.mkdir()

        for subset in dataset:
            print(f'Doing {subset}')
            DATASET = DATA / subset
            if not args['--skip_gdal']:
                do_gdal_calls(DATASET)

            # Convert data to pytorch tensor files (pt) for efficient data loading
            for img in tqdm(list(DATASET.glob('tiles/data/*.tif'))):
                mask, tcvis = others_from_img(img)

                with rio.open(img) as raster:
                    imgdata = raster.read()
                    # Skip nodata tiles
                    if (imgdata == 0).all(axis=0).mean() > THRESHOLD:
                        continue
                    # Assert data can safely be coerced to int16
                    assert imgdata.max() < 2 ** 15

                with rio.open(tcvis) as raster:
                    # Throw away alpha channel
                    tcdata = raster.read()[:3] 
                    # Assert data can safely be coerced to int16
                    assert tcdata.max() < 2 ** 15

                full_data = np.concatenate([imgdata, tcdata], axis=0)
                imgtensor = torch.from_numpy(full_data.astype(np.int16))

                with rio.open(mask) as raster:
                    maskdata = raster.read()
                    assert maskdata.max() <= 1, "Mask can't contain values > 1"
                    masktensor = torch.from_numpy(maskdata.astype(np.bool))

                # gdal_retile leaves narrow stripes at the right and bottom border,
                # which are filtered out here:
                if imgtensor.shape != (7, 256, 256):
                    continue
                if masktensor.shape != (1, 256, 256):
                    continue

                # Write the tensor files
                filename = img.stem + '.pt'
                torch.save(imgtensor, data_dir / filename)
                torch.save(masktensor, mask_dir / filename)
                make_info_picture(imgtensor, masktensor,
                        (info_dir / filename).with_suffix('.jpg'))

                if setname == 'train' and masktensor.max() > 0:
                    torch.save(imgtensor, slump_data_dir / filename)
                    torch.save(masktensor, slump_mask_dir / filename)
                    make_info_picture(imgtensor, masktensor,
                            (slump_info_dir / filename).with_suffix('.jpg'))
