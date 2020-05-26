#!/usr/bin/env python
# flake8: noqa: E501
from pathlib import Path
from tqdm import tqdm
import rasterio as rio
import numpy as np
import torch
import os

# Tiling Settings
XSIZE = 256
YSIZE = 256
OVERLAP = 25

# Paths setup
RASTERFILTER = '*3B_AnalyticMS_SR.tif'
VECTORFILTER = '*.shp'

gdal_path = '/usr/bin'
gdal_bin = '/usr/bin'
gdal_merge = os.path.join(gdal_path, 'gdal_merge.py')
gdal_retile = os.path.join(gdal_path, 'gdal_retile.py')
gdal_rasterize = os.path.join(gdal_bin, 'gdal_rasterize')
gdal_translate = os.path.join(gdal_bin, 'gdal_translate')

DATA = Path('data')
datasets = list(sorted(DATA.glob('*/tiles')))

# Train-val-test split
setnames = list(map(lambda x: x.parent.name, datasets))
val_set = ['20190727_160426_104e']
test_set = ['20190709_042959_08_1057']
train_set = [t for t in setnames if t not in val_set + test_set]

sets = dict(train=train_set, val=val_set, test=test_set)


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
    maskfile = DATASET / 'mask.tif'
    maskfile2 = DATASET / f'{DATASET.name}_mask.tif'
    tcvisfile = DATASET / 'tcvis.tif'

    tile_dir_data = DATASET / 'tiles' / 'data'
    tile_dir_tcvis = DATASET / 'tiles' / 'tcvis'
    tile_dir_mask = DATASET / 'tiles' / 'mask'

    # Create parents on the first data folder
    tile_dir_data.mkdir(exist_ok=True, parents=True)
    tile_dir_tcvis.mkdir(exist_ok=True)
    tile_dir_mask.mkdir(exist_ok=True)

    rasterfile = glob_file(DATASET, RASTERFILTER)
    vectorfile = glob_file(DATASET, VECTORFILTER)

    # Create temporary raster maskfile
    os.system(f'python {gdal_merge} -createonly -init 0 -o {maskfile} -ot Byte -co COMPRESS=DEFLATE {rasterfile}')
    # Add empty band to mask
    os.system(f'{gdal_translate} -of GTiff -ot Byte -co COMPRESS=DEFLATE -b 1 {maskfile} {maskfile2}')
    # Burn digitized polygons into mask
    os.system(f'{gdal_rasterize} -l {DATASET.name} -a label {vectorfile} {maskfile2}')
    # Retile data, mask and tcvis
    os.system(f'python {gdal_retile} -ps {XSIZE} {YSIZE} -overlap {OVERLAP} -targetDir {tile_dir_data} {rasterfile}')
    os.system(f'python {gdal_retile} -ps {XSIZE} {YSIZE} -overlap {OVERLAP} -targetDir {tile_dir_mask} {maskfile2}')
    os.system(f'python {gdal_retile} -ps {XSIZE} {YSIZE} -overlap {OVERLAP} -targetDir {tile_dir_tcvis} {tcvisfile}')


for setname, dataset in sets.items():
    dest = DATA / f'tiles_{setname}'
    dest.mkdir(exist_ok=False)

    data_dir = dest / 'data'
    data_dir.mkdir()
    mask_dir = dest / 'mask'
    mask_dir.mkdir()

    i = 0
    for subset in dataset:
        print(f'Doing {subset}')
        DATASET = DATA / subset
        do_gdal_calls(DATASET)

        # Convert data to pytorch tensor files (pt) for efficient data loading
        # Sort for reproducibility
        for img in tqdm(list(sorted(DATASET.glob('tiles/data/*.tif')))):
            mask, tcvis = others_from_img(img)

            with rio.open(img) as raster:
                imgdata = raster.read()
                # Skip nodata tiles
                if imgdata.max() == 0:
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
            i += 1

    # Optional Compression for quicker uploading
    # os.system(f'cd data && tar -cJf tiles_{setname}.tar.xz tiles_{setname}')
