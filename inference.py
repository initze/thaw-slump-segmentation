"""
Usecase 2 Inference Script

Usage:
    inference.py [options] <model_path> <tile_to_predict> ... 

Options:
    -h --help          Show this screen
    --ckpt=<CKPT>      Checkpoint to use [default: latest]
    --gdal_path=PATH        Path to gdal scripts (only needed if tile isn't already preprocessed) [default: ]
    --gdal_bin=PATH         Path to gdal binaries (only needed if tile isn't already preprocessed) [default: ]
"""
from pathlib import Path
import shutil

import rasterio as rio
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from rasterio.transform import from_gcps

from deep_learning.utils.images import extract_patches
from deep_learning.models import get_model
from deep_learning.utils.plot_info import flatui_cmap
from data_loading import get_loader

from setup_raw_data import preprocess_directory
from prepare_data import read_and_assert_imagedata
from data_loading import make_transform, DATA_SOURCES, get_sources

import os

from docopt import docopt
import yaml

cmap_dem = flatui_cmap('Alizarin', 'Clouds', 'Peter River')
cmap_slope = flatui_cmap('Clouds', 'Midnight Blue')
cmap_ndvi = 'RdYlGn'

DEM = '/hdd/AntarcticDEM/Antarctica4326.tif'

FIGSIZE_MAX = 20
PATCHSIZE = 2048
PS_DEM = PATCHSIZE // 16


def predict(imagery):
    prediction = torch.zeros(1, *imagery.shape[2:])
    num_preds  = torch.zeros(1, *imagery.shape[2:])

    for y in tqdm(np.arange(0, imagery.shape[2], PATCHSIZE // 2)):
        for x in np.arange(0, imagery.shape[3], PATCHSIZE // 2):
            if y + PATCHSIZE > imagery.shape[2]:
                y = imagery.shape[2] - PATCHSIZE
            if x + PATCHSIZE > imagery.shape[3]:
                x = imagery.shape[3] - PATCHSIZE
            patch_imagery = imagery[:, :, y:y+PATCHSIZE, x:x+PATCHSIZE]
            if patch_imagery[0, 0].max() == 0:
                continue
            patch_imagery = patch_imagery.to(dev)
            patch_pred = model(patch_imagery).cpu()
            prediction[:, y:y+PATCHSIZE, x:x+PATCHSIZE] += patch_pred
            num_preds[:, y:y+PATCHSIZE, x:x+PATCHSIZE] += 1
    final_pred = prediction / torch.clamp_min(num_preds, 1)
    return final_pred


def do_inference(tilename):
    # ===== PREPARE THE DATA =====
    data_directory = Path('data') / tilename
    if not data_directory.exists():
        print('Preprocessing directory')
        raw_directory = Path('data_input') / tilename
        if not raw_directory.exists():
            print(f"Couldn't find tile '{tilename}' in data/ or data_input/. Skipping this tile")
            return
        preprocess_directory(raw_directory, gdal_bin=args['--gdal_bin'], gdal_path=args['--gdal_path'])
        # After this, data_directory should contain all the stuff that we need.
    output_directory = Path('inference') / tilename
    output_directory.mkdir(exist_ok=True)

    planet_imagery_path = next(data_directory.glob('*_AnalyticMS_SR.tif'))
    # TODO: Enable subsetting of the channels from the config file

    data = []
    for source in sources:
        print(f'loading {source.name}')
        if source.name == 'planet':
            tif_path = planet_imagery_path
        else:
            tif_path = data_directory / f'{source.name}.tif'

        data_part = rio.open(tif_path).read().astype(np.float32)

        if source.name == 'tcvis':
            if data_part.shape[0] == 3:
                print('TCVIS alpha channel was fixed, you can delete this if-block now')
            data_part = data_part[:3]  # TODO: make this redundant!

        data_part = data_part / np.array(source.normalization_factors, dtype=np.float32).reshape(-1, 1, 1)
        data.append(data_part)

    name_to_source = {src.name: src for src in sources}
    def make_img(filename, source, colorbar=False, **kwargs):
        idx = sources.index(source)

        h, w = data[idx].shape[1:]
        if h > w:
            figsize = (FIGSIZE_MAX * w / h, FIGSIZE_MAX)
        else:
            figsize = (FIGSIZE_MAX, FIGSIZE_MAX * h / w)
        fig, ax = plt.subplots(figsize=figsize)
        ax.axis('off')

        if source.channels >= 3:
            rgb = np.stack([data[idx][i, ::10, ::10] for i in range(3)], axis=-1)
            rgb = np.clip(rgb, 0, 1)
            ax.imshow(rgb, aspect='equal')
        elif source.channels == 1:
            image = ax.imshow(data[idx][0], aspect='equal', **kwargs)
            if colorbar:
                plt.colorbar(image)
        plt.savefig(output_directory / filename, bbox_inches='tight', pad_inches=0)
        plt.close()

    for src in sources:
        kwargs = dict()
        if src.name == 'ndvi':
            kwargs = dict(colorbar=True, cmap=cmap_ndvi, vmin=0, vmax=1)
        elif src.name == 'relative_elevation':
            kwargs = dict(colorbar=True, cmap=cmap_dem, vmin=-0.01, vmax=0.01)
        elif src.name == 'slope':
            kwargs = dict(colorbar=True, cmap=cmap_slope, vmin=0, vmax=0.5)
        make_img(f'{src.name}.jpg', src, **kwargs)

    full_data = torch.from_numpy(np.concatenate(data, axis=0))
    full_data = full_data.unsqueeze(0)  # Pretend this is a batch of size 1

    res = predict(full_data)
    del full_data

    torch.save(res, safe_path / 'prediction.pt')

    # TODO: make Geo-Tiffs for Probability Map and Binarized Predictions

    # TODO: vvv
    fig, ax = plt.subplots(figsize=figsize)
    ax.axis('off')
    ax.imshow(res[0], vmin=-1, vmax=1, cmap=flatui, aspect='equal')
    plt.savefig(safe_path / 'inference_seg.jpg', bbox_inches='tight', pad_inches=0)
    plt.close()


if __name__ == "__main__":
    args = docopt(__doc__, version="Usecase 3 Inference Script 1.0")

    # ===== LOAD THE MODEL =====
    cuda = True if torch.cuda.is_available() else False
    dev = torch.device("cpu") if not cuda else torch.device("cuda")
    print(f'Running on {dev} device')

    if not args['<model_path>']:
        last_modified = 0
        last_modeldir = None

        for config_file in Path('logs').glob('*/config.yml'):
            modified = config_file.stat().st_mtime
            if modified > last_modified:
                last_modified = modified
                last_modeldir = config_file.parent
        args['<model_path>'] = last_modeldir

    model_dir = Path(args['<model_path>'])
    model_name = model_dir.name
    config = yaml.load((model_dir / 'config.yml').open(), Loader=yaml.SafeLoader)

    modelclass = get_model(config['model'])
    model = modelclass(**config['model_args'])

    if args['--ckpt'] == 'latest':
        ckpt_nums = [int(ckpt.stem) for ckpt in model_dir.glob('checkpoints/*.pt')]
        last_ckpt = max(ckpt_nums)
    else:
        last_ckpt = int(args['--ckpt'])
    ckpt = model_dir / 'checkpoints' / f'{last_ckpt:02d}.pt'
    print(f"Loading checkpoint {ckpt}")
    model.load_state_dict(torch.load(ckpt, map_location=dev))
    model = model.to(dev)

    sources = get_sources(config['data_sources'])

    for tilename in args['<tile_to_predict>']:
        do_inference(tilename)
