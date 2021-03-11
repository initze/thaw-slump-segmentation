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

import rasterio as rio
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from tqdm import tqdm

from deep_learning.models import create_model, create_loss
from deep_learning.utils.plot_info import flatui_cmap

from setup_raw_data import preprocess_directory
from data_loading import DataSources

from docopt import docopt
import yaml

cmap_prob = flatui_cmap('Midnight Blue', 'Alizarin')
cmap_dem = flatui_cmap('Alizarin', 'Clouds', 'Peter River')
cmap_slope = flatui_cmap('Clouds', 'Midnight Blue')
cmap_ndvi = 'RdYlGn'

FIGSIZE_MAX = 20
PATCHSIZE = 1024
MARGIN = 256 # Margin

def predict(model, imagery, device='cpu'):
    prediction = torch.zeros(1, *imagery.shape[2:])
    weights     = torch.zeros(1, *imagery.shape[2:])

    PS = PATCHSIZE

    for y in np.arange(0, imagery.shape[2], (PS - MARGIN)):
        for x in np.arange(0, imagery.shape[3], (PS - MARGIN)):
            if y + PS > imagery.shape[2]:
                y = imagery.shape[2] - PS
            if x + PS > imagery.shape[3]:
                x = imagery.shape[3] - PS
            patch_imagery = imagery[:, :, y:y+PS, x:x+PS]
            if patch_imagery[0, 0].max() == 0:
                continue
            patch_pred  = torch.sigmoid(model(patch_imagery.to(device))[0].cpu())
            margin_ramp = torch.cat([
                torch.linspace(0, 1, MARGIN),
                torch.ones(PS - 2 * MARGIN),
                torch.linspace(1, 0, MARGIN),
            ])

            soft_margin = margin_ramp.reshape(1, 1, PS) * \
                          margin_ramp.reshape(1, PS, 1)

            # Essentially premultiplied alpha blending
            prediction[:, y:y+PS, x:x+PS] += patch_pred * soft_margin
            weights[   :, y:y+PS, x:x+PS]  += soft_margin

    # Avoid division by zero
    weights = torch.where(weights == 0, torch.ones_like(weights), weights)
    return prediction / weights


def do_inference(tilename):
    # ===== PREPARE THE DATA =====
    data_directory = Path('data') / tilename
    if not data_directory.exists():
        print('Preprocessing directory')
        raw_directory = Path('data_input') / tilename
        if not raw_directory.exists():
            print(f"Couldn't find tile '{tilename}' in data/ or data_input/. Skipping this tile")
            return
        preprocess_directory(raw_directory, gdal_bin=args['--gdal_bin'],
                             gdal_path=args['--gdal_path'], label_required=False)
        # After this, data_directory should contain all the stuff that we need.
    output_directory = Path('inference') / tilename
    output_directory.mkdir(exist_ok=True)

    planet_imagery_path = next(data_directory.glob('*_AnalyticMS_SR.tif'))

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

    full_data = np.concatenate(data, axis=0)
    nodata = np.all(full_data == 0, axis=0, keepdims=True)
    full_data = torch.from_numpy(full_data)
    full_data = full_data.unsqueeze(0)  # Pretend this is a batch of size 1

    res = predict(model, full_data, dev).numpy()
    del full_data

    res[nodata] = np.nan

    with rio.open(planet_imagery_path) as input_raster:
        profile = input_raster.profile
        profile.update(
            dtype=rio.float32,
            count=1,
            compress='lzw'
        )
        out_path = output_directory / 'pred_probability.tif'
        with rio.open(out_path, 'w', **profile) as output_raster:
            output_raster.write(res.astype(np.float32))

        out_path = output_directory / 'pred_binarized.tif'
        profile.update(
            dtype=rio.uint8,
            nodata=255
        )
        with rio.open(out_path, 'w', **profile) as output_raster:
            binarized = (res > 0.5).astype(np.uint8)
            binarized[nodata] = 255
            output_raster.write(binarized)

    h, w = res.shape[1:]
    if h > w:
        figsize = (FIGSIZE_MAX * w / h, FIGSIZE_MAX)
    else:
        figsize = (FIGSIZE_MAX, FIGSIZE_MAX * h / w)

    fig, ax = plt.subplots(figsize=figsize)
    ax.axis('off')
    mappable = ax.imshow(res[0], vmin=0, vmax=1, cmap=cmap_prob, aspect='equal')
    plt.colorbar(mappable)
    plt.savefig(output_directory / 'pred_probability.jpg', bbox_inches='tight', pad_inches=0)
    plt.close()

    fig, ax = plt.subplots(figsize=figsize)
    ax.axis('off')
    mappable = ax.imshow(res[0] > 0.5, vmin=0, vmax=1, cmap=cmap_prob, aspect='equal')
    plt.colorbar(mappable)
    plt.savefig(output_directory / 'pred_binarized.jpg', bbox_inches='tight', pad_inches=0)
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

    m = config['model']
    model = create_model(
        arch=m['architecture'],
        encoder_name=m['encoder'],
        encoder_weights=None if m['encoder_weights'] == 'random' else m['encoder_weights'],
        classes=1,
        in_channels=m['input_channels']
    )

    if args['--ckpt'] == 'latest':
        ckpt_nums = [int(ckpt.stem) for ckpt in model_dir.glob('checkpoints/*.pt')]
        last_ckpt = max(ckpt_nums)
    else:
        last_ckpt = int(args['--ckpt'])
    ckpt = model_dir / 'checkpoints' / f'{last_ckpt:02d}.pt'
    print(f"Loading checkpoint {ckpt}")
    model.load_state_dict(torch.load(ckpt, map_location=dev))
    model = model.to(dev)

    sources = DataSources(config['data_sources'])

    torch.set_grad_enabled(False)
    for tilename in args['<tile_to_predict>']:
        do_inference(tilename)
