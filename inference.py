#!/usr/bin/env python
# flake8: noqa: E501
# Copyright (c) Ingmar Nitze and Konrad Heidler

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
Usecase 2 Inference Script
"""

import argparse
from pathlib import Path

import rasterio as rio
import rioxarray
import numpy as np
import matplotlib.pyplot as plt
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from datetime import datetime
import geopandas as gpd

from lib.models import create_model
from lib.utils import init_logging, get_logger, extract_contours, Compositor
from lib.data.loading import NCDataset
import cv2

import yaml

cmap_ndvi = 'RdYlGn'

FIGSIZE_MAX = 20

def predict(model, imagery, device='cpu'):
    prediction = torch.zeros(1, *imagery.shape[2:])
    weights = torch.zeros(1, *imagery.shape[2:])

    PS = args.patch_size
    MARGIN = args.margin_size

    margin_ramp = torch.cat([
        torch.linspace(0, 1, MARGIN),
        torch.ones(PS - 2 * MARGIN),
        torch.linspace(1, 0, MARGIN),
    ])

    soft_margin = margin_ramp.reshape(1, 1, PS) * \
                  margin_ramp.reshape(1, PS, 1)

    for y in np.arange(0, imagery.shape[2], (PS - MARGIN)):
        for x in np.arange(0, imagery.shape[3], (PS - MARGIN)):
            if y + PS > imagery.shape[2]:
                y = imagery.shape[2] - PS
            if x + PS > imagery.shape[3]:
                x = imagery.shape[3] - PS
            patch_imagery = imagery[:, :, y:y + PS, x:x + PS]
            patch_pred = torch.sigmoid(model(patch_imagery.to(device))[0].cpu())

            # Essentially premultiplied alpha blending
            prediction[:, y:y + PS, x:x + PS] += patch_pred * soft_margin
            weights[:, y:y + PS, x:x + PS] += soft_margin

    # Avoid division by zero
    weights = torch.where(weights == 0, torch.ones_like(weights), weights)
    return prediction / weights


def flush_rio(filepath):
    """For some reason, rasterio doesn't actually finish writing
    a file after finishing a `with rio.open(...) as ...:` block
    Trying to open the file for reading seems to force a flush"""

    with rio.open(filepath) as f:
        pass


def do_inference(tile_path, model, args, config, dev, log_path=None):
    model = model.to(dev)
    tile_logger = get_logger(f'inference.{tilename}')
    # ===== PREPARE THE DATA =====
    DATA_ROOT = args.data_dir
    INFERENCE_ROOT = args.inference_dir
    data_directory = DATA_ROOT / 'tiles' / tilename

    if args.name:
        output_directory = INFERENCE_ROOT / args.name / tilename
    else:
        output_directory = INFERENCE_ROOT / tilename

    output_directory.mkdir(exist_ok=True, parents=True)

    # file check, check for suffix and if exists
    file_path = (DATA_ROOT / f'{tile_path}.nc').as_posix()
    # TODO: Overlap mode
    # set config automatically to deterministic
    config['sampling_mode'] = 'deterministic'
    data = NCDataset(file_path, config, inference=True)
    loader = DataLoader(data, batch_size=config['batch_size'], num_workers=1, pin_memory=True)

    result = Compositor()
    nodata = Compositor()
    for batch in loader:
      #img = batch[0].to(dev)
      img = torch.tensor(batch[0]).to(torch.float32).to(dev)
      nds = (batch[0].numpy() == 0).all(axis=1)
      metadata = batch[-1]
      preds = torch.sigmoid(model(img))[:, 0]
      preds = preds.cpu().numpy()
      for pred, nd, y, x in zip(preds, nds, metadata['y0'], metadata['x0']):
        result.add_tile((y, x), pred)
        nodata.add_tile((y, x), nd)

    result = result.compose()
    nodata = nodata.compose() > 0.5

    input_nc = rioxarray.open_rasterio(file_path)
    profile = {
        'driver': 'COG',
        'dtype': np.float32,
        'width': result.shape[1],
        'height': result.shape[0],
        'count': 1,
        'transform': input_nc.rio.transform(),
        'crs': input_nc.rio.crs,
        'compress': 'LZW',
    }

    print(profile)
    result[nodata] = np.nan
    with rio.open(output_directory / 'pred_probability.tif', 'w', **profile) as out:
      out.write(result, 1)

    contours = extract_contours(result, 0.5)
    contours = gpd.GeoSeries(contours, crs=input_nc.rio.crs)

    tx = input_nc.rio.transform()
    mat = [tx.a, tx.b, tx.d, tx.e, tx.c, tx.f]
    contours = contours.buffer(0)
    contours = contours.affine_transform(mat)
    contours.to_file(output_directory / 'predictions.gpkg', driver='GPKG')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_jobs", default=-1, type=int, help="number of parallel joblib jobs")
    parser.add_argument("--ckpt", default='latest', type=str, help="Checkpoint to use")
    parser.add_argument("--data_dir", default='data', type=Path, help="Path to data processing dir")
    parser.add_argument("--log_dir", default='logs', type=Path, help="Path to log dir")
    parser.add_argument("--inference_dir", default='inference', type=Path, help="Main inference directory")
    parser.add_argument("-n", "--name", default=None, type=str, help="Name of inference run, data will be stored in subdirectory")
    parser.add_argument("-m", "--margin_size", default=256, type=int, help="Size of patch overlap")
    parser.add_argument("-p", "--patch_size", default=1024, type=int, help="Size of patches")
    parser.add_argument("model_path", type=str, help="path to model")
    parser.add_argument("tile_to_predict", type=str, help="path to model", nargs='+')

    args = parser.parse_args()

    timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    log_path = Path(args.log_dir) / f'inference-{timestamp}.log'
    if not Path(args.log_dir).exists():
        os.mkdir(Path(args.log_dir))
    init_logging(log_path)
    logger = get_logger('inference')

    # ===== LOAD THE MODEL =====
    cuda = True if torch.cuda.is_available() else False
    dev = torch.device("cpu") if not cuda else torch.device("cuda")
    logger.info(f'Running on {dev} device')

    if not args.model_path:
        last_modified = 0
        last_modeldir = None

        for config_file in Path(args.log_dir).glob('*/config.yml'):
            modified = config_file.stat().st_mtime
            if modified > last_modified:
                last_modified = modified
                last_modeldir = config_file.parent
        args.model_path = last_modeldir

    model_dir = Path(args.model_path)
    model_name = model_dir.name
    config = yaml.load((model_dir / 'config.yml').open(), Loader=yaml.Loader)

    m = config['model']
    model = create_model(
        arch=m['architecture'],
        encoder_name=m['encoder'],
        encoder_weights=None if m['encoder_weights'] == 'random' else m['encoder_weights'],
        classes=1,
        in_channels=m['input_channels']
    )

    if args.ckpt == 'latest':
        ckpt_nums = [int(ckpt.stem) for ckpt in model_dir.glob('checkpoints/*.pt')]
        last_ckpt = max(ckpt_nums)
    else:
        last_ckpt = int(args.ckpt)
    ckpt = model_dir / 'checkpoints' / f'{last_ckpt:02d}.pt'
    logger.info(f"Loading checkpoint {ckpt}")

    # Parallelized Model needs to be declared before loading
    try:
        model.load_state_dict(torch.load(ckpt, map_location=dev))
    except:
        model = nn.DataParallel(model)
        model.load_state_dict(torch.load(ckpt, map_location=dev))

    torch.set_grad_enabled(False)

    for tilename in tqdm(args.tile_to_predict):
        do_inference(tilename, model, args, config, dev, log_path)
