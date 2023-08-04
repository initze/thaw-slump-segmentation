# Copyright (c) Ingmar Nitze and Konrad Heidler

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import xarray
import torch
import numpy as np
from torch.utils.data import DataLoader, ConcatDataset, Dataset
from math import ceil
from einops import rearrange
from tqdm import tqdm
from pathlib import Path
from skimage.measure import find_contours
from .base import _LAYER_REGISTRY


class NCDataset(Dataset):
  def __init__(self, netcdf_path, config):
    self.netcdf_path = netcdf_path
    self.tile_size = config['tile_size']
    self.data_sources = config['data_sources']
    self.data = xarray.open_dataset(netcdf_path, cache=False)
    self.sampling_mode = config['sampling_mode']

    self.H, self.W = len(self.data.y), len(self.data.x)
    self.H_tile = self.H // self.tile_size
    self.W_tile = self.W // self.tile_size

    if self.sampling_mode == 'targets_only':
      targets = (self.data.Mask == 1).squeeze('mask_band').values
      # Find Bounding boxes of targets
      contours = find_contours(targets)

      self.bboxes = []
      for contour in contours:
        ymin, xmin = np.floor(contour.min(axis=0)).astype(int)
        ymax, xmax = np.ceil(contour.max(axis=0)).astype(int)
        self.bboxes.append([ymin, xmin, ymax, xmax])

  def __getitem__(self, idx):
    if self.sampling_mode == 'deterministic':
      y_tile, x_tile = divmod(idx, self.W_tile)
      y0 = y_tile * self.tile_size
      x0 = x_tile * self.tile_size
    elif self.sampling_mode == 'random':
      y0 = int(torch.randint(0, self.H - self.tile_size, ()))
      x0 = int(torch.randint(0, self.W - self.tile_size, ()))
    elif self.sampling_mode == 'targets_only':
      bbox_idx = int(torch.randint(0, len(self.bboxes), ()))
      ymin, xmin, ymax, xmax = self.bboxes[bbox_idx]

      y_start = max(0, ymin - self.tile_size)
      y_end   = min(self.H - self.tile_size, ymax)

      x_start = max(0, xmin - self.tile_size)
      x_end   = min(self.W - self.tile_size, xmax )

      if y_start >= y_end or x_start >= x_end:
        print("Nasty BBox!")
        print(f'y range: {ymin} -- {ymax}')
        print(f'x range: {xmin} -- {xmax}')
        print('Derived:')
        print(f'Sample y from [{y_start}, {y_end})')
        print(f'Sample x from [{x_start}, {x_end})')
        print(f'Image size: {self.H} x {self.W}')

      y0 = int(torch.randint(y_start, y_end, ()))
      x0 = int(torch.randint(x_start, x_end, ()))
    else:
      raise ValueError(f'Unsupported tiling mode: {self.sampling_mode!r}')
    y1 = y0 + self.tile_size
    x1 = x0 + self.tile_size

    metadata = {
      'source_file': self.netcdf_path,
      'y0': y0, 'x0': x0,
      'y1': y1, 'x1': x1,
    }
    tile = {k: self.data[k][:, y0:y1, x0:x1].fillna(0).values for k in self.data_sources}
    tile = {k: _LAYER_REGISTRY[k].normalize(v) for k, v in tile.items()}
    
    if 'Mask' in tile:
      return (
        np.concatenate([tile[k] for k in tile if k != 'Mask'], axis=0),
        tile['Mask'],
        metadata
      )
    else:
      return (
          np.concatenate([tile[k] for k in tile if k != 'Mask'], axis=0),
          metadata
      )

  def __len__(self):
    return self.H_tile * self.W_tile


def get_loader(config):
  root = config['data_root']
  scene_names = config['scenes']
  scenes = [NCDataset(f'{root}/{scene}.nc', config) for scene in scene_names]
  all_data = ConcatDataset(scenes)

  return DataLoader(
    all_data,
    shuffle = (config['sampling_mode'] != 'deterministic'),
    batch_size=config['batch_size'],
    num_workers=config['num_workers'],
    persistent_workers=True,
    pin_memory=True
  )


if __name__ == '__main__':
  import yaml
  from munch import munchify
  import argparse

  parser = argparse.ArgumentParser()
  parser.add_argument("--data_dir", default='data', type=Path, help="Path to data processing dir")
  parser.add_argument('-c', '--config', default='config.yml', type=Path,
                      help='Specify run config to use.')
  args = parser.parse_args()

  config = munchify(yaml.safe_load(args.config.open()))

  ds_config = config.datasets.val
  ds_config.batch_size = 16
  ds_config.num_workers = config.data_threads
  ds_config.data_sources = config.data_sources
  ds_config.data_root = args.data_dir
  loader = get_loader(ds_config)

  print('Starting Loop')
  for i, (img, mask, metadata) in enumerate(tqdm(loader)):
    print(f'Iteration {i}')
    for l in metadata['source_file']:
      print(' ', l)
