#!/usr/bin/env python
# flake8: noqa: E501
# Copyright (c) Ingmar Nitze and Konrad Heidler

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
Usecase 2 Training Script
"""
import argparse
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from collections import defaultdict
import wandb

import numpy as np
import torch
import torch.nn as nn
import yaml
from tqdm import tqdm
from PIL import Image, ImageDraw
from skimage.measure import find_contours
from einops import rearrange

from lib import Metrics, Accuracy, Precision, Recall, F1, IoU
from lib.models import create_model, create_loss
from lib.data.loading import get_loader
from lib.utils import showexample, plot_metrics, plot_precision_recall, init_logging, get_logger, yaml_custom

parser = argparse.ArgumentParser()
parser.add_argument('-s', '--summary', action='store_true',
                    help='Only print model summary and return.')
parser.add_argument("--data_dir", default='data', type=Path, help="Path to data processing dir")
parser.add_argument("--log_dir", default='logs', type=Path, help="Path to log dir")
parser.add_argument('-n', '--name', default='',
                    help='Give this run a name, so that it will be logged into logs/<NAME>_<timestamp>.')
parser.add_argument('-c', '--config', default='config.yml', type=Path,
                    help='Specify run config to use.')
parser.add_argument('-r', '--resume', default='',
                    help='Resume from the specified checkpoint.'
                         'Can be either a run-id (e.g. "2020-06-29_18-12-03") to select the last'
                         'checkpoint of that run, or a direct path to a checkpoint to be loaded.'
                         'Overrides the resume option in the config file if given.'
                    )
parser.add_argument('-wp', '--wandb_project', default='RTS Sentinel2',
                    help='Set a project name for weights and biases')
parser.add_argument('-wn', '--wandb_name', default=None,
                    help='Set a run name for weights and biases')


class Engine:
  def __init__(self):
      args = parser.parse_args()
      self.config = yaml.load(args.config.open(), Loader=yaml_custom.SaneYAMLLoader)
      self.DATA_ROOT = args.data_dir
      # Logging setup
      timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
      if args.name:
          log_dir_name = f'{args.name}_{timestamp}'
      else:
          log_dir_name = timestamp
      self.log_dir = Path(args.log_dir) / log_dir_name
      self.log_dir.mkdir(exist_ok=False)

      init_logging(self.log_dir / 'train.log')
      self.logger = get_logger('train')

      self.data_sources = self.config['data_sources']
      if 'Mask' not in self.data_sources:
        self.data_sources += ['Mask']
      self.dataset_cache = {}

      # Sanity check: No scene should be in train AND val at the same time
      train_scenes = set(self.config['datasets']['train']['scenes'])
      val_scenes = set(self.config['datasets']['val']['scenes'])
      intersection = train_scenes & val_scenes
      if intersection:
        self.logger.warn(f'The following scenes are in train and val: {intersection}')

      for (img, mask, _) in self.get_dataloader('val'):
        self.config['model']['input_channels'] = img.shape[1]
        break

      m = self.config['model']
      self.model = create_model(
          arch=m['architecture'],
          encoder_name=m['encoder'],
          encoder_weights=None if m['encoder_weights'] == 'random' else m['encoder_weights'],
          classes=1,
          in_channels=m['input_channels']
      )

      # make parallel
      self.model = nn.DataParallel(self.model)

      if args.resume:
          self.config['resume'] = args.resume

      if 'resume' in self.config and self.config['resume']:
          checkpoint = Path(self.config['resume'])
          if not checkpoint.exists():
              raise ValueError(f"There is no Checkpoint at {self.config['resume']} to resume from!")
          if checkpoint.is_dir():
              # Load last checkpoint in run dir
              ckpt_nums = [int(ckpt.stem) for ckpt in checkpoint.glob('checkpoints/*.pt')]
              last_ckpt = max(ckpt_nums)
              self.config['resume'] = str(checkpoint / 'checkpoints' / f'{last_ckpt:02d}.pt')
          self.logger.info(f"Resuming training from checkpoint {self.config['resume']}")
          self.model.load_state_dict(torch.load(self.config['resume']))

      self.dev = torch.device("cpu") if not torch.cuda.is_available() else torch.device("cuda")
      self.logger.info(f'Training on {self.dev} device')

      self.model = self.model.to(self.dev)

      self.learning_rate = self.config['learning_rate']
      self.opt = torch.optim.AdamW(self.model.parameters(), lr=self.learning_rate)
      self.setup_lr_scheduler()

      self.epoch = 0
      self.metrics = Metrics(Accuracy, Precision, Recall, F1, IoU)

      if args.summary:
          from torchinfo import summary
          summary(self.model, [(self.config['model']['input_channels'], 256, 256)])
          sys.exit(0)

      self.vis_predictions = None

      # Write the config YML to the run-folder
      self.config['run_info'] = dict(
          timestamp=timestamp,
          git_head=subprocess.check_output(["git", "describe"], encoding='utf8').strip()
      )
      with open(self.log_dir / 'config.yml', 'w') as f:
          yaml.dump(self.config, f)

      self.checkpoints = self.log_dir / 'checkpoints'
      self.checkpoints.mkdir()

      # Weights and Biases initialization
      if True:
        wandb.init(project=args.wandb_project, name=args.wandb_name, config=self.config)

  def run(self):
      for phase in self.config['schedule']:
          self.logger.info(f'Starting phase "{phase["phase"]}"')
          for epoch in range(phase['epochs']):
              # Epoch setup
              self.loss_function = create_loss(scoped_get('loss_function', phase, self.config)).to(self.dev)

              for step in phase['steps']:
                  if type(step) is dict:
                      assert len(step) == 1
                      (command, key), = step.items()
                  else:
                      command = step

                  if command == 'train_on':
                      # Training step
                      data_loader = self.get_dataloader(key)
                      self.train_epoch(data_loader)
                  elif command == 'validate_on':
                      # Validation step
                      data_loader = self.get_dataloader(key)
                      self.val_epoch(data_loader, key)
              if self.scheduler:
                  print("before step:", self.scheduler.get_last_lr())
                  self.scheduler.step()
                  print("after step:", self.scheduler.get_last_lr())

  def get_dataloader(self, name):
      if name in self.dataset_cache:
          return self.dataset_cache[name]

      if name in self.config['datasets']:
          ds_config = self.config['datasets'][name]
          if 'batch_size' not in ds_config:
              ds_config['batch_size'] = self.config['batch_size']
          ds_config['num_workers'] = self.config['data_threads']
          ds_config['data_sources'] = self.data_sources
          ds_config['data_root'] = self.DATA_ROOT
          ds_config['sampling_mode'] = self.config['sampling_mode']
          ds_config['tile_size'] = self.config['tile_size']
          self.dataset_cache[name] = get_loader(ds_config)

      return self.dataset_cache[name]

  def train_epoch(self, train_loader):
      self.epoch += 1
      wandb.log({'epoch': self.epoch}, step=self.epoch)
      self.logger.info(f'Epoch {self.epoch} - Training Started')
      progress = tqdm(train_loader)
      self.model.train(True)
      #for iteration, (img, target, metadata) in enumerate(progress):
      for iteration, (img, target) in enumerate(progress):
          img = img.to(self.dev, torch.float)
          target = target.to(self.dev, torch.long, non_blocking=True)#[:,[0]]

          self.opt.zero_grad()
          y_hat = self.model(img)

          metrics_terms = {}
          loss = self.loss_function(y_hat, target)
          metrics_terms['Loss'] = loss.detach()

          loss.backward()
          self.opt.step()

          with torch.no_grad():
              self.metrics.step(y_hat, target, **metrics_terms)

      metrics_vals = self.metrics.evaluate()
      progress.set_postfix(metrics_vals)
      self.metrics_vals_train = metrics_vals
      logstr = f'{self.epoch},' + ','.join(f'{val}' for key, val in metrics_vals.items())
      logfile = self.log_dir / 'train.csv'
      self.logger.info(f'Epoch {self.epoch} - Training Metrics: {metrics_vals}')
      if not logfile.exists():
          # Print header upon first log print
          header = 'Epoch,' + ','.join(f'{key}' for key, val in metrics_vals.items())
          with logfile.open('w') as f:
              print(header, file=f)
              print(logstr, file=f)
      else:
          with logfile.open('a') as f:
              print(logstr, file=f)

      wandb.log({f'trn/{k}': v for k, v in metrics_vals.items()}, step=self.epoch)

      # Save model Checkpoint
      torch.save(self.model.state_dict(), self.checkpoints / f'{self.epoch:02d}.pt')

  @torch.no_grad()
  def val_epoch(self, val_loader, tag):
    self.logger.info(f'Epoch {self.epoch} - Validation Started')
    self.metrics.reset()
    val_outputs = defaultdict(list)
    self.model.train(False)
    for iteration, (raw_img, raw_target, metadata) in enumerate(tqdm(val_loader)):
      img = raw_img.to(self.dev, torch.float)
      target = raw_target.to(self.dev, torch.long, non_blocking=True)
      y_hat = self.model(img)
      loss = self.loss_function(y_hat, target)
      if target.min() < 255:
        self.metrics.step(y_hat, target, Loss=loss.detach())

      for i in range(y_hat.shape[0]):
        name = Path(metadata['source_file'][i]).stem
        val_outputs[name].append({
          'Prediction': y_hat[i].cpu().numpy(),
          'Image': raw_img[i].numpy(),
          'Target': raw_target[i].numpy(),
          **{k: metadata[k][i] for k in metadata}
        })

    m = self.metrics.evaluate()
    self.metrics_vals_val = m
    logstr = f'{self.epoch},' + ','.join(f'{val}' for key, val in m.items())
    logfile = self.log_dir / f'{tag}.csv'
    self.logger.info(f'Epoch {self.epoch} - Validation Metrics: {m}')
    if not logfile.exists():
      # Print header upon first log print
      header = 'Epoch,' + ','.join(f'{key}' for key, val in m.items())
      with logfile.open('w') as f:
        print(header, file=f)
        print(logstr, file=f)
    else:
      with logfile.open('a') as f:
        print(logstr, file=f)

    wandb.log({f'{tag}/{k}': v for k, v in m.items()}, step=self.epoch)
    self.log_images(val_outputs)

  def log_images(self, val_outputs):
    self.logger.debug(f'Epoch {self.epoch} - Image Logging')

    for tile, data in val_outputs.items():
      y_max = max(d['y1'] for d in data)
      x_max = max(d['x1'] for d in data)

      rgb    = np.zeros([y_max, x_max, 3], dtype=np.uint8)
      target = np.zeros([y_max, x_max, 1], dtype=np.uint8)
      pred   = np.zeros([y_max, x_max, 1], dtype=np.uint8)

      for patch in data:
        y0, x0, y1, x1 = [patch[k] for k in ['y0', 'x0', 'y1', 'x1']]
        patch_rgb = patch['Image'][[3,2,1]]
        patch_rgb = np.clip(2 * 255 * patch_rgb, 0, 255).astype(np.uint8)
        patch_target = np.clip(255 * patch['Target'], 0, 255).astype(np.uint8)
        patch_pred = np.clip(255 * patch['Prediction'], 0, 255).astype(np.uint8)

        rgb[y0:y1, x0:x1]    = rearrange(patch_rgb, 'C H W -> H W C')
        target[y0:y1, x0:x1] = rearrange(patch_target, 'C H W -> H W C')
        pred[y0:y1, x0:x1]   = rearrange(patch_pred, 'C H W -> H W C')

      stacked = np.concatenate([
        rgb,
        np.concatenate([
          target,
          pred,
          np.zeros_like(target)
        ], axis=-1),
      ], axis=1)

      stacked = Image.fromarray(stacked)

      target_img = Image.new("L", (x_max, y_max), 0)
      target_draw = ImageDraw.Draw(target_img)
      for contour in find_contours(target[..., 0], 0.5):
        target_draw.polygon([(x,y) for y,x in contour],
                          fill=0, outline=255, width=3)

      pred_img = Image.new("L", (x_max, y_max), 0)
      pred_draw = ImageDraw.Draw(pred_img)
      for contour in find_contours(pred[..., 0], 0.7):
        pred_draw.polygon([(x,y) for y,x in contour],
                          fill=0, outline=255, width=3)
      target_img = np.asarray(target_img)
      pred_img = np.asarray(pred_img)
      annot = np.stack([
        target_img,
        pred_img,
        np.zeros_like(target_img),
      ], axis=-1)
      rgb_with_annot = np.where(np.all(annot == 0, axis=-1, keepdims=True),
                                rgb, annot)
      rgb_with_annot = Image.fromarray(rgb_with_annot)

      (self.log_dir / 'tile_predictions').mkdir(exist_ok=True)
      rgb_with_annot.save(self.log_dir / 'tile_predictions' / f'{tile}_contour_{self.epoch}.jpg')
      stacked.save(self.log_dir / 'tile_predictions' / f'{tile}_masks_{self.epoch}.jpg')

      outdir = self.log_dir / 'metrics_plots'
      outdir.mkdir(exist_ok=True)

  def setup_lr_scheduler(self):
      # Scheduler
      if 'learning_rate_scheduler' not in self.config.keys():
          print("running without learning rate scheduler")
          self.scheduler = None
      elif self.config['learning_rate_scheduler'] == 'StepLR':
          if 'lr_step_size' not in self.config.keys():
              step_size = 10
          else:
              step_size = self.config['lr_step_size']
          if 'lr_gamma' not in self.config.keys():
              gamma = 0.1
          else:
              gamma = self.config['lr_gamma']
          self.scheduler = torch.optim.lr_scheduler.StepLR(self.opt, step_size=step_size, gamma=gamma)
          print(f"running with 'StepLR' learning rate scheduler with step_size = {step_size} and gamma = {gamma}")
      
      elif self.config['learning_rate_scheduler'] == 'ExponentialLR':
          if 'lr_gamma' not in self.config.keys():
              gamma = 0.9
          else:
              gamma = self.config['lr_gamma']
          self.scheduler = torch.optim.lr_scheduler.ExponentialLR(self.opt, gamma=gamma)
          print(f"running with 'ExponentialLR' learning rate scheduler with gamma = {gamma}")
      
      elif self.config['learning_rate_scheduler'] == 'ReduceLROnPlateau':
          if 'lr_factor' not in self.config.keys():
              factor = 0.1
          else:
              factor = self.config['lr_factor']          
          if 'lr_patience' not in self.config.keys():
              patience = 10
          else:
              patience = self.config['lr_patience']
          self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.opt, factor=factor, patience=patience)
          print(f"running with 'ReduceLROnPlateau' learning rate scheduler with factor={factor} and patience={patience}")


def scoped_get(key, *scopestack):
    for scope in scopestack:
        value = scope.get(key)
        if value is not None:
            return value
    raise ValueError(f'Could not find "{key}" in any scope.')


def safe_append(dictionary, key, value):
    try:
        dictionary[key].append(value)
    except KeyError:
        dictionary[key] = [value]


if __name__ == "__main__":
    args = parser.parse_args()
    Engine().run()
