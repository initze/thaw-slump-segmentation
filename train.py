#!/usr/bin/env python
# flake8: noqa: E501
# Copyright (c) Ingmar Nitze and Konrad Heidler

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
Usecase 2 Training Script
"""
import argparse
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import torch
import yaml
from tqdm import tqdm

from data_loading import get_loader, get_vis_loader, get_slump_loader, DataSources
from lib import Metrics, Accuracy, Precision, Recall, F1, IoU
from lib.models import create_model, create_loss
from lib.utils import showexample, plot_metrics, plot_precision_recall, init_logging, get_logger, yaml_custom

parser = argparse.ArgumentParser()
parser.add_argument('-s', '--summary', action='store_true',
        help='Only print model summary and return.')
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


class Engine:
    def __init__(self):
        cli_args = parser.parse_args()
        self.config = yaml.load(cli_args.config.open(), Loader=yaml_custom.SaneYAMLLoader)

        # Logging setup
        timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        if cli_args.name:
            log_dir_name = f'{cli_args.name}_{timestamp}'
        else:
            log_dir_name = timestamp
        self.log_dir = Path('logs') / log_dir_name
        self.log_dir.mkdir(exist_ok=False)

        init_logging(self.log_dir / 'train.log')
        self.logger = get_logger('train')

        self.data_sources = DataSources(self.config['data_sources'])
        self.config['model']['input_channels'] = sum(src.channels for src in self.data_sources)

        m = self.config['model']
        self.model = create_model(
            arch=m['architecture'],
            encoder_name=m['encoder'],
            encoder_weights=None if m['encoder_weights'] == 'random' else m['encoder_weights'],
            classes=1,
            in_channels=m['input_channels']
        )

        if cli_args.resume:
            self.config['resume'] = cli_args.resume

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

        self.opt = torch.optim.AdamW(self.model.parameters(), lr=self.config['learning_rate'])

        self.board_idx = 0
        self.epoch = 0
        self.metrics = Metrics(Accuracy, Precision, Recall, F1, IoU)

        if cli_args.summary:
            from torchsummary import summary
            summary(self.model, [(self.config['model']['input_channels'], 256, 256)])
            sys.exit(0)

        self.dataset_cache = {}

        self.vis_predictions = None
        self.vis_loader, self.vis_names = get_vis_loader(self.config['visualization_tiles'],
                                                         batch_size=self.config['batch_size'],
                                                         data_sources=self.data_sources)

        # Write the config YML to the run-folder
        self.config['run_info'] = dict(
            timestamp=timestamp,
            git_head=subprocess.check_output(["git", "describe"], encoding='utf8').strip()
        )
        with open(self.log_dir / 'config.yml', 'w') as f:
            yaml.dump(self.config, f)

        self.checkpoints = self.log_dir / 'checkpoints'
        self.checkpoints.mkdir()

        # Tensorboard initialization
        from torch.utils.tensorboard import SummaryWriter
        self.trn_writer = SummaryWriter(self.log_dir / 'train')
        self.val_writer = SummaryWriter(self.log_dir / 'val')
        self.trn_metrics = {}
        self.val_metrics = {}

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
                        self.val_epoch(data_loader)
                    elif command == 'log_images':
                        self.log_images()

    def get_dataloader(self, name):
        if name in self.dataset_cache:
            return self.dataset_cache[name]

        if name in self.config['datasets']:
            ds_config = self.config['datasets'][name]
            if 'batch_size' not in ds_config:
                ds_config['batch_size'] = self.config['batch_size']
            ds_config['num_workers'] = self.config['data_threads']
            ds_config['augment_types'] = self.config['datasets']
            self.dataset_cache[name] = get_loader(data_sources=self.data_sources, **ds_config)
        else:
            func, arg = re.search(r'(\w+)\((\w+)\)', name).groups()
            if func == 'slump_tiles':
                ds_config = self.config['datasets'][arg]
                if 'batch_size' not in ds_config:
                    ds_config['batch_size'] = self.config['batch_size']
                ds_config['num_workers'] = self.config['data_threads']
                self.dataset_cache[name] = get_slump_loader(data_sources=self.data_sources, **ds_config)

        return self.dataset_cache[name]

    def train_epoch(self, train_loader):
        self.epoch += 1
        self.logger.info(f'Epoch {self.epoch} - Training Started')
        progress = tqdm(train_loader)
        self.model.train(True)
        for iteration, (img, target) in enumerate(progress):
            self.board_idx += img.shape[0]
            # TBD: Should we use learning rate decay?
            # lr_factor = min(board_idx / 50000, 1, 2 ** ((50000 - board_idx) / 50000))
            # for i in range(len(opt.param_groups)):
            #     lr = self.config['learning_rate'] * lr_factor
            #     opt.param_groups[i]['lr'] = lr

            img = img.to(self.dev, torch.float)
            target = target.to(self.dev, torch.long, non_blocking=True)

            self.opt.zero_grad()
            y_hat = self.model(img).squeeze(1)
            loss = self.loss_function(y_hat, target)
            loss.backward()
            self.opt.step()

            with torch.no_grad():
                self.metrics.step(y_hat, target, Loss=loss.detach())

        metrics_vals = self.metrics.evaluate()
        progress.set_postfix(metrics_vals)
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

        for key, val in metrics_vals.items():
            self.trn_writer.add_scalar(key, val, self.board_idx)
            safe_append(self.trn_metrics, key, val)
        safe_append(self.trn_metrics, 'step', self.board_idx)
        safe_append(self.trn_metrics, 'epoch', self.epoch)
        self.trn_writer.flush()

        # Save model Checkpoint
        torch.save(self.model.state_dict(), self.checkpoints / f'{self.epoch:02d}.pt')

    def val_epoch(self, val_loader):
        self.logger.info(f'Epoch {self.epoch} - Validation Started')
        self.metrics.reset()
        self.model.train(False)
        with torch.no_grad():
            for iteration, (img, target) in enumerate(val_loader):
                img = img.to(self.dev, torch.float)
                target = target.to(self.dev, torch.long, non_blocking=True)
                y_hat = self.model(img).squeeze(1)

                loss = self.loss_function(y_hat, target)
                self.metrics.step(y_hat, target, Loss=loss.detach())

            m = self.metrics.evaluate()
            logstr = f'{self.epoch},' + ','.join(f'{val}' for key, val in m.items())
            logfile = self.log_dir / 'val.csv'
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
            for key, val in m.items():
                self.val_writer.add_scalar(key, val, self.board_idx)
                safe_append(self.val_metrics, key, val)
            safe_append(self.val_metrics, 'step', self.board_idx)
            safe_append(self.val_metrics, 'epoch', self.epoch)
            self.val_writer.flush()

    def log_images(self):
        self.logger.debug(f'Epoch {self.epoch} - Image Logging')
        with torch.no_grad():
            preds = []
            for vis_imgs, vis_masks in self.vis_loader:
                preds.append(self.model(vis_imgs.to(self.dev)).cpu().squeeze(1))
            preds = torch.cat(preds).unsqueeze(1)
            if self.vis_predictions is None:
                self.vis_predictions = preds
            else:
                self.vis_predictions = torch.cat([self.vis_predictions, preds], dim=1)
        (self.log_dir / 'tile_predictions').mkdir(exist_ok=True)
        for i, tile in enumerate(self.vis_names):
            filename = self.log_dir / 'tile_predictions' / f'{tile}.jpg'
            showexample(self.vis_loader.dataset[i], self.vis_predictions[i],
                        filename, self.data_sources, self.val_writer)

        outdir = self.log_dir / 'metrics_plots'
        outdir.mkdir(exist_ok=True)
        plot_metrics(self.trn_metrics, self.val_metrics, outdir=outdir)
        plot_precision_recall(self.trn_metrics, self.val_metrics, outdir=outdir)
        self.val_writer.flush()


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
    cli_args = parser.parse_args()
    Engine().run()
