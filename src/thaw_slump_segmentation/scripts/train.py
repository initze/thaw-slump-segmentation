#!/usr/bin/env python
# flake8: noqa: E501
# Copyright (c) Ingmar Nitze and Konrad Heidler

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
Usecase 2 Training Script
"""

import argparse
import gc
import re
import subprocess
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import typer
import yaml
from rich import pretty, traceback
from torchmetrics import (
    AUROC,
    ROC,
    Accuracy,
    AveragePrecision,
    CohenKappa,
    ConfusionMatrix,
    F1Score,
    HammingDistance,
    HingeLoss,
    JaccardIndex,
    MetricCollection,
    Precision,
    PrecisionRecallCurve,
    Recall,
    Specificity,
)
from tqdm import tqdm
from typing_extensions import Annotated

import wandb

from thaw_slump_segmentation.data_loading import DataSources, get_loader, get_slump_loader, get_vis_loader
from thaw_slump_segmentation.models import create_loss, create_model
from thaw_slump_segmentation.utils import get_logger, init_logging, showexample, yaml_custom

traceback.install(show_locals=True)
pretty.install()


class Engine:
    def __init__(
        self,
        config: Path,
        data_dir: Path,
        name: str,
        log_dir: Path,
        resume: str,
        summary: bool,
        wandb_project: str,
        wandb_name: str,
    ):
        self.config = yaml.load(config.open(), Loader=yaml_custom.SaneYAMLLoader)
        self.DATA_ROOT = data_dir
        # Logging setup
        timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        if name:
            log_dir_name = f'{name}_{timestamp}'
        else:
            log_dir_name = timestamp
        self.log_dir = Path(log_dir) / log_dir_name
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
            in_channels=m['input_channels'],
        )

        # make parallel
        self.model = nn.DataParallel(self.model)

        if resume:
            self.config['resume'] = resume

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

        self.dev = torch.device('cpu') if not torch.cuda.is_available() else torch.device('cuda')
        self.logger.info(f'Training on {self.dev} device')

        self.model = self.model.to(self.dev)
        self.model = torch.compile(self.model, mode='max-autotune', fullgraph=True)

        self.learning_rate = self.config['learning_rate']
        self.opt = torch.optim.AdamW(self.model.parameters(), lr=self.learning_rate)
        self.setup_lr_scheduler()

        self.board_idx = 0
        self.epoch = 0

        if summary:
            from torchsummary import summary

            summary(self.model, [(self.config['model']['input_channels'], 256, 256)])
            sys.exit(0)

        self.dataset_cache = {}

        self.vis_predictions = None
        self.vis_loader, self.vis_names = get_vis_loader(
            self.config['visualization_tiles'],
            batch_size=self.config['batch_size'],
            data_sources=self.data_sources,
            data_root=self.DATA_ROOT,
        )

        # Write the config YML to the run-folder
        self.config['run_info'] = {
            'timestamp': timestamp,
            'git_head': subprocess.check_output(['git', 'describe'], encoding='utf8').strip(),
        }
        with open(self.log_dir / 'config.yml', 'w') as f:
            yaml.dump(self.config, f)

        self.checkpoints = self.log_dir / 'checkpoints'
        self.checkpoints.mkdir()

        # Weights and Biases initialization
        print('wandb project:', wandb_project)
        print('wandb name:', wandb_name)
        print('config:', self.config)
        print('entity:', 'ml4earth')
        wandb.init(project=wandb_project, name=wandb_name, config=self.config, entity='ingmarnitze_team')

    def run(self):
        for phase in self.config['schedule']:
            self.logger.info(f'Starting phase "{phase["phase"]}"')

            self.phase_should_log_images = phase.get('log_images', False)
            self.setup_metrics(phase)
            self.current_key = None
            for epoch in range(phase['epochs']):
                # Epoch setup
                self.epoch = epoch
                self.loss_function = create_loss(scoped_get('loss_function', phase, self.config)).to(self.dev)

                for step in phase['steps']:
                    command, key = parse_step(step)
                    self.current_key = key

                    if command == 'train_on':
                        # Training step
                        data_loader = self.get_dataloader(key)
                        self.train_epoch(data_loader)
                    elif command == 'validate_on':
                        # Validation step
                        data_loader = self.get_dataloader(key)
                        self.val_epoch(data_loader)
                    elif command == 'log_images':
                        self.logger.warn(
                            "Step 'log_images' is deprecated. Please use 'log_images' in the phase instead."
                        )
                        continue
                    else:
                        raise ValueError(f"Unknown command '{command}'")

                    # Reset the metrics after each step
                    if self.current_key:
                        self.metrics[self.current_key].reset()
                        gc.collect()
                        torch.cuda.empty_cache()

                # Log images after each epoch
                if self.phase_should_log_images:
                    self.log_images()

                if self.scheduler:
                    print('before step:', self.scheduler.get_last_lr())
                    self.scheduler.step()
                    print('after step:', self.scheduler.get_last_lr())

    def setup_metrics(self, phase: dict):
        # Setup Metrics for this phase
        # Check if the phase has a log_images command -> Relevant for memory management of metrics (PRC, ROC and Confusion Matrix)
        nthresholds = (
            100  # Make sure that the thresholds for the PRC-based metrics are equal to benefit from grouped computing
        )
        metrics = MetricCollection(
            {
                'Accuracy': Accuracy(task='binary'),
                'Precision': Precision(task='binary'),
                'Specificity': Specificity(task='binary'),
                'Recall': Recall(task='binary'),
                'F1Score': F1Score(task='binary'),
                'JaccardIndex': JaccardIndex(task='binary'),
                'AUROC': AUROC(task='binary', thresholds=nthresholds),
                'AveragePrecision': AveragePrecision(task='binary', thresholds=nthresholds),
                # Calibration errors: https://arxiv.org/abs/1909.10155, they take a lot of memory!
                #'ExpectedCalibrationError': CalibrationError(task='binary', norm='l1'),
                #'RootMeanSquaredCalibrationError': CalibrationError(task='binary', norm='l2'),
                #'MaximumCalibrationError': CalibrationError(task='binary', norm='max'),
                'CohenKappa': CohenKappa(task='binary'),
                'HammingDistance': HammingDistance(task='binary'),
                'HingeLoss': HingeLoss(task='binary'),
                # MCC raised a weired error one time, skipping to be save
                # 'MatthewsCorrCoef': MatthewsCorrCoef(task='binary'),
            }
        )
        self.metrics = {}
        self.rocs = {}
        self.prcs = {}
        self.confmats = {}
        if self.phase_should_log_images:
            self.metric_tracker = defaultdict(list)

        for step in phase['steps']:
            command, key = parse_step(step)
            if not key:
                continue

            self.metrics[key] = metrics.clone(f'{key}/').to(self.dev)
            if command == 'validate_on':
                # We don't want to log the confusion matrix, PRC and ROC curve for every step in the training loop
                # We assign them seperately to have easy access to them in the log_images phase but still benefit from the MetricCollection in terms of compute_groups and update, compute & reset calls
                self.rocs[key] = ROC(task='binary', thresholds=nthresholds).to(self.dev)
                self.prcs[key] = PrecisionRecallCurve(task='binary', thresholds=nthresholds).to(self.dev)
                self.confmats[key] = ConfusionMatrix(task='binary', normalize='true').to(self.dev)
                self.metrics[key].add_metrics(
                    {'ROC': self.rocs[key], 'PRC': self.prcs[key], 'ConfusionMatrix': self.confmats[key]}
                )

        # Create the plot directory
        if self.phase_should_log_images:
            phase_name = phase['phase'].lower()
            self.metric_plot_dir = self.log_dir / f'metrics_plots_{phase_name}'
            self.metric_plot_dir.mkdir(exist_ok=True)

    def get_dataloader(self, name):
        if name in self.dataset_cache:
            return self.dataset_cache[name]

        if name in self.config['datasets']:
            ds_config = self.config['datasets'][name]
            if 'batch_size' not in ds_config:
                ds_config['batch_size'] = self.config['batch_size']
            ds_config['num_workers'] = self.config['data_threads']
            ds_config['augment_types'] = self.config['datasets']
            ds_config['data_sources'] = self.data_sources
            ds_config['data_root'] = self.DATA_ROOT
            self.dataset_cache[name] = get_loader(**ds_config)
        else:
            func, arg = re.search(r'(\w+)\((\w+)\)', name).groups()
            if func == 'slump_tiles':
                ds_config = self.config['datasets'][arg]
                if 'batch_size' not in ds_config:
                    ds_config['batch_size'] = self.config['batch_size']
                ds_config['num_workers'] = self.config['data_threads']
                ds_config['data_sources'] = self.data_sources
                ds_config['data_root'] = self.DATA_ROOT
                self.dataset_cache[name] = get_slump_loader(**ds_config)

        return self.dataset_cache[name]

    def train_epoch(self, train_loader):
        self.logger.info(f'Epoch {self.epoch} - Training Started')
        progress = tqdm(train_loader)
        self.model.train(True)
        epoch_loss = {
            'Loss': [],
            'Deep Supervision Loss': [],
        }
        for iteration, (img, target) in enumerate(progress):
            self.board_idx += img.shape[0]

            img = img.to(self.dev, torch.float)
            target = target.to(self.dev, torch.long, non_blocking=True)

            self.opt.zero_grad()
            y_hat = self.model(img)

            if isinstance(y_hat, (tuple, list)):
                # Deep Supervision
                deep_super_losses = [self.loss_function(pred.squeeze(1), target) for pred in y_hat]
                y_hat = y_hat[0].squeeze(1)
                loss = sum(deep_super_losses)
                for dsl in deep_super_losses:
                    epoch_loss['Loss'].append(dsl.detach())
                epoch_loss['Deep Supervision Loss'].append(loss.detach())
            else:
                loss = self.loss_function(y_hat, target)
                epoch_loss['Loss'].append(loss.detach())

            loss.backward()
            self.opt.step()

            with torch.no_grad():
                self.metrics[self.current_key].update(y_hat.squeeze(1), target)

        # Compute epochs loss and metrics
        epoch_loss = {k: torch.stack(v).mean().item() for k, v in epoch_loss.items() if v}
        wandb.log({'train/Loss': epoch_loss}, step=self.board_idx)
        epoch_metrics = self.log_metrics()
        self.logger.info(f'Epoch {self.epoch} - Loss {epoch_loss} Training Metrics: {epoch_metrics}')
        self.log_csv(epoch_metrics, epoch_loss)

        # Update progress bar
        progress.set_postfix(epoch_metrics)

        # Save model Checkpoint
        torch.save(self.model.state_dict(), self.checkpoints / f'{self.epoch:02d}.pt')

    def val_epoch(self, val_loader):
        self.logger.info(f'Epoch {self.epoch} - Validation of {self.current_key} Started')
        self.model.train(False)
        with torch.no_grad():
            epoch_loss = []
            for iteration, (img, target) in enumerate(val_loader):
                img = img.to(self.dev, torch.float)
                target = target.to(self.dev, torch.long, non_blocking=True)
                y_hat = self.model(img).squeeze(1)

                loss = self.loss_function(y_hat, target)
                epoch_loss.append(loss.detach())
                self.metrics[self.current_key].update(y_hat, target)

            # Compute epochs loss and metrics
            epoch_loss = torch.stack(epoch_loss).mean().item()
            epoch_metrics = self.log_metrics()
        wandb.log({'val/Loss': epoch_loss}, step=self.board_idx)
        self.logger.info(f'Epoch {self.epoch} - Loss {epoch_loss} Validation Metrics: {epoch_metrics}')
        self.log_csv(epoch_metrics, epoch_loss)

        # Plot roc, prc and confusion matrix to disk and wandb
        fig_roc, _ = self.rocs[self.current_key].plot(score=True)
        fig_prc, _ = self.prcs[self.current_key].plot(score=True)
        fig_confmat, _ = self.confmats[self.current_key].plot(cmap='Blues')

        # We need to wrap the figures into a wandb.Image to save storage -> Maybe we can find a better solution in the future, e.g. using plotly
        wandb.log({f'{self.current_key}/ROC': wandb.Image(fig_roc)}, step=self.board_idx)
        wandb.log({f'{self.current_key}/PRC': wandb.Image(fig_prc)}, step=self.board_idx)
        wandb.log({f'{self.current_key}/Confusion Matrix': wandb.Image(fig_confmat)}, step=self.board_idx)

        if self.phase_should_log_images:
            fig_roc.savefig(self.metric_plot_dir / f'{self.current_key}_roc_curve.png')
            fig_prc.savefig(self.metric_plot_dir / f'{self.current_key}_precision_recall_curve.png')
            fig_confmat.savefig(self.metric_plot_dir / f'{self.current_key}_confusion_matrix.png')
        fig_roc.clear()
        fig_prc.clear()
        fig_confmat.clear()
        plt.close('all')

    def log_metrics(self) -> dict:
        epoch_metrics = self.metrics[self.current_key].compute()
        # We need to filter our ROC, PRC and Confusion Matrix from the metrics because they are not scalar metrics

        scalar_epoch_metrics = {
            k: v.item() for k, v in epoch_metrics.items() if isinstance(v, torch.Tensor) and v.numel() == 1
        }

        # Log to WandB
        wandb.log(scalar_epoch_metrics, step=self.board_idx)

        # Plot the metrics to disk
        if self.phase_should_log_images:
            scalar_epoch_metrics_tensor = {
                k: v for k, v in epoch_metrics.items() if isinstance(v, torch.Tensor) and v.numel() == 1
            }
            self.metric_tracker[self.current_key].append(scalar_epoch_metrics_tensor)
            fig, ax = plt.subplots(figsize=(20, 10))
            self.metrics[self.current_key].plot(self.metric_tracker[self.current_key], together=True, ax=ax)
            fig.savefig(self.metric_plot_dir / f'{self.current_key}_metrics.png')
            fig.clear()
            plt.close('all')

        return scalar_epoch_metrics

    def log_csv(self, epoch_metrics: dict, epoch_loss: int | dict):
        # Log to CSV
        logfile = self.log_dir / f'{self.current_key}.csv'

        if isinstance(epoch_loss, dict):
            logstr = (
                f'{self.epoch},'
                + ','.join(f'{val}' for key, val in epoch_metrics.items())
                + ','
                + ','.join(f'{val}' for key, val in epoch_loss.items())
            )
        else:
            logstr = f'{self.epoch},' + ','.join(f'{val}' for key, val in epoch_metrics.items()) + f',{epoch_loss}'

        if not logfile.exists():
            # Print header upon first log print
            header = 'Epoch,' + ','.join(f'{key}' for key, val in epoch_metrics.items()) + ',Loss'
            with logfile.open('w') as f:
                print(header, file=f)
                print(logstr, file=f)
        else:
            with logfile.open('a') as f:
                print(logstr, file=f)

    def log_images(self):
        self.logger.debug(f'Epoch {self.epoch} - Image Logging')
        self.model.train(False)
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
            showexample(
                self.vis_loader.dataset[i], self.vis_predictions[i], filename, self.data_sources, step=self.board_idx
            )
        plt.close('all')

    def setup_lr_scheduler(self):
        # Scheduler
        if 'learning_rate_scheduler' not in self.config.keys():
            print('running without learning rate scheduler')
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


def scoped_get(key, *scopestack):
    for scope in scopestack:
        value = scope.get(key)
        if value is not None:
            return value
    raise ValueError(f'Could not find "{key}" in any scope.')


def parse_step(step) -> tuple[str, str | None]:
    if isinstance(step, dict):
        assert len(step) == 1
        ((command, key),) = step.items()
    else:
        command, key = step, None
    return command, key


def train(
    name: Annotated[
        str,
        typer.Option(
            '--name',
            '-n',
            prompt=True,
            help='Give this run a name, so that it will be logged into logs/<NAME>_<timestamp>.',
        ),
    ],
    data_dir: Annotated[Path, typer.Option('--data_dir', help='Path to data processing dir')] = Path('data'),
    log_dir: Annotated[Path, typer.Option('--log_dir', help='Path to log dir')] = Path('logs'),
    config: Annotated[Path, typer.Option('--config', '-c', help='Specify run config to use.')] = Path('config.yml'),
    resume: Annotated[
        str,
        typer.Option(
            '--resume',
            '-r',
            help='Resume from the specified checkpoint. Can be either a run-id (e.g. "2020-06-29_18-12-03") to select the last. Overrides the resume option in the config file if given.',
        ),
    ] = None,
    summary: Annotated[bool, typer.Option('--summary', '-s', help='Only print model summary and return.')] = False,
    wandb_project: Annotated[
        str, typer.Option('--wandb_project', '-wp', help='Set a project name for weights and biases')
    ] = 'thaw-slump-segmentation',
    wandb_name: Annotated[
        str, typer.Option('--wandb_name', '-wn', help='Set a run name for weights and biases')
    ] = None,
):
    """Training script"""
    engine = Engine(config, data_dir, name, log_dir, resume, summary, wandb_project, wandb_name)
    engine.run()


# ! Moving legacy argparse cli to main to maintain compatibility with the original script
if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Training script', formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('-s', '--summary', action='store_true', help='Only print model summary and return.')
    parser.add_argument('--data_dir', default='data', type=Path, help='Path to data processing dir')
    parser.add_argument('--log_dir', default='logs', type=Path, help='Path to log dir')
    parser.add_argument(
        '-n', '--name', default='', help='Give this run a name, so that it will be logged into logs/<NAME>_<timestamp>.'
    )
    parser.add_argument('-c', '--config', default='config.yml', type=Path, help='Specify run config to use.')
    parser.add_argument(
        '-r',
        '--resume',
        default='',
        help='Resume from the specified checkpoint.'
        'Can be either a run-id (e.g. "2020-06-29_18-12-03") to select the last'
        'Can be either a run-id (e.g. "2020-06-29_18-12-03") to select the last'
        'Overrides the resume option in the config file if given.',
    )
    parser.add_argument(
        '-wp', '--wandb_project', default='thaw-slump-segmentation', help='Set a project name for weights and biases'
    )
    parser.add_argument('-wn', '--wandb_name', default=None, help='Set a run name for weights and biases')

    args = parser.parse_args()
    Engine(
        args.config,
        args.data_dir,
        args.name,
        args.log_dir,
        args.resume,
        args.summary,
        args.wandb_project,
        args.wandb_name,
    ).run()
