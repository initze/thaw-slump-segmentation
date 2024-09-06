#!/usr/bin/env python
# flake8: noqa: E501
# Copyright (c) Ingmar Nitze and Konrad Heidler

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
Usecase 2 Training Script
"""

import gc
import re
import subprocess
import sys
from collections import defaultdict
from copy import deepcopy
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import typer
import yaml
from rich import pretty, print, traceback
from torchmetrics import (
    AUROC,
    ROC,
    Accuracy,
    AveragePrecision,
    CohenKappa,
    ConfusionMatrix,
    F1Score,
    HammingDistance,
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
from thaw_slump_segmentation.metrics import (
    BinaryBoundaryIoU,
    BinaryInstanceAccuracy,
    BinaryInstanceAveragePrecision,
    BinaryInstanceConfusionMatrix,
    BinaryInstanceF1Score,
    BinaryInstancePrecision,
    BinaryInstancePrecisionRecallCurve,
    BinaryInstanceRecall,
)
from thaw_slump_segmentation.models import create_loss, create_model
from thaw_slump_segmentation.utils import get_logger, init_logging, showexample, yaml_custom

traceback.install(show_locals=True)
pretty.install()


class Engine:
    def __init__(
        self,
        config: dict,
        data_dir: Path,
        name: str,
        log_dir: Path,
        resume: str,
        summary: bool,
    ):
        self.config = config
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
        # self.model = torch.compile(self.model, mode='max-autotune', fullgraph=True)

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

    def run(self):
        print('\n\n=== Using Config ===\n')
        print(self.config)

        for phase in self.config['schedule']:
            self.logger.info(f'Starting phase "{phase["phase"]}"')

            self.phase_should_log_images = phase.get('log_images', False)
            self.phase_should_save_plots = phase.get('save_plots_to_file', True)
            self.setup_metrics(phase)
            self.current_key = None
            for epoch in range(phase['epochs']):
                # Epoch setup
                self.epoch = epoch
                self.loss_function = create_loss(scoped_get('loss_function', phase, self.config)).to(self.dev)

                # Check if this epoch should log images and heavy metrics
                self.is_heavy_epoch = epoch % phase.get('heavy_metrics_nsteps', 1) == 0
                if self.is_heavy_epoch:
                    self.logger.info(f'Epoch {epoch} - Heavy Metrics Enabled')
                else:
                    self.logger.info(f'Epoch {epoch} - Heavy Metrics Disabled')

                for step in phase['steps']:
                    self.run_step(step)

                # Log images after each epoch
                if self.phase_should_log_images:
                    self.log_images()

                if self.scheduler:
                    print('\nUpdating LR-Scheduler: ')
                    print(f' - Before step: {self.scheduler.get_last_lr()}')
                    self.scheduler.step()
                    print(f' - After step: {self.scheduler.get_last_lr()}')

    def run_step(self, step: str):
        command, key = parse_step(step)

        # Fail early if the command is unknown or deprecated
        assert command in ['train_on', 'validate_on', 'log_images'], f"Unknown command '{command}'"
        if command == 'log_images':
            self.logger.warn("Step 'log_images' is deprecated. Please use 'log_images' in the phase instead.")
            return

        self.current_key = key
        self.current_command = command
        data_loader = self.get_dataloader(key)

        if command == 'train_on':
            self.train_epoch(data_loader)
        elif command == 'validate_on':
            self.val_epoch(data_loader)

        # Reset the metrics after each step
        if self.current_key:
            self.metrics[self.current_key].reset()
            self.metrics_heavy[self.current_key].reset()
            gc.collect()
            torch.cuda.empty_cache()

    def setup_metrics(self, phase: dict):
        # Setup Metrics for this phase
        # Check if the phase has a log_images command -> Relevant for memory management of metrics (PRC, ROC and Confusion Matrix)
        nthresholds = (
            100  # Make sure that the thresholds for the PRC-based metrics are equal to benefit from grouped computing
        )
        nthresholds_instance = 10  # Less for instance metrics because they are more memory intensive
        # Make sure that the matching args are the same for all instance metrics
        matching_threshold = 0.5
        matching_metric = 'iou'
        boundary_dilation = 0.02

        metrics = MetricCollection(
            {
                'Accuracy': Accuracy(task='binary', validate_args=False),
                'Precision': Precision(task='binary', validate_args=False),
                'Specificity': Specificity(task='binary', validate_args=False),
                'Recall': Recall(task='binary', validate_args=False),
                'F1Score': F1Score(task='binary', validate_args=False),
                'JaccardIndex': JaccardIndex(task='binary', validate_args=False),
                # Calibration errors: https://arxiv.org/abs/1909.10155, they take a lot of memory!
                #'ExpectedCalibrationError': CalibrationError(task='binary', norm='l1'),
                #'RootMeanSquaredCalibrationError': CalibrationError(task='binary', norm='l2'),
                #'MaximumCalibrationError': CalibrationError(task='binary', norm='max'),
                'CohenKappa': CohenKappa(task='binary', validate_args=False),
                'HammingDistance': HammingDistance(task='binary', validate_args=False),
                # 'HingeLoss': HingeLoss(task='binary', validate_args=False),
                # MCC raised a weired error one time, skipping to be save
                # 'MatthewsCorrCoef': MatthewsCorrCoef(task='binary'),
            }
        )

        # Introducing a second collection, which should store hard-to-compute metrics
        heavy_metrics = MetricCollection(
            {
                'AUROC': AUROC(task='binary', thresholds=nthresholds, validate_args=False),
                'AveragePrecision': AveragePrecision(task='binary', thresholds=nthresholds, validate_args=False),
            }
        )

        self.metrics = {}
        self.metrics_heavy = {}
        self.rocs = {}
        self.prcs = {}
        self.confmats = {}
        self.instance_prcs = {}
        self.instance_confmats = {}

        if self.phase_should_log_images:
            self.metric_tracker = defaultdict(list)
            self.metric_heavy_tracker = defaultdict(list)

        for step in phase['steps']:
            command, key = parse_step(step)
            if not key:
                continue

            self.metrics[key] = metrics.clone(f'{key}/').to(self.dev)
            self.metrics_heavy[key] = heavy_metrics.clone(f'{key}/').to(self.dev)
            if command == 'validate_on':
                # We don't want to log the confusion matrix, PRC and ROC curve for every step in the training loop
                # We assign them seperately to have easy access to them in the log_images phase but still benefit from the MetricCollection in terms of compute_groups and update, compute & reset calls
                self.rocs[key] = ROC(task='binary', thresholds=nthresholds, validate_args=False).to(self.dev)
                self.prcs[key] = PrecisionRecallCurve(task='binary', thresholds=nthresholds, validate_args=False).to(
                    self.dev
                )
                self.confmats[key] = ConfusionMatrix(task='binary', normalize='true', validate_args=False).to(self.dev)
                self.instance_prcs[key] = BinaryInstancePrecisionRecallCurve(
                    thresholds=nthresholds_instance,
                    matching_threshold=matching_threshold,
                    matching_metric=matching_metric,
                    validate_args=False,
                ).to(self.dev)
                self.instance_confmats[key] = BinaryInstanceConfusionMatrix(
                    matching_threshold=matching_threshold, matching_metric=matching_metric, validate_args=False
                ).to(self.dev)

                self.metrics[key].add_metrics(
                    {
                        'ConfusionMatrix': self.confmats[key],
                        'Instance-ConfusionMatrix': self.instance_confmats[key],
                    }
                )
                self.metrics_heavy[key].add_metrics(
                    {
                        'ROC': self.rocs[key],
                        'PRC': self.prcs[key],
                        'Instance-PRC': self.instance_prcs[key],
                    }
                )

                # We don't want to log the instance metrics for every step in the training loop, because they are memory intensive (and not very useful for training)
                self.metrics[key].add_metrics(
                    {
                        'Instance-Accuracy': BinaryInstanceAccuracy(
                            matching_threshold=matching_threshold, matching_metric=matching_metric, validate_args=False
                        ).to(self.dev),
                        'Instance-Precision': BinaryInstancePrecision(
                            matching_threshold=matching_threshold, matching_metric=matching_metric, validate_args=False
                        ).to(self.dev),
                        'Instance-Recall': BinaryInstanceRecall(
                            matching_threshold=matching_threshold, matching_metric=matching_metric, validate_args=False
                        ).to(self.dev),
                        'Instance-F1Score': BinaryInstanceF1Score(
                            matching_threshold=matching_threshold, matching_metric=matching_metric, validate_args=False
                        ).to(self.dev),
                        'BoundaryIoU': BinaryBoundaryIoU(dilation=boundary_dilation, validate_args=False).to(self.dev),
                    }
                )
                self.metrics_heavy[key].add_metrics(
                    {
                        'Instance-AveragePrecision': BinaryInstanceAveragePrecision(
                            thresholds=nthresholds_instance,
                            matching_threshold=matching_threshold,
                            matching_metric=matching_metric,
                            validate_args=False,
                        ).to(self.dev),
                    }
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
                if self.is_heavy_epoch:
                    self.metrics_heavy[self.current_key].update(y_hat.squeeze(1), target)

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
                if self.is_heavy_epoch:
                    self.metrics_heavy[self.current_key].update(y_hat, target)

            # Compute epochs loss and metrics
            epoch_loss = torch.stack(epoch_loss).mean().item()
            epoch_metrics = self.log_metrics()
        wandb.log({'val/Loss': epoch_loss}, step=self.board_idx)
        self.logger.info(f'Epoch {self.epoch} - Loss {epoch_loss} Validation Metrics: {epoch_metrics}')
        self.log_csv(epoch_metrics, epoch_loss)

        # Plot roc, prc and confusion matrix to disk and wandb
        fig_confmat, _ = self.confmats[self.current_key].plot(cmap='Blues')
        fig_instance_confmat, _ = self.instance_confmats[self.current_key].plot(cmap='Blues')
        # We need to wrap the figures into a wandb.Image to save storage -> Maybe we can find a better solution in the future, e.g. using plotly
        wandb.log({f'{self.current_key}/Confusion Matrix': wandb.Image(fig_confmat)}, step=self.board_idx)
        wandb.log(
            {f'{self.current_key}/Instance-Confusion Matrix': wandb.Image(fig_instance_confmat)}, step=self.board_idx
        )
        if self.phase_should_log_images and self.phase_should_save_plots:
            fig_confmat.savefig(self.metric_plot_dir / f'{self.current_key}_confusion_matrix.png')
            fig_instance_confmat.savefig(self.metric_plot_dir / f'{self.current_key}_instance_confusion_matrix.png')

        fig_confmat.clear()
        fig_instance_confmat.clear()

        if self.is_heavy_epoch:
            fig_roc, _ = self.rocs[self.current_key].plot(score=True)
            fig_prc, _ = self.prcs[self.current_key].plot(score=True)
            fig_instance_prc, _ = self.instance_prcs[self.current_key].plot(score=True)

            wandb.log({f'{self.current_key}/ROC': wandb.Image(fig_roc)}, step=self.board_idx)
            wandb.log({f'{self.current_key}/PRC': wandb.Image(fig_prc)}, step=self.board_idx)
            wandb.log({f'{self.current_key}/Instance-PRC': wandb.Image(fig_instance_prc)}, step=self.board_idx)

            if self.phase_should_log_images and self.phase_should_save_plots:
                fig_roc.savefig(self.metric_plot_dir / f'{self.current_key}_roc_curve.png')
                fig_prc.savefig(self.metric_plot_dir / f'{self.current_key}_precision_recall_curve.png')
                fig_instance_prc.savefig(
                    self.metric_plot_dir / f'{self.current_key}_instance_precision_recall_curve.png'
                )

            fig_roc.clear()
            fig_prc.clear()
            # fig_instance_prc.clear()
        plt.close('all')

    def log_metrics(self) -> dict:
        epoch_metrics = self.metrics[self.current_key].compute()
        epoch_metrics_heavy = {}
        if self.is_heavy_epoch and self.current_command == 'validate_on':
            self.logger.info(f'*** Calculating Heavy Metrics ({self.current_key}) in epoch {self.epoch}')
            epoch_metrics_heavy = self.metrics_heavy[self.current_key].compute()

        # Plot the metrics to disk
        if self.phase_should_log_images and self.phase_should_save_plots:
            scalar_epoch_metrics_tensor = {
                k: v for k, v in epoch_metrics.items() if isinstance(v, torch.Tensor) and v.numel() == 1
            }
            self.metric_tracker[self.current_key].append(scalar_epoch_metrics_tensor)
            fig, ax = plt.subplots(figsize=(30, 10))
            self.metrics[self.current_key].plot(self.metric_tracker[self.current_key], together=True, ax=ax)
            fig.savefig(self.metric_plot_dir / f'{self.current_key}_metrics.png')
            fig.clear()

            if self.is_heavy_epoch and self.current_command == 'validate_on':
                scalar_epoch_heavy_metrics_tensor = {
                    k: v for k, v in epoch_metrics_heavy.items() if isinstance(v, torch.Tensor) and v.numel() == 1
                }
                self.metric_heavy_tracker[self.current_key].append(scalar_epoch_heavy_metrics_tensor)
                fig, ax = plt.subplots(figsize=(30, 10))
                self.metrics_heavy[self.current_key].plot(
                    self.metric_heavy_tracker[self.current_key], together=True, ax=ax
                )
                fig.savefig(self.metric_plot_dir / f'{self.current_key}_heavy_metrics.png')
                fig.clear()
            plt.close('all')

        if self.is_heavy_epoch:
            epoch_metrics = epoch_metrics | epoch_metrics_heavy

        # We need to filter our ROC, PRC and Confusion Matrix from the metrics because they are not scalar metrics
        scalar_epoch_metrics = {
            k: v.item() for k, v in epoch_metrics.items() if isinstance(v, torch.Tensor) and v.numel() == 1
        }

        # Log to WandB
        wandb.log(scalar_epoch_metrics, step=self.board_idx)

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
                self.vis_loader.dataset[i],
                self.vis_predictions[i],
                filename,
                self.data_sources,
                step=self.board_idx,
                complex=self.phase_should_save_plots,
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
    config = yaml.load(config.open(), Loader=yaml_custom.SaneYAMLLoader)
    engine = Engine(config, data_dir, name, log_dir, resume, summary)

    if wandb_project is not None:
        wandb_entity = 'ingmarnitze_team'
        print('\n\n=== Using Weights & Biases ===\n')
        print(f'Entitiy: {wandb_entity}')
        print(f'Project: {wandb_project}')
        print(f'Name: {wandb_name}')

        wandb.init(project=wandb_project, name=wandb_name, config=config, entity=wandb_entity)

    engine.run()


def sweep(
    config: Annotated[Path, typer.Option('--config', '-c', help='Specify run config to use.')] = Path('config.yml'),
    wandb_project: Annotated[
        str, typer.Option('--wandb_project', '-wp', help='Set a project name for weights and biases')
    ] = 'thaw-slump-segmentation',
    wandb_name: Annotated[
        str, typer.Option('--wandb_name', '-wn', help='Set a run name for weights and biases')
    ] = None,
):
    # Create Sweep configuration
    cfg = yaml.load(config.open(), Loader=yaml_custom.SaneYAMLLoader)
    sweep_configuration = cfg['sweep']
    sweep_configuration['name'] = wandb_name

    # Start the sweep
    wandb_entity = 'ingmarnitze_team'
    sweep_id = wandb.sweep(sweep=sweep_configuration, project=wandb_project, entity=wandb_entity)

    print(f'Started Sweep at project {wandb_project} with ID: {sweep_id}')
    print('Please run the following command to run an agent:\n')
    print(f'    $ rye run thaw-slump-segmentation tune --wandb_project {wandb_project} --config {config} {sweep_id}')


def tune(
    sweep_id: Annotated[
        str,
        typer.Argument(
            help="The sweep id from weights and biases. If you don't know one, create a sweep with 'rye run thaw-slump-segmentation sweep' first."
        ),
    ],
    sweep_count: Annotated[int, typer.Option(help='Set how often this worker should train')] = 10,
    data_dir: Annotated[Path, typer.Option('--data_dir', help='Path to data processing dir')] = Path('data'),
    log_dir: Annotated[Path, typer.Option('--log_dir', help='Path to log dir')] = Path('logs'),
    config: Annotated[Path, typer.Option('--config', '-c', help='Specify run config to use.')] = Path('config.yml'),
    wandb_project: Annotated[
        str, typer.Option('--wandb_project', '-wp', help='Set a project name for weights and biases')
    ] = 'thaw-slump-segmentation',
):
    # Some features are disabled by default for tuning
    summary = False
    resume = None

    config = yaml.load(config.open(), Loader=yaml_custom.SaneYAMLLoader)

    def tune_run():
        wandb_entity = 'ingmarnitze_team'
        wandb.init(project=wandb_project, config=config, entity=wandb_entity)
        name = wandb.run.name
        print('\n\n=== Using Weights & Biases ===\n')
        print(f'Entitiy: {wandb_entity}')
        print(f'Project: {wandb_project}')
        print(f'Name: {name}')

        # Overwrite config with WandB config
        engine_config = deepcopy(config)
        engine_config['learning_rate'] = wandb.config['learning_rate']
        engine_config['model']['architecture'] = wandb.config['architecture']
        engine_config['model']['encoder'] = wandb.config['encoder']

        engine = Engine(engine_config, data_dir, name, log_dir, resume, summary)
        engine.run()

    wandb_entity = 'ingmarnitze_team'
    wandb.agent(sweep_id, function=tune_run, count=sweep_count, project=wandb_project, entity=wandb_entity)


def overwrite_config(self, key: str | list[str], wandb_key: str = None):
    if isinstance(key, str):
        # If key is matching, just use the key for both, only applicable if not nested
        wandb_key = wandb_key or key
        if self.config[key] == 'TUNE':
            self.config[key] = wandb.config[wandb_key]
    else:
        assert isinstance(key, list), f'Expect type of key either be str or list, got {type(key)}'
        if wandb_key is None:
            raise ValueError(f'Type of key is {type(key)}, expected wandb_key to be present but got None!')
        cfg = self.config
        # Recurse down to nested key
        for k in key[:-2]:
            cfg = cfg[k]
        k = key[-1]
        if cfg[k] == 'TUNE':
            cfg[k] = wandb.config[wandb_key]
