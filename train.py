"""
Usecase 2 Training Script

Usage:
    train.py [options]

Options:
    -h --help          Show this screen
    --summary          Only print model summary and return (Requires the torchsummary package)
    --config=CONFIG    Specify run config to use [default: config.yml]
    --resume=CHKPT     Resume from the specified checkpoint [default: ]
                       Can be either a run-id (e.g. "2020-06-29_18-12-03") to select the last
                       checkpoint of that run, or a direct path to a checkpoint to be loaded.
                       Overrides the resume option in the config file if given.
"""
import sys
from datetime import datetime
from pathlib import Path
import shutil

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchsummary
from tqdm import tqdm

from deep_learning import get_model, get_loss, Metrics, Accuracy, Precision, Recall, F1, IoU
from deep_learning.utils import showexample, read_metrics_file, plot_metrics, plot_precision_recall
from data_loading import get_loader, get_filtered_loader, get_batch, make_sift_transform

import re

from docopt import docopt
import yaml


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


def get_dataloader(name):
    if name in dataset_cache:
        return dataset_cache[name]

    if name in config['datasets']:
        ds_config = config['datasets'][name]
        if 'batch_size' not in ds_config:
            ds_config['batch_size'] = config['batch_size']
        ds_config['num_workers'] = config['data_threads']
        dataset_cache[name] = get_loader(channels=config['channels_used'], **ds_config)
    else:
        func, arg = re.search(r'(\w+)\((\w+)\)', name).groups()
        if func == 'slump_tiles':
            ds_config = config['datasets'][arg]
            if 'batch_size' not in ds_config:
                ds_config['batch_size'] = config['batch_size']
            ds_config['num_workers'] = config['data_threads']
            dataset_cache[name] = get_filtered_loader(**ds_config)
        elif func == 'sift':
            ds_config = dict(**config['datasets'][arg])
            ds_config['names'] = ['data']
            if 'batch_size' not in ds_config:
                ds_config['batch_size'] = config['batch_size']
            ds_config['num_workers'] = config['data_threads']
            transform = make_sift_transform(config['channels_used'])
            dataset_cache[name] = get_loader(channels=config['channels_used'], transform=transform, **ds_config)

    return dataset_cache[name]


def with_edges(mask):
    with torch.no_grad():
        avg = F.avg_pool2d(mask, 3, padding=1, stride=1)
        edge = mask != avg
        mask[edge] = 2
    return mask


def make_sift_classifier(feature_shape, sift_shape, meta_shape):
    global sift_classifier, sift_opt
    from deep_learning.models import Merger

    params = (feature_shape[-1], sift_shape[-1], meta_shape[-1])
    print(f'Initializing a Merger-Block with feature dimensions {params}')
    sift_classifier = Merger(*params).to(dev)
    sift_opt = torch.optim.SGD(sift_classifier.parameters(), config['learning_rate'], momentum=0.99, nesterov=True)


def sift_train_epoch(train_loader):
    global epoch, board_idx, sift_classifier
    epoch += 1
    progress = tqdm(train_loader)
    metrics.reset()
    model.train(True)
    for iteration, (img, sift, meta) in enumerate(progress):
        img = img.to(dev, torch.float)
        sift = sift.to(dev, torch.float)
        meta = meta.to(dev, torch.float)

        features = model.encode(img)
        features = features.reshape(features.shape[0], 1, -1)

        if sift_classifier is None:
            make_sift_classifier(features.shape, sift.shape, meta.shape)

        res_pos, res_neg = sift_classifier(features, sift, meta)
        loss_pos = torch.mean(-F.logsigmoid(res_pos))
        loss_neg = torch.mean(-F.logsigmoid(-res_neg))

        opt.zero_grad()
        sift_opt.zero_grad()

        loss = loss_pos + loss_neg
        loss.backward()

        opt.step()
        sift_opt.step()

        with torch.no_grad():
            pos_acc = (res_pos > 0).float().mean()
            neg_acc = (res_neg < 0).float().mean()
            metrics.step(SIFTLoss=loss, PosAcc=pos_acc, NegAcc=neg_acc)
            board_idx += img.shape[0]

            if (iteration+1) % 50 == 0:
                metrics_vals = metrics.evaluate()
                logstr = ', '.join(f'{key}: {val:.2f}' for key, val in metrics_vals.items())
                progress.set_description(logstr)
                with (log_dir / 'metrics.txt').open('a+') as f:
                    print(logstr, file=f)
                for key, val in metrics_vals.items():
                    trn_writer.add_scalar(key, val, board_idx)
                trn_writer.flush()
    # Save model Checkpoint
    torch.save(model.state_dict(), checkpoints / f'{epoch:02d}.pt')


def train_epoch(train_loader):
    global epoch, board_idx
    epoch += 1
    progress = tqdm(train_loader)
    model.train(True)
    for iteration, (img, target) in enumerate(progress):
        board_idx += img.shape[0]
        lr_factor = min(board_idx / 50000, 1, 2 ** ((50000 - board_idx) / 50000))
        for i in range(len(opt.param_groups)):
            lr = config['learning_rate'] * lr_factor
            opt.param_groups[i]['lr'] = lr

        img = img.to(dev, torch.float)
        target = target.to(dev, torch.long, non_blocking=True)

        opt.zero_grad()
        y_hat = model(img)
        loss = loss_function(y_hat, target)
        loss.backward()
        opt.step()

        with torch.no_grad():
            metrics.step(y_hat, target, Loss=loss.detach(), lr=lr)
            if (iteration+1) % 10 == 0:
                metrics_vals = metrics.evaluate()
                logstr = ', '.join(f'{key}: {val:.2f}' for key, val in metrics_vals.items())
                progress.set_description(logstr)
                with (log_dir / 'metrics.txt').open('a+') as f:
                    print(logstr, file=f)
                for key, val in metrics_vals.items():
                    trn_writer.add_scalar(key, val, board_idx)
                    safe_append(trn_metrics, key, val)
                safe_append(trn_metrics, 'step', board_idx)
                trn_writer.flush()

    # Save model Checkpoint
    torch.save(model.state_dict(), checkpoints / f'{epoch:02d}.pt')


def val_epoch(val_loader):
    metrics.reset()
    model.train(False)
    with torch.no_grad():
        for iteration, (img, target) in enumerate(val_loader):
            img = img.to(dev, torch.float)
            target = target.to(dev, torch.long, non_blocking=True)
            y_hat = model(img)

            loss = loss_function(y_hat, target)
            metrics.step(y_hat, target, Loss=loss.detach())

        m = metrics.evaluate()
        logstr = f'Epoch {epoch:02d} - Val: ' \
               + ', '.join(f'{key}: {val:.2f}' for key, val in m.items())
        print(logstr)
        with (log_dir / 'metrics.txt').open('a+') as f:
            print(logstr, file=f)
        for key, val in m.items():
            val_writer.add_scalar(key, val, board_idx)
            safe_append(val_metrics, key, val)
        safe_append(val_metrics, 'step', board_idx)
        val_writer.flush()


def log_images():
    with torch.no_grad():
        vis_predictions.append(model(vis_imgs).cpu())
    (log_dir / 'tile_predictions').mkdir(exist_ok=True)
    for i, tile in enumerate(config['visualization_tiles']):
        filename = log_dir / 'tile_predictions' / f'{tile}.jpg'
        showexample(vis_batch, vis_predictions, i, filename, config['channels_used'], val_writer)

        outdir = log_dir / 'metrics_plots'
        outdir.mkdir(exist_ok=True)
        metrics_file = log_dir / 'metrics.txt'
        plot_metrics(trn_metrics, val_metrics, outdir=outdir)
        plot_precision_recall(trn_metrics, val_metrics, outdir=outdir)
    val_writer.flush()


if __name__ == "__main__":
    cli_args = docopt(__doc__, version="Usecase 2 Training Script 1.0")
    config_file = Path(cli_args['--config'])
    config = yaml.load(config_file.open(), Loader=yaml.SafeLoader)
    config['model_args']['input_channels'] = len(config['channels_used'])

    modelclass = get_model(config['model'])
    model = modelclass(**config['model_args'])

    if cli_args['--resume']:
        config['resume'] = cli_args['--resume']

    if 'resume' in config and config['resume']:
        checkpoint = Path(config['resume'])
        if not checkpoint.exists():
            raise ValueError(f"There is no Checkpoint at {config['resume']} to resume from!")
        if checkpoint.is_dir():
            # Load last checkpoint in run dir
            ckpt_nums = [int(ckpt.stem) for ckpt in checkpoint.glob('checkpoints/*.pt')]
            last_ckpt = max(ckpt_nums)
            config['resume'] = checkpoint / 'checkpoints' / f'{last_ckpt:02d}.pt'
        print(f"Resuming training from checkpoint {config['resume']}")
        model.load_state_dict(torch.load(config['resume']))

    dev = torch.device("cpu") if not torch.cuda.is_available() else torch.device("cuda")
    print(f'Training on {dev} device')
    model = model.to(dev)

    board_idx = 0
    opt = torch.optim.AdamW(model.parameters(), lr=config['learning_rate'])

    epoch = 0
    sift_classifier = None
    sift_opt = None

    metrics = Metrics(Accuracy, Precision, Recall, F1, IoU)

    if cli_args['--summary']:
        from torchsummary import summary
        summary(model, [(len(config['channels_used']), 256, 256)])
        sys.exit(0)

    dataset_cache = {}

    vis_batch = list(get_batch(config['visualization_tiles'], config['channels_used']))
    vis_batch[1] = with_edges(vis_batch[1].to(torch.long))
    vis_imgs = vis_batch[0].to(dev)
    vis_predictions = []

    log_dir = Path('logs') / datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    log_dir.mkdir(exist_ok=False)

    # Write the config YML to the run-folder
    with open(log_dir / 'config.yml', 'w') as f:
        yaml.dump(config, f)

    checkpoints = log_dir / 'checkpoints'
    checkpoints.mkdir()

    # Tensorboard initialization
    from torch.utils.tensorboard import SummaryWriter
    trn_writer = SummaryWriter(log_dir / 'train')
    val_writer = SummaryWriter(log_dir / 'val')
    trn_metrics = {}
    val_metrics = {}


    for phase in config['schedule']:
        print(f'Starting phase "{phase["phase"]}"')
        with (log_dir / 'metrics.txt').open('a+') as f:
            print(f'Phase {phase["phase"]}', file=f)
        for epoch in range(phase['epochs']):
            # Epoch setup
            loss_function = get_loss(scoped_get('loss_function', phase, config))
            try:
                loss_function = loss_function.to(dev)
            except:
                # Loss function is functional style, so it lives on the GPU already... :)
                pass

            datasets_config = scoped_get('datasets', phase, config)

            for step in phase['steps']:
                if type(step) is dict:
                    assert len(step) == 1
                    (command, key), = step.items()
                else:
                    command = step

                if command == 'train_on':
                    # Training step
                    data_loader = get_dataloader(key)
                    train_epoch(data_loader)
                if command == 'sift_train_on':
                    # SIFT-Pretraining Training step
                    data_loader = get_dataloader(key)
                    sift_train_epoch(data_loader)
                elif command == 'validate_on':
                    # Validation step
                    data_loader = get_dataloader(key)
                    val_epoch(data_loader)
                elif command == 'log_images':
                    log_images()
