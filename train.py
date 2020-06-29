"""
Usecase 2 Training Script

Usage:
    train.py [options]

Options:
    -h --help          Show this screen
    --summary          Only print model summary and return (Requires the torchsummary package)
    --config=CONFIG    Specify run config to use [default: config.yml]
"""
import sys
import distutils.util
from datetime import datetime
from pathlib import Path
import shutil

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from tqdm import tqdm

from deep_learning.models import get_model
from deep_learning.loss_functions import get_loss
from deep_learning.metrics import Metrics, Accuracy, Precision, Recall, F1
from data_loading import get_loader, get_filtered_loader, get_batch

import re

from docopt import docopt
import yaml


def showexample(idx, filename):
    ## First plot
    ROWS = 4
    m = 0.02
    gridspec_kw = dict(left=m, right=1 - m, top=1 - m, bottom=m,
                       hspace=0.12, wspace=m)
    N = 1 + int(np.ceil(len(val_predictions) / ROWS))
    fig, ax = plt.subplots(ROWS, N, figsize=(3*N, 3*ROWS), gridspec_kw=gridspec_kw)
    ax = ax.T.reshape(-1)
    heatmap_args = dict(cmap='coolwarm', vmin=0, vmax=1)

    batch_img, batch_target = vis_batch
    batch_img = batch_img.to(torch.float)

    # Clear all axes
    for axis in ax:
        axis.imshow(np.ones([1, 1, 3]))
        axis.axis('off')

    rgb = batch_img[idx, [3, 2, 1]].cpu().numpy()
    tcvis = batch_img[idx, [4, 5, 6]].cpu().numpy()
    dem = batch_img[idx, [7, 7, 8]].cpu().numpy()
    dem[0] = -dem[0] # Red is negative, green is positive

    ax[0].imshow(np.clip(rgb.transpose(1, 2, 0), 0, 1))
    ax[0].set_title('B-G-NIR')
    ax[1].imshow(np.clip(tcvis.transpose(1, 2, 0), 0, 1))
    ax[1].set_title('TCVis')
    ax[2].imshow(np.clip(dem.transpose(1, 2, 0), 0, 1))
    ax[2].set_title('DEM')
    ax[3].imshow(batch_target[idx, 0].cpu(), **heatmap_args)
    ax[3].set_title('Target')

    for i, pred in enumerate(val_predictions):
        ax[i+ROWS].imshow(pred[idx, 0].cpu(), **heatmap_args)
        ax[i+ROWS].set_title(f'Epoch {i} Prediction')

    filename.parent.mkdir(exist_ok=True)
    plt.savefig(filename, bbox_inches='tight')
    plt.close()


def scoped_get(key, *scopestack):
    for scope in scopestack:
        value = scope.get(key)
        if value is not None:
            return value
    raise ValueError(f'Could not find "{key}" in any scope.')


def get_dataloader(name):
    if name in dataset_cache:
        return dataset_cache[name]
    if name in config['datasets']:
        ds_config = config['datasets'][name]
        if 'batch_size' not in ds_config:
            ds_config['batch_size'] = config['batch_size']
        dataset_cache[name] = get_loader(**ds_config)
        return dataset_cache[name]
    else:
        func, arg = re.search(r'(\w+)\((\w+)\)', name).groups()
        return COMMANDS[func](arg)
    return dataset_cache[name]


def train(dataset):
    global epoch
    # Training step
    data_loader = get_dataloader(dataset)

    epoch += 1
    model.train(True)
    for iteration, (img, target) in enumerate(tqdm(data_loader)):
        img = img.to(dev, torch.float)
        target = target.to(dev, torch.float, non_blocking=True)

        opt.zero_grad()
        y_hat = model(img)

        loss = loss_function(y_hat, target)
        loss.backward()
        opt.step()

        with torch.no_grad():
            metrics.step(y_hat, target, Loss=loss.detach())

    metrics_vals = metrics.evaluate()
    if 'Iterations' not in train_metrics:
        train_metrics['Iterations'] = [iteration + 1]
    else:
        train_metrics['Iterations'].append(train_metrics['Iterations'][-1] + iteration + 1)
    for k, v in metrics_vals.items():
        if k not in train_metrics:
            train_metrics[k] = []
        train_metrics[k].append(v)
    logstr = f'Epoch {epoch:02d} - Train: ' \
           + ', '.join(f'{key}: {val:.2f}' for key, val in metrics_vals.items())
    print(logstr)
    with (log_dir / 'metrics.txt').open('a+') as f:
        print(logstr, file=f)
    # Save model Checkpoint
    torch.save(model.state_dict(), checkpoints / f'{epoch:02d}.pt')


def val(dataset):
    # Validation step
    data_loader = get_dataloader(dataset)

    model.train(False)
    with torch.no_grad():
        for iteration, (img, target) in enumerate(data_loader):
            img = img.to(dev, torch.float)
            target = target.to(dev, torch.float, non_blocking=True)
            y_hat = model(img)

            loss = loss_function(y_hat, target)
            metrics.step(y_hat, target, Loss=loss.detach())

    metrics_vals = metrics.evaluate()
    if 'Iterations' not in val_metrics:
        val_metrics['Iterations'] = []
    val_metrics['Iterations'].append(train_metrics['Iterations'][-1])
    for k, v in metrics_vals.items():
        if k not in val_metrics:
            val_metrics[k] = []
        val_metrics[k].append(v)

    logstr = f'Epoch {epoch:02d} - Val: ' \
           + ', '.join(f'{key}: {val:.2f}' for key, val in metrics_vals.items())
    print(logstr)
    with (log_dir / 'metrics.txt').open('a+') as f:
        print(logstr, file=f)


def log_images():
    # Training Loss Curves
    (log_dir / 'metrics').mkdir(exist_ok=True)
    for metric in train_metrics:
        if metric == 'Iterations':
            continue
        fig, ax = plt.subplots()
        ax.set_title(metric)
        plt.plot(train_metrics['Iterations'], train_metrics[metric], '--', label=f'Train {metric}')
        plt.plot(val_metrics['Iterations'], val_metrics[metric], label=f'Val {metric}')
        plt.ylabel(metric)
        plt.xlabel('Iterations')
        plt.legend()
        plt.savefig(log_dir / 'metrics' / f'{metric}.jpg', bbox_inches='tight')
        plt.close()


    (log_dir / 'vis_tiles').mkdir(exist_ok=True)
    # Prediction Images
    model.train(False)
    with torch.no_grad():
        val_predictions.append(model(vis_imgs).cpu())
    for i, tile in enumerate(config['visualization_tiles']):
        filename = log_dir / 'vis_tiles' / f"{tile.replace('/', '_')}.jpg"
        filename.parent.mkdir(exist_ok=True)
        showexample(i, filename)


def slump_tiles(dataset):
    name = f'slump_tiles({dataset})'
    ds_config = config['datasets'][dataset]
    if 'batch_size' not in ds_config:
        ds_config['batch_size'] = config['batch_size']
    dataset_cache[name] = get_filtered_loader(**ds_config)
    return dataset_cache[name]


COMMANDS = dict(
    train_on=train,
    validate_on=val,
    log_images=log_images,
    slump_tiles=slump_tiles,
)


if __name__ == "__main__":
    cli_args = docopt(__doc__, version="Usecase 2 Training Script 1.0")
    config_file = Path(cli_args['--config'])
    config = yaml.load(config_file.open(), Loader=yaml.SafeLoader)

    modelclass = get_model(config['model'])
    model = modelclass(config['input_channels'], 1, base_channels=config['modelscale'])

    cuda = True if torch.cuda.is_available() else False
    dev = torch.device("cpu") if not cuda else torch.device("cuda")
    print(f'Training on {dev} device')
    model = model.to(dev)

    opt = torch.optim.Adam(model.parameters(), lr=1e-3)

    epoch = 0
    train_metrics = {}
    val_metrics = {}
    val_predictions = []

    loss_function = F.binary_cross_entropy_with_logits
    metrics = Metrics(Accuracy, Precision, Recall, F1)

    lr = config['learning_rate']
    optimizer = torch.optim.Adam(model.parameters(), lr)

    if cli_args['--summary']:
        from torchsummary import summary
        summary(model, [(7, 256, 256)])
        sys.exit(0)

    dataset_cache = {}

    vis_batch = get_batch(config['visualization_tiles'])
    vis_imgs = vis_batch[0].to(dev)

    log_dir = Path('logs') / datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    log_dir.mkdir(exist_ok=False)

    shutil.copy(config_file, log_dir / 'config.yml')

    checkpoints = log_dir / 'checkpoints'
    checkpoints.mkdir()

    for phase in config['schedule']:
        print(f'Starting phase "{phase["phase"]}"')
        with (log_dir / 'metrics.txt').open('a+') as f:
            print(f'Phase {phase["phase"]}', file=f)
        for _ in range(phase['epochs']):
            # Epoch setup
            loss_fn = get_loss(scoped_get('loss_function', phase, config))
            loss_function = loss_fn.to(dev)

            datasets_config = scoped_get('datasets', phase, config)

            for step in phase['steps']:
                if type(step) is dict:
                    assert len(step) == 1
                    (command, arg), = step.items()
                    COMMANDS[command](arg)
                else:
                    command = step
                    COMMANDS[command]()
