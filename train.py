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
from datetime import datetime
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from tqdm import tqdm

from deep_learning import Trainer
from deep_learning.models import get_model
from deep_learning.loss_functions import get_loss
from data_loading import get_loader, get_batch

from docopt import docopt
import yaml

def showexample(batch, pred, idx, filename):
    m = 0.02
    gridspec_kw = dict(left=m, right=1 - m, top=1 - m, bottom=m,
                       hspace=m, wspace=m)
    fig, axes = plt.subplots(2, 2, figsize=(8, 8), gridspec_kw=gridspec_kw)
    ((a1, a2), (a3, a4)) = axes
    heatmap_args = dict(cmap='coolwarm', vmin=0, vmax=1)

    batch_img, batch_target = batch
    batch_img = batch_img.to(torch.float)

    rgb = batch_img[idx].cpu().numpy()[[2, 1, 0]]
    a1.imshow(np.clip(rgb.transpose(1, 2, 0), 0, 1))
    a1.axis('off')
    a2.imshow(batch_target[idx, 0].cpu(), **heatmap_args)
    a2.axis('off')
    tcvis = batch_img[idx].cpu().numpy()[[4, 5, 6]]
    a3.imshow(np.clip(tcvis.transpose(1, 2, 0), 0, 1))
    a3.axis('off')
    a4.imshow(torch.sigmoid(pred[idx, 0]).cpu(), **heatmap_args)
    # a4.imshow((pred[idx, 0] > 0).cpu(), cmap='coolwarm', vmin=0, vmax=1)
    a4.axis('off')
    filename.parent.mkdir(exist_ok=True)
    plt.savefig(filename)
    plt.close()


if __name__ == "__main__":
    cli_args = docopt(__doc__, version="Usecase 2 Training Script 1.0")
    config = yaml.load(open(cli_args['--config']), Loader=yaml.SafeLoader)

    modelclass = get_model(config['model'])
    model = modelclass(config['input_channels'], 1, base_channels=config['modelscale'])

    # TODO: Resume from checkpoint

    trainer = Trainer(model)
    loss_fn = get_loss(config['loss_function'])
    trainer.loss_function = loss_fn.to(trainer.dev)

    lr = config['learning_rate']
    trainer.optimizer = torch.optim.Adam(trainer.model.parameters(), lr)

    if cli_args['--summary']:
        from torchsummary import summary
        summary(trainer.model, [(7, 256, 256)])
        sys.exit(0)

    batch_size = config['batchsize']
    train_loader = get_loader(config['train_data'], train=True, batch_size=batch_size)
    val_loader   = get_loader(config['val_data'], train=False, batch_size=batch_size)

    vis_batch = get_batch(config['visualization_tiles'])
    vis_imgs = vis_batch[0].to(trainer.dev)

    log_dir = Path('logs') / datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    log_dir.mkdir(exist_ok=False)
    checkpoints = log_dir / 'checkpoints'
    checkpoints.mkdir()

    EPOCHS = config['epochs']
    for epoch in range(EPOCHS):
        # Train epoch
        trainer.train_epoch(tqdm(train_loader))
        metrics = trainer.metrics.evaluate()
        logstr = f'Epoch {trainer.epoch:02d} - Train: ' \
               + ', '.join(f'{key}: {val:.2f}' for key, val in metrics.items())
        print(logstr)
        with (log_dir / 'metrics.txt').open('a+') as f:
            print(logstr, file=f)

        # Save model Checkpoint
        torch.save(trainer.model.state_dict(), checkpoints / f'{trainer.epoch:02d}.pt')

        # Val epoch
        trainer.val_epoch(val_loader)
        metrics = trainer.metrics.evaluate()
        logstr = f'Epoch {trainer.epoch:02d} - Val: ' \
               + ', '.join(f'{key}: {val:.2f}' for key, val in metrics.items())
        print(logstr)
        with (log_dir / 'metrics.txt').open('a+') as f:
            print(logstr, file=f)

        with torch.no_grad():
            pred = trainer.model(vis_imgs)
        for i, tile in enumerate(config['visualization_tiles']):
            filename = log_dir / tile / f'{trainer.epoch}.png'
            filename.parent.mkdir(exist_ok=True)
            showexample(vis_batch, pred, i, filename)
