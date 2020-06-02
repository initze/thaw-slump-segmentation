"""
Usecase 2 Training Script

Usage:
    train.py [options]

Options:
    -h --help          Show this screen
    --summary          Only print model summary and return (Requires the torchsummary package)
    --epochs=EPOCHS    Number of epochs to train [default: 20]
    --batchsize=BS     Specify batch size [default: 8]
    --modelscale=MS    Model feature space scale [default: 32]
    --augment=bool     Whether to use data augmentation [default: True]
    --resume=CHKPT     If a checkpoint path is given, resume training from a given checkpoint.
                       Otherwise, training starts from scratch
    --lr=LR            Learning rate to use [default: 1e-4]
    --posweight=PW     Weighting for positive examples [default: 150]
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
from deep_learning.models import UNet
from data_loading import get_loaders

from docopt import docopt


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
    args = docopt(__doc__, version="Usecase 2 Training Script 1.0")

    modelscale = int(args['--modelscale'])
    model = UNet(7, 1, base_channels=modelscale)
    if args['--resume']:
        model.load_state_dict(torch.load(args['--resume']))

    trainer = Trainer(model)
    posweight = float(args['--posweight'])
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=posweight * torch.ones([]))
    trainer.loss_function = loss_fn.to(trainer.dev)

    lr = float(args['--lr'])
    trainer.optimizer = torch.optim.Adam(trainer.model.parameters(), lr)

    if args['--summary']:
        from torchsummary import summary
        summary(trainer.model, [(7, 256, 256)])
        sys.exit(0)

    batch_size = int(args['--batchsize'])
    assert args['--augment'] == 'True' or args['--augment'] == 'False'
    augment = args['--augment'] == 'True'
    train_loader, val_loader = get_loaders(batch_size=batch_size, augment=augment)

    vis_tiles = [
        '20190727_160426_104e_3B_AnalyticMS_SR_06_31',
        '20190727_160426_104e_3B_AnalyticMS_SR_07_29',
        '20190727_160426_104e_3B_AnalyticMS_SR_12_16',
        '20190727_160426_104e_3B_AnalyticMS_SR_13_18'
    ]
    val_names = [n.stem for n, *_ in val_loader.dataset.index]
    vis_idx = [val_names.index(tile) for tile in vis_tiles]
    vis_batch = list(zip(*[val_loader.dataset[i] for i in vis_idx]))
    vis_batch = [torch.stack(i, dim=0) for i in vis_batch]
    vis_imgs = vis_batch[0].to(trainer.dev)

    log_dir = Path('logs') / datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    log_dir.mkdir(exist_ok=False)
    checkpoints = log_dir / 'checkpoints'
    checkpoints.mkdir()

    EPOCHS = int(args['--epochs'])

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
        for i, tile in enumerate(vis_tiles):
            filename = log_dir / tile / f'{trainer.epoch}.png'
            filename.parent.mkdir(exist_ok=True)
            showexample(vis_batch, pred, i, filename)
