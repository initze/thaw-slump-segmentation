import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import torch
from matplotlib.ticker import MaxNLocator


def imageize(tensor):
    return np.clip(tensor.cpu().numpy().transpose(1, 2, 0), 0, 1)


def get_channel_offset(data_sources, target_source):
    offset = 0
    for src in data_sources:
        if src.name == target_source:
            break
        offset += src.channels
    return offset


def showexample(data, preds, filename, data_sources, writer=None):
    # First plot
    ROWS = 6
    m = 0.02
    gridspec_kw = dict(left=m, right=1 - m, top=1 - m, bottom=m,
                       hspace=0.12, wspace=m)
    N = 1 + int(np.ceil(len(preds) / ROWS))
    fig, ax = plt.subplots(ROWS, N, figsize=(3*N, 3*ROWS), gridspec_kw=gridspec_kw)
    ax = ax.T.reshape(-1)

    heatmap_args = dict(cmap=plt.cm.Greys_r, vmin=0, vmax=1)

    img, target = data
    img = img.to(torch.float)

    # Clear all axes
    for axis in ax:
        axis.imshow(np.ones([1, 1, 3]))
        axis.axis('off')

    ds_names = set(src.name for src in data_sources)
    if 'planet' in ds_names:
        offset = get_channel_offset(data_sources, 'planet')
        b, g, r, nir = np.arange(4) + offset
        bgnir = imageize(img[[nir, b, g]])
        ax[0].imshow(bgnir)
        ax[0].set_title('NIR-R-G')
    if 'ndvi' in ds_names:
        c = get_channel_offset(data_sources, 'ndvi')
        ndvi = img[[c, c, c]].cpu().numpy()
        ax[1].imshow(ndvi[0], cmap=plt.cm.RdYlGn, vmin=0, vmax=1)
        ax[1].set_title('NDVI')
    if 'tcvis' in ds_names:
        offset = get_channel_offset(data_sources, 'tcvis')
        r, g, b = np.arange(3) + offset
        tcvis = imageize(img[[r, g, b]])
        ax[2].imshow(tcvis)
        ax[2].set_title('TCVIS')
    if 'relative_elevation' in ds_names:
        c = get_channel_offset(data_sources, 'relative_elevation')
        dem = img[[c, c, c]].cpu().numpy()
        ax[3].imshow(np.clip(dem[0], 0, 1), cmap='RdBu', vmin=0, vmax=1)
        ax[3].set_title('DEM')
    if 'slope' in ds_names:
        c = get_channel_offset(data_sources, 'slope')
        # TODO: loading predominantly nan values
        slope = img[[c, c, c]].cpu().numpy()
        ax[4].imshow(np.clip(slope[0], 0, 1), cmap='Reds', vmin=0, vmax=1)
        ax[4].set_title('Slope')

    ax[5].imshow(target[0], **heatmap_args)
    ax[5].set_title('Target')

    for i, pred in enumerate(preds):
        p = pred.argmax(dim=0).cpu()
        ax[i+ROWS].imshow(p, **heatmap_args)
        ax[i+ROWS].set_title(f'Epoch {i+1} Prediction')

    filename.parent.mkdir(exist_ok=True)
    plt.savefig(filename, bbox_inches='tight')
    plt.close()
    if writer is not None:
        fig, ax = plt.subplots(1, 3, figsize=(9, 4), gridspec_kw=gridspec_kw)
        if 'planet' in ds_names:
            offset = get_channel_offset(data_sources, 'planet')
            b, g, r, nir = np.arange(4) + offset
            bgnir = imageize(img[[nir, b, g]])
            ax[0].imshow(bgnir)
            ax[0].set_title('NIR-R-G')
        ax[1].imshow(target[0].cpu(), **heatmap_args)
        ax[1].set_title('Ground Truth')

        pred = preds[-1].argmax(dim=0)
        ax[2].imshow(pred, **heatmap_args)
        ax[2].set_title('Prediction')
        for axis in ax:
            axis.axis('off')
        writer.add_figure(filename.stem, fig, len(preds))


def read_metrics_file(file_path):
    with open(file_path) as src:
        lines = src.readlines()
    data = []
    for line in lines:
        if line.startswith('Phase'):
            continue
        epoch, remain = line.replace(' ', '').replace('\n', '').split('-')
        val_type = remain.split(',')[0].split(':')[0].strip()
        vals = remain.replace('Train:', '').replace('Val:', '').split(',')
        acc_vals = [float(v.split(':')[1]) for v in vals]

        data.append([int(epoch.replace('Epoch', '')), str(val_type), *acc_vals])

    df = pd.DataFrame(columns=['epoch', 'val_type', 'accuracy', 'precision', 'recall', 'f1', 'iou', 'loss'],
                      data=data)
    return df


def plot_metrics(train_metrics, val_metrics, outdir='.'):
    metrics = (set(train_metrics) | set(val_metrics)) - set(['step', 'epoch'])
    for metric in metrics:
        outfile = os.path.join(outdir, f'{metric}.png')
        ax = plt.subplot()
        fig = ax.get_figure()

        if metric in train_metrics:
            ax.plot(train_metrics['epoch'], train_metrics[metric], c='C0', label='Train')
        if metric in val_metrics:
            ax.plot(val_metrics['epoch'], val_metrics[metric], c='C1', label='Val')
        ax.set_xlabel('Epoch')
        ax.get_xaxis().set_major_locator(MaxNLocator(integer=True))
        ax.set_ylabel(metric)
        ax.legend()
        ax.grid()
        fig.savefig(outfile, bbox_inches='tight')
        plt.close()


def plot_precision_recall(train_metrics, val_metrics, outdir='.'):
    fig = plt.figure(figsize=(10, 3))
    ax1, ax2 = fig.subplots(1, 2)

    ax1.plot(train_metrics['epoch'], train_metrics['Precision'])
    ax1.plot(train_metrics['epoch'], train_metrics['Recall'])
    ax1.set_title('Train')
    ax2.plot(val_metrics['epoch'], val_metrics['Precision'])
    ax2.plot(val_metrics['epoch'], val_metrics['Recall'])
    ax2.set_title('Val')

    for ax in [ax1, ax2]:
        ax.set_xlabel('Epoch')
        ax.get_xaxis().set_major_locator(MaxNLocator(integer=True))
        ax.legend(['Precision', 'Recall'])
        ax.grid()

    fig.tight_layout()

    outfile = os.path.join(outdir, f'precision_recall.png')
    fig.savefig(outfile)
    fig.clear()