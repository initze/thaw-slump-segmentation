import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import torch

def imageize(tensor):
    return np.clip(tensor.cpu().numpy().transpose(1, 2, 0), 0, 1)


def showexample(batch, preds, idx, filename, writer=None):
    ## First plot
    ROWS = 5
    m = 0.02
    gridspec_kw = dict(left=m, right=1 - m, top=1 - m, bottom=m,
                       hspace=0.12, wspace=m)
    N = 1 + int(np.ceil(len(preds) / ROWS))
    fig, ax = plt.subplots(ROWS, N, figsize=(3*N, 3*ROWS), gridspec_kw=gridspec_kw)
    ax = ax.T.reshape(-1)
    heatmap_args = dict(cmap='coolwarm', vmin=0, vmax=1)
    heatmap_dem = dict(cmap='RdBu', vmin=0, vmax=0.9)

    batch_img, batch_target = batch
    batch_img = batch_img.to(torch.float)

    # Clear all axes
    for axis in ax:
        axis.imshow(np.ones([1, 1, 3]))
        axis.axis('off')

    rgb = imageize(batch_img[idx, [3, 2, 1]])
    ndvi = imageize(batch_img[idx, [4, 4, 4]])
    tcvis = imageize(batch_img[idx, [5, 6, 7]])
    dem = batch_img[idx, [8, 8, 8]].cpu().numpy()
    target = batch_target[idx, 0].cpu()

    ax[0].imshow(rgb)
    ax[0].set_title('B-G-NIR')
    ax[1].imshow(ndvi)
    ax[1].set_title('NDVI')
    ax[2].imshow(tcvis)
    ax[2].set_title('TCVis')
    ax[3].imshow(np.clip(dem[0], 0, 1), **heatmap_dem)
    ax[3].set_title('DEM')
    ax[4].imshow(target , **heatmap_args)
    ax[4].set_title('Target')

    for i, pred in enumerate(preds):
        ax[i+ROWS].imshow(pred[idx, 0].cpu(), **heatmap_args)
        ax[i+ROWS].set_title(f'Epoch {i} Prediction')

    filename.parent.mkdir(exist_ok=True)
    plt.savefig(filename, bbox_inches='tight')
    plt.close()
    if writer is not None:
        fig, ax = plt.subplots(1, 3, figsize=(9, 4), gridspec_kw=gridspec_kw)
        ax[0].imshow(rgb)
        ax[0].set_title('B-G-NIR')
        ax[1].imshow(batch_target[idx, 0].cpu(), **heatmap_args)
        ax[1].set_title('Ground Truth')
        ax[2].imshow(preds[-1][idx, 0].cpu(), **heatmap_args)
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


def plot_metrics(df, outdir='.'):
    df_val = df.query('val_type == "Val"')
    df_train = df.query('val_type == "Train"')
    for metric in ['accuracy', 'precision', 'recall', 'f1', 'iou', 'loss']:
        outfile = os.path.join(outdir, f'{metric}.png')
        ax = plt.subplot()
        fig = ax.get_figure()

        ax.plot(df_train['epoch'], df_train[metric])
        ax.plot(df_val['epoch'], df_val[metric])
        ax.set_xlabel('Epoch')
        ax.set_ylabel(metric)
        ax.legend(['Train', 'Val'])
        ax.grid()
        fig.savefig(outfile, bbox_inches='tight')
        plt.close()


def plot_precision_recall(df, outdir='.'):
    df_val = df.query('val_type == "Val"')
    df_train = df.query('val_type == "Train"')

    fig = plt.figure(figsize=(10, 3))
    ax1, ax2 = fig.subplots(1, 2)

    ax1.plot(df_train['epoch'], df_train['precision'])
    ax1.plot(df_train['epoch'], df_train['recall'])
    ax1.set_title('Train')
    ax2.plot(df_val['epoch'], df_val['precision'])
    ax2.plot(df_val['epoch'], df_val['recall'])
    ax2.set_title('Val')

    for ax in [ax1, ax2]:
        ax.set_xlabel('Epoch')
        ax.legend(['precision', 'recall'])
        ax.grid()

    outfile = os.path.join(outdir, f'precision_recall.png')
    fig.savefig(outfile)
    fig.clear()
