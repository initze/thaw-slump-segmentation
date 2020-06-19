import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import torch

def showexample(batch, pred, idx, filename):
    m = 0.02
    gridspec_kw = dict(left=m, right=1 - m, top=1 - m, bottom=m,
                       hspace=m, wspace=m)
    fig, axes = plt.subplots(2, 2, figsize=(8, 8), gridspec_kw=gridspec_kw)
    ((a1, a2), (a3, a4)) = axes
    heatmap_args = dict(cmap='coolwarm', vmin=0, vmax=1)

    batch_img, batch_target = batch
    batch_img = batch_img.to(torch.float)

    rgb = batch_img[idx].cpu().numpy()[[4, 3, 2]]
    a1.imshow(np.clip(rgb.transpose(1, 2, 0), 0, 1))
    a1.axis('off')
    a2.imshow(batch_target[idx, 0].cpu(), **heatmap_args)
    a2.axis('off')
    tcvis = batch_img[idx].cpu().numpy()[[5, 6, 7]]
    a3.imshow(np.clip(tcvis.transpose(1, 2, 0), 0, 1))
    a3.axis('off')
    a4.imshow(torch.sigmoid(pred[idx, 0]).cpu(), **heatmap_args)
    a4.axis('off')
    filename.parent.mkdir(exist_ok=True)
    plt.savefig(filename)
    plt.close()


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

    df = pd.DataFrame(columns=['epoch', 'val_type', 'accuracy', 'precision', 'recall', 'f1', 'loss'],
                      data=data)
    return df


def plot_metrics(df, outdir='.'):
    df_val = df.query('val_type == "Val"')
    df_train = df.query('val_type == "Train"')
    for metric in ['accuracy', 'precision', 'recall', 'f1', 'loss']:
        outfile = os.path.join(outdir, f'{metric}.png')
        ax = plt.subplot()
        fig = ax.get_figure()

        ax.plot(df_train['epoch'], df_train[metric])
        ax.plot(df_val['epoch'], df_val[metric])
        ax.set_xlabel('Epoch')
        ax.set_ylabel(metric)
        ax.legend(['Train', 'Val'])
        ax.grid()
        fig.savefig(outfile)
        fig.clear()


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