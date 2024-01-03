# Copyright (c) Ingmar Nitze and Konrad Heidler

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import matplotlib.pyplot as plt
from einops import rearrange
from skimage.measure import find_contours
import torch
import numpy as np
import os
import pandas as pd
from PIL import Image, ImageDraw
import wandb
from matplotlib.ticker import MaxNLocator
from matplotlib.colors import LinearSegmentedColormap


def imageize(tensor):
    return np.clip(tensor.cpu().numpy().transpose(1, 2, 0), 0, 1)


def get_channel_offset(data_sources, target_source):
    offset = 0
    for src in data_sources:
        if src.name == target_source:
            break
        offset += src.channels
    return offset


def grid(imgs, padding=8):
    rows = []
    for row in imgs:
        row_elements = []
        for img in row:
            if img.ndim == 2:
                img = np.stack([img] * 3, axis=-1)
            row_elements.append(np.pad(img, [(8, 8), (8, 8), (0, 0)]))
        rows.append(np.concatenate(row_elements, axis=1))
    return np.concatenate(rows, axis=0)


def showexample(data, preds, tag, step):
    # First plot
    img, mask = data
    img  = img.numpy()
    mask = mask.numpy()

    mask = np.where(mask == 255, np.uint8(127),
           np.where(mask == 1,   np.uint8(255),
                                 np.uint8(  0)))

    img = rearrange(img[[3,2,7]], 'C H W -> H W C')
    img = np.clip(2 * 255 * img, 0, 255).astype(np.uint8)
    pred = (255 * torch.sigmoid(preds[-1]).numpy()).astype(np.uint8)

    combined = Image.fromarray(grid([[img, pred, mask]]))
    wandb.log({f'imgs/{tag}': wandb.Image(combined)}, step=step)


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


def log_image(tile, data, epoch, log_dir, image_bands=[3,2,1]):
    y_max = max(d['y1'] for d in data)
    x_max = max(d['x1'] for d in data)

    rgb    = np.zeros([y_max, x_max, 3], dtype=np.uint8)
    target = np.zeros([y_max, x_max, 1], dtype=np.uint8)
    pred   = np.zeros([y_max, x_max, 1], dtype=np.uint8)
    for patch in data:
        y0, x0, y1, x1 = [patch[k] for k in ['y0', 'x0', 'y1', 'x1']]
        patch_rgb = patch['Image'][image_bands]
        patch_rgb = np.clip(255 * patch_rgb, 0, 255).astype(np.uint8)
        patch_target = np.clip(255 * patch['Target'], 0, 255).astype(np.uint8)
        patch_pred = np.clip(255 * patch['Prediction'], 0, 255).astype(np.uint8)

        rgb[y0:y1, x0:x1]    = rearrange(patch_rgb, 'C H W -> H W C')
        # check for dimensions of target and add dimenstion if necessary
        if patch_target.ndim == 3:
            target[y0:y1, x0:x1] = rearrange(patch_target, 'C H W -> H W C')
        else:
            target[y0:y1, x0:x1] = np.expand_dims(patch_target, axis=2)
        pred[y0:y1, x0:x1]   = rearrange(patch_pred, 'C H W -> H W C')

    stacked = np.concatenate([
    rgb,
    np.concatenate([
        target,
        pred,
        np.zeros_like(target)
    ], axis=-1),
    ], axis=1)

    stacked = Image.fromarray(stacked)

    target_img = Image.new("L", (x_max, y_max), 0)
    target_draw = ImageDraw.Draw(target_img)
    for contour in find_contours(target[..., 0], 0.5):
      target_draw.polygon([(x,y) for y,x in contour],
                        fill=0, outline=255, width=3)

    pred_img = Image.new("L", (x_max, y_max), 0)
    pred_draw = ImageDraw.Draw(pred_img)
    for contour in find_contours(pred[..., 0], 0.7):
      pred_draw.polygon([(x,y) for y,x in contour],
                        fill=0, outline=255, width=3)
    target_img = np.asarray(target_img)
    pred_img = np.asarray(pred_img)
    annot = np.stack([
    target_img,
    pred_img,
    np.zeros_like(target_img),
    ], axis=-1)
    rgb_with_annot = np.where(np.all(annot == 0, axis=-1, keepdims=True),
                            rgb, annot)
    rgb_with_annot = Image.fromarray(rgb_with_annot)

    (log_dir / 'tile_predictions').mkdir(exist_ok=True)
    rgb_with_annot.save(log_dir / 'tile_predictions' / f'{tile}_contour_{epoch}.jpg')
    stacked.save(log_dir / 'tile_predictions' / f'{tile}_masks_{epoch}.jpg')

    outdir = log_dir / 'metrics_plots'
    outdir.mkdir(exist_ok=True)
