import torch
import numpy as np
import cv2
from torch.utils.data import DataLoader, ConcatDataset, Subset
from deep_learning.utils.data import PTDataset, Augment, Transform
from pathlib import Path


def make_transform(channels=None):
    factors = [3000, 3000, 3000, 3000, 20000, 255, 255, 255, 20000, 90]
    if channels is None:
        normalize = 1 / torch.Tensor(factors).reshape(-1, 1, 1)
        def transform_fn(sample):
            data, *rest = sample
            data = data.to(torch.float) * normalize
            return data, *rest
        return transform_fn
    else:
        normalize = 1 / torch.Tensor(factors)[channels].reshape(-1, 1, 1).contiguous()
        def transform_fn(sample):
            data, *rest = sample
            data = data[channels].to(torch.float) * normalize[channels]
            return data, *rest
        return transform_fn


def _get_dataset(dataset, names=['data', 'mask'], channels=None, augment=False, transform=None):
    ds_path = 'data_pytorch/' + dataset
    dataset = PTDataset(ds_path, names)
    if augment:
        dataset = Augment(dataset)
    if transform is not None:
        dataset = Transform(dataset, transform)
    return dataset


def get_loader(folders, batch_size, augment=False, shuffle=False, num_workers=0, names=['data', 'mask'], channels=None, transform=None):
    if transform is None:
        transform = make_transform(channels)
    folders = [_get_dataset(ds, names, augment=augment, transform=transform) for ds in folders]
    concatenated = ConcatDataset(folders)
    return DataLoader(concatenated, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=True)


def get_filtered_loader(folders, batch_size, augment=False, shuffle=False):
    filtered_sets = []
    print("Calculating slump only dataset...", end='', flush=True)
    for folder in folders:
        unfiltered = _get_dataset(folder, augment=augment)
        subset = []
        for i in range(len(unfiltered)):
            # Load just the mask
            if type(unfiltered) is Augment:
                mask = unfiltered._get_nth_tensor_raw(i, 1)
            elif type(unfiltered) is PTDataset:
                mask = unfiltered.get_nth_tensor(i, 1)
            else:
                raise ValueError(f"Can't filter dataset of type {type(unfiltered)}")
            if mask.max() > 0:
                subset.append(i)
        filtered_sets.append(Subset(unfiltered, subset))
    full_dataset = ConcatDataset(filtered_sets)
    print("Done.")
    return DataLoader(full_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=0, pin_memory=True)


def get_batch(data_names, channels):
    base_dir = Path('data_pytorch')
    tensors = []
    masks = []

    transform_fn = make_transform(channels)
    for sample in data_names:
        tensor_file, = base_dir.glob(f'*/data/{sample}.pt')
        mask_file, = base_dir.glob(f'*/mask/{sample}.pt')
        sample = torch.load(tensor_file), torch.load(mask_file)
        tensor, mask = transform_fn(sample)
        tensors.append(tensor)
        masks.append(mask)

    data = torch.stack(tensors, dim=0)
    mask = torch.stack(masks, dim=0)
    return data, mask
