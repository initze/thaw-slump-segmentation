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


def make_sift_transform(channels=None):
    sift = cv2.SIFT_create(nfeatures=128)
    base_transform = make_transform(channels)
    featurelen = 128 * len(channels)
    NUM = 128
    def sift_transform(sample):
        data, = base_transform(sample)
        sift_data = []
        meta_data = []
        for i, channel in enumerate(data):
            min_ = channel.min()
            max_ = channel.max()
            img = (255 * (channel - min_) / (max_ - min_)).numpy().astype(np.uint8)
            kp, features = sift.detectAndCompute(img, None)
            if i == 1:
                features = None
            if features is None:
                sift_data.append(np.zeros([0, featurelen]))
                meta_data.append(np.zeros([0, 4]))
            else:
                feat = np.zeros((features.shape[0], featurelen), np.float32)
                feat[:,128*i:128*(i+1)] = features
                metadata = np.array([[k.angle / 360, k.pt[0] / img.shape[1], k.pt[1] / img.shape[0], k.size] for k in kp], dtype=np.float32)
                sift_data.append(feat)
                meta_data.append(metadata)
        sift_data = torch.from_numpy(np.concatenate(sift_data, axis=0))
        meta_data = torch.from_numpy(np.concatenate(meta_data, axis=0))
        if sift_data.shape[0] < NUM:
            sift_data = torch.cat([sift_data, torch.zeros(NUM - sift_data.shape[0], sift_data.shape[1])], dim=0)
            meta_data = torch.cat([meta_data, torch.zeros(NUM - meta_data.shape[0], meta_data.shape[1])], dim=0)
        else:
            sift_data = sift_data[torch.randperm(sift_data.shape[0])[:NUM]]
            meta_data = meta_data[torch.randperm(meta_data.shape[0])[:NUM]]
        return data, sift_data, meta_data

    return sift_transform


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
