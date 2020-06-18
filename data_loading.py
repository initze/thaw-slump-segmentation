import torch
from torch.utils.data import DataLoader, ConcatDataset, Subset
from deep_learning.utils.data import PTDataset, Augment
from pathlib import Path


factors = [3000, 3000, 3000, 3000, 20000, 255, 255, 255, 40000, 90]
normalize = 1 / torch.Tensor(factors).reshape(-1, 1, 1)


def transform_fn(sample):
    data, mask = sample
    data = data.to(torch.float) * normalize
    mask = mask.to(torch.float)
    return data, mask


def _get_dataset(dataset, names=['data', 'mask'], augment=False):
    ds_path = 'data_pytorch/' + dataset
    dataset = PTDataset(ds_path, names, transform=transform_fn)
    if augment:
        dataset = Augment(dataset)
    return dataset


def get_loader(folders, batch_size, augment=False, shuffle=False):
    folders = [_get_dataset(ds, ['data', 'mask'], augment=augment) for ds in folders]
    concatenated = ConcatDataset(folders)
    return DataLoader(concatenated, batch_size=batch_size, shuffle=shuffle, num_workers=0, pin_memory=True)


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


def get_batch(data_names):
    base_dir = Path('data_pytorch')
    tensors = []
    masks = []
    for sample in data_names:
        tensor_file, = base_dir.glob(f'*/data/{sample}.pt')
        tensors.append(torch.load(tensor_file))
        mask_file, = base_dir.glob(f'*/mask/{sample}.pt')
        masks.append(torch.load(mask_file))

    data = torch.stack(tensors, dim=0).float()

    return data * normalize, torch.stack(masks, dim=0).float()
