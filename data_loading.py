import torch
from torch.utils.data import DataLoader, ConcatDataset
from deep_learning.utils.data import PTDataset, Augment
from pathlib import Path

factors = [3000, 3000, 3000, 3000, 255, 255, 255]
normalize = 1 / torch.Tensor(factors).reshape(-1, 1, 1)


def transform_fn(sample):
    data, mask = sample
    data = data.to(torch.float) * normalize
    mask = mask.to(torch.float)
    return data, mask


def get_loader(datasets, batch_size, train=False):
    pt_datasets = []
    for dataset in datasets:
        dataset = 'data_pytorch/' + dataset
        pt_data = PTDataset(dataset, ['data', 'mask'], transform=transform_fn)
        if train:
            pt_data = Augment(pt_data)
        pt_datasets.append(pt_data)

    concatenated = ConcatDataset(pt_datasets)
    return DataLoader(concatenated, batch_size=batch_size, shuffle=train, num_workers=0, pin_memory=True)


def get_batch(data_names):
    base_dir = Path('data_pytorch')
    tensors = []
    masks = []
    for sample in data_names:
        tensor_file, = base_dir.glob(f'*/data/{sample}.pt')
        tensors.append(torch.load(tensor_file))
        mask_file, = base_dir.glob(f'*/mask/{sample}.pt')
        masks.append(torch.load(mask_file))

    return torch.stack(tensors, dim=0).float(), torch.stack(masks, dim=0).float()
