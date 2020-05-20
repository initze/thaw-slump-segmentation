import torch
from torch.utils.data import DataLoader
from deep_learning.utils.data import PTDataset, Augment


factors = [3000, 3000, 3000, 3000, 255, 255, 255]
normalize = 1 / torch.Tensor(factors).reshape(-1, 1, 1)


def transform_fn(sample):
    data, mask = sample
    data = data.to(torch.float) * normalize
    mask = mask.to(torch.float)
    return data, mask


def get_loaders(batch_size, augment=True):
    train_data = PTDataset('data/tiles_train',
                           ['data', 'mask'],
                           transform=transform_fn)

    if augment:
        train_data = Augment(train_data)

    train_loader = DataLoader(train_data,
                              batch_size=batch_size,
                              num_workers=0,
                              pin_memory=True)

    val_data = PTDataset('data/tiles_val',
                         ['data', 'mask'],
                         transform=transform_fn)

    val_loader = DataLoader(val_data,
                            batch_size=batch_size,
                            num_workers=0,
                            pin_memory=True)

    return train_loader, val_loader
