import torch
from torch.utils.data import DataLoader
from deep_learning.utils.data import PTDataset


def transform(augment=False):
    factors = [3000, 3000, 3000, 3000, 255, 255, 255]
    normalize = 1 / torch.Tensor(factors).reshape(-1, 1, 1)

    def transform_fn(sample):
        data, mask = sample
        data = data.to(torch.float) * normalize
        mask = mask.to(torch.float)
        if augment:
            # 8-fold augmentation
            transpose = torch.randint(0, 2, []).item() == 1
            dir1 = 2 * torch.randint(0, 2, []).item() - 1
            dir2 = 2 * torch.randint(0, 2, []).item() - 1
            data = torch.from_numpy(data.numpy()[:, ::dir1, ::dir2].copy())
            mask = torch.from_numpy(mask.numpy()[:, ::dir1, ::dir2].copy())
            if transpose:
                data = data.transpose(1, 2)
                mask = mask.transpose(1, 2)
        return data, mask
    return transform_fn


def get_loaders(batch_size, augment=True):
    train_data = PTDataset('data/tiles_train',
                           ['data', 'mask'],
                           transform=transform(augment=augment))
    train_loader = DataLoader(train_data,
                              batch_size=batch_size,
                              num_workers=4,
                              pin_memory=True)

    val_data = PTDataset('data/tiles_val',
                         ['data', 'mask'],
                         transform=transform(augment=False))
    val_loader = DataLoader(val_data,
                            batch_size=batch_size,
                            num_workers=4,
                            pin_memory=True)

    return train_loader, val_loader
