import torch
from torch.utils.data import DataLoader
from deep_learning.utils.data import PTDataset


def transform(augment=False):
    normalize = 1 / torch.Tensor([3000, 3000, 3000, 3000, 255, 255, 255]).reshape(-1, 1, 1)

    def transformer(sample):
        data, mask = sample
        data = data.to(torch.float) * normalize
        # data = data[:4]
        mask = mask.to(torch.float)
        # mask = torch.argmax(mask)
        if augment:
            transpose = torch.randint(0, 2, []).item() == 1
            stride1 = 2 * torch.randint(0, 2, []).item() - 1
            stride2 = 2 * torch.randint(0, 2, []).item() - 1
            data = torch.from_numpy(data.numpy()[:, ::stride1, ::stride2].copy())
            mask = torch.from_numpy(mask.numpy()[:, ::stride1, ::stride2].copy())
            if transpose:
                data = data.transpose(1, 2)
                mask = mask.transpose(1, 2)
        # mask = mask - smooth * (2 * mask - 1)
        return data, mask
    return transformer


def get_loaders(batch_size, augment=True):
    train_data = PTDataset('data/tiles_train', ['data', 'mask'], transform=transform(augment=augment))
    train_loader = DataLoader(train_data,
                              batch_size=batch_size,
                              num_workers=4,
                              pin_memory=True)

    val_data = PTDataset('data/tiles_val', ['data', 'mask'], transform=transform(augment=False))
    val_loader = DataLoader(val_data,
                            batch_size=batch_size,
                            num_workers=4,
                            pin_memory=True)

    return train_loader, val_loader
