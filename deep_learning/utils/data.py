import torch
from torch.utils.data import Dataset
from pathlib import Path


class PTDataset(Dataset):
    """
    Random access Dataset for datasets of pytorch tensors stored like this:
        data/images/file1.pt
        data/images/file1.pt
        ...
        data/masks/file1.pt
        data/masks/file2.pt
        ...
    """
    def __init__(self, root, parts, transform=None, suffix='.pt'):
        self.root = Path(root)
        self.parts = parts

        first = self.root / parts[0]
        filenames = list(sorted([x.name for x in first.glob('*' + suffix)]))
        self.index = [[self.root / p / x for p in parts] for x in filenames]
        self.transform = transform

    def __getitem__(self, idx):
        files = self.index[idx]
        data = [torch.load(f) for f in files]
        if self.transform is not None:
            data = self.transform(data)
        return data

    def get_nth_tensor(self, idx, n):
        """Loads just the n-th of the tensors belonging to idx, untransformed!"""
        files = self.index[idx]
        return torch.load(files[n])

    def __len__(self):
        return len(self.index)


class Augment(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset
        self.len = 8 * len(self.dataset)

    def __getitem__(self, idx):
        idx, (flipx, flipy, transpose) = self._augmented_idx_and_ops(idx)
        diry = 2 * flipy - 1
        dirx = 2 * flipx - 1
        base = self.dataset[idx]
        augmented = []
        for field in base:
            field = field.numpy()
            field = field[:, ::diry, ::dirx]
            if transpose == 1:
                field = field.transpose(0, 2, 1)
            augmented.append(torch.from_numpy(field.copy()))
        return tuple(augmented)

    def _augmented_idx_and_ops(self, idx):
        idx, carry = divmod(idx, 8)
        carry, flipx = divmod(carry, 2)
        transpose, flipy = divmod(carry, 2)

        return idx, (flipx, flipy, transpose)

    def _get_nth_tensor_raw(self, idx, n):
        """Hacky way of transparently accessing the underlying get_nth_tensor of a PTDataset"""
        idx, ops = self._augmented_idx_and_ops(idx)
        return self.dataset.get_nth_tensor(idx, n)

    def __len__(self):
        return len(self.dataset) * 8
