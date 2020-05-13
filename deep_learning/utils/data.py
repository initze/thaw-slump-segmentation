import torch
from torch.utils.data import Dataset
import h5py
from pathlib import Path


class H5Dataset(Dataset):
    """
    Random access Dataset for datasets stored in HDF5 files as large tensors
    """
    def __init__(self, parts, h5_path):
        self.h5_path = h5_path
        self.parts = parts
        self.h5_file = None

    def assert_open(self):
        if self.h5_file is None:
            self.h5_file = h5py.File(self.h5_path, 'r')
            self.length = len(self.h5_file[self.parts[0]])

    def __getitem__(self, idx):
        self.assert_open()
        return [self.h5_file[part][idx] for part in self.parts]

    def __len__(self):
        self.assert_open()
        return self.length


class PTDataset(Dataset):
    """
    Random access Dataset for datasets of pytorch tensors stored like this:
        data/images/1.pt
        data/images/2.pt
        ...
        data/masks/1.pt
        data/masks/2.pt
        ...
    """
    def __init__(self, root, parts, transform=None, suffix='.pt'):
        self.root = Path(root)
        self.parts = parts
        filenames = list(sorted([x.name for x in (self.root / parts[0]).glob('*' + suffix)]))
        self.index = [[self.root / part / x for part in parts] for x in filenames]
        self.transform = transform

    def __getitem__(self, idx):
        files = self.index[idx]
        data = [torch.load(f) for f in files]
        if self.transform is not None:
            data = self.transform(data)
        return data

    def __len__(self):
        return len(self.index)
