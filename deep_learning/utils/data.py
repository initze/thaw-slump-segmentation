import torch
from torch.utils.data import Dataset
from pathlib import Path


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

    def __len__(self):
        return len(self.index)
