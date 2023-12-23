# Copyright (c) Ingmar Nitze and Konrad Heidler

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch
import numpy as np
import h5py
from torch.utils.data import Dataset
from pathlib import Path
import albumentations as A
from torchvision.transforms import v2
from torchvision import transforms

from albumentations.pytorch import ToTensorV2


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
    def __init__(self, root, parts, suffix='.pt'):
        self.root = Path(root)
        self.parts = parts

        first = self.root / parts[0]
        filenames = list(sorted([x.name for x in first.glob('*' + suffix)]))
        self.index = [[self.root / p / x for p in parts] for x in filenames]

    def __getitem__(self, idx):
        files = self.index[idx]
        data = [torch.load(f) for f in files]
        return data

    def get_nth_tensor(self, idx, n):
        """Loads just the n-th of the tensors belonging to idx, untransformed!"""
        files = self.index[idx]
        return torch.load(files[n])

    def __len__(self):
        return len(self.index)


class H5Dataset(Dataset):
    def __init__(self, dataset_path, data_sources):
        super().__init__()
        self.dataset_path = dataset_path
        self.sources = [src.name for src in data_sources]
        self.h5 = None
        self.length = None

    def assert_open(self):
        # Needed for multi-threading to work
        if self.h5 is None:
            self.h5 = h5py.File(self.dataset_path, 'r',
                rdcc_nbytes = 2*(1<<30), # 2 GiB
                rdcc_nslots = 200003,
            )

    def close_fd(self):
        # We need to close the h5 file descriptor before going multi-threaded
        # to allow for pickling on windows
        self.h5.close()
        del self.h5 
        self.h5 = None

    def __getitem__(self, idx):
        self.assert_open()
        features = [self.h5[source][idx] for source in self.sources]
        features = torch.from_numpy(np.concatenate(features, axis=0))

        mask = torch.from_numpy(self.h5['mask'][idx])[0]
        return features, mask

    def __len__(self):
        # Three branches for this, because __getitem__ should be the only
        # operation to open the h5 in the long-term (otherwise, threading will fail on windows)
        # 1. length was already determined -> just use that
        # 2. we don't know the length & h5 is open -> get length from open h5
        # 3. we don't know the length & h5 is closed -> open h5, get length and CLOSE AGAIN!
        if self.length is None:
            if self.h5 is None:
                self.assert_open()
                self.length = self.h5['mask'].shape[0]
                self.close_fd()
            else:
                self.length = self.h5['mask'].shape[0]
        return self.length


class Augment_TV(Dataset):
    def __init__(self, dataset, augment_types=None, tile_size=256):
        self.dataset = dataset
        self.tile_size = tile_size
        if not augment_types:
            self.augment_types = ['RandomHorizontalFlip', 'RandomVerticalFlip', 'Blur', 'RandomRotate90', 'Cutout']
        else:
            self.augment_types = augment_types
        
    def __getitem__(self, idx):
        idx, (flipx, flipy, transpose) = self._augmented_idx_and_ops(idx)
        base = self.dataset[idx]
        transforms_geom_list = ["RandomCrop", "RandomResizedCrop", "RandomHorizontalFlip", "RandomVerticalFlip", "RandomRotation", "RandomAffine"]
        # add Augmentation types
        augment_list_geom = []
        augment_list_visual = []
        if self.augment_types:
            for aug_type in self.augment_types:
                if aug_type == 'RandomRotation':
                    kwargs = dict(degrees=[0,180], interpolation=v2.InterpolationMode.BILINEAR)
                else:
                    kwargs = dict(p=0.5)
                
                if aug_type in transforms_geom_list:
                    augment_list_geom.append(getattr(v2, aug_type)(**kwargs))
            augment_list = augment_list_geom + augment_list_visual
        else:
            return base
        # scale data
        #transform = transforms.Compose([transforms.RandomHorizontalFlip(), transforms.RandomVerticalFlip()])
        transform = v2.Compose(augment_list)
        image = torch.tensor(base[0])
        mask = torch.tensor(base[1])
        input = torch.cat((image, mask.unsqueeze(0)), dim=0)
        augmented = transform(input)
        #print('Image equals augmented', (augmented_image == image).all())
        return (augmented[:-1].clone(), augmented[-1].round().byte().clone())
        #return transform(base)

    def _augmented_idx_and_ops(self, idx):
        idx, carry = divmod(idx, 8)
        carry, flipx = divmod(carry, 2)
        transpose, flipy = divmod(carry, 2)

        return idx, (flipx, flipy, transpose)

    def __len__(self):
        return len(self.dataset)
    

class Augment_A2(Dataset):
    def __init__(self, dataset, augment_types=None, tile_size=256):
        self.dataset = dataset
        self.tile_size = tile_size
        if not augment_types:
            self.augment_types = ['HorizontalFlip', 'VerticalFlip', 'Blur', 'RandomRotate90']
        else:
            self.augment_types = augment_types
        
    def __getitem__(self, idx):
        idx, (flipx, flipy, transpose) = self._augmented_idx_and_ops(idx)
        base = self.dataset[idx]


        # add Augmentation types
        augment_list = []
        if self.augment_types:
            for aug_type in self.augment_types:
                augment_list = []
                kwargs_aug = {}
                for aug_type in self.augment_types:
                    kwargs_aug = {}
                    if aug_type in ['HorizontalFlip','VerticalFlip','RandomRotate90']:
                        kwargs_aug = dict(p=0.5)
                    elif aug_type in ['Cutout']:
                        kwargs_aug = dict(max_h_size=20, max_w_size=20, p=0.2)
                    elif aug_type in ['CropAndPad']:
                        kwargs_aug = dict(p=0.5, percent=20)
                    elif aug_type in ['RandomResizedCrop', 'RandomCrop']:
                        # TODO: get tile size here
                        kwargs_aug = dict(height=self.tile_size, width=self.tile_size, p=0.5)
                    elif aug_type in ['CoarseDropout']:
                        # TODO: get tile size here
                        kwargs_aug = dict(max_height=100, max_width=100, min_height=20, min_width=20, max_holes=20, p=0.2)

                    augment_list.append(getattr(A, aug_type)(**kwargs_aug))
        else:
            return base
        # scale data
        transform = A.Compose(augment_list)

        augmented = transform(image=np.array(base[0].transpose(1,2,0)), mask=np.array(base[1]))
        data = torch.from_numpy(np.ascontiguousarray(augmented['image']).transpose(2,0,1).copy())
        mask = torch.from_numpy(np.ascontiguousarray(augmented['mask']).copy())

        return (data, mask)

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


class Transformed(Dataset):
    "Wrap a dataset and apply a given transformation to every sample (e.g. Scaling)"
    def __init__(self, dataset, transform):
        self.dataset = dataset
        self.transform = transform

    def __getitem__(self, idx):
        sample = self.dataset[idx]
        return self.transform(sample)

    def __len__(self):
        return len(self.dataset)


class Scaling():
    "Scales tensors by predefined normalization factors"
    def __init__(self, normalize):
        self.normalize = normalize
        # TODO: Also allow for a shifting factor?

    def __call__(self, sample):
        sample = list(sample)
        # Imagery is sample[0]
        sample[0] = sample[0].to(torch.float) * self.normalize
        return sample
    

class Normalize(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset
        
    def __getitem__(self, idx):
        base = self.dataset[idx]
        # calculate mean and std
        #####
        # 
        # scale data
        image = torch.tensor(base[0])
        mask = torch.tensor(base[1])
        transform = v2.Compose([v2.ToDtype(torch.float32, scale=True)])
        return (transform(image.clone(), mask.clone()))
    
    def __len__(self):
        return len(self.dataset)
