import torch
import numpy as np
from torch.utils.data import DataLoader, ConcatDataset, Subset
from deep_learning.utils.data import H5Dataset, Augment, Transform
from pathlib import Path
from collections import namedtuple


DataSource = namedtuple('DataSource', ['name', 'channels', 'normalization_factors'])

Planet = DataSource('planet', 4, [3000, 3000, 3000, 3000])
NDVI = DataSource('ndvi', 1, [20000])
TCVIS = DataSource('tcvis', 3, [255, 255, 255])
RelativeElevation = DataSource('relative_elevation', 1, [20000])
Slope = DataSource('slope', 1, [90])

DATA_SOURCES = list(sorted([Planet, NDVI, TCVIS, RelativeElevation, Slope]))
SOURCE_FROM_NAME = {src.name: src for src in DATA_SOURCES}

def get_sources(source_names):
    # Always sort the source list
    return list(sorted(SOURCE_FROM_NAME[name] for name in source_names))


def make_transform(data_sources=None):
    if data_sources is None:
        # Use all data sources by default
        data_sources = DATA_SOURCES
    # Sort Data Sources to be independent of the order given in the config
    data_sources = list(sorted(data_sources))

    factors = []
    for source in data_sources:
        factors += source.normalization_factors

    normalize = 1 / torch.tensor(factors, dtype=torch.float32).reshape(-1, 1, 1)
    def transform_fn(sample):
        data, *rest = sample
        data = data.to(torch.float) * normalize
        return data, *rest
    return transform_fn


def get_dataset(dataset, data_sources=None, augment=False, transform=None):
    ds_path = 'data_h5/' + dataset + '.h5'
    dataset = H5Dataset(ds_path, data_sources=data_sources)
    if augment:
        dataset = Augment(dataset)
    if transform is not None:
        dataset = Transform(dataset, transform)
    return dataset


def get_loader(scenes, batch_size, augment=False, shuffle=False, num_workers=0, data_sources=None, transform=None):
    if transform is None:
        transform = make_transform(data_sources)
    scenes = [get_dataset(ds, data_sources=data_sources, augment=augment, transform=transform) for ds in scenes]
    concatenated = ConcatDataset(scenes)
    return DataLoader(concatenated, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=True)


def get_slump_loader(scenes, batch_size, augment=False, shuffle=False, num_workers=0, names=['data', 'mask'], data_sources=None, transform=None):
    if transform is None:
        transform = make_transform(data_sources)
    filtered_sets = []
    print("Calculating slump only dataset...", end='', flush=True)
    for scene in scenes:
        mask_only = get_dataset(scene, names=['mask'])
        subset = [i for i, (mask,) in mask_only if mask.max() > 0]
        unfiltered = get_dataset(scene)
        filtered_sets.append(Subset(unfiltered, subset))

    dataset = ConcatDataset(filtered_sets)
    if augment:
        dataset = Augment(dataset)
    dataset = Transform(dataset, transform)
    print("Done.")
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=True)


def get_vis_loader(vis_config, batch_size, data_sources=None):
    vis_names = []
    vis_datasets = []
    for scene, indices in vis_config.items():
        dataset = get_dataset(scene, data_sources=data_sources)
        vis_names += [f'{scene}|{i}' for i in indices]
        filtered = Subset(dataset, indices)
        vis_datasets.append(filtered)
    vis_data = ConcatDataset(vis_datasets)
    vis_data = Transform(vis_data, make_transform(data_sources))

    loader = DataLoader(vis_data, batch_size=batch_size, shuffle=False, pin_memory=True)
    return loader, vis_names
