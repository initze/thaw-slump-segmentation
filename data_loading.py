# Copyright (c) Ingmar Nitze and Konrad Heidler

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging
import torch
from torch.utils.data import DataLoader, ConcatDataset, Subset
from lib.utils import get_logger
from lib.utils.data import H5Dataset, Augment, Transformed, Scaling
from collections import namedtuple
from tqdm import tqdm


class DataSources:
    DataSource = namedtuple('DataSource',
            ['name', 'channels', 'normalization_factors'])
    Planet = DataSource('planet', 4, [3000, 3000, 3000, 3000])
    NDVI = DataSource('ndvi', 1, [20000])
    TCVIS = DataSource('tcvis', 3, [255, 255, 255])
    RelativeElevation = DataSource('relative_elevation', 1, [30000])
    Slope = DataSource('slope', 1, [90])
    LIST = list(sorted([Planet, NDVI, TCVIS, RelativeElevation, Slope]))
    NAME2SOURCE = {src.name: src for src in LIST}

    def __init__(self, sources):
        sources = (self._get_source(source) for source in sources)
        # Sort Data Sources to be independent of the order given in the config
        self.sources = tuple(sorted(sources))

    def _get_source(self, source):
        if type(source) is str:
            return DataSources.NAME2SOURCE[source]
        elif type(source) is DataSources.DataSource:
            return source
        else:
            raise ValueError(f"Can't convert object {source}"
                             f"of type {type(source)} to DataSource")

    @staticmethod
    def all():
        return DataSources([src.name for src in DataSources.LIST])

    def __iter__(self):
        return self.sources.__iter__()

    def index(self, element):
        return self.sources.index(element)


def make_scaling(data_sources=None):
    if data_sources is None:
        # Use all data sources by default
        data_sources = DataSources.all()
    # Sort Data Sources to be independent of the order given in the config
    data_sources = DataSources(data_sources)

    factors = []
    for source in data_sources:
        factors += source.normalization_factors

    normalize = 1 / torch.tensor(factors, dtype=torch.float32).reshape(-1, 1, 1)
    return Scaling(normalize)


def get_dataset(dataset, data_sources=None, augment=False, transform=None, augment_types=None):
    if data_sources is None:
        # Use all data sources by default
        data_sources = DataSources.all()
    data_sources = DataSources(data_sources)

    ds_path = 'data_h5/' + str(dataset) + '.h5'
    dataset = H5Dataset(ds_path, data_sources=data_sources)
    if augment:
        dataset = Augment(dataset, augment_types=augment_types)
    if transform is not None:
        dataset = Transformed(dataset, transform)
    return dataset


def get_loader(scenes, batch_size, augment=False, augment_types=None, shuffle=False, num_workers=0, data_sources=None, transform=None):
    if transform is None:
        transform = make_scaling(data_sources)
    scenes = [get_dataset(ds, data_sources=data_sources, augment=augment, augment_types=None, transform=transform) for ds in scenes]
    concatenated = ConcatDataset(scenes)
    return DataLoader(concatenated, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=True)


def get_slump_loader(scenes, batch_size, augment=False, augment_types=None, shuffle=False, num_workers=0, data_sources=None, transform=None):
    logger = get_logger('data_loading')

    if transform is None:
        transform = make_scaling(data_sources)
    filtered_sets = []
    logger.info("Start calculating slump only dataset.")
    for scene in tqdm(scenes):
        data = get_dataset(scene, data_sources=data_sources)
        subset = [i for i in range(len(data)) if data[i][1].max() > 0]
        filtered_sets.append(Subset(data, subset))

    dataset = ConcatDataset(filtered_sets)
    if augment:
        dataset = Augment(dataset, augment_types=augment_types)
    dataset = Transformed(dataset, transform)
    logger.info("Done calculating slump only dataset.")
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=True)


def get_vis_loader(vis_config, batch_size, data_sources=None):
    vis_names = []
    vis_datasets = []
    for scene, indices in vis_config.items():
        dataset = get_dataset(scene, data_sources=data_sources)
        vis_names += [f'{scene}-{i}' for i in indices]
        filtered = Subset(dataset, indices)
        vis_datasets.append(filtered)
    vis_data = ConcatDataset(vis_datasets)
    vis_data = Transformed(vis_data, make_scaling(data_sources))

    loader = DataLoader(vis_data, batch_size=batch_size, shuffle=False, pin_memory=True)
    return loader, vis_names
