"""
Usecase 2 Performance Debugging Script

Usage:
    debug_performance.py [options]

Options:
    -h --help          Show this screen
    --config=CONFIG    Specify run config to use [default: config.yml]
    --dataset=DATASET  Dataset to use for benchmarking, should correspond to
                       a config section under "datasets" [default: train]
"""

import time
import yaml
from docopt import docopt

import torch
import numpy as np

from lib import get_model
from data_loading import get_loader, DataSources

steps = [
    (0, 'Disk cache warmup'),
    (4, 'GPU/CUDA warmup'),
 
    (1, 'Benchmarking Disk -> RAM                              '),
    (2, 'Benchmarking Disk -> RAM -> GPU                       '),
    (3, 'Benchmarking Disk -> RAM -> GPU -> Model -> GPU       '),
    (4, 'Benchmarking Disk -> RAM -> GPU -> Model -> GPU -> RAM'),
    (5, 'Benchmarking         RAM -> GPU                       '),
    (6, 'Benchmarking         RAM -> GPU -> Model -> GPU       '),
    (7, 'Benchmarking         RAM -> GPU -> Model -> GPU -> RAM'),
    (8, 'Benchmarking                GPU -> Model -> GPU -> RAM'),
]

def main():
    cli_args = docopt(__doc__, version="Usecase 2 Training Script 1.0")
    config = yaml.load(open(cli_args['--config']), Loader=yaml.SafeLoader)

    data_sources = DataSources(config['data_sources'])

    ds_config = config['datasets'][cli_args['--dataset']]
    if 'batch_size' not in ds_config:
        ds_config['batch_size'] = config['batch_size']
    ds_config['num_workers'] = config['data_threads']
    loader = get_loader(data_sources=data_sources, **ds_config)

    config['model_args']['input_channels'] = sum(src.channels for src in data_sources)
    modelclass = get_model(config['model'])
    model = modelclass(**config['model_args']).cuda()

    img_shape = None
    target_shape = None

    for (step, description) in steps:
        samples_done = 0
        print(description, end=' – ')
        tic = time.time()

        if step >= 5:
            cuda = step >= 8
            loader = RandomDataGenerator(img_shape, target_shape, cuda=cuda)
        for iteration, (img, target) in enumerate(loader):
            if iteration == 0:
                img_shape = img.shape
                target_shape = target.shape

            if step >= 2:
                img = img.cuda()
                target = target.cuda()

            if step >= 3 and step != 5:
                prediction = model(img)

            if step >= 4:
                prediction = prediction.cpu()

            samples_done += img.shape[0]
            if samples_done >= 512:
                break

        toc = time.time()
        time_elapsed = toc - tic
        throughput = 4 * 512 * np.prod(img.shape[1:]) / time_elapsed
        print(f'{time_elapsed:.02f} seconds. Pipeline Throughput: {throughput / 1e6:.02f}MB/s')


class RandomDataGenerator():
    def __init__(self, *data_shapes, cuda=False):
        self.data_shapes = data_shapes
        self.device = torch.device('cpu') if not cuda else torch.device('cuda')

    def __iter__(self):
        for _ in range(100):
            data = [torch.zeros(shp, device=self.device) for shp in self.data_shapes]
            yield data


if __name__ == "__main__":
    main()
