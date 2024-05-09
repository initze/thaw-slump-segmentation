# Copyright (c) Ingmar Nitze and Konrad Heidler

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os
import sys
from pathlib import Path

import yaml

_module = sys.modules[__name__]


def initialize(args=None, *, bin=None, path=None):
    # If command line arguments are given, use those:
    system_yml = Path('system.yml')

    if args is not None:
        # print('Manually set path')
        _module.gdal_path = args.gdal_path
        _module.gdal_bin = args.gdal_bin
    elif bin is not None and path is not None:
        # print('Manually set path')
        _module.gdal_path = path
        _module.gdal_bin = bin
    # Otherwise, fall back to the ones from system.yml
    elif system_yml.exists():
        # print('yml file')
        system_config = yaml.load(system_yml.open(), Loader=yaml.SafeLoader)
        if 'gdal_path' in system_config:
            _module.gdal_path = system_config['gdal_path']
        if 'gdal_bin' in system_config:
            _module.gdal_bin = system_config['gdal_bin']

    else:
        # print('Empty path')
        _module.gdal_path = ''
        _module.gdal_bin = ''

    # print(_module.gdal_bin)
    # print(_module.gdal_path)
    _module.rasterize = os.path.join(_module.gdal_bin, 'gdal_rasterize')
    _module.translate = os.path.join(_module.gdal_bin, 'gdal_translate')
    _module.warp = os.path.join(_module.gdal_bin, 'gdalwarp')

    _module.merge = os.path.join(_module.gdal_path, 'gdal_merge.py')
    _module.retile = os.path.join(_module.gdal_path, 'gdal_retile.py')
    _module.polygonize = os.path.join(_module.gdal_path, 'gdal_polygonize.py')
