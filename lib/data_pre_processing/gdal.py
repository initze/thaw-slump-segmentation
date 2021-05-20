import os
import sys
import yaml
from pathlib import Path
_module = sys.modules[__name__]


def initialize(args=None):
    # If command line arguments are given, use those:
    if args is not None:
        _module.gdal_path = args.gdal_path
        _module.gdal_bin = args.gdal_bin

    # Otherwise, fall back to the ones from system.yml
    system_yml = Path('system.yml')
    if system_yml.exists():
        system_config = yaml.load(system_yml.open(), Loader=yaml.SafeLoader)
        if not _module.gdal_path and 'gdal_path' in system_config:
            _module.gdal_path = system_config['gdal_path']
        if not _module.gdal_bin and 'gdal_bin' in system_config:
            _module.gdal_bin = system_config['gdal_bin']

    _module.rasterize  = os.path.join(_module.gdal_bin, 'gdal_rasterize')
    _module.translate  = os.path.join(_module.gdal_bin, 'gdal_translate')
    _module.warp       = os.path.join(_module.gdal_bin, 'gdalwarp')

    _module.merge      = os.path.join(_module.gdal_path, 'gdal_merge.py')
    _module.retile     = os.path.join(_module.gdal_path, 'gdal_retile.py')
    _module.polygonize = os.path.join(_module.gdal_path, 'gdal_polygonize.py')
