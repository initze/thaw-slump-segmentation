from abc import ABC, abstractmethod, abstractstaticmethod
from pathlib import Path
from typing import Union
import numpy as np
import xarray as xr
import rioxarray
import ee
import geedim as gd
from shapely.geometry import box, Polygon
from shapely.ops import transform
from pyproj import Transformer

_root_path: Path = Path('data/')

_LAYER_REGISTRY = {}

def get_source(source_name):
  return _LAYER_REGISTRY[source_name]


def init_data_paths(root_path: Union[str, Path]):
    global _root_path
    _root_path = Path(root_path)


class TileSource(ABC):
  @abstractmethod
  def get_raster_data(self, scene: 'Scene') -> xr.Dataset:
    ...

  @abstractstaticmethod
  def normalize(tile):
    ...


class EETileSource(TileSource):
    def get_raster_data(self, scene: 'Scene') -> xr.Dataset:
        _cache_path = cache_path(class_name(self), f'{scene.id}.tif')
        _cache_path.parent.mkdir(parents=True, exist_ok=True)

        if not _cache_path.exists():
            gd.Initialize()
            img = gd.MaskedImage(self.get_ee_image())
            safe_download(img, _cache_path,
                region=scene.ee_bounds().getInfo(),
                crs=str(scene.crs),
                crs_transform=scene.transform,
                shape=scene.size
            )

        data = rioxarray.open_rasterio(_cache_path)
        data = data.isel(band=slice(0, -1))
        data = data.rename(band=f'{class_name(self)}_band')
        return data

    @abstractmethod
    def get_ee_image(self):
        ...

    @abstractmethod
    def get_dtype(self):
        ...


class Scene:
    """
    A Scene object encapsulates the following things:
        * Unique scene id
        * Projection / CRS
        * Transformation
        * Extents
        * Data Sources (e.g. PlanetScope, Sentinel2, ...)

    Based on this information, it can be used to do a number of things.

    (1) Download and merge data from different sources:

        scene = Scene(...)
        scene.add_layer(Sentinel2)
        scene.add_layer(TCVIS)
            ...
        scene.save('scene.nc')

    (2) More to come :)

    """
    def __init__(self, id, crs, transform, size, layers=[], data_mask=None):
        self.id = id
        self.crs = crs
        self.transform = transform
        self.size = size
        self.layers = layers
        self.data_mask = data_mask

    def add_layer(self, source: TileSource):
        self.layers.append(source)

    def to_xarray(self):
        ds = xr.Dataset({
            type(ly).__name__: ly.get_raster_data(self)
            for ly in self.layers
        })
        ds.attrs['id'] = self.id
        ds.attrs['size'] = self.size
        return ds

    def save(self, path: Union[str, Path]):
        xarray = self.to_xarray()
        # mask here
        # metadata are being lost

        # writes mask but bloats up data
        for key in list(xarray.keys()):
            xarray[key].data = xarray[key].where(self.data_mask).data

        xarray.to_netcdf(path, engine='h5netcdf')

    def bounds(self, crs=None) -> Polygon:
        if crs is None:
            crs = self.crs

        h, w = self.size
        top, left     = self.transform * (0, 0)
        bottom, right = self.transform * (w, h)
        native_bbox = box(top, left, bottom, right)
        reprojection = Transformer.from_crs(self.crs, crs).transform
        return transform(reprojection, native_bbox)

    def ee_bounds(self, crs=None):
        if crs is None:
            crs = self.crs
        return ee.Geometry.Polygon(list(self.bounds(crs).exterior.coords),
                    proj=str(crs), evenOdd=False)

    def get_coords(self):
        H, W = self.size
        y = np.arange(H) + 0.5
        x = np.arange(W) + 0.5

        tx_y = self.transform.e * y + self.transform.f
        tx_x = self.transform.a * x + self.transform.c

        return {'y': tx_y, 'x': tx_x}

    def __repr__(self):
        layer_repr = ',\n'.join(f'  {ly}' for ly in self.layers)
        return f"Scene(id={self.id}, crs={self.crs}, size={self.size}, layers=[\n{layer_repr}\n])"

# UTILS

def cache_path(prefix, filename):
    _cache_path = _root_path / 'cache' / prefix / filename
    _cache_path.parent.mkdir(exist_ok=True, parents=True)
    return _cache_path


def class_name(obj):
    return type(obj).__name__


def safe_download(img, out_path, **kwargs):
    if out_path.exists():
        raise FileExistsError("Output file exists: {out_path}")
    tmp_path = out_path.parent / f'{out_path.stem}_incomplete{out_path.suffix}'
    if tmp_path.exists():
        # Incomplete Download (i.e. script crashed earlier)
        tmp_path.unlink()
        print(f'Removing incomplete download at {tmp_path}')
        # TODO: Debug Log Message
    img.download(tmp_path, **kwargs)
    tmp_path.rename(out_path)

