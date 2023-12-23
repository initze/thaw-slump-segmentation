from typing import Union
from pathlib import Path
import numpy as np
import xarray as xr
import rasterio as rio
import rioxarray
from pyproj import Proj


from .base import TileSource, Scene, cache_path
from lib.data_pre_processing import udm

def scale_array_ignore_zero_add_constant_per_band_3D(arr, constant=1e-5, perc_min=2, perc_max=98):
    if (arr == 0).all():
        return arr

    non_zero_mask = arr != 0
    min_vals = np.percentile(arr[non_zero_mask], perc_min, axis=0)
    max_vals = np.percentile(arr[non_zero_mask], perc_max, axis=0)
    if max_vals == min_vals:
      max_vals += 1e-4
  
    out = np.clip(np.where(non_zero_mask, (arr - min_vals) / (max_vals - min_vals), 0), constant, 1)
    out[~non_zero_mask] = 0
    return out

class PlanetScope(TileSource):
  def __init__(self, tile_path: Union[str, Path]):
    self.tile_path = Path(tile_path)

  def get_raster_data(self, scene: Scene) -> xr.DataArray:
    data = rioxarray.open_rasterio(self.tile_path)
    # Usually, PlanetScope will be the master imagery.
    # But if not, we'll need to reproject it to match whatever master we're using.
    same_size = scene.size == data.shape[-2:]
    same_crs  = Proj(data.rio.crs) == Proj(scene.crs)
    same_transform = data.rio.transform() == scene.transform
    if not (same_size and same_crs and same_transform):
      print('Reprojecting PlanetScope data. This has never been tested, so please double-check!')
      data = data.reproject(
        dst_crs=scene.crs,
        shape=scene.size,
        transform=scene.transform,
        resampling=rio.enums.Resampling.average,
      )

    # We could transfer the band names as coordinate labels here
    # But other tools don't seem to be compatible with that (i.e. QGIS)
    # data = data.assign_coords({'band': list(data.long_name[:13])})
    data = data.rename(band='PlanetScope_band')
    return data


  @staticmethod
  def build_scene(tile_path):
    tile_path = Path(tile_path)
    tile_id = tile_path.parent.name
    print(tile_id)
    ds = rioxarray.open_rasterio(tile_path, decode_coords='all')
    udm_file = [f for f in list(tile_path.parent.glob('*udm*.tif')) if 'udm2' not in f.name][0]
    data_mask = udm.get_mask_from_udm(udm_file)
    #ds = ds * data_mask
    #ds.attrs['_FillValue'] = 0

    scene = Scene(
      id=tile_id,
      crs=ds.rio.crs,
      transform=ds.rio.transform(),
      size=ds.shape[-2:],
      layers=[PlanetScope(tile_path)],
      data_mask=data_mask,
      red_band = 3,
      nir_band = 4)
    return scene

  def __repr__(self):
    return f'PlanetScope({self.tile_path.stem})'

  @staticmethod
  def normalize(tile):
    clipped = np.clip((tile / 3000), 0, 1)
    #np.clip(scale_array_ignore_zero_add_constant(clipped), 0, 1)
    return clipped
    #return scale_array_ignore_zero_add_constant_per_band_3D(clipped)
    #return np.clip((tile / 3000), 0, 1)

