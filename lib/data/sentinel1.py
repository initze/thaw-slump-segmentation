import xarray as xr
import rioxarray
import ee
import geedim as gd
import pandas as pd

from .base import TileSource, Scene, cache_path, safe_download


class Sentinel1(TileSource):
    def __init__(self, s1sceneid: str):
        self.s1sceneid = s1sceneid

    def get_raster_data(self, scene: Scene) -> xr.Dataset:
        _cache_path = cache_path('Sentinel1', f'{scene.id}.tif')
        _cache_path.parent.mkdir(parents=True, exist_ok=True)
        Sentinel1.download_tile(_cache_path, s1sceneid, scene.ee_bounds())

        data = rioxarray.open_rasterio(_cache_path)
        data = data.isel(band=[0,1])
        # We could transfer the band names as coordinate labels here
        # But other tools don't seem to be compatible with that (i.e. QGIS)
        # data = data.assign_coords({'band': list(data.long_name[:13])})
        data = data.rename(band='Sentinel1_band')
        data.attrs['date'] = str(pd.to_datetime(scene.id.split('_')[-3]))
        return data

    @staticmethod
    def download_tile(out_path, s1sceneid, bounds=None, crs=None):
        if not out_path.exists():
            gd.Initialize()
            img = ee.Image(s1sceneid)
            img = img.select(['HH', 'HV'])
            img = gd.MaskedImage(img)
            safe_download(img, out_path,
                scale=40,
                max_tile_size=2,
                max_tile_dim=2000,
            )

    @staticmethod
    def build_scene(bounds, crs, start_date, end_date, prefix, min_coverage=90):
        gd.Initialize()
        s1 = ee.ImageCollection("COPERNICUS/S1_GRD")
        s1 = s1.filter(ee.Filter.eq('instrumentMode', 'EW'))
        s1 = s1.filter(ee.Filter.eq('orbitProperties_pass', 'DESCENDING'))
        s1 = s1.filter(ee.Filter.eq('transmitterReceiverPolarisation', ['HH', 'HV']))
        s1 = gd.MaskedCollection(s1)

        bounds = ee.Geometry.Polygon(
            list(bounds.exterior.coords),
            proj=None if crs is None else str(crs),
            evenOdd=False)

        imgs = s1.search(
          start_date=start_date,
          end_date=end_date,
          region=bounds,
          fill_portion=min_coverage,
        )

        metadata = imgs.properties
        if len(metadata) == 0:
          return None
        best = max(metadata, key=lambda x: metadata[x]['FILL_PORTION'])
        assert metadata[best]['FILL_PORTION'] > 90
        print(metadata)

        s1_id = best.split('/')[-1]

        scene_id = f'{prefix}_{s1_id}'
        _cache_path = cache_path('Sentinel1', f'{scene_id}.tif')
        Sentinel1.download_tile(_cache_path, best, bounds)
        ds = rioxarray.open_rasterio(_cache_path)

        scene = Scene(
            id=scene_id,
            crs=ds.rio.crs,
            transform=ds.rio.transform(),
            size=ds.shape[-2:],
            layers=[Sentinel1(scene_id)])
        return scene

    def __repr__(self):
        return f'Sentinel1({self.s1sceneid})'
