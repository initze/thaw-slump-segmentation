import xarray as xr
import rioxarray
import ee
import geedim as gd
import pandas as pd

from .base import TileSource, Scene, cache_path, safe_download


class Sentinel2(TileSource):
    def __init__(self, s2sceneid: str):
        self.s2sceneid = s2sceneid

    def get_raster_data(self, scene: Scene) -> xr.Dataset:
        _cache_path = cache_path('Sentinel2', f'{scene.id}.tif')
        _cache_path.parent.mkdir(parents=True, exist_ok=True)
        if not _cache_path.exists():
            utm_zone = s2sceneid.split('_')[-1][1:3]
            crs = f'EPSG:326{utm_zone}'
            Sentinel2.download_tile(_cache_path, self.s2sceneid, scene.ee_bounds(crs), crs=crs)

        data = rioxarray.open_rasterio(_cache_path)
        data = data.isel(band=[0,1,2,3,4,5,6,7,8,9,10,11,12])
        # We could transfer the band names as coordinate labels here
        # But other tools don't seem to be compatible with that (i.e. QGIS)
        # data = data.assign_coords({'band': list(data.long_name[:13])})
        data = data.rename(band='Sentinel2_band')
        data.attrs['date'] = str(pd.to_datetime(scene.id.split('_')[-3]))
        return data

    @staticmethod
    def download_tile(out_path, s2sceneid, bounds, crs=None):
        if not out_path.exists():
            gd.Initialize()
            img = ee.Image(s2sceneid)
            img = img.select(['B1','B2','B3','B4','B5','B6','B7','B8','B8A','B9','B10','B11','B12'])
            img = gd.MaskedImage(img)
            safe_download(img, out_path,
                region=bounds.getInfo(),
                # crs=None if crs is None else str(crs),
                crs=crs,
                scale=10,
                dtype='uint16',
                max_tile_size=2,
                max_tile_dim=2000,
            )

    @staticmethod
    def build_scenes(bounds, crs, start_date, end_date, prefix, min_coverage=90, max_cloudy_pixels=20):
        gd.Initialize()
        s2 = ee.ImageCollection("COPERNICUS/S2_HARMONIZED")
        s2 = gd.MaskedCollection(s2)

        bounds = ee.Geometry.Polygon(
            list(bounds.exterior.coords),
            proj=None if crs is None else str(crs),
            evenOdd=False)

        imgs = s2.search(
          start_date=start_date,
          end_date=end_date,
          region=bounds,
          fill_portion=min_coverage,
          custom_filter=f'CLOUDY_PIXEL_PERCENTAGE < {max_cloudy_pixels}'
        )

        scenes = []

        for img in imgs.properties:
            s2_id = img.split('/')[-1]
            scene_id = f'{prefix}_{s2_id}'
            _cache_path = cache_path('Sentinel2', f'{scene_id}.tif')
            Sentinel2.download_tile(_cache_path, img, bounds)
            ds = rioxarray.open_rasterio(_cache_path)

            scene = Scene(
                id=scene_id,
                crs=ds.rio.crs,
                transform=ds.rio.transform(),
                size=ds.shape[-2:],
                layers=[Sentinel2(s2_id)])
            scenes.append(scene)
        return scenes

    def __repr__(self):
        return f'Sentinel2({self.s2sceneid})'

# Determine the S2-Tile's projection
# cf. https://forum.step.esa.int/t/epsg-code-of-sentinel-2-images/17787
# utm_zone = props['MGRS_TILE'][2]
# utm_strip = props['MGRS_TILE'][:2]
# if utm_zone in 'CDEFGHIJKLM':5000
#     # UTM South
#     crs = f'EPSG:327{utm_strip}'
# elif utm_zone in 'NOPQRSTUVWX':
#     # UTM North
#     crs = f'EPSG:326{utm_strip}'
# else:
#     raise ValueError(
#         f'Getting CRS from MGRS Tile {props["MGRS_TILE"]} not yet implemented.'
#         ' It is probably UPS North/South.'
#     )
