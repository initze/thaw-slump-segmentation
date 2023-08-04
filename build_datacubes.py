#!/usr/bin/env python
# Copyright (c) Ingmar Nitze and Konrad Heidler

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
Usecase 2 Data Preprocessing Script
"""
import argparse
from datetime import datetime
from pathlib import Path
import os
from collections import defaultdict
import xarray
from tqdm import tqdm
import random

import pandas as pd
import geopandas as gpd
from shapely.ops import unary_union
from joblib import Parallel, delayed
import fsspec

from lib import data
from lib.utils import init_logging, get_logger
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("--n_jobs", default=-1, type=int, help="number of parallel joblib jobs")
parser.add_argument("--data_dir", default='data', type=Path, help="Path to data processing dir")
parser.add_argument("--log_dir", default='logs', type=Path, help="Path to log dir")
parser.add_argument("--mode", default='planet',
                    choices=['planet', 's1', 's2', 's2_4w', 's2_timeseries',
                             's2_unlabelled', 's2_unlabelled_4w', 'qinghai', 's2_ts_inference'],
        help="The type of data cubes to build.")


def complete_scene(scene):
    # scene.add_layer(data.RelativeElevation())
    scene.add_layer(data.AbsoluteElevation())
    scene.add_layer(data.Slope())
    scene.add_layer(data.TCVIS())


def build_planet_cube(planet_file: Path, out_dir: Path):
    # We need to re-initialize the data paths every time
    # because of the multi-processing
    data.init_data_paths(out_dir.parent)

    scene = data.PlanetScope.build_scene(planet_file)
    labels = gpd.read_file(next(planet_file.parent.glob('*.shp')))
    scene.add_layer(data.Mask(labels.geometry))
    complete_scene(scene)
    out_dir.mkdir(exist_ok=True, parents=True)
    scene.save(out_dir / f'{scene.id}.nc')


def build_sentinel1_cubes(site_poly, image_id, image_date, tiles, targets, out_dir: Path):
    if any(out_dir.glob(f'{image_id}*')):
      print(f'Skipping {image_id} as it has already been built')
      return
    data.init_data_paths(out_dir.parent)

    start_date = image_date - pd.to_timedelta('14 days')
    end_date = image_date + pd.to_timedelta('14 days')
    scene = data.Sentinel1.build_scene(
        bounds=site_poly, crs='EPSG:3995',
        start_date=start_date, end_date=end_date,
        prefix=image_id,
    )
    if scene is None:
      print(f'No valid image found for {image_id}')
      return
    # TODO: Re-add this for git push
    # complete_scene(scene)
    scene.add_layer(data.Mask(targets.geometry, tiles.geometry))
    scene.save(out_dir / f'{scene.id}.nc')


def build_sentinel2_cubes(site_poly, image_id, image_date, tiles, targets, out_dir: Path):
    if any(out_dir.glob(f'{image_id}*')):
      print(f'Skipping {image_id} as it has already been built')
      return
    data.init_data_paths(out_dir.parent)

    start_date = image_date - pd.to_timedelta('14 days')
    end_date = image_date + pd.to_timedelta('14 days')
    scene = data.Sentinel2.build_scene(
        bounds=site_poly, crs='EPSG:3995',
        start_date=start_date, end_date=end_date,
        prefix=image_id,
    )
    if scene is None:
      print(f'No valid image found for {image_id}')
      return
    # TODO: Re-add this for git push
    # complete_scene(scene)
    scene.add_layer(data.Mask(targets.geometry, tiles.geometry))
    scene.save(out_dir / f'{scene.id}.nc')


def build_sentinel2_4w(site_poly, image_id, image_date, tiles, targets, out_dir: Path):
    if any(out_dir.glob(f'{image_id}*')):
      print(f'Skipping {image_id} as it has already been built')
      return
    data.init_data_paths(out_dir.parent)

    start_date = image_date - pd.to_timedelta('30 days')
    end_date = image_date + pd.to_timedelta('30 days')
    scenes = data.Sentinel2.build_scenes(
        bounds=site_poly, crs='EPSG:3995',
        start_date=start_date, end_date=end_date,
        prefix=image_id,
    )
    print(f'Found {len(scenes)} valid image found for {image_id}')
    if not scenes:
      return

    mask = data.Mask(targets.geometry, tiles.geometry)

    xarrs = []
    print('Starting extracting Scenes')
    for scene in tqdm(scenes):
        xarr = scene.to_xarray()
        date = pd.to_datetime(xarr.Sentinel2.date)
        xarr = xarr.expand_dims({'time': [date]}, axis=0)
        xarrs.append(xarr)
    print('Rastering Mask...')
    mask = mask.get_raster_data(scenes[0])
    xarrs.append(xarray.Dataset({'Mask': mask}))
    print('Combining...')
    combined = xarray.combine_by_coords(xarrs, combine_attrs='drop_conflicts')

    print('Writing...')
    combined.to_netcdf(out_dir / f'{image_id}.nc', engine='h5netcdf')
    print('Done')


def build_sentinel2_timeseries(site_poly, image_id, tiles, targets, out_dir: Path):
    if any(out_dir.glob(f'{image_id}*')):
      print(f'Skipping {image_id} as it has already been built')
      return
    data.init_data_paths(out_dir.parent)

    # Get tile dates
    img2date = dict(zip(targets.image_id, targets.image_date))
    tiles['image_date'] = tiles.image_id.apply(img2date.get)
    tiles = tiles.groupby('image_date').agg({'geometry': unary_union}).reset_index().set_crs(tiles.crs)
    targets.groupby('image_date').agg({'geometry': unary_union}).reset_index().set_crs(targets.crs)

    # Build S2 Scenes
    # start_date = pd.to_datetime('2015-01-01')
    start_date = pd.to_datetime('2015-06-24')
    end_date   = pd.to_datetime('2022-10-01')

    scenes = data.Sentinel2.build_scenes(
        bounds=site_poly, crs='EPSG:3995', 
        start_date=start_date, end_date=end_date,
        prefix=image_id,
        max_cloudy_pixels=5
    )

    x_scenes = defaultdict(list)
    for scene in tqdm(scenes):
        xarr = scene.to_xarray()
        date = pd.to_datetime(xarr.Sentinel2.date)
        xarr = xarr.expand_dims({'time': [date]}, axis=0)
        x_scenes[scene.crs].append((scene, xarr))

    # encoding = dict(Sentinel2={
    #   'scale_factor': 1/255, 'add_offset': 0, 'dtype': 'uint8','_FillValue': 0})
    for crs in x_scenes:
      # Build Masks
      sample_scene = x_scenes[crs][0][0]
      xmasks = []
      for i in tiles.index:
        tile_targets = targets[targets.image_date == tiles.image_date[i]]
        mask = data.Mask(tile_targets.geometry, tiles.loc[[i]].geometry)
        mask_scene = data.Scene(f'{tiles.image_date.loc[i]}_{crs}', crs,
            sample_scene.transform, sample_scene.size, layers=[mask])
        date = pd.to_datetime(tile_targets.image_date.iloc[0])
        xmask = mask_scene.to_xarray()
        xmask = xmask.expand_dims({'mask_time': [date]}, axis=0)
        xmasks.append(xmask)

      xarrs = [xarr for scene, xarr in x_scenes[crs]]
      combined = xarray.combine_by_coords(xarrs + xmasks, combine_attrs='drop_conflicts')
      combined.to_netcdf(out_dir / f'{image_id}_{crs}.nc', engine='h5netcdf')

    print(f'Done with build for scene {image_id}')


def build_unlabelled_sentinel2_timeseries(site_poly, site_name, out_dir: Path):
    if any(out_dir.glob(f'{site_name}*')):
      print(f'Skipping {site_name} as it has already been built')
      return
    data.init_data_paths(out_dir.parent)

    # Build S2 Scenes
    # start_date = pd.to_datetime('2015-06-24')
    # end_date   = pd.to_datetime('2022-10-01')
    start_date = pd.to_datetime('2015-06-01')
    end_date   = pd.to_datetime('2022-10-01')

    scenes = data.Sentinel2.build_scenes(
        bounds=site_poly, crs='EPSG:3995', 
        start_date=start_date, end_date=end_date,
        prefix=site_name,
        max_cloudy_pixels=20
    )

    print(f'Downloaded all the data...')

    x_scenes = defaultdict(list)
    for scene in tqdm(scenes):
        xarr = scene.to_xarray()
        date = pd.to_datetime(xarr.Sentinel2.date)
        xarr = xarr.expand_dims({'time': [date]}, axis=0)
        x_scenes[scene.crs].append((scene, xarr))

    for crs in x_scenes:
      xarrs = [xarr for scene, xarr in x_scenes[crs]]
      combined = xarray.combine_by_coords(xarrs, combine_attrs='drop_conflicts')

      raise NotImplementedError('Saving unscaled!')
      # opts = dict(zlib=True, shuffle=True, complevel=1)
      # for var in combined.data_vars:
      #   combined[var].encoding.update(opts)
      combined.to_netcdf(out_dir / f'{site_name}_{crs}.nc', engine='h5netcdf')

    print(f'Done with build for scene {site_name}')


def build_sentinel2_unlabelled(tile_id, out_dir: Path):
    if any(out_dir.glob(f'{tile_id}*')):
      print(f'Skipping {tile_id} as it has already been built')
      return
    data.init_data_paths(out_dir.parent)

    random.seed(tile_id)
    year = random.choice([2017, 2018, 2019, 2020, 2021])
    start_date = f'{year}-07-01'
    end_date = f'{year}-10-01'
    scene = data.Sentinel2.scene_for_tile(tile_id,
      start_date=start_date, end_date=end_date,
    )
    scene.save(out_dir / f'{scene.id}.nc')


def build_s2_unlabelled_multi(tile_id, out_dir: Path):
    if any(out_dir.glob(f'{tile_id}*')):
      print(f'Skipping {tile_id} as it has already been built')
      return
    data.init_data_paths(out_dir.parent)

    random.seed(tile_id)
    year = random.choice([2017, 2018, 2019, 2020, 2021])
    start_date = f'{year}-07-01'
    end_date = f'{year}-10-01'
    scenes = data.Sentinel2.scenes_for_tile(tile_id,
      start_date=start_date, end_date=end_date,
    )

    xarrs = []
    for scene in scenes:
        xarr = scene.to_xarray()
        date = pd.to_datetime(xarr.Sentinel2.date)
        xarr = xarr.expand_dims({'time': [date]}, axis=0)
        xarrs.append(xarr)
    combined = xarray.combine_by_coords(xarrs, combine_attrs='drop_conflicts')

    combined.to_netcdf(out_dir / f'{tile_id}_{year}.nc', engine='h5netcdf')


def load_annotations(shapefile_root):
    targets = map(gpd.read_file, shapefile_root.glob('*/TrainingLabel*.shp'))
    targets = pd.concat(targets).to_crs('EPSG:3995').reset_index(drop=True)
    targets['geometry'] = targets['geometry'].buffer(0)
    targets['image_date'] = pd.to_datetime(targets['image_date'])
    id2date = dict(targets[['image_id', 'image_date']].values)

    scenes = map(gpd.read_file, shapefile_root.glob('*/ImageFootprints*.shp'))
    scenes = pd.concat(scenes).to_crs('EPSG:3995').reset_index(drop=True)
    scenes = scenes[scenes.image_id.isin(targets.image_id)]
    scenes['geometry'] = scenes['geometry'].buffer(0)
    scenes['image_date'] = scenes.image_id.apply(id2date.get)

    # Semijoin targets and sites:
    targets = targets[targets.image_id.isin(scenes.image_id)]

    sites = unary_union(scenes.geometry)
    sites = list(sites.geoms)

    return targets, scenes, sites


def load_qinghai_annotations(data_folder):
  SHAPEFILE_FOLDER = data_folder / 'cache' / 'qinghai'
  origin = fsspec.get_mapper("zip::simplecache::https://download.pangaea.de/dataset/933957/files/RTS_inventory.zip")
  target = fsspec.get_mapper(str(SHAPEFILE_FOLDER))
  target.update(origin)
  targets = gpd.read_file(next(SHAPEFILE_FOLDER.glob('**/RTS_Inventory.shp')))
  return targets, ['46SDC', '46SDD', '46SDE']


def build_sentinel2_qinghai(tile_id, targets, out_dir):
    if any(out_dir.glob(f'qinghai_{tile_id}*')):
      print(f'Skipping qinghai_{tile_id} as it has already been built')
      return
    data.init_data_paths(out_dir.parent)

    scene = data.Sentinel2.scene_for_tile(tile_id,
      start_date='2019-07-01', end_date='2019-08-31',
    )
    if scene is None:
      return

    scene.add_layer(
        data.Mask(geometries=targets.geometry, bounds=None)
    )
    scene.save(out_dir / f'qinghai_{tile_id}.nc')


def run_jobs(function, n_jobs, out_dir, args_list):
  if n_jobs == 0:
    for args in tqdm(args_list):
      function(*args, out_dir)
  else:
    Parallel(n_jobs=n_jobs)(delayed(function)(*args, out_dir) for args in args_list)


if __name__ == "__main__":
    args = parser.parse_args()

    global DATA_ROOT, INPUT_DATA_DIR, BACKUP_DIR, DATA_DIR, AUX_DIR

    DATA_ROOT = Path(args.data_dir)
    timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    log_path = Path(args.log_dir) / f'build_data_cubes-{timestamp}.log'
    if not Path(args.log_dir).exists():
        os.mkdir(Path(args.log_dir))
    init_logging(log_path)
    logger = get_logger('setup_raw_data')
    logger.info('################################')
    logger.info('# Starting Building Data Cubes #')
    logger.info('################################')

    if args.mode == 'planet':
        planet_files = list((DATA_ROOT / 'input').glob('*/*_SR*.tif'))
        out_dir = DATA_ROOT / 'planet'

        run_jobs(build_planet_cube, args.n_jobs, out_dir,
                 [(f, ) for f in planet_files])
    elif args.mode == 's1':
        shapefile_root = DATA_ROOT / 'ML_training_labels' / 'retrogressive_thaw_slumps'
        out_dir = DATA_ROOT / 's1'
        out_dir.mkdir(exist_ok=True)
        targets, scenes, sites = load_annotations(shapefile_root)

        scene_infos = []
        for _, row in scenes.iterrows():
          scene_infos.append((row.geometry, row.image_id, row.image_date,
            scenes[scenes.image_id == row.image_id], # scenes
            targets[targets.image_id == row.image_id], # targets
          ))
        run_jobs(build_sentinel1_cubes, args.n_jobs, out_dir, scene_infos)
    elif args.mode == 's2':
        shapefile_root = DATA_ROOT / 'ML_training_labels' / 'retrogressive_thaw_slumps'
        out_dir = DATA_ROOT / 's2'
        out_dir.mkdir(exist_ok=True)
        targets, scenes, sites = load_annotations(shapefile_root)

        scene_infos = []
        for _, row in scenes.iterrows():
          scene_infos.append((row.geometry, row.image_id, row.image_date,
            scenes[scenes.image_id == row.image_id], # scenes
            targets[targets.image_id == row.image_id], # targets
          ))
        run_jobs(build_sentinel2_cubes, args.n_jobs, out_dir, scene_infos)
    elif args.mode == 's2_4w':
        shapefile_root = DATA_ROOT / 'ML_training_labels' / 'retrogressive_thaw_slumps'
        out_dir = DATA_ROOT / 's2_4w'
        out_dir.mkdir(exist_ok=True)
        targets, scenes, sites = load_annotations(shapefile_root)

        scene_infos = []
        for _, row in scenes.iterrows():
          scene_infos.append((row.geometry, row.image_id, row.image_date,
            scenes[scenes.image_id == row.image_id], # scenes
            targets[targets.image_id == row.image_id], # targets
          ))
        run_jobs(build_sentinel2_4w, args.n_jobs, out_dir, scene_infos[:1])
    elif args.mode == 's2_timeseries':
        shapefile_root = DATA_ROOT / 'ML_training_labels' / 'retrogressive_thaw_slumps'
        out_dir = DATA_ROOT / 's2_timeseries_cloudfree'
        out_dir.mkdir(exist_ok=True)
        targets, scenes, sites = load_annotations(shapefile_root)

        site_info = []
        for site_poly in sites:
            tiles = scenes[scenes.intersects(site_poly)].copy()
            site_id = tiles.iloc[tiles.area.argmax()].image_id
            site_info.append((
              site_poly, site_id, tiles, targets[targets.image_id.isin(tiles.image_id)].copy()))
        run_jobs(build_sentinel2_timeseries, args.n_jobs, out_dir, site_info)
    elif args.mode == 's2_ts_inference':
        out_dir = DATA_ROOT / 's2_timeseries'
        out_dir.mkdir(exist_ok=True)
        sites = gpd.read_file(DATA_ROOT / 'active_sites.geojson')
        sites = sites.to_crs('EPSG:3995')

        RADIUS = 5
        sites['geometry'] = sites.buffer(1000 * RADIUS)

        site_info = []
        for _, site in sites.iterrows():
            site_tag = site.site.replace(' ', '_') + f'_{RADIUS}km'
            site_info.append((site.geometry, site_tag))
        run_jobs(build_unlabelled_sentinel2_timeseries, args.n_jobs, out_dir, site_info)
    elif args.mode == 'qinghai':
      out_dir = DATA_ROOT / 's2'
      out_dir.mkdir(exist_ok=True)
      targets, tile_ids = load_qinghai_annotations(DATA_ROOT)
      site_info = [(tile_id, targets) for tile_id in tile_ids]
      run_jobs(build_sentinel2_qinghai, args.n_jobs, out_dir, site_info)
    elif args.mode == 's2_unlabelled':
      out_dir = DATA_ROOT / 's2_unlabelled'
      out_dir.mkdir(exist_ok=True)

      tiles = gpd.read_file(DATA_ROOT / 's2_unlabelled/index_v2.geojson')
      site_info = [(x,) for x in tiles.Name]
      run_jobs(build_sentinel2_unlabelled, args.n_jobs, out_dir, site_info)
    elif args.mode == 's2_unlabelled_4w':
      out_dir = DATA_ROOT / 's2_unlabelled_4w'
      out_dir.mkdir(exist_ok=True)

      tiles = gpd.read_file(DATA_ROOT / 'active_tiles.geojson')
      site_info = [(x,) for x in tiles.Name]
      run_jobs(build_s2_unlabelled_multi, args.n_jobs, out_dir, site_info)
