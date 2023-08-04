from typing import Union, List
from pathlib import Path
import os
import geopandas as gpd
import pandas as pd
import shutil
import numpy as np
import rasterio


def run_inference(df, model, processing_dir, inference_dir, model_dir=Path('/isipd/projects/p_aicore_pf/initze/models/thaw_slumps'), gpu=0, run=False, patch_size=1024, margin_size=256):
    if len(df) == 0:
        print('Empty dataframe')
    else:
        tiles = ' '.join(df.name.values)
        run_string = f"CUDA_VISIBLE_DEVICES='{gpu}' python inference.py -n {model} --data_dir {processing_dir} --inference_dir {inference_dir}  --patch_size {patch_size} --margin_size {margin_size} {model_dir/model} {tiles}"
        print(run_string)
        if run:
            os.system(run_string)

def listdirs(rootdir):
    dirs = []
    for path in Path(rootdir).iterdir():
        if path.is_dir():
            #print(path)
            dirs.append(path)
    return dirs

def listdirs2(rootdir, depth=0):
    dirs = []
    for path in Path(rootdir).iterdir():
        if path.is_dir():
            if depth == 1:
                for path2 in Path(path).iterdir():
                    if path2.is_dir():
                        dirs.append(path2)
            else:
                dirs.append(path)
    return dirs

def get_PS_products_type(name):
    if len(name.split('_')) == 3:
        return 'PSScene'
    elif len(name.split('_')) == 4:
        return 'PSOrthoTile'
    else:
        None
        
def get_date_from_PSfilename(name):
    date = name.split('_')[2]
    return date
    

def get_datasets(path, depth=0, preprocessed=False):
    dirs = listdirs2(path, depth=depth)
    df = pd.DataFrame(data=dirs, columns=['path'])

    df['name'] = df.apply(lambda x: x['path'].name, axis=1)
    df['preprocessed'] = preprocessed
    df['PS_product_type'] = df.apply(lambda x: get_PS_products_type(x['name']), axis=1)
    df['image_date'] = df.apply(lambda x: get_date_from_PSfilename(x['name']), axis=1)
    df['tile_id'] = df.apply(lambda x: x['name'].split('_')[1], axis=1)
    return df

def copy_unprocessed_files(row, processing_dir, quiet=True):
    inpath = row['path']
    outpath = processing_dir / 'input' / inpath.name

    if not outpath.exists():
        if not quiet:
            print (f'Start copying {inpath.name} to {outpath}')
        shutil.copytree(inpath, outpath)
    else:
        if not quiet:
            print(f'Skipped copying {inpath.name}')

def update_DEM(vrt_target_dir):
    """
    Function to update elevation vrts
    """
    os.system('./create_ArcticDEM.sh')
    shutil.copy('elevation.vrt', vrt_target_dir)
    shutil.copy('slope.vrt', vrt_target_dir)
    

def get_processing_status(raw_data_dir, procesing_dir, inference_dir, model):
    # get raw tiles
    df_raw = get_datasets(raw_data_dir, depth=1)
    # get processed
    df_processed = get_datasets(procesing_dir / 'tiles', depth=0, preprocessed=True)
    # calculate prperties
    diff = df_raw[~df_raw['name'].isin(df_processed['name'])]
    df_merged = pd.concat([df_processed, diff]).reset_index()
    
    products_list = [prod.name for prod in list((inference_dir / model).glob('*'))]
    df_merged['inference_finished'] = df_merged.apply(lambda x: x['name'] in (products_list), axis=1)
    
    return df_merged


def get_processing_status_ensemble(inference_dir, model_input_names=['RTS_v5_notcvis','RTS_v5_tcvis'], model_ensemble_name='RTS_v5_ensemble'):
    """
    Get processing status for a model ensemble and its individual models based on available data.

    This function examines the contents of specified directories within the 'inference_dir'
    to determine the processing status of a model ensemble and its constituent models.
    It constructs DataFrames indicating whether data is available for each model, and whether
    the processing has been completed for both the ensemble and individual models.

    Args:
        inference_dir (Path-like): Path to the directory containing inference data.
        model_input_names (list, optional): List of model input directory names.
            Default values are ['RTS_v5_notcvis', 'RTS_v5_tcvis'].
        model_ensemble_name (str, optional): Name of the model ensemble directory.
            Default value is 'RTS_v5_ensemble'.

    Returns:
        pandas.DataFrame: A DataFrame containing the processing status for each model
        and the ensemble. Columns include 'name', 'data_available', and 'process'.

    Example:
        >>> inference_dir = Path('/path/to/inference_data')
        >>> status_df = get_processing_status_ensemble(inference_dir)

    """
    dfs = []
    for model in model_input_names:
        df = pd.DataFrame(data=[prod.name for prod in list((inference_dir / model).glob('*'))], columns=['name']).set_index('name')
        df['model_name'] = model
        dfs.append(df)
    df_ensemble = pd.DataFrame(data=[prod.name for prod in list((inference_dir / model_ensemble_name).glob('*'))], columns=['name']).set_index('name')
    df_ensemble['ensemble_name'] = model_ensemble_name
    dfs.append(df_ensemble)

    df_process = pd.concat(dfs, axis=1)
    df_process['data_available'] = ~df_process['model_name'].isna().all(axis=1)
    df_process['process'] = df_process['data_available'] & (df_process['ensemble_name'].isna())

    return df_process[['data_available', 'process']].reset_index(drop=False).rename(columns={'index':'name'})


#def create_ensemble(inference_dir: Path, modelnames: List[str], ensemblename: str, image_id: str, binary_threshold: list[float]=[0.3,0.4,0.5], delete_proba=True, delete_binary=True) -> None:
    
def create_ensemble(inference_dir: Path, modelnames: List[str], ensemblename: str, image_id: str, binary_threshold: list=[0.3,0.4,0.5], delete_proba=True, delete_binary=True):
    """
    Calculate the mean of two model predictions and write the output to disk.
    
    Args:
    modelnames (List[str]): A list of two model names.
    ensemblename (str): The name of the ensemble model.
    image_id (str): The ID of the image.
    binary_threshold (float): The binary threshold value.
    
    Returns:
    None
    """
    try:
    # setup
        outpath = inference_dir / ensemblename / image_id / f'{image_id}_{ensemblename}_proba.tif'
        os.makedirs(outpath.parent, exist_ok=True)

        # calculate
        image1 = inference_dir / modelnames[0] / image_id / 'pred_probability.tif'
        image2 = inference_dir / modelnames[1] / image_id / 'pred_probability.tif'

        with rasterio.open(image1) as src1:
            with rasterio.open(image2) as src2:
                a1 = src1.read()
                a2 = src2.read()

            out_meta = src1.meta.copy()
            out_meta_binary = out_meta.copy()
            out_meta_binary['dtype'] = 'uint8'

        out = np.mean([a1, a2], axis=0)
        with rasterio.open(outpath, 'w', **out_meta) as target:
            target.write(out)


        # write binary raster
        for threshold in binary_threshold:
            thresh_str = str(threshold).replace('.','')
            outpath_class = Path(str(outpath).replace('proba', f'class_{thresh_str}'))
            outpath_shp = outpath_class.with_suffix('.gpkg')

            out_binary = (out >= threshold)

            with rasterio.open(outpath_class, 'w', **out_meta_binary, compress='deflate') as target:
                target.write(out_binary)

            # make vector
            s_polygonize = f'gdal_polygonize.py {outpath_class} -q -mask {outpath_class} -f "GPKG" {outpath_shp}'
            os.system(s_polygonize)
            if delete_binary:
                os.remove(outpath_class)

        # delete files
        if delete_proba:
            os.remove(outpath)
            
        return 0
    
    except:
        return 1
    
    
def load_and_parse_vector(file_path: Union[str, Path]) -> gpd.GeoDataFrame:
    """
    Load a GeoDataFrame from a given file path, reproject it to EPSG:4326,
    and parse image metadata from the file path to add as attributes.

    This function reads a GeoDataFrame from the specified file path, converts
    the GeoDataFrame's coordinate reference system to EPSG:4326, and parses
    the image ID from the parent directory name of the file path. It then
    extracts components from the image ID (take ID, tile ID, date, and satellite)
    and adds them as new columns in the GeoDataFrame.

    Args:
        file_path (str or pathlib.Path): Path to the vector file.

    Returns:
        geopandas.GeoDataFrame: A GeoDataFrame with added attributes representing
        parsed image metadata.

    Example:
        >>> file_path = '/path/to/your/vector_file.geojson'
        >>> parsed_gdf = load_and_parse_vector(file_path)
    """
    gdf = gpd.read_file(file_path).to_crs('EPSG:4326')

    image_id = file_path.parent.name
    take_id, tile_id, date, satellite = image_id.split('_')

    gdf['image_id'] = image_id
    gdf['take_id'] = take_id
    gdf['tile_id'] = tile_id
    gdf['date'] = date
    gdf['year'] = pd.to_datetime(gdf['date'], infer_datetime_format=True).dt.year
    gdf['satellite'] = satellite
    
    return gdf