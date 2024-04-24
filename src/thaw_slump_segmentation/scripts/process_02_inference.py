
from pathlib import Path
import torch
import pandas as pd
import os
import numpy as np
import tqdm
from joblib import delayed, Parallel
import shutil
from tqdm import tqdm
import swifter
from datetime import datetime

from ..postprocessing import *

# ### Settings 

def main():
    # Local code dir
    CODE_DIR = Path('/isipd/projects/p_aicore_pf/initze/code/aicore_inference')
    # Location of raw data
    #RAW_DATA_DIR = Path('/isipd/projects/p_aicore_pf/initze/data/planet/planet_data_inference_grid/tiles')
    RAW_DATA_DIR = Path('/isipd/projects/p_aicore_pf/initze/data/planet/planet_data_inference_grid/scenes')
    # Location data processing
    PROCESSING_DIR = Path('/isipd/projects/p_aicore_pf/initze/processing')
    # Target directory for
    INFERENCE_DIR = Path('/isipd/projects/p_aicore_pf/initze/processed/inference')

    # Target to models - RTS
    MODEL_DIR = Path('/isipd/projects/p_aicore_pf/initze/models/thaw_slumps')

    MODEL='RTS_v6_notcvis'
    models = ['RTS_v6_notcvis', 'RTS_v6_tcvis']

    #USE_GPU = [1,2,3,4]
    USE_GPU = [1,2]
    RUNS_PER_GPU = 5
    MAX_IMAGES = None

    # ### List all files with properties
    df_processing_status = get_processing_status(RAW_DATA_DIR, PROCESSING_DIR, INFERENCE_DIR, MODEL)

    df_final = df_processing_status

    total_images = len(df_final)
    preprocessed_images = df_final.preprocessed.sum()
    finished_images = df_final.inference_finished.sum()
    print(f'Number of images: {total_images}')
    print(f'Number of preprocessed images: {preprocessed_images}')
    print(f'Number of finished images: {finished_images}')
    print(f'Number of image to process: {preprocessed_images - finished_images}')

    ## Preprocessing

    # #### Update Arctic DEM data 
    print('Updating Elevation VRTs!')
    dem_data_dir = Path('/isipd/projects/p_aicore_pf/initze/data/ArcticDEM')
    vrt_target_dir = Path('/isipd/projects/p_aicore_pf/initze/processing/auxiliary/ArcticDEM')
    #update_DEM(vrt_target_dir)
    update_DEM2(dem_data_dir=dem_data_dir, vrt_target_dir=vrt_target_dir)

    # #### Copy data for Preprocessing 
    # make better documentation

    df_preprocess = df_final[~df_final.preprocessed]
    print(f'Number of images to preprocess: {len(df_preprocess)}')

    # Cleanup processing directories to avoid incomplete processing
    input_dir_dslist = list((PROCESSING_DIR / 'input').glob('*'))
    if len(input_dir_dslist) > 0:
        print(input_dir_dslist)
        for d in input_dir_dslist:
            print('Delete', d)
            shutil.rmtree(d)
    else:
        print('Processing directory is ready, nothing to do!')

    # Copy Data
    _ = df_preprocess.swifter.apply(lambda x: copy_unprocessed_files(x, PROCESSING_DIR), axis=1)

    # #### Run Preprocessing 
    import warnings
    warnings.filterwarnings('ignore')

    N_JOBS=40
    print(f'Preprocessing {len(df_preprocess)} images') #fix this
    if len(df_preprocess) > 0:
        pp_string = f'python setup_raw_data.py --data_dir {PROCESSING_DIR} --n_jobs {N_JOBS} --nolabel'
        os.system(pp_string)

    # ## Processing/Inference
    # rerun processing status
    df_processing_status2 = get_processing_status(RAW_DATA_DIR, PROCESSING_DIR, INFERENCE_DIR, MODEL)

    # Filter to images that are not preprocessed yet
    df_process = df_final[~df_final.inference_finished]
    # update overview and filter accordingly - really necessary?
    df_process_final = df_process.set_index('name').join(df_processing_status2[df_processing_status2['preprocessed']][['name']].set_index('name'), how='inner').reset_index(drop=False).iloc[:MAX_IMAGES]
    # validate if images are correctly preprocessed
    df_process_final['preprocessing_valid'] = (df_process_final.apply(lambda x: len(list(x['path'].glob('*'))), axis=1) >= 5)
    # final filtering process to remove incorrectly preprocessed data
    df_process_final = df_process_final[df_process_final['preprocessing_valid']]

    print(f'Number of images:', len(df_process_final))

    # #### Parallel runs 
    # Make splits to distribute the processing
    n_splits = len(USE_GPU) * RUNS_PER_GPU
    df_split = np.array_split(df_process_final, n_splits)
    gpu_split = USE_GPU * RUNS_PER_GPU

    #for split in df_split:
    #    print(f'Number of images: {len(split)}')

    print('Run inference!')
    # ### Parallel Inference execution
    _ = Parallel(n_jobs=n_splits)(delayed(run_inference)(df_split[split], model=MODEL, processing_dir=PROCESSING_DIR, inference_dir=INFERENCE_DIR, model_dir=MODEL_DIR, gpu=gpu_split[split], run=True) for split in range(n_splits))
    # #### Merge output files

    # read all files which following the above defined threshold
    flist = list((INFERENCE_DIR / MODEL).glob(f'*/*pred_binarized.shp'))
    len(flist)
    # TODO:uncomment here
    if len(df_process_final) > 0:
        # load them in parallel
        out = Parallel(n_jobs=6)(delayed(load_and_parse_vector)(f) for f in tqdm(flist[:]))
        
        # merge them and save to geopackage file
        merged_gdf = gpd.pd.concat(out)
        save_file = INFERENCE_DIR / MODEL / f'{MODEL}_merged.gpkg'
        
        # check if file already exists, create backup file if exists
        if save_file.exists():
            # Get the current timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            # Create the backup file name
            save_file_bk = INFERENCE_DIR / MODEL / f"{MODEL}_merged_bk_{timestamp}.gpkg"
            print (f'Creating backup of file {save_file} to {save_file_bk}')
            shutil.move(save_file, save_file_bk)
        
        # save to files
        print(f'Saving vectors to {save_file}')
        merged_gdf.to_file(save_file)


if __name__ == "__main__":
    main()
