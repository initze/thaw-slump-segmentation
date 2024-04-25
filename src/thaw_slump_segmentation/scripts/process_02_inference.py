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
import argparse

from ..postprocessing import *

# ### Settings 
# Add argument definitions
parser = argparse.ArgumentParser(description="Script to run auto inference for RTS", 
                    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--code_dir", type=Path, default=Path('/isipd/projects/p_aicore_pf/initze/code/aicore_inference'),
                    help="Local code directory")
parser.add_argument("--raw_data_dir", type=Path, default=Path('/isipd/projects/p_aicore_pf/initze/data/planet/planet_data_inference_grid/scenes'),
                    help="Location of raw data")
parser.add_argument("--processing_dir", type=Path, default=Path('/isipd/projects/p_aicore_pf/initze/processing'),
                    help="Location for data processing")
parser.add_argument("--inference_dir", type=Path, default=Path('/isipd/projects/p_aicore_pf/initze/processed/inference'),
                    help="Target directory for inference results")
parser.add_argument("--model_dir", type=Path, default=Path('/isipd/projects/p_aicore_pf/initze/models/thaw_slumps'),
                    help="Target directory for models")
parser.add_argument("--model", type=str, default='RTS_v6_tcvis',
                    help="Model name, examples ['RTS_v6_tcvis', 'RTS_v6_notcvis']")
parser.add_argument("--use_gpu", nargs="+", type=int, default=[0],
                    help="List of GPU IDs to use, space separated")
parser.add_argument("--runs_per_gpu", type=int, default=5,
                    help="Number of runs per GPU")
parser.add_argument("--max_images", type=int, default=None,
                    help="Maximum number of images to process (optional)")
parser.add_argument("--skip_vrt", action="store_false", 
                    help="set to skip DEM vrt creation")
parser.add_argument("--skip_vector_save", action="store_true", 
                    help="set to skip output vector creation")

# TODO, make flag to skip vrt
args = parser.parse_args()

def main():
    # ### List all files with properties
    df_processing_status = get_processing_status(args.raw_data_dir, args.processing_dir, args.inference_dir, args.model)

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
    if args.skip_vrt == False:
        print('Updating Elevation VRTs!')
        dem_data_dir = Path('/isipd/projects/p_aicore_pf/initze/data/ArcticDEM')
        vrt_target_dir = Path('/isipd/projects/p_aicore_pf/initze/processing/auxiliary/ArcticDEM')
        #update_DEM(vrt_target_dir)
        update_DEM2(dem_data_dir=dem_data_dir, vrt_target_dir=vrt_target_dir)
    else:
        print('Skipping Elevation VRT creation!')


    # #### Copy data for Preprocessing 
    # make better documentation

    df_preprocess = df_final[~df_final.preprocessed]
    print(f'Number of images to preprocess: {len(df_preprocess)}')

    # Cleanup processing directories to avoid incomplete processing
    input_dir_dslist = list((args.processing_dir / 'input').glob('*'))
    if len(input_dir_dslist) > 0:
        print(input_dir_dslist)
        for d in input_dir_dslist:
            print('Delete', d)
            shutil.rmtree(d)
    else:
        print('Processing directory is ready, nothing to do!')

    # Copy Data
    _ = df_preprocess.swifter.apply(lambda x: copy_unprocessed_files(x, args.processing_dir), axis=1)

    # #### Run Preprocessing 
    import warnings
    warnings.filterwarnings('ignore')

    N_JOBS=40
    print(f'Preprocessing {len(df_preprocess)} images') #fix this
    if len(df_preprocess) > 0:
        pp_string = f'setup_raw_data --data_dir {args.processing_dir} --n_jobs {N_JOBS} --nolabel'
        os.system(pp_string)

    # ## Processing/Inference
    # rerun processing status
    df_processing_status2 = get_processing_status(args.raw_data_dir, args.processing_dir, args.inference_dir, args.model)

    # Filter to images that are not preprocessed yet
    df_process = df_final[~df_final.inference_finished]
    # update overview and filter accordingly - really necessary?
    df_process_final = df_process.set_index('name').join(df_processing_status2[df_processing_status2['preprocessed']][['name']].set_index('name'), how='inner').reset_index(drop=False).iloc[:args.max_images]
    # validate if images are correctly preprocessed
    df_process_final['preprocessing_valid'] = (df_process_final.apply(lambda x: len(list(x['path'].glob('*'))), axis=1) >= 5)
    # final filtering process to remove incorrectly preprocessed data
    df_process_final = df_process_final[df_process_final['preprocessing_valid']]

    print(f'Number of images:', len(df_process_final))

    # #### Parallel runs 
    # Make splits to distribute the processing
    n_splits = len(args.use_gpu) * args.runs_per_gpu
    df_split = np.array_split(df_process_final, n_splits)
    gpu_split = args.use_gpu * args.runs_per_gpu

    #for split in df_split:
    #    print(f'Number of images: {len(split)}')

    print('Run inference!')
    # ### Parallel Inference execution
    _ = Parallel(n_jobs=n_splits)(delayed(run_inference)(df_split[split], model=args.model, processing_dir=args.processing_dir, inference_dir=args.inference_dir, model_dir=args.model_dir, gpu=gpu_split[split], run=True) for split in range(n_splits))
    # #### Merge output files

    if not args.skip_vector_save:
    # read all files which following the above defined threshold
        flist = list((args.inference_dir / args.model).glob(f'*/*pred_binarized.shp'))
        len(flist)
        # TODO:uncomment here
        if len(df_process_final) > 0:
            # load them in parallel
            out = Parallel(n_jobs=6)(delayed(load_and_parse_vector)(f) for f in tqdm(flist[:]))
            
            # merge them and save to geopackage file
            merged_gdf = gpd.pd.concat(out)
            save_file = args.inference_dir / args.model / f'{args.model}_merged.gpkg'
            
            # check if file already exists, create backup file if exists
            if save_file.exists():
                # Get the current timestamp
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                # Create the backup file name
                save_file_bk = args.inference_dir / args.model / f"{args.model}_merged_bk_{timestamp}.gpkg"
                print (f'Creating backup of file {save_file} to {save_file_bk}')
                shutil.move(save_file, save_file_bk)
            
            # save to files
            print(f'Saving vectors to {save_file}')
            merged_gdf.to_file(save_file)
    else:
        print('Skipping output vector creation!')

if __name__ == "__main__":
    main()