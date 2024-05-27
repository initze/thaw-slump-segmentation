import argparse
import os
import shutil
from datetime import datetime
from pathlib import Path
from typing import List

import geopandas as gpd
import numpy as np
import pandas as pd
import typer
from joblib import Parallel, delayed
from tqdm import tqdm
from typing_extensions import Annotated
import swifter

from ..postprocessing import (
    copy_unprocessed_files,
    get_processing_status,
    load_and_parse_vector,
    run_inference,
    update_DEM2,
)


def process_02_inference(
    code_dir: Annotated[Path, typer.Option(help='Local code directory')] = Path(
        '/isipd/projects/p_aicore_pf/initze/code/aicore_inference'
    ),
    raw_data_dir: Annotated[List[Path], typer.Option(help='Location of raw data')] = [
        Path('/isipd/projects/p_aicore_pf/initze/data/planet/planet_data_inference_grid/scenes'),
        Path('/isipd/projects/p_aicore_pf/initze/data/planet/planet_data_inference_grid/tiles'),
    ],
    processing_dir: Annotated[Path, typer.Option(help='Location for data processing')] = Path(
        '/isipd/projects/p_aicore_pf/initze/processing'
    ),
    inference_dir: Annotated[Path, typer.Option(help='Target directory for inference results')] = Path(
        '/isipd/projects/p_aicore_pf/initze/processed/inference'
    ),
    model_dir: Annotated[Path, typer.Option(help='Target directory for models')] = Path(
        '/isipd/projects/p_aicore_pf/initze/models/thaw_slumps'
    ),
    model: Annotated[
        str, typer.Option(help="Model name, examples ['RTS_v6_tcvis', 'RTS_v6_notcvis']")
    ] = 'RTS_v6_tcvis',
    use_gpu: Annotated[List[int], typer.Option(help='List of GPU IDs to use, space separated')] = [0],
    runs_per_gpu: Annotated[int, typer.Option(help='Number of runs per GPU')] = 5,
    max_images: Annotated[int, typer.Option(help='Maximum number of images to process (optional)')] = None,
    skip_vrt: Annotated[bool, typer.Option(help='set to skip DEM vrt creation')] = False,
    skip_vector_save: Annotated[bool, typer.Option(help='set to skip output vector creation')] = False,
):
    """Script to run auto inference for RTS"""
    # ### List all files with properties
    # TODO: run double for both paths
    print('Checking processing status!')
    # read processing status for raw data list
    # TODO: check here - produces very large output when double checking
    df_processing_status_list = [
        get_processing_status(rdd, processing_dir, inference_dir, model) for rdd in raw_data_dir
    ]

    # get df for preprocessing
    df_final = pd.concat(df_processing_status_list).drop_duplicates()

    # TODO: move to function
    # print basic information
    total_images = int(len(df_final))
    preprocessed_images = int(df_final.preprocessed.sum())
    preprocessing_images = int(total_images - preprocessed_images)
    finished_images = int(df_final.inference_finished.sum())
    print(f'Number of images: {total_images}')
    print(f'Number of preprocessed images: {preprocessed_images}')
    print(f'Number of images for preprocessing: {preprocessing_images}')
    print(f'Number of images for inference: {preprocessed_images - finished_images}')
    print(f'Number of finished images: {finished_images}')

    # TODO: images with processing status True but Inference False are crappy

    if total_images == finished_images:
        print('No processing needed: all images are already processed!')
        return 0

    ## Preprocessing
    # #### Update Arctic DEM data
    if skip_vrt:
        print('Skipping Elevation VRT creation!')
    else:
        print('Updating Elevation VRTs!')
        dem_data_dir = Path('/isipd/projects/p_aicore_pf/initze/data/ArcticDEM')
        vrt_target_dir = Path('/isipd/projects/p_aicore_pf/initze/processing/auxiliary/ArcticDEM')
        update_DEM2(dem_data_dir=dem_data_dir, vrt_target_dir=vrt_target_dir)

    # #### Copy data for Preprocessing
    # make better documentation

    df_preprocess = df_final[df_final.preprocessed == False]
    print(f'Number of images to preprocess: {len(df_preprocess)}')

    # Cleanup processing directories to avoid incomplete processing
    input_dir_dslist = list((processing_dir / 'input').glob('*'))
    if len(input_dir_dslist) > 0:
        print(f"Cleaning up {(processing_dir / 'input')}")
        for d in input_dir_dslist:
            print('Delete', d)
            shutil.rmtree(d)
    else:
        print('Processing directory is ready, nothing to do!')

    # Copy Data
    _ = df_preprocess.swifter.apply(lambda x: copy_unprocessed_files(x, processing_dir), axis=1)

    # #### Run Preprocessing
    import warnings

    warnings.filterwarnings('ignore')

    N_JOBS = 40
    print(f'Preprocessing {len(df_preprocess)} images')  # fix this
    if len(df_preprocess) > 0:
        pp_string = f'setup_raw_data --data_dir {processing_dir} --n_jobs {N_JOBS} --nolabel'
        os.system(pp_string)

    # ## Processing/Inference
    # rerun processing status
    df_processing_status2 = pd.concat(
        [get_processing_status(rdd, processing_dir, inference_dir, model) for rdd in raw_data_dir]
    ).drop_duplicates()
    # Filter to images that are not preprocessed yet
    df_process = df_final[df_final.inference_finished == False]
    # update overview and filter accordingly - really necessary?
    df_process_final = (
        df_process.set_index('name')
        .join(df_processing_status2[df_processing_status2['preprocessed']][['name']].set_index('name'), how='inner')
        .reset_index(drop=False)
        .iloc[:max_images]
    )
    # validate if images are correctly preprocessed
    df_process_final['preprocessing_valid'] = (
        df_process_final.apply(lambda x: len(list(x['path'].glob('*'))), axis=1) >= 5
    )
    # final filtering process to remove incorrectly preprocessed data
    df_process_final = df_process_final[df_process_final['preprocessing_valid']]

    # TODO: check for empty files and processing
    print(f'Number of images: {len(df_process_final)}')

    # #### Parallel runs
    # Make splits to distribute the processing
    n_splits = len(use_gpu) * runs_per_gpu
    df_split = np.array_split(df_process_final, n_splits)
    gpu_split = use_gpu * runs_per_gpu

    # for split in df_split:
    #    print(f'Number of images: {len(split)}')

    print('Run inference!')
    # ### Parallel Inference execution
    _ = Parallel(n_jobs=n_splits)(
        delayed(run_inference)(
            df_split[split],
            model=model,
            processing_dir=processing_dir,
            inference_dir=inference_dir,
            model_dir=model_dir,
            gpu=gpu_split[split],
            run=True,
        )
        for split in range(n_splits)
    )
    # #### Merge output files

    if not skip_vector_save:
        if len(df_process_final) > 0:
            # read all files which following the above defined threshold
            flist = list((inference_dir / model).glob('*/*pred_binarized.shp'))
            len(flist)

            # Save output vectors to merged file
            # load them in parallel
            out = Parallel(n_jobs=6)(delayed(load_and_parse_vector)(f) for f in tqdm(flist[:]))

            # merge them and save to geopackage file
            merged_gdf = gpd.pd.concat(out)
            save_file = inference_dir / model / f'{model}_merged.gpkg'

            # check if file already exists, create backup file if exists
            if save_file.exists():
                # Get the current timestamp
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                # Create the backup file name
                save_file_bk = inference_dir / model / f'{model}_merged_bk_{timestamp}.gpkg'
                print(f'Creating backup of file {save_file} to {save_file_bk}')
                shutil.move(save_file, save_file_bk)

            # save to files
            print(f'Saving vectors to {save_file}')
            merged_gdf.to_file(save_file)
    else:
        print('Skipping output vector creation!')


def main():
    # ### Settings

    # Add argument definitions
    parser = argparse.ArgumentParser(
        description='Script to run auto inference for RTS', formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        '--code_dir',
        type=Path,
        default=Path('/isipd/projects/p_aicore_pf/initze/code/aicore_inference'),
        help='Local code directory',
    )
    parser.add_argument(
        '--raw_data_dir',
        type=Path,
        nargs='+',
        default=[
            Path('/isipd/projects/p_aicore_pf/initze/data/planet/planet_data_inference_grid/scenes'),
            Path('/isipd/projects/p_aicore_pf/initze/data/planet/planet_data_inference_grid/tiles'),
        ],
        help='Location of raw data',
    )
    parser.add_argument(
        '--processing_dir',
        type=Path,
        default=Path('/isipd/projects/p_aicore_pf/initze/processing'),
        help='Location for data processing',
    )
    parser.add_argument(
        '--inference_dir',
        type=Path,
        default=Path('/isipd/projects/p_aicore_pf/initze/processed/inference'),
        help='Target directory for inference results',
    )
    parser.add_argument(
        '--model_dir',
        type=Path,
        default=Path('/isipd/projects/p_aicore_pf/initze/models/thaw_slumps'),
        help='Target directory for models',
    )
    parser.add_argument(
        '--model', type=str, default='RTS_v6_tcvis', help="Model name, examples ['RTS_v6_tcvis', 'RTS_v6_notcvis']"
    )
    parser.add_argument('--use_gpu', nargs='+', type=int, default=[0], help='List of GPU IDs to use, space separated')
    parser.add_argument('--runs_per_gpu', type=int, default=5, help='Number of runs per GPU')
    parser.add_argument('--max_images', type=int, default=None, help='Maximum number of images to process (optional)')
    parser.add_argument('--skip_vrt', action='store_true', help='set to skip DEM vrt creation')
    parser.add_argument('--skip_vector_save', action='store_true', help='set to skip output vector creation')

    # TODO, make flag to skip vrt
    args = parser.parse_args()

    process_02_inference(
        code_dir=args.code_dir,
        raw_data_dir=args.raw_data_dir,
        processing_dir=args.processing_dir,
        inference_dir=args.inference_dir,
        model_dir=args.model_dir,
        model=args.model,
        use_gpu=args.use_gpu,
        runs_per_gpu=args.runs_per_gpu,
        max_images=args.max_images,
        skip_vrt=args.skip_vrt,
        skip_vector_save=args.skip_vector_save,
    )


if __name__ == '__main__':
    main()
