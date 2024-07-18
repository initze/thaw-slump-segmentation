# # Create ensemble results from several model outputs
# ### Imports

import argparse
import shutil
from datetime import datetime
from pathlib import Path
from typing import List

import geopandas as gpd
import typer
from joblib import Parallel, delayed
from tqdm import tqdm
from typing_extensions import Annotated

# from tqdm.notebook import tqdm
from thaw_slump_segmentation.postprocessing import (
    create_ensemble_v2,
    get_processing_status,
    get_processing_status_ensemble,
    load_and_parse_vector,
)


def process_03_ensemble(
    raw_data_dir: Annotated[Path, typer.Option(help='Location of raw data')] = Path(
        '/isipd/projects/p_aicore_pf/initze/data/planet/planet_data_inference_grid/tiles'
    ),
    processing_dir: Annotated[Path, typer.Option(help='Location for data processing')] = Path(
        '/isipd/projects/p_aicore_pf/initze/processing'
    ),
    inference_dir: Annotated[Path, typer.Option(help='Target directory for inference results')] = Path(
        '/isipd/projects/p_aicore_pf/initze/processed/inference'
    ),
    model_dir: Annotated[Path, typer.Option(help='Target directory for models')] = Path(
        '/isipd/projects/p_aicore_pf/initze/models/thaw_slumps'
    ),
    ensemble_name: Annotated[str, typer.Option(help='Target directory for models')] = 'RTS_v6_ensemble_v2',
    model_names: Annotated[List[str], typer.Option(help="Model name, examples ['RTS_v6_tcvis', 'RTS_v6_notcvis']")] = [
        'RTS_v6_tcvis',
        'RTS_v6_notcvis',
    ],
    gpu: Annotated[int, typer.Option(help='GPU IDs to use for edge cleaning')] = 0,
    n_jobs: Annotated[int, typer.Option(help='number of CPU jobs for ensembling')] = 15,
    n_vector_loaders: Annotated[int, typer.Option(help='number of parallel vector loaders for final merge')] = 6,
    max_images: Annotated[int, typer.Option(help='Maximum number of images to process (optional)')] = None,
    vector_output_format: Annotated[
        List[str], typer.Option(help='Output format extension of ensembled vector files')
    ] = ['gpkg', 'parquet'],
    ensemble_thresholds: Annotated[
        List[float],
        typer.Option(help='Thresholds for polygonized outputs of the ensemble, needs to be string, see examples'),
    ] = [0.4, 0.45, 0.5],
    ensemble_border_size: Annotated[
        int, typer.Option(help='Number of pixels to remove around the border and no data')
    ] = 10,
    ensemble_mmu: Annotated[int, typer.Option(help='Minimum mapping unit of output objects in pixels')] = 32,
    try_gpu: Annotated[bool, typer.Option(help='set to try image processing with gpu')] = False,
    force_vector_merge: Annotated[
        bool, typer.Option(help='force merging of output vectors even if no new ensemble tiles were processed')
    ] = False,
    save_binary: Annotated[bool, typer.Option(help='set to keep intermediate binary rasters')] = False,
    save_probability: Annotated[bool, typer.Option(help='set to keep intermediate probability rasters')] = False,
    filter_water: Annotated[bool, typer.Option(help='set to remove polygons over water')] = False,
):
    ### Start
    # check if cucim is available
    try:
        import cucim  # type: ignore  # noqa: F401

        if try_gpu:
            try_gpu = True
            print('Running ensembling with GPU!')
        else:
            try_gpu = False
            print('Running ensembling with CPU!')
    except Exception as e:
        try_gpu = False
        print(f'Cucim import failed: {e}')

    # setup all params
    kwargs_ensemble = {
        'ensemblename': ensemble_name,
        'inference_dir': inference_dir,
        'modelnames': model_names,
        'binary_threshold': ensemble_thresholds,
        'border_size': ensemble_border_size,
        'minimum_mapping_unit': ensemble_mmu,
        'save_binary': save_binary,
        'save_probability': save_probability,
        'try_gpu': try_gpu,  # currently default to CPU only
        'gpu': gpu,
    }

    # Check for finalized products
    get_processing_status(raw_data_dir, processing_dir, inference_dir, model=kwargs_ensemble['ensemblename'])
    df_ensemble_status = get_processing_status_ensemble(
        inference_dir,
        model_input_names=kwargs_ensemble['modelnames'],
        model_ensemble_name=kwargs_ensemble['ensemblename'],
    )
    # Check which need to be process - check for already processed and invalid files
    process = df_ensemble_status[df_ensemble_status['process']]
    n_images = len(process.iloc[:max_images])
    # #### Run Ensemble Merging
    if len(process) > 0:
        print(f'Start running ensemble for {n_images} images with {n_jobs} parallel jobs!')
        print(f'Target ensemble name: {ensemble_name}')
        print(f'Source model output {model_names}')
        _ = Parallel(n_jobs=n_jobs)(
            delayed(create_ensemble_v2)(image_id=process.iloc[row]['name'], **kwargs_ensemble)
            for row in tqdm(range(n_images))
        )
    else:
        print(f'Skipped ensembling, all files ready for {ensemble_name}!')

    # # #### run parallelized batch

    if (len(process) > 0) or force_vector_merge:
        # ### Merge vectors to complete dataset
        # set probability levels: 'class_05' means 50%, 'class_045' means 45%. This is the regex to search for vector naming
        # proba_strings = args.ensemble_thresholds
        # TODO: needs to be 'class_04',
        proba_strings = [f'class_{thresh}'.replace('.', '') for thresh in ensemble_thresholds]

        for proba_string in proba_strings:
            # read all files which follow the above defined threshold
            flist = list((inference_dir / ensemble_name).glob(f'*/*_{proba_string}.gpkg'))
            len(flist)
            # load them in parallel
            print(f'Loading results {proba_string}')
            out = Parallel(n_jobs=6)(
                delayed(load_and_parse_vector)(f, filter_water=filter_water) for f in tqdm(flist[:max_images])
            )
            # merge them and save to geopackage file
            print('Merging results')
            merged_gdf = gpd.pd.concat(out)

            # if filter_water:
            #     # water removal here
            #     merged_gdf = filter_remove_water(merged_gdf)

            for vector_format in vector_output_format:
                # Save output to vector
                save_file = inference_dir / ensemble_name / f'merged_{proba_string}.{vector_format}'

                # make file backup if necessary
                if save_file.exists():
                    # Get the current timestamp
                    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                    # Create the backup file name
                    save_file_bk = (
                        inference_dir / ensemble_name / f'merged_{proba_string}_bk_{timestamp}.{vector_format}'
                    )
                    print(f'Creating backup of file {save_file} to {save_file_bk}')
                    shutil.move(save_file, save_file_bk)

                # save to files
                print(f'Saving vectors to {save_file}')
                if vector_format in ['shp', 'gpkg']:
                    merged_gdf.to_file(save_file)
                elif vector_format in ['parquet']:
                    merged_gdf.to_parquet(save_file)
                else:
                    print(f'Unknown format {vector_format}!')


def main():
    # Add argument definitions
    parser = argparse.ArgumentParser(
        description='Script to run auto inference for RTS', formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        '--raw_data_dir',
        type=Path,
        default=Path('/isipd/projects/p_aicore_pf/initze/data/planet/planet_data_inference_grid/tiles'),
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
    parser.add_argument('--ensemble_name', type=str, default='RTS_v6_ensemble_v2', help='Target directory for models')
    parser.add_argument(
        '--model_names',
        type=str,
        nargs='+',
        default=['RTS_v6_tcvis', 'RTS_v6_notcvis'],
        help="Model name, examples ['RTS_v6_tcvis', 'RTS_v6_notcvis']",
    )
    parser.add_argument('--gpu', type=int, default=0, help='GPU IDs to use for edge cleaning')
    parser.add_argument('--n_jobs', type=int, default=15, help='number of CPU jobs for ensembling')
    parser.add_argument(
        '--n_vector_loaders', type=int, default=6, help='number of parallel vector loaders for final merge'
    )
    parser.add_argument('--max_images', type=int, default=None, help='Maximum number of images to process (optional)')
    parser.add_argument(
        '--vector_output_format',
        type=str,
        nargs='+',
        default=['gpkg', 'parquet'],
        help='Output format extension of ensembled vector files',
    )
    parser.add_argument(
        '--ensemble_thresholds',
        type=float,
        nargs='+',
        default=[0.4, 0.45, 0.5],
        help='Thresholds for polygonized outputs of the ensemble, needs to be string, see examples',
    )
    parser.add_argument(
        '--ensemble_border_size', type=int, default=10, help='Number of pixels to remove around the border and no data'
    )
    parser.add_argument('--ensemble_mmu', type=int, default=32, help='Minimum mapping unit of output objects in pixels')
    parser.add_argument('--save_binary', action='store_true', help='set to keep intermediate binary rasters')
    parser.add_argument('--save_probability', action='store_true', help='set to keep intermediate probability rasters')
    parser.add_argument('--try_gpu', action='store_true', help='set to try image processing with gpu')
    parser.add_argument(
        '--force_vector_merge',
        action='store_true',
        help='force merging of output vectors even if no new ensemble tiles were processed',
    )
    parser.add_argument('--filter_water', action='store_true', help='set to remove polygons over water')

    args = parser.parse_args()

    process_03_ensemble(
        raw_data_dir=args.raw_data_dir,
        processing_dir=args.processing_dir,
        inference_dir=args.inference_dir,
        model_dir=args.model_dir,
        ensemble_name=args.ensemble_name,
        model_names=args.model_names,
        gpu=args.gpu,
        n_jobs=args.n_jobs,
        n_vector_loaders=args.n_vector_loaders,
        max_images=args.max_images,
        vector_output_format=args.vector_output_format,
        ensemble_thresholds=args.ensemble_thresholds,
        ensemble_border_size=args.ensemble_border_size,
        ensemble_mmu=args.ensemble_mmu,
        try_gpu=args.try_gpu,
        force_vector_merge=args.force_vector_merge,
        save_binary=args.save_binary,
        save_probability=args.save_probability,
        filter_water=args.filter_water,
    )


if __name__ == '__main__':
    main()
