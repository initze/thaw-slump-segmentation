# # Create ensemble results from several model outputs
# ### Imports 

from pathlib import Path
import pandas as pd
from joblib import delayed, Parallel
#from tqdm.notebook import tqdm
from tqdm import tqdm
from ..postprocessing import *
import geopandas as gpd
from datetime import datetime
import argparse

# Add argument definitions
parser = argparse.ArgumentParser(description="Script to run auto inference for RTS")
parser.add_argument("--raw_data_dir", type=Path, default=Path('/isipd/projects/p_aicore_pf/initze/data/planet/planet_data_inference_grid/tiles'),
                    help="Location of raw data")
parser.add_argument("--processing_dir", type=Path, default=Path('/isipd/projects/p_aicore_pf/initze/processing'),
                    help="Location for data processing")
parser.add_argument("--inference_dir", type=Path, default=Path('/isipd/projects/p_aicore_pf/initze/processed/inference'),
                    help="Target directory for inference results")
parser.add_argument("--model_dir", type=Path, default=Path('/isipd/projects/p_aicore_pf/initze/models/thaw_slumps'),
                    help="Target directory for models")
parser.add_argument("--ensemble_name", type=str, default='RTS_v6_ensemble_v2',
                    help="Target directory for models")
parser.add_argument("--model_names", type=str, nargs='+', default=['RTS_v6_tcvis', 'RTS_v6_notcvis'],
                    help="Model name, examples ['RTS_v6_tcvis', 'RTS_v6_notcvis']")
parser.add_argument("--use_gpu", type=int, default=0,
                    help="GPU IDs to use for edge cleaning")
parser.add_argument("--n_jobs", type=int, default=15,
                    help="number of CPU jobs for ensembling")
parser.add_argument("--n_vector_loaders", type=int, default=6,
                    help="number of parallel vector loaders for final merge")
parser.add_argument("--max_images", type=int, default=None,
                    help="Maximum number of images to process (optional)")
parser.add_argument("--ensemble_thresholds", type=str, nargs='+', default=[0.4, 0.45, 0.5],
                    help="Thresholds for polygonized outputs of the ensemble")
parser.add_argument("--ensemble_border_size", type=int, default=10,
                    help="Number of pixels to remove around the border and no data")
parser.add_argument("--ensemble_mmu", type=int, default=32,
                    help="Minimum mapping unit of output objects in pixels")
parser.add_argument("--try_gpu", action="store_true", help="set to try image processing with gpu")
args = parser.parse_args()

# Location of raw data
# TODO: make support for multiple sources
RAW_DATA_DIR = args.raw_data_dir
# Location data processing
PROCESSING_DIR = args.processing_dir
# Target directory for
INFERENCE_DIR = args.inference_dir
# Target to models - RTS
MODEL_DIR = args.model_dir
# Ensemble Target
ENSEMBLE_NAME = args.ensemble_name
MODEL_NAMES = args.model_names
N_IMAGES = args.max_images# automatically run full set
N_JOBS = args.n_jobs # number of cpu jobs for ensembling
N_VECTOR_LOADERS = args.n_vector_loaders # number of parallel vector loaders for final merge

### Select Data 
# check if cucim is available
try:
    import cucim
    try_gpu = True
    print ('Running ensembling with GPU!')
except:
    try_gpu = False
    print ('Cucim import failed')

# setup all params
kwargs_ensemble = {
    'ensemblename': ENSEMBLE_NAME,
    'inference_dir': INFERENCE_DIR,
    'modelnames': MODEL_NAMES,
    'binary_threshold': args.ensemble_thresholds,
    'border_size': args.ensemble_border_size,
    'minimum_mapping_unit': args.ensemble_mmu,
    'delete_binary': True,
    'try_gpu': args.try_gpu, # currently default to CPU only
    'gpu' : 0,
}

# Check for finalized products
df_processing_status = get_processing_status(RAW_DATA_DIR, PROCESSING_DIR, INFERENCE_DIR, model=kwargs_ensemble['ensemblename'])
df_ensemble_status = get_processing_status_ensemble(INFERENCE_DIR, model_input_names=kwargs_ensemble['modelnames'], model_ensemble_name=kwargs_ensemble['ensemblename'])
# Check which need to be process - check for already processed and invalid files
process = df_ensemble_status[df_ensemble_status['process']]

# #### Run Ensemble Merging

print(f'Start running ensemble with {N_JOBS} jobs!')
print(f'Target ensemble name:', kwargs_ensemble['ensemblename'])
print(f'Source model output', kwargs_ensemble['modelnames'])
_ = Parallel(n_jobs=N_JOBS)(delayed(create_ensemble_v2)(image_id=process.iloc[row]['name'], **kwargs_ensemble) for row in tqdm(range(len(process.iloc[:N_IMAGES]))))

# # #### run parallelized batch 

# ### Merge vectors to complete dataset 
ensemblename = ENSEMBLE_NAME
# set probability levels: 'class_05' means 50%, 'class_045' means 45%. This is the regex to search for vector naming
proba_strings = args.ensemble_thresholds

for proba_string in proba_strings:
    # read all files which followiw the above defined threshold
    flist = list((INFERENCE_DIR / ensemblename).glob(f'*/*_{proba_string}.gpkg'))
    len(flist)
    # load them in parallel
    print (f'Loading results {proba_string}')
    out = Parallel(n_jobs=6)(delayed(load_and_parse_vector)(f) for f in tqdm(flist[:N_IMAGES]))
    # merge them and save to geopackage file
    print ('Merging results')
    merged_gdf = gpd.pd.concat(out)

    # Save output to vector
    save_file = INFERENCE_DIR / ensemblename / f'merged_{proba_string}.gpkg'    
    
    # make file backup if necessary
    if save_file.exists():
        # Get the current timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        # Create the backup file name
        save_file_bk = INFERENCE_DIR / ensemblename / f'merged_{proba_string}_bk_{timestamp}.gpkg'
        print (f'Creating backup of file {save_file} to {save_file_bk}')
        shutil.move(save_file, save_file_bk)
    
    # save to files
    print(f'Saving vectors to {save_file}')
    merged_gdf.to_file(save_file)
    merged_gdf.to_file(INFERENCE_DIR / ensemblename / f'merged_{proba_string}.gpkg')
    