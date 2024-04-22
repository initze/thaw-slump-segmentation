# # Create ensemble results from several model outputs
# ### Imports 

from pathlib import Path
import pandas as pd
from joblib import delayed, Parallel
#from tqdm.notebook import tqdm
from tqdm import tqdm
from lib.postprocessing import *
import geopandas as gpd
from datetime import datetime

# ### Settings 
# Local code dir
CODE_DIR = Path('.')
BASE_DIR = Path('../..')
# Location of raw data
# TODO: make support for multiple sources
RAW_DATA_DIR = BASE_DIR / Path('data/planet/planet_data_inference_grid/tiles')
# Location data processing
PROCESSING_DIR = BASE_DIR / 'processing'
# Target directory for
INFERENCE_DIR = BASE_DIR / Path('processed/inference')
# Target to models - RTS
MODEL_DIR = BASE_DIR / Path('models/thaw_slumps')
# Ensemble Target
ENSEMBLE_NAME = 'RTS_v6_ensemble_v2'
MODEL_NAMES = ['RTS_v6_notcvis', 'RTS_v6_tcvis']
N_IMAGES = None # automatically run full set
N_JOBS = 15 # number of cpu jobs for ensembling
N_VECTOR_LOADERS = 6 # number of parallel vector loaders for final merge

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
    'binary_threshold': [0.4, 0.45, 0.5],
    'border_size': 10,
    'minimum_mapping_unit': 32,
    'delete_binary': True,
    'try_gpu': False, # currently default to CPU only
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
proba_strings = ['class_05', 'class_045','class_04']

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