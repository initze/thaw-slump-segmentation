import rasterio
import os
from pathlib import Path
import numpy as np
import ee
import eemont
import geemap
ee.Initialize(opt_url='https://earthengine-highvolume.googleapis.com')

import argparse


def download_S2image_preprocessed(s2_image_id, outfile, outbands=['B2', 'B3', 'B4', 'B8'], factor=1e4):
    ic = ee.ImageCollection(ee.Image(f'COPERNICUS/S2_SR_HARMONIZED/{s2_image_id}'))
    # load basic image and preprocess (maks clouds, scale and offset)
    image = ee.Image(f'COPERNICUS/S2_SR_HARMONIZED/{s2_image_id}').preprocess()#.spectralIndices(['NDVI'])
    # select corresponding bands
    image_4Band = image.select(outbands)
    # scale by 10k and convert to uint16
    image_out = image_4Band.multiply(1e4).uint16().copyProperties(image_4Band, image_4Band.propertyNames())
    # download
    if not Path(outfile).parent.exists():
        Path(outfile).parent.mkdir()
    geemap.download_ee_image(ee.Image(image_out), outfile)
    return 0

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Download preprocessed S2 image.', 
                    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--s2id', type=str, nargs='+', help='S2 image ID, you can use several separated by space')
    parser.add_argument('--data_dir', type=str, help='Output directory')
    args = parser.parse_args()
    
    outdir = Path(args.data_dir)
    s2id = args.s2id
    
    for s2id in args.s2id:
        # Call the function with the provided s2id
        outfile = outdir/ s2id / f'{s2id}_SR.tif'
        if not outdir.exists():
            print('Creating output directory', outdir)
            outdir.mkdir()
        download_S2image_preprocessed(s2id, outfile)
