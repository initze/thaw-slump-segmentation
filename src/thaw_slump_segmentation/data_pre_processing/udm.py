#!/usr/bin/env python
# Copyright (c) Ingmar Nitze and Konrad Heidler

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


import numpy as np
import rasterio as rio


def get_mask_from_udm2(infile, bands=[1,2,4,5], nodata=1):
    with rio.open(infile) as src:
        a = src.read()
        mask = a[bands].max(axis=0) < nodata
    return mask

def get_mask_from_udm2_v2(infile, band=1):
    with rio.open(infile) as src:
        a = src.read(1)
    return a


def get_mask_from_udm(infile, nodata=1):
    with rio.open(infile) as src:
        mask = src.read()[0] >= nodata
    return np.array(mask, dtype=np.uint8)


def burn_mask(file_src, file_dst, file_udm, file_udm2=None, mask_value=0):
    with rio.Env():
        # checks 
        
        if file_udm:
            mask_udm = get_mask_from_udm(file_udm)
            
        if file_udm2:
            mask_udm2 = get_mask_from_udm2(file_udm2)
            #mask_udm2 = get_mask_from_udm2_v2(file_udm2)

        # merge masks if both
        if ((file_udm is not None) & (file_udm2 is not None)):
            clear_mask = np.array([mask_udm, mask_udm2]).max(axis=0) == mask_value
        # only use one of udm or udm if only one exists
        elif file_udm is not None:
            clear_mask = mask_udm
        elif file_udm2 is not None:   
            clear_mask = mask_udm2
        else:
            raise ValueError

        with rio.open(file_src) as ds_src:
            data = ds_src.read()*clear_mask
            profile = ds_src.profile
            with rio.open(file_dst, 'w', **profile) as ds_dst:
                ds_dst.write(data)
    return 1
