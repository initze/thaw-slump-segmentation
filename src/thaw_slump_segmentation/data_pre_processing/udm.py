#!/usr/bin/env python
# Copyright (c) Ingmar Nitze and Konrad Heidler

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


import numpy as np
import rasterio as rio


def get_mask_from_udm2(infile, bands=[1,2,4,5], nodata=1):
    """
    return masked array, where True = NoData
    """
    with rio.open(infile) as src:
        a = src.read()
        mask = a[bands].max(axis=0) == nodata
    return mask


def get_mask_from_udm2_v2(infile):
    """
    Create a data mask from a UDM2 V2 file.

    The data mask is a boolean array where True values represent good data,
    and False values represent no data or unusable data.

    Args:
        infile (str): Path to the input UDM2 V2 file.

    Returns:
        numpy.ndarray: A boolean array representing the data mask.

    Notes:
        - The function assumes that the input file is a UDM2 V2 file with
          specific band meanings:
            - Band 0: Clear data
            - Bands 1, 2, 4, 5: Unusable data (if any of these bands has a value of 1)
            - Band 7: No data

        - The data mask is created by combining the following conditions:
            - Clear data (band 0 == 1)
            - Not no data (band 7 != 1)
            - Not unusable data (maximum of bands 1, 2, 4, 5 != 1)

        - The function uses the rasterio library to read the input file.
    """
    with rio.open(infile) as src:
        a = src.read()
        unusable_data = a[[1,2,4,5]].max(axis=0) == 1
        clear_data = a[[0]] == 1
        no_data = a[[7]] == 1
        # final data mask: 0 = no or crappy data, 1 = good data
        data_mask = np.logical_and(clear_data, ~np.logical_or(no_data, unusable_data))
    return data_mask


def get_mask_from_udm(infile, nodata=1):
    with rio.open(infile) as src:
        mask = src.read()[0] >= nodata
    return np.array(mask, dtype=np.uint8)


# TODO: inconsistent application what the mask value is
def burn_mask(file_src, file_dst, file_udm, file_udm2=None, mask_value=0):
    """"
    """
    with rio.Env(): 
        if file_udm2:
            mask_udm2 = get_mask_from_udm2_v2(file_udm2)
        else:
            raise ValueError

        with rio.open(file_src) as ds_src:
            # apply data mask (multiply)
            data = ds_src.read() * mask_udm2
            profile = ds_src.profile
            with rio.open(file_dst, 'w', **profile) as ds_dst:
                ds_dst.write(data)
    return 1
