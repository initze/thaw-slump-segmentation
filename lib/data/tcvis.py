import ee
import numpy as np
from .base import EETileSource


class TCVIS(EETileSource):
  def get_ee_image(self):
    return ee.ImageCollection("users/ingmarnitze/TCTrend_SR_2000-2019_TCVIS").mosaic()

  def get_dtype(self):
    return 'uint8'

  def __repr__(self):
    return f'TCVIS()'

  @staticmethod
  def replace_zeros(data):
    data.loc[dict(band=[1, 2, 3])] = replace_single_band_zeros(data.sel(band=[1, 2, 3]).to_numpy())
    return data


  @staticmethod
  def normalize(tile):
    #tile =
    return replace_single_band_zeros(tile, 0, 1)
    #return tile / 255

def replace_single_band_zeros(inarray, value_to_replace=0, replace_with=1):
  print(inarray.shape)
  outarray = inarray.copy()
  # get locations of change value
  zero_mask = (inarray == value_to_replace)
  # check for value in all bands
  zero_all = np.all(zero_mask, axis=0)
  # check for value in single bands only
  zero_any = np.any(zero_mask, axis=0)
  # keep where only individual bands
  change_mask0 = (zero_any & ~zero_all)
  # make final multi-band mask
  change_mask = zero_mask * change_mask0
  # replace values
  outarray[change_mask] = replace_with

  return outarray