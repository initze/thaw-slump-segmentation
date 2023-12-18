import ee
import numpy as np
from .base import EETileSource


class Hillshade(EETileSource):
  def get_ee_image(self):
    return ee.Image(ee.Terrain.hillshade(ee.Image("UMN/PGC/ArcticDEM/V3/2m_mosaic")))

  def get_dtype(self):
    return 'uint8'

  def __repr__(self):
    return f'Hillshade()'

  @staticmethod
  def normalize(tile):
    return np.clip(tile.astype(np.float32) / 255., 0, 1)
