# Copyright (c) Ingmar Nitze and Konrad Heidler

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
from itertools import zip_longest
from functools import reduce, lru_cache
import cv2
import shapely as shp
from collections import defaultdict


def extract_patches(images, patch_size=512, max_nodata=0.0, stride=None):
    """
    Extracts corresponding patches from the given images.
    Use as e.g. extract_patches([image, mask]).
    Assumes the first image to be the source image for nodata-filtering.
    """
    if type(images) is not list:
        images = [images]
    if stride is None:
        stride = patch_size // 2

    base = images[0]
    yvals = np.arange(0, base.shape[-2] - patch_size, stride)
    xvals = np.arange(0, base.shape[-1] - patch_size, stride)

    for y_i, y0 in enumerate(yvals):
        y1 = y0 + patch_size
        for x_i, x0 in enumerate(xvals):
            x1 = x0 + patch_size
            cutouts = [x[..., y0:y1, x0:x1] for x in images]
            nodata = cutouts[0] == 0
            while len(nodata.shape) > 2:
                nodata = nodata.all(axis=0)
            if nodata.mean() <= max_nodata:
                yield [x_i, y_i, *cutouts]



@lru_cache(16)
def make_stencil(dims):
  dims = dims[:-1]

  parts = []
  for i, dim in enumerate(dims):
    l1 = dim // 2
    l2 = dim - l1
    part = np.concatenate([np.linspace(0.1, 1, l1), np.linspace(1, 0.1, l2)])
    for j in range(len(dims)):
      if j != i:
        part = np.expand_dims(part, j)
    parts.append(part)

  return reduce(np.minimum, parts)[..., np.newaxis]



class Compositor:
  tiles: dict
  max_coords: list[int]

  def __init__(self):
    self.tiles = {}
    self.max_coords = []

  def add_tile(self, coords: tuple[int, ...], tile: np.ndarray):
    extent = [coord + shp for coord, shp in zip(coords, tile.shape)]
    self.max_coords = [max(c, e) for c, e in
                       zip_longest(self.max_coords, extent, fillvalue=0)]
    self.tiles[coords] = tile

  def compose(self):
    assert self.max_coords is not None
    wgt = np.zeros([*self.max_coords], dtype=np.float64)
    out = np.zeros([*self.max_coords], dtype=np.float64)

    for coords, tile in self.tiles.items():
      stencil = make_stencil(tile.shape)
      slices = tuple(slice(c, c+e) for c, e in zip(coords, tile.shape))
      # We could just write out[*slices], but only starting from Py3.11 :)
      out.__setitem__(slices, out.__getitem__(slices) + stencil * tile)
      wgt.__setitem__(slices, wgt.__getitem__(slices) + stencil)
    wgt = np.where(wgt == 0, 1, wgt)

    composed = out / wgt
    return composed



def extract_contours(image, threshold):
    mask = (image > threshold).astype(np.uint8)
    contours, hierarchy = cv2.findContours(mask, mode=cv2.RETR_CCOMP, method=cv2.CHAIN_APPROX_TC89_KCOS)
    # Match children and parents, cf. https://michhar.github.io/masks_to_polygons_and_back/
    if not contours:
      return []

    assert hierarchy.shape[0] == 1
    children = set()
    cnt_children = defaultdict(list)
    for idx, (_, _, _, parent_idx) in enumerate(hierarchy[0]):
      if parent_idx != -1:
        children.add(idx)
        cnt_children[parent_idx].append(contours[idx])

    # create actual polygons filtering by area (removes artifacts)
    all_polygons = []
    for idx, cnt in enumerate(contours):
      if idx not in children and cv2.contourArea(cnt) >= 2.0:
        assert cnt.shape[1] == 1
        holes = [c[:, 0, :] for c in cnt_children.get(idx, []) if cv2.contourArea(c) >= 2.]
        poly = shp.Polygon(shell=cnt[:, 0, :], holes=holes)
        all_polygons.append(poly)
    return all_polygons
