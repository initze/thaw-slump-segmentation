import numpy as np
import rasterio as rio


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

    for y0 in yvals:
        y1 = y0 + patch_size
        for x0 in xvals:
            x1 = x0 + patch_size
            cutouts = [x[..., y0:y1, x0:x1] for x in images]
            nodata = cutouts[0] == 0
            while len(nodata.shape) > 2:
                nodata = nodata.all(axis=0)
            if nodata.mean() <= max_nodata:
                if len(cutouts) > 1:
                    yield cutouts
                else:
                    yield cutouts[0]
