import numpy as np
from numba import jit, prange


@jit(nopython=True, parallel=True)
def make_sdf(mask, truncation=16):
    """
    Calculates a truncated signed distance field on the given binary mask
    """
    _, H, W = mask.shape
    sdf = np.zeros((1, H, W), np.float32)

    for y in range(0, H):
        for x in prange(0, W):
            base = mask[0, y, x]

            lo_y = max(0, y - truncation)
            hi_y = min(H, y + 1 + truncation)  # exclusive
            lo_x = max(0, x - truncation)
            hi_x = min(W, x + 1 + truncation)  # exclusive

            best = truncation * truncation
            for y2 in range(lo_y, hi_y):
                for x2 in range(lo_x, hi_x):
                    if not base == mask[0, y2, x2]:
                        dy = y - y2
                        dx = x - x2
                        best = min(best, dy * dy + dx * dx)

            best = np.sqrt(best)
            if not base:
                best = -best
            sdf[0, y, x] = best

    return sdf / truncation
