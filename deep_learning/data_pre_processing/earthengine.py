import ee


def get_ArcticDEM_rel_el(kernel_size=300):
    dem = ee.Image("UMN/PGC/ArcticDEM/V3/2m_mosaic")
    conv = dem.convolve(ee.Kernel.circle(kernel_size, 'meters'))
    diff = dem.subtract(conv)
    return diff


def get_ArcticDEM_slope():
    dem = ee.Image("UMN/PGC/ArcticDEM/V3/2m_mosaic")
    slope = ee.Terrain.slope(dem)
    return slope
