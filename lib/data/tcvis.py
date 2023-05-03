import ee
from .base import EETileSource


class TCVIS(EETileSource):
    def get_ee_image(self):
        return ee.ImageCollection("users/ingmarnitze/TCTrend_SR_2000-2019_TCVIS").mosaic()

    def __repr__(self):
        return f'TCVIS()'
