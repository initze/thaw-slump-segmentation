from .base import Scene, TileSource, init_data_paths
from .base import _LAYER_REGISTRY
from .mask import Mask
from .sentinel2 import Sentinel2
from .sentinel1 import Sentinel1
from .relative_elevation import RelativeElevation
from .absolute_elevation import AbsoluteElevation
from .slope import Slope
from .tcvis import TCVIS
from .planet import PlanetScope

for layer_type in [Mask, Sentinel1, Sentinel2, RelativeElevation,
                   AbsoluteElevation, Slope, TCVIS, PlanetScope]:
  _LAYER_REGISTRY[layer_type.__name__] = layer_type
