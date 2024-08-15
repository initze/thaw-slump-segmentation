from pathlib import Path
import pytest
import tempfile
import shutil

from thaw_slump_segmentation.data_pre_processing.dem import buildDemVrt

# check if we have the standard test data available


def testWithData(data_dir:Path):
    arcticdem_base_dir = data_dir / "auxiliary/ArcticDEM"

    if not arcticdem_base_dir.exists():
        pytest.skip(f"{arcticdem_base_dir} does not exist")

    output_dir = tempfile.mkdtemp()
    output_path = Path(output_dir)

    buildDemVrt( arcticdem_base_dir, output_path)

    assert (output_path / "elevation.vrt" ).exists()
    assert (output_path / "slope.vrt" ).exists()

    with (output_path / "elevation.vrt").open("r") as f:
        assert f.read(11) == "<VRTDataset"
    with (output_path / "slope.vrt").open("r") as f:
        assert f.read(11) == "<VRTDataset"

    shutil.rmtree(output_dir)

