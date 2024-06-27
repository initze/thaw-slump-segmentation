import pytest
from pathlib import Path
import os, warnings, subprocess

# adds a test parameter --data-dir to pass a directory where predefined testfiles reside
# the testfiles will be configured as a fixture
# tests which request this directory will be skipped if this parameter isn't passed.
def pytest_addoption(parser):
    parser.addoption(
        "--data_dir",
        action="store",
        default=None,
        help="Directory containing the predefined data files"
    )
    parser.addoption(
        "--gdal_bin",
        action="store",
        default=None,
        help='Path to gdal binaries'
    )
    parser.addoption(
        "--gdal_path",
        action="store",
        default=None,
        help='Path to gdal python scripts (e.g. gdal_retile.py)'
    )
    parser.addoption(
        "--proj_data_env",
        action="store",
        default=None,
        help='path to data of the proj library, will be set as PROJ_DATA environment variable'
    )

@pytest.fixture()
def proj_data(request):
    proj_data = request.config.getoption("--proj_data_env")
    if proj_data is not None:
        os.environ["PROJ_DATA"] = proj_data        

# the fixture for the data dir:
@pytest.fixture
def data_dir(request):
    p = request.config.getoption("--data_dir")
    
    if p is None:
        pytest.skip( f"parameter --data_dir needed for this test" )

    data_path_root = Path(p)

    if not data_path_root.exists():
        pytest.skip( f" {data_path_root} does not exist " )

    return data_path_root

GDAL_TEST = 0

@pytest.fixture
def gdal_bin(request, proj_data):
    global GDAL_TEST
    gdal_bin = request.config.getoption("--gdal_bin")
    if gdal_bin is None:
        pytest.skip( f"parameter --gdal_bin needed for this test" )


    gdal_bin_path = Path(gdal_bin)

    for bin_tool in [ 'gdal_rasterize', 'gdal_translate', 'gdalwarp' ]:
        bin_tool_path = gdal_bin_path / bin_tool

        if not bin_tool_path.exists():
            pytest.skip( f"--gdal_bin is required to point to the folder of the gdal binaries (e.g. gdalwarp) " )

    if GDAL_TEST == 0:
        # check if gdal via subprocess (as we do it in the scripts) works:
        gdaltransform = gdal_bin_path / "gdaltransform"
        shell_command = f'echo "12 34" | {gdaltransform} -s_srs EPSG:4326 -t_srs EPSG:3857'
        proc = subprocess.run(shell_command, shell=True, capture_output=True)

        if proc.returncode > 0:
            GDAL_TEST = -1
            warning_message = f"testing GDAL with {shell_command} failed: \n{str(proc.stderr)}\nYou might want to pass --proj_data_env pointing to the proj database folder"
            warnings.warn(warning_message)

        else:
            GDAL_TEST = 1

    if GDAL_TEST == -1:
        pytest.skip( f"a working gdal is required for this test, but the testcommand {shell_command} failed"  )

    return gdal_bin

@pytest.fixture
def gdal_path(request, proj_data):
    gdal_path = request.config.getoption("--gdal_path")
    if gdal_path is None:
        pytest.skip( f"parameter --gdal_path needed for this test" )


    pygdal_bin_path = Path(gdal_path)

    for py_tool in [ 'gdal_merge.py', 'gdal_retile.py', 'gdal_polygonize.py' ]:
        pygdal_tool_path = pygdal_bin_path / py_tool

        if not pygdal_tool_path.exists():
            pytest.skip( f"--gdal_path is required to point to the folder of the gdal python scripts (e.g. gdal_merge.py) " )

    return gdal_path


