import pytest
from pathlib import Path
import tempfile, shutil

from thaw_slump_segmentation.scripts.setup_raw_data import preprocess_directory


def testCompleteProcessing(data_dir:Path, gdal_bin, gdal_path):

    # check if data_dir this is just a basic image folder formatted from planet download

    image_dir = data_dir / "raw_data_dir" / "scenes" / "20230807_191420_44_241d"
    aux_dir = data_dir / "auxiliary"
    if not image_dir.exists():
        pytest.skip(f"could not find predefined image dir {image_dir}")

    # create a working directory and copy images there
    temp_path = Path(tempfile.mkdtemp())

    source_dir = temp_path / "input"
    source_dir.mkdir()
    [shutil.copy(f, source_dir) for f in image_dir.glob("*")]

    # target directory of the preprocess_directory processing
    target_dir = temp_path / "output"
    target_dir.mkdir()

    backup_dir = temp_path / "backup"
    backup_dir.mkdir()

    preprocess_result = preprocess_directory(
        image_dir=source_dir, 
        data_dir=target_dir, 
        aux_dir=aux_dir,
        backup_dir=backup_dir,
        log_path=temp_path / "preprocess.log",
        gdal_bin=gdal_bin, gdal_path=gdal_path, label_required=False
        )

    # {'rename': 2, 'label': 2, 'ndvi': 1, 'tcvis': 1, 'rel_dem': 1, 'slope': 1, 'mask': 1, 'move': 1}
    # assert that every result entry except "rename" (was for chopping of _clip) and "label" is 1
    for k,v in preprocess_result.items():
        if k not in ["rename", "label"]:
            assert v == 1

    #assert that the files ndvi.tif, relative_elevation.tif, slope.tif and tcvis.tif exist in backup/input
    # and their size is non-zero

    for f in ["ndvi.tif", "relative_elevation.tif", "slope.tif", "tcvis.tif"]:
        testfile = backup_dir / "input" / f
        assert testfile.exists()
        assert testfile.stat().st_size > 0


    shutil.rmtree(temp_path)

