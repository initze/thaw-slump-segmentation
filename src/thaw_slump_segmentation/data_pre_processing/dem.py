# The original Script
# 
# ARCTIC_DEM=path/to/base/folder
# ls $ARCTIC_DEM/tiles_rel_el/*.tif > flist_rel_el.txt;
# ls $ARCTIC_DEM/tiles_slope/*.tif > flist_slope.txt;
# gdalbuildvrt -input_file_list flist_rel_el.txt -srcnodata "0" -vrtnodata "0" elevation.vrt;
# gdalbuildvrt -input_file_list flist_slope.txt -srcnodata "nan" -vrtnodata "0" slope.vrt;
#
#
# The Notebook to download ArtcicDEM data is here: https://github.com/initze/thaw_slump_preprocessing/blob/main/DEM_01_ProcessArcticDEM.ipynb

import os
from typing_extensions import Annotated
from pathlib import Path
import logging
import typer

subdirs = {
    "elevation" : "tiles_rel_el",
    "slope" : "tiles_slope"
}

l = logging.getLogger("thaw_slump_segmentation.preprocess.dem")

buildDemVrtMain = typer.Typer()

@buildDemVrtMain.command()
def buildDemVrt(
    dem_data_dir:Annotated[ Path, typer.Option("--dem_data_dir") ], 
    vrt_target_dir:Annotated[ Path, typer.Option("--vrt_target_dir") ] 
    ):
    """parses the subfolders 'tiles_rel_el' and 'tiles_slope' of `dem_data_dir` to
    create VRT (virtual raster tile) files (https://gdal.org/drivers/raster/vrt.html).

    A working installation of the python gdal bindings is required!

    Args:
        dem_data_dir (Path): The folder containing the source folders
        vrt_target_dir (Path): The folder where to write the VRT files

    Raises:
        EnvironmentError: if gdal python bindings could not be imported
        IOError: If target files are not writable
    """
    
    try:
        from osgeo import gdal
    except ModuleNotFoundError:
        raise EnvironmentError("The python GDAL bindings where not found. Please install those which are appropriate for your platform.")

    l.info(f"GDAL found: {gdal.__version__}")

    # decide on the exception behavior of GDAL to supress a warning if we dont
    # don't know if this is necessary in all GDAL versions
    try:
        gdal.UseExceptions()
    except AttributeError():
        pass
    
    # check first if BOTH files are writable
    non_writable_files = []
    for name in subdirs.keys():
        output_file_path = vrt_target_dir / f"{name}.vrt"
        if not (
            (os.access(output_file_path, os.F_OK) and os.access(output_file_path, os.W_OK)) # f exists + writable
            or os.access(output_file_path.parent, os.W_OK) # ...OR folder writable
        ):
            non_writable_files.append(output_file_path)
    if len(non_writable_files) > 0:
        raise IOError(f"cannot write to {', '.join([ f.name for f in non_writable_files ])}")

    for name,subdir in subdirs.items():
        output_file_path = vrt_target_dir / f"{name}.vrt"
        # check the file first if we can write to it

        ds_path = dem_data_dir / subdir
        filelist = [ str(f.absolute().resolve()) for f in ds_path.glob("*.tif") ]
        l.info(f"{len(filelist)} files for {name}\n --> '{output_file_path}'")
        src_nodata = "nan" if name=="slope" else 0
        gdal.BuildVRT( str(output_file_path.absolute()), filelist, options=gdal.BuildVRTOptions(srcNodata=src_nodata, VRTNodata=0) )

if __name__ == "__main__":
    buildDemVrtMain()
