import typer

from thaw_slump_segmentation.scripts.download_s2_4band_planet_format import download_s2_4band_planet_format
from thaw_slump_segmentation.scripts.inference import inference
from thaw_slump_segmentation.scripts.prepare_data import prepare_data
from thaw_slump_segmentation.scripts.prepare_s2_4band_planet_format import prepare_s2_4band_planet_format
from thaw_slump_segmentation.scripts.setup_raw_data import setup_raw_data
from thaw_slump_segmentation.scripts.train import train

cli = typer.Typer()

cli.command()(train)
cli.command()(inference)


data_cli = typer.Typer()

data_cli.command('download-planet')(download_s2_4band_planet_format)
data_cli.command('prepare-planet')(prepare_s2_4band_planet_format)
data_cli.command('setup-raw')(setup_raw_data)
data_cli.command('prepare')(prepare_data)

cli.add_typer(data_cli, name='data')
