import typer

from thaw_slump_segmentation.scripts.inference import inference
from thaw_slump_segmentation.scripts.prepare_data import prepare_data
from thaw_slump_segmentation.scripts.process_02_inference import process_02_inference
from thaw_slump_segmentation.scripts.process_03_ensemble import process_03_ensemble
from thaw_slump_segmentation.scripts.setup_raw_data import setup_raw_data
from thaw_slump_segmentation.scripts.train import sweep, train, tune

# TODO: Move this comment to docs
# GEE is used for:
# - Setup of raw data (init call inside function)
# - download of S2 images (init call global at module level)
# - prepare of S2 images (init call global at module level)
# GDAL is used for:
# - Setup of raw data (in threaded function)
# - prepare data (in main function)
# - inference (in main function)
# - prepare of S2 images (but its not implemented via gdal module but hardcoded)

cli = typer.Typer(pretty_exceptions_show_locals=False)


@cli.command()
def hello(name: str):
    typer.echo(f'Hello, {name}!')
    return f'Hello, {name}!'

cli.command()(sweep)
cli.command()(tune)
cli.command()(train)
cli.command()(inference)


data_cli = typer.Typer()

#data_cli.command('download-planet')(download_s2_4band_planet_format)
#data_cli.command('prepare-planet')(prepare_s2_4band_planet_format)
data_cli.command('setup-raw')(setup_raw_data)
data_cli.command('prepare')(prepare_data)

cli.add_typer(data_cli, name='data')

process_cli = typer.Typer()

process_cli.command('inference')(process_02_inference)
process_cli.command('ensemble')(process_03_ensemble)

cli.add_typer(process_cli, name='process')
