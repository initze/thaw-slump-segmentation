import typer

from thaw_slump_segmentation.scripts.download_s2_4band_planet_format import download_s2_4band_planet_format
from thaw_slump_segmentation.scripts.inference import inference
from thaw_slump_segmentation.scripts.setup_raw_data import setup_raw_data
from thaw_slump_segmentation.scripts.train import train

cli = typer.Typer()

cli.command()(train)
cli.command()(inference)


@cli.command()
def hello(name: str):
    typer.echo(f'Hello {name}')


@cli.command()
def goodbye(name: str):
    typer.echo(f'Goodbye {name}')


data_cli = typer.Typer()

data_cli.command('download-planet')(download_s2_4band_planet_format)
data_cli.command('setup-raw')(setup_raw_data)

cli.add_typer(data_cli, name='data')
