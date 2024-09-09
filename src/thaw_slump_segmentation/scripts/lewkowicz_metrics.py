from pathlib import Path

import geopandas as gpd
import pandas as pd
import typer
from rich import print
from rich.progress import track
from shapely.geometry import Point
from typing_extensions import Annotated

LEWKOWICZ_SITES = [
    'Banks_Island',
    'Bluenose_moraine',
    'NW_Victoria_Island',
    'Paulatuk_region',
    'Richardson_Mts_Peel_Plateau',
]

NITZE_COVERAGE_FNAME = 'RTS_INitze_v1_coverage_2018-2023_annual.parquet'
NITZE_LEVEL2_FNAME = 'RTS_INitze_v1_rts_2018-2023_v1_05_level2.parquet'


def check_lewkowicz_data(lewkowicz_data: Path):
    site_exists = {site: (lewkowicz_data / f'{site}_RTS_activity.txt').exists() for site in LEWKOWICZ_SITES}
    if any(not exists for exists in site_exists.values()):
        not_existing_sites = [site for site, exists in site_exists.items() if not exists]
        err_str = 'The following sites are missing from the lewkowicz data:\n\t' + '\n\t'.join(not_existing_sites)
        err_str += '\nPlease ensure that the data is present in the correct location.'
        err_str += f'\nData should be located in {lewkowicz_data}'
        example_site = lewkowicz_data / f'{not_existing_sites[0]}_RTS_activity.txt'
        err_str += f', e.g. {example_site.absolute()}'
        raise ValueError(err_str)


def check_lewkowicz_boundaries(lewkowicz_data: Path):
    site_exists = {site: (lewkowicz_data / 'sites' / f'{site}.geojson').exists() for site in LEWKOWICZ_SITES}
    if any(not exists for exists in site_exists.values()):
        not_existing_sites = [site for site, exists in site_exists.items() if not exists]
        err_str = 'The following sites are missing from the lewkowicz data:\n\t' + '\n\t'.join(not_existing_sites)
        err_str += '\nPlease ensure that the data is present in the correct location.'
        err_str += f'\nData should be located in {lewkowicz_data}'
        example_site = lewkowicz_data / 'sites' / f'{not_existing_sites[0]}.geojson'
        err_str += f', e.g. {example_site.absolute()}'
        raise ValueError(err_str)


def check_nitze_data(nitze_data: Path):
    coverage_exists = (nitze_data / NITZE_COVERAGE_FNAME).exists()
    if not coverage_exists:
        err_str = 'The coverage data is missing from the nitze data.'
        err_str += f'\nData should be located in {nitze_data}'
        example_path = nitze_data / NITZE_COVERAGE_FNAME
        err_str += f', e.g. {example_path.absolute()}'
        raise ValueError(err_str)

    level2_exists = (nitze_data / NITZE_LEVEL2_FNAME).exists()
    if not level2_exists:
        err_str = 'The level2 data is missing from the nitze data.'
        err_str += f'\nData should be located in {nitze_data}'
        example_path = nitze_data / NITZE_LEVEL2_FNAME
        err_str += f', e.g. {example_path.absolute()}'
        raise ValueError(err_str)


def match(lewkowicz: pd.DataFrame, nitze: pd.DataFrame, buffer: int, year: int, region: str):
    # Match the lewkowicz data with the nitze data
    matches = gpd.sjoin(lewkowicz, nitze, predicate='intersects')
    # Matches can contain duplicate lewkowicz sites AND duplicate nitze sites
    # We drop only the duplicate lewkowicz sites, since they are the gold truth
    matches = matches.drop_duplicates(subset='Identifier')

    # Calculate the metrics
    tp = len(matches)

    # Avoid division by zero
    if tp == 0:
        return [
            {'metric': 'precision', 'buffer': buffer, 'year': year, 'value': 0, 'region': region},
            {'metric': 'recall', 'buffer': buffer, 'year': year, 'value': 0, 'region': region},
            {'metric': 'f1', 'buffer': buffer, 'year': year, 'value': 0, 'region': region},
            {'metric': 'tp', 'buffer': buffer, 'year': year, 'value': 0, 'region': region},
            {'metric': 'fp', 'buffer': buffer, 'year': year, 'value': len(lewkowicz), 'region': region},
            {'metric': 'fn', 'buffer': buffer, 'year': year, 'value': len(nitze), 'region': region},
            {'metric': 'n_lewkowicz', 'buffer': buffer, 'year': year, 'value': len(lewkowicz), 'region': region},
            {'metric': 'n_nitze', 'buffer': buffer, 'year': year, 'value': len(nitze), 'region': region},
        ]
    fp = len(nitze) - tp
    fn = len(lewkowicz) - tp
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    f1 = 2 * (precision * recall) / (precision + recall)
    return [
        {'metric': 'precision', 'buffer': buffer, 'year': year, 'value': precision, 'region': region},
        {'metric': 'recall', 'buffer': buffer, 'year': year, 'value': recall, 'region': region},
        {'metric': 'f1', 'buffer': buffer, 'year': year, 'value': f1, 'region': region},
        {'metric': 'tp', 'buffer': buffer, 'year': year, 'value': tp, 'region': region},
        {'metric': 'fp', 'buffer': buffer, 'year': year, 'value': fp, 'region': region},
        {'metric': 'fn', 'buffer': buffer, 'year': year, 'value': fn, 'region': region},
        {'metric': 'n_lewkowicz', 'buffer': buffer, 'year': year, 'value': len(lewkowicz), 'region': region},
        {'metric': 'n_nitze', 'buffer': buffer, 'year': year, 'value': len(nitze), 'region': region},
    ]


def match_lewkowicz(
    lewkowicz_data: Annotated[Path, typer.Option(help='Path to the unzipped lewkowicz data')] = Path(
        'data/publication/Lewkowicz'
    ),
    nitze_data: Annotated[Path, typer.Option(help='Path to the our data')] = Path('data/publication/v5'),
):
    # Make sure the data paths are passes right
    check_lewkowicz_data(lewkowicz_data)
    check_lewkowicz_boundaries(lewkowicz_data)
    check_nitze_data(nitze_data)

    # Load nitze data
    nitze_coverage = gpd.read_parquet(nitze_data / NITZE_COVERAGE_FNAME)
    nitze_level2 = gpd.read_parquet(nitze_data / NITZE_LEVEL2_FNAME)

    metrics = []
    for region in track(LEWKOWICZ_SITES, description='Processing regions'):
        # Load lewkowicz data
        fname = lewkowicz_data / f'{region}_RTS_activity.txt'
        df = pd.read_csv(fname, sep='\t')
        geometry = [Point(xy) for xy in zip(df['Longitude'], df['Latitude'])]
        lewkowicz_full = gpd.GeoDataFrame(df, geometry=geometry)
        print(f'Number of total Lewkowicz sites in {region}: {len(lewkowicz_full)}')

        # Set the coordinate reference system (CRS) WGS84
        lewkowicz_full.set_crs(epsg=4326, inplace=True)

        # Rename the the 5th to Start date and 6th column to End date
        lewkowicz_full.rename(columns={lewkowicz_full.columns[4]: 'Start date'}, inplace=True)
        lewkowicz_full.rename(columns={lewkowicz_full.columns[5]: 'End date'}, inplace=True)

        # Filter out inactive sites ("End date" != 2100)
        end_data_column = 'End date'
        lewkowicz_full = lewkowicz_full[lewkowicz_full[end_data_column] != 2100]
        print(f'Number of active Lewkowicz sites in {region}: {len(lewkowicz_full)} ')

        # Filter out sites that are not part of our coverage
        region_boundary = gpd.read_file(lewkowicz_data / 'sites' / f'{region}.geojson')
        nitze_coverage_region = gpd.overlay(nitze_coverage, region_boundary, how='intersection')

        lewkowicz_intercov = []
        nitze_intercov = []
        for year in nitze_coverage_region['year'].unique():
            nitze_coverage_region_year = nitze_coverage_region[nitze_coverage_region['year'] == year]

            lewkowicz_year = gpd.overlay(lewkowicz_full, nitze_coverage_region_year, how='intersection')
            lewkowicz_year['year'] = year
            lewkowicz_intercov.append(lewkowicz_year)

            nitze_year = nitze_level2[nitze_level2['year'] == year]
            nitze_year = gpd.overlay(nitze_year, nitze_coverage_region_year, how='intersection')
            nitze_intercov.append(nitze_year)
        lewkowicz_intercov = pd.concat(lewkowicz_intercov)
        nitze_intercov = pd.concat(nitze_intercov)

        # rename year_1 to year and delete year_2 (they are the same)
        nitze_intercov.rename(columns={'year_1': 'year'}, inplace=True)
        nitze_intercov.drop(columns=['year_2'], inplace=True)

        # Delete area_km2 column, since it has lost its meaning after the intersection
        lewkowicz_intercov.drop(columns=['area_km2'], inplace=True)
        nitze_intercov.drop(columns=['area_km2'], inplace=True)

        print(f'Number of active Lewkowicz sites in {region} that are part of our coverage: {len(lewkowicz_intercov)}')
        print(f'Number of active Nitze sites in {region}: {len(nitze_intercov)}')

        # Match the lewkowicz data with the nitze data
        for buffer in [1, 100, 250, 500, 1000, 1500, 2000]:
            for year in nitze_coverage_region['year'].unique():
                lewkowicz_year = lewkowicz_intercov[lewkowicz_intercov['year'] == year]
                nitze_year = nitze_intercov[nitze_intercov['year'] == year]
                # Convert the geometry of lewkowicz to a m projection
                lewkowicz_year = lewkowicz_year.to_crs(epsg=5937)
                # Buffer the geometry
                lewkowicz_year['geometry'] = lewkowicz_year['geometry'].buffer(buffer)
                # Convert back to WGS84
                lewkowicz_year = lewkowicz_year.to_crs(epsg=4326)

                metrics += match(lewkowicz_year, nitze_year, buffer, year, region)

            # Maximum Dissolve between 2021 and 2023
            lewkowicz_max_dissolve = lewkowicz_intercov[
                lewkowicz_intercov['year'].isin([2021, 2022, 2023])
            ].drop_duplicates(subset='Identifier')
            nitze_max_dissolve = nitze_intercov[nitze_intercov['year'].isin([2021, 2022, 2023])].dissolve().explode()
            # Convert the geometry of lewkowicz to a m projection
            lewkowicz_max_dissolve = lewkowicz_max_dissolve.to_crs(epsg=5937)
            # Buffer the geometry
            lewkowicz_max_dissolve['geometry'] = lewkowicz_max_dissolve['geometry'].buffer(buffer)
            # Convert back to WGS84
            lewkowicz_max_dissolve = lewkowicz_max_dissolve.to_crs(epsg=4326)

            metrics += match(lewkowicz_max_dissolve, nitze_max_dissolve, buffer, 0, region)

    metrics_fpath = nitze_data / 'lewkowicz_metrics.parquet'
    metrics = pd.DataFrame(metrics)
    metrics.to_parquet(metrics_fpath)
    print(f'Metrics saved to {metrics_fpath.absolute()}')
    print('Done!')
    return metrics
