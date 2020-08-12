"""
Share library of constants and functions that
can be used across notebooks.
"""
import matplotlib.pyplot as p
import matplotlib
import numpy as np
import pandas as pd
import os
from datetime import datetime, timedelta, date
from typing import Dict, Sequence, Tuple, Union, NoReturn
from dataclasses import dataclass
import math
import csv
from enum import Enum
import re
from collections import namedtuple
from IPython.core.magic import (register_line_magic, register_cell_magic)


# Ignore data before the starting date. This makes all the data sources
# cover the same period.
STARTING_DATE = datetime(year=2020, month=4, day=12, hour=0, minute=0, second=0)

IMAGES_PATH = 'images'

# Column names for normalized data. Not all data sources
# support every column, so the unsupported ones should be
# filled with default values (e.g., 0 for numeric columns).
COL_DATE             = 'date'
COL_MONTH_DAY        = 'month_day'
COL_REGION           = 'region'            # state name, or "United States"
COL_COUNTY           = 'county'            # only applicable to county data sets
COL_DEATHS           = 'deaths'
COL_CASES            = 'cases'
COL_TESTS            = 'tests'
COL_HOSPITALIZATIONS = 'hospitalizations'
COL_RECOVERIES       = 'recoveries'
COL_POSITIVITY       = 'positivity_rate'


class MetricType(Enum):
    """
    The type of metric to plot. Not all metrics are supported by
    all data sets.
    """
    DEATHS = 'deaths'
    CASES = 'cases'
    RECOVERIES = 'recoveries'
    HOSPITALIZATIONS = 'hospitalizations'
    TESTED = 'tested'
    POSITIVITY = 'positivity' # calculated on the fly

METRIC_LABELS = {
    MetricType.DEATHS: 'Deaths',
    MetricType.CASES: 'Cases',
    MetricType.RECOVERIES: 'Recoveries',
    MetricType.HOSPITALIZATIONS: 'Hospitalizations',
    MetricType.TESTED: 'Tested',
    MetricType.POSITIVITY: 'Positivity Rate',
}

METRIC_COLUMNS = {
    MetricType.DEATHS: COL_DEATHS,
    MetricType.CASES: COL_CASES,
    MetricType.RECOVERIES: COL_RECOVERIES,
    MetricType.HOSPITALIZATIONS: COL_HOSPITALIZATIONS,
    MetricType.TESTED: COL_TESTS,
    MetricType.POSITIVITY: COL_POSITIVITY,
}

# Note: XKCD colors are listed at https://xkcd.com/color/rgb/
# Prefix them with "xkcd:" to use in plotting.
METRIC_COLORS = {
    MetricType.DEATHS: 'red',
    MetricType.CASES: 'blue',
    MetricType.RECOVERIES: 'green',
    MetricType.HOSPITALIZATIONS: 'xkcd:purple',
    MetricType.TESTED: 'xkcd:almost black',
    MetricType.POSITIVITY: 'xkcd:sandy',
}

METRIC_MOVING_AVERAGE_COLORS = {
    MetricType.DEATHS: 'pink',
    MetricType.CASES: 'xkcd:pale blue',
    MetricType.RECOVERIES: 'xkcd:pale green',
    MetricType.HOSPITALIZATIONS: 'xkcd:light lavender',
    MetricType.TESTED: 'xkcd:light grey',
    MetricType.POSITIVITY: 'xkcd:buff',
}

@dataclass(frozen=True)
class StateInfo:
    state_name: str
    abbreviation: str
    fips_code: int

        
@dataclass(frozen=True)
class StateCountyInfo:
    state_name: str # full state name
    county_name: str
    fips_code: str


@register_line_magic
def head(path: str, n: int=10, encoding: str='utf-8') -> NoReturn:
    """
    Convenience routine kind of like the Unix head(1)
    command: Opens a file (presumed to be text) and displays
    the first N lines of it to standard output.
    
    You can also call this function as %head, but only the
    first argument (the file name) is honored in that case.
    """
    with open(path, mode='r', encoding=encoding) as f:
        for i, line in enumerate(f.readlines()):
            if i >= n:
                break
            print(line, end="")


def make_month_day_column(df: pd.DataFrame):
    """
    Take a Pandas DataFrame containing normalized
    COVID-19 data, and create the month-day column.
    """
    df[COL_MONTH_DAY] = df[COL_DATE].dt.strftime('%m/%d')


def date_to_datetime(date: date, hour: int = 0, minute: int = 0, second: int = 0) -> datetime:
    """
    Converts a Python date object to a datetime object,
    adding the specified hour, minute, and second.
    """
    from datetime import datetime
    return datetime(year=date.year, month=date.month, day=date.day,
                    hour=hour, minute=minute, second=second)


def datestr(d: date, include_year: bool = False) -> str:
    """
    Format a date in a consistent fashion.
    """
    pat = "%m/%d/%Y" if include_year else "%m/%d"
    return datetime.strftime(d, pat)


def textbox(ax: matplotlib.axes.Axes, x: Union[int, float], y: Union[int, float], 
            contents: str, fontsize: int = 12, boxstyle: str = 'round', bg: str = 'xkcd:pale green'):
    """
    Place text in a box on a plot.
    
    Note on coordinates: (0, 0) is lower left. (1, 1) is upper right. Floats are allowed.
    The coordinates refer to the upper left corner of the text box. A good starting pair
    is (0.01, 0.98)
    
    Parameters:
    
    ax        - The plot
    x         - The X location for the box.
    y         - The Y location for the box.
    contents  - The text. Can be multiline.
    fontsize  - The size of the text font. Defaults to 12.
    boxstyle  - The style of the box. Defaults to 'round'.
    bg        - The background color. Defaults to pale green.    
    """
    props = {'boxstyle': boxstyle, 'facecolor': bg, 'alpha': 0.3}
    ax.text(x, y, contents, transform=ax.transAxes, fontsize=fontsize, bbox=props, va='top', ha='left')
    

def csv_int_field(row: Dict[str, str], key: str) -> int:
    """
    Get an integer value from a csv.DictReader row. If the value
    is empty or not there, return 0.
    
    Parameters:
    
    row - the row from the csv.DictReader
    key - the key to retrieve
    """
    s = row.get(key, '0').strip()
    if len(s) == 0:
        return 0
    return int(s)


def csv_float_field(row: Dict[str, str], key: str) -> float:
    """
    Get a float value from a csv.DictReader row. If the value
    is empty or not there, return 0.0.
    
    Parameters:
    
    row - the row from the csv.DictReader
    key - the key to retrieve
    """
    s = row.get(key, '0.0').strip()
    if len(s) == 0:
        return 0.0
    return float(s)


def determine_ymax_and_stride(max_value: Union[int, float]) -> Tuple[int, int]:
    """
    Given a maximum value to be plotted on the Y axis, use a simple
    heuristic to determine (a) the upper bound to be shown (i.e.,
    the maximum Y) value to show on the graph, and (b) the "stride",
    or number by which to increment for each y-tick.
    
    Returns a (maximum_y, stride) tuple.
    """
    magnitude = int(math.log(max_value, 10)) # 10 to the what?
    # Use the next lowest magnitude, unless the magnitude is already 1.
    if magnitude > 1:
        magnitude -= 1
    stride = 5 * (10 ** magnitude)
    return (max_value + stride, stride)


def get_per_capita_float(n: int, population: int, per_n: int=100_000) -> float:
    """
    Get the per-N per capita rate, given a value (say, total deaths
    and a population figure. For instance, passing a per_n value of
    1_000_000 scales the "n" value to a one-per-million unit.
    
    Parameters:
    
    n:          the value to scale
    population: the estimated 2020 population of the entity
    per_n:      The "per" value (e.g., 1_000_000 for a per-million
                result).
    
    Returns: the per-capita value, as a float
    """
    per_n_factor = per_n / population
    return n * per_n_factor

def get_per_capita_int(n: int, population: int, per_n: int=100_000) -> int:
    """
    Get the per-N per capita rate, given a value (say, total deaths
    and a population figure. For instance, passing a per_n value of
    1_000_000 scales the "n" value to a one-per-million unit.
    
    Parameters:
    
    n:          the value to scale
    population: the estimated 2020 population of the entity
    per_n:      The "per" value (e.g., 1_000_000 for a per-million
                result). 
                
    Returns: the per-capita value, rounded up to the nearest integer
    """
    return int(round(get_per_capita_float(n, population, per_n)))


def load_united_states_population_data() -> Dict[str, int]:
    """
    Load state population data. Returns a dict indexed by (full) state name,
    with the estimated population as the integer value. The summed up figure,
    for the United States as a whole, is available under key "United States".
    """
    populations = dict()
    with open('data/state-populations.csv', mode='r', encoding='utf-8') as f:
        r = csv.DictReader(f)
        for row in r:
            state = row['State']
            population = int(row['Pop'])
            populations[state] = population

    populations['United States'] = sum(populations.values())
    return populations


def load_state_info() -> Dict[str, StateInfo]:
    """
    Loads some information about each US state. Returns
    a dictionary indexed by full state name; the dictionary
    values are StateInfo objects, which provide the postal
    code state abbreviation and the FIPS code.
    """
    results = dict()
    with open('data/states-fips.csv', mode='r') as f:
        for row in csv.DictReader(f):
            state = row['State Name']
            results[state] = StateInfo(
                state_name=state,
                abbreviation=row['Abbreviation'],
                fips_code=int(row['FIPS'])
            )

    return results


def load_county_info(state_info=None) -> Dict[str, Dict[str, StateCountyInfo]]:
    """
    Load the state county information for all states.
    Returns a dictionary indexed by state; each value is a
    dictionary indexed by county name with a StateCountyInfo
    as the value.
    
    You can pass, as the first parameter, the results of a
    prior call to load_state_info(). If you don't do that,
    this function will call load_state_info() itself.
    """
    if state_info is None:
        state_info = load_state_info()

    # The county info lists states by abbreviation. We want to
    # map to full name. Build a lookup table.
    states_by_abbrev = dict()
    for si in state_info.values():
        states_by_abbrev[si.abbreviation] = si.state_name
        
    results = dict()
    with open('data/state-county-fips.csv', mode='r', encoding='utf-8') as f:
        for row in csv.DictReader(f):
            state = states_by_abbrev[row['state_abbrev']]
            state_data = results.get(state, {})
            county = row['county']
            state_data[county] = StateCountyInfo(
                state_name = state,
                county_name = county,
                fips_code=csv_int_field(row, 'fips')
            )
            results[state] = state_data
    return results