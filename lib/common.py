"""
Common definitions for all notebooks.
"""
from datetime import datetime, timedelta, date
from typing import Dict, Sequence, Tuple, Union
import math
import csv

IMAGES_PATH = 'images'

# suggested line colors and styles for a multi-line plot.
# xkcd colors, supported by matplotlib, come from here:
# https://xkcd.com/color/rgb/
LINE_COLORS_AND_STYLES = (
    ('red', 'solid'),
    ('blue', 'dashed'),
    ('green', 'solid'),
    ('cyan', 'solid'),
    ('orange', 'solid'),
    ('xkcd:aquamarine', 'dotted'),
    ('xkcd:taupe', 'solid'),
    ('magenta', 'dashed'),
    ('black', 'solid'),
    ('grey', 'dotted'),
    ('xkcd:violet', 'dotted'),
    ('xkcd:pale green', 'dashdot'),
)

# Just the line colors.
LINE_COLORS = tuple([c for c, _ in LINE_COLORS_AND_STYLES])

# A list of states to compare.
STATES_TO_COMPARE = (
    'Arizona',
    'California',
    'Connecticut',
    'Florida',
    'Georgia',
    'Illinois',
    'Massachusetts',
    'New York',
    'Ohio',
    'Pennsylvania',
    'Texas',
    'Washington',
)

# A dict of states and territories, with abbreviations.
STATES_AND_ABBREVIATIONS = {
    "Alabama": "AL",
    "Alaska": "AK",
    "Arizona": "AZ",
    "Arkansas": "AR",
    "California": "CA",
    "Colorado": "CO",
    "Connecticut": "CT",
    "Delaware": "DE",
    "District of Columbia": "DC",
    "Florida": "FL",
    "Georgia": "GA",
    "Hawaii": "HI",
    "Idaho": "ID",
    "Illinois": "IL",
    "Indiana": "IN",
    "Iowa": "IA",
    "Kansas": "KS",
    "Kentucky": "KY",
    "Louisiana": "LA",
    "Maine": "ME",
    "Maryland": "MD",
    "Massachusetts": "MA",
    "Michigan": "MI",
    "Minnesota": "MN",
    "Mississippi": "MS",
    "Missouri": "MO",
    "Montana": "MT",
    "Nebraska": "NE",
    "Nevada": "NV",
    "New Hampshire": "NH",
    "New Jersey": "NJ",
    "New Mexico": "NM",
    "New York": "NY",
    "North Carolina": "NC",
    "North Dakota": "ND",
    "Ohio": "OH",
    "Oklahoma": "OK",
    "Oregon": "OR",
    "Pennsylvania": "PA",
    "Rhode Island": "RI",
    "South Carolina": "SC",
    "South Dakota": "SD",
    "Tennessee": "TN",
    "Texas": "TX",
    "Utah": "UT",
    "Vermont": "VT",
    "Virginia": "VA",
    "Washington": "WA",
    "West Virginia": "WV",
    "Wisconsin": "WI",
    "Wyoming": "WY",
    "Puerto Rico": "PR",
    "American Samoa": "AS",
    "Virgin Islands": "VI",
    "Guam": "GU",
}

assert len(LINE_COLORS) >= len(STATES_TO_COMPARE)


def datestr(d: datetime.date, include_year: bool=False) -> str:
    """
    Format a date in a consistent fashion.
    """
    pat = "%m/%d/%Y" if include_year else "%m/%d"
    return datetime.strftime(d, pat)


def textbox(ax, x, y, contents, fontsize=12, boxstyle='round', bg='xkcd:pale green'):
    """
    Place text in a box on a plot.
    
    Note on coordinates: (0, 0) is lower left. (1, 1) is upper right. Floats are allowed.
    
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
    ax.text(x, y, contents, transform=ax.transAxes, fontsize=fontsize, bbox=props)
    

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


def get_per_capita_value(n: int, population: int, per_n=100_000) -> float:
    """
    Get the per-N per capita rate, given a value (say, total deaths
    and a population figure. For instance, passing a per_n value of
    1_000_000 scales the "n" value to a one-per-million unit.
    
    Parameters:
    
    n:          the value to scale
    population: the estimated 2020 population of the entity
    per_n:      The "per" value (e.g., 1_000_000 for a per-million
                result).
    
    """
    per_n_factor = per_n / population
    return n * per_n_factor


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
    