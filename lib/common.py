"""
Common definitions for all notebooks.
"""
from datetime import datetime, timedelta, date
import math

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

assert len(LINE_COLORS) >= len(STATES_TO_COMPARE)


def datestr(d, include_year=False):
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
    

def csv_int_field(row, key):
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


def csv_float_field(row, key):
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


def determine_ymax_and_stride(max_value):
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