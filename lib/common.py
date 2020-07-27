"""
Share library of constants and functions that
can be used across notebooks.
"""
import matplotlib.pyplot as p
import numpy as np
import pandas as pd
import os
from datetime import datetime, timedelta, date
from typing import Dict, Sequence, Tuple, Union
from dataclasses import dataclass
import math
import csv
from enum import Enum
import re
from collections import namedtuple

IMAGES_PATH = 'images'

# Column names for normalized data. Not all data sources
# support every column, so the unsupported ones should be
# filled with default values (e.g., 0 for numeric columns).
COL_DATE             = 'date'
COL_MONTH_DAY        = 'month_day'
COL_REGION           = 'region'
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

def make_month_day_column(df):
    """
    Take a Pandas DataFrame containing normalized
    COVID-19 data, and create the month-day colun.
    """
    df[COL_MONTH_DAY] = df[COL_DATE].dt.strftime('%m/%d')


def date_to_datetime(date, hour=0, minute=0, second=0):
    """
    Converts a Python date object to a datetime object,
    adding the specified hour, minute, and second.
    """
    from datetime import datetime
    return datetime(year=date.year, month=date.month, day=date.day,
                    hour=hour, minute=minute, second=second)


def datestr(d: datetime.date, include_year: bool=False) -> str:
    """
    Format a date in a consistent fashion.
    """
    pat = "%m/%d/%Y" if include_year else "%m/%d"
    return datetime.strftime(d, pat)


def fix_pandas_multiplot_legend(ax, legend_loc):
    """
    When plotting multiple pieces of data, the Pandas-generated
    plot legend will often look like "(metric, place)" (e.g.,
    "(Deaths, Connecticut)".
    
    This function corrects the legend, by extracting just the place.
    
    Parameters:
    
    ax         - the plot axis
    legend_loc - the desired location for the legend
    """
    patches, labels = ax.get_legend_handles_labels()
    pat = re.compile(r'^\([^,\s]+,\s+(.*)\)$')
    labels2 = []
    for label in labels:
        m = pat.match(label)
        assert m is not None
        labels2.append(m.group(1))
    ax.legend(patches, labels2, loc=legend_loc)


def plot_stats_by_date(df, 
                       source,
                       metrics={MetricType.DEATHS}, 
                       region='United States',
                       moving_average=False,
                       per_n=1, 
                       populations=None,
                       textbox_heading=None, 
                       textbox_loc=None, 
                       marker=None, 
                       figsize=(20, 12), 
                       image_file=None,
                       legend_loc=None):
    """
    Takes a Pandas DataFrame with normalized data, groups the data by
    the month-day column and sums up the values for all metrics. Then,
    plots the results. If an image file is specified, saves the plot in
    the appropriate image.
    
    Parameters:
    
    df              - The Pandas DataFrame to plot
    source          - Description of data source
    metrics         - A set containing the metrics to plot. Defaults to deaths.
    region          - The state name, or 'United States' for everything
    moving_average  - If True, plot a 7-day moving average along side the data. If
                      False, plot the data as is.
    per_n           - If set to 1, plot the data as is. Otherwise, do a per-capita
                      plot (i.e., number of X per n people). If per_n is not 1,
                      then population must be defined.
    populations     - The loaded populations data. Only necessary if per_n is
                      greater than 1.
    figsize         - The size of the plot.
    marker          - matplotlib marker to use for data points, or None.
    textbox_heading - An optional heading to add to the textbox annotation
    textbox_loc     - An (x, y) tuple for the location of the text box's top
                      left corner. Defaults to the upper left.
    image_file      - Name of image file in which to save plot, or None.
    legend_loc      - Overrides the legend location
    """
    assert (per_n == 1) or (population is not None)

    MOVING_AVERAGE_COLUMNS = {
        MetricType.CASES:            'Cases 7-day Moving Average',
        MetricType.DEATHS:           'Deaths 7-day Moving Average',
        MetricType.TESTED:           'Tests 7-day Moving Average',
        MetricType.HOSPITALIZATIONS: 'Hospitalization 7-day Moving Average',
        MetricType.RECOVERIES:       'Recoveries 7-day Moving Average',
        MetricType.POSITIVITY:       'Positivity Rate 7-day Moving Average',
    }

    FirstLast = namedtuple('FirstLast', ('first', 'last'))

    def maybe_plot_metric(m, ax, df, first_last, errors):
        """
        Given a particular metric, calculate the appropriate data
        and, if the metric was specified, plot it.
        
        Parameters:
        m          - the metric (e.g., MetricType.DEATHS)
        ax         - the plot axis to use
        df         - the DataFrame to query
        first_last - the dictionary containing first and last values
                     for each metric. The key is a MetricType. The
                     value is a FirstLast object. The first and last
                     values for all metrics will be added here, even
                     if the metric isn't plotted.
        errors     - an array to which error messages can be appended
        """
        col = METRIC_COLUMNS[m]
        ma_col = MOVING_AVERAGE_COLUMNS[m]

        def handle_per_capita(df, col):
            if per_n > 1:
                pop = populations[region]
                df[col] = df.apply(lambda row: get_per_capita_float(row[col], pop), axis=1)

        handle_per_capita(df, col)

        first_value = int(round(df[col].iloc[0]))
        last_value = int(round(df[col].iloc[-1]))
        first_last[m] = FirstLast(first=first_value, last=last_value)

        if m not in metrics:
            return

        # Since these values are cumulative, if the last value is 0, we
        # can't trust the data.
        last_val = df[col].iloc[-1]
        if last_val <= 0:
            errors.append(f"No data for {METRIC_LABELS[m]} in this data set.")
            return

        df.plot(x=COL_MONTH_DAY, y=col, ax=ax, color=METRIC_COLORS[m], marker=marker, zorder=2)
        if moving_average:
            df[ma_col] = df[col].rolling(7).mean()
            handle_per_capita(df, ma_col)
            color = METRIC_MOVING_AVERAGE_COLORS[m]
            df.plot(x=COL_MONTH_DAY, y=ma_col, ax=ax, color=color, linewidth=10, zorder=1)

        return
    
    def build_text_for_texbox(first_last, start_date, end_date):
        """
        Builds the text block that will fill the explanatory
        text box in the plot.
        
        Parameters:
        
        first_last - the dictionary containing first and last values
                     for each metric. The key is a MetricType. The
                     value is a FirstLast object.
        start_date - the starting date, formatted as a string
        end_date   - the ending date, formatted as a string                     
        """
        def metric_summary_text(metric, first_last_dict, date_start, date_end):
            fl = first_last_dict[metric]
            label = METRIC_LABELS[metric]
            return f"{label}: {fl.first:,} ({date_start}) to {fl.last:,} ({date_end})"

        heading = f"{textbox_heading}: " if textbox_heading else ""
        text_lines = [f"{heading}{start_date} to {end_date}"]
        text_lines.append("\nValues represent running totals as of each\n"
                          "week, not totals just for that week.")
        if per_n > 1:
            text_lines.append(f"All numbers are per {per_n:,} people.")

        # Don't include hospitalizations, recoveries, or tested totals unless
        # they're in the metrics. This avoids cluttering up the textbox.

        text_lines.append("")
        text_lines.append(metric_summary_text(MetricType.DEATHS, first_last, start_date, end_date))
        text_lines.append(metric_summary_text(MetricType.CASES, first_last, start_date, end_date))
        if MetricType.RECOVERIES in metrics:
            text_lines.append(metric_summary_text(MetricType.RECOVERIES, first_last, start_date, end_date))
        if MetricType.HOSPITALIZATIONS in metrics:
            text_lines.append(metric_summary_text(MetricType.HOSPITALIZATIONS, first_last, start_date, end_date))
        if MetricType.TESTED in metrics:
            text_lines.append(metric_summary_text(MetricType.TESTED, first_last, start_date, end_date))
        
        return '\n'.join(text_lines)

    # Main logic

    first_last = dict()
    errors = []

    fig, ax = p.subplots(figsize=figsize)

    if region == 'United States':
        # Without reset_index(), the grouping column gets lost.
        # The non-summable columns (e.g., Month_Day) get lost,
        # regardless.
        df = df.groupby(by=COL_DATE).sum().reset_index()
        df[COL_REGION] = 'United States'
        # Month_Day gets lost. Recreate it.
        make_month_day_column(df)
    else:
        df = df.loc[df[COL_REGION] == region].sort_values(by=[COL_DATE], inplace=False)

    if MetricType.POSITIVITY in metrics:
        # Calculate the positivity rate (cases / tests)
        df[COL_POSITIVITY] = df[COL_CASES] / df[COL_TESTS]

    all_metrics = {x for x in MetricType}
    # Skip positivity if not explicitly requested.
    # Some data sources don't have test values, making
    # it impossible to calculate.
    if MetricType.POSITIVITY not in metrics:
        all_metrics = all_metrics - {MetricType.POSITIVITY}

    for m in all_metrics:
        df_by_date = df.sort_values(by=[COL_DATE], inplace=False)
        maybe_plot_metric(m, ax, df_by_date, first_last, errors)

    if legend_loc is None:
        if (len(metrics) > 1) or moving_average:
            legend_loc = 'upper center'
        else:
            legend_loc = 'best'

    ax.legend(loc=legend_loc)

    x_label = f"Week\n\n(Source: {source})"
    ax.set_xlabel(x_label)

    y_label = ', '.join(METRIC_LABELS[m] for m in metrics)
    if per_n > 1:
        y_label = f"{y_label} per {per_n:,} people"
    ax.set_ylabel(y_label)

    # Build the explanatory text box.
    if len(errors) > 0:
        text_lines = ["ERROR: Can't plot all requested metrics.\n"]
        text_lines.extend(errors)
        text = '\n'.join(text_lines)
    else:
        start_date = df_by_date[COL_DATE].min().date().strftime('%B %d')
        end_date = df_by_date[COL_DATE].max().date().strftime('%B %d')
        text = build_text_for_texbox(first_last, start_date, end_date)

    text_x, text_y = textbox_loc or (0.01, 0.98)
    textbox(ax, text_x, text_y, text)

    # Save the plot, if desired.
    if image_file is not None:
        fig.savefig(os.path.join(IMAGES_PATH, image_file))

    return None


def plot_state(df, source, region, image_file, metrics, moving_average=False, legend_loc=None,
               textbox_loc=None):
    """
    Convenience front-end to plot_stats_by_date() that puts a heading for the state
    in the textbox.
    """
    return plot_stats_by_date(df, source,
                              region=region,
                              textbox_heading=region, 
                              image_file=image_file, 
                              metrics=metrics,
                              moving_average=moving_average,
                              textbox_loc=textbox_loc,
                              legend_loc=legend_loc)


def plot_states(df, source, states, metric=MetricType.DEATHS, per_n=1,
                populations=None, textbox_heading=None, textbox_loc=None, 
                figsize=(20, 12), legend_loc="lower right", image_file=None):
    """
    Takes a Pandas DataFrame with the normalized data, and plots a particular
    metric once for each of a group of states, across all the dates in the
    DataFrame.
    
    Parameters:
    
    df              - The Pandas DataFrame from which to select the data.
    source          - Description of data source
    states          - The names of the states.
    metric          - The metric to plot. Defaults to deaths.
    per_n           - If set to 1, plot the data as is. Otherwise, do a per-capita
                      plot (i.e., number of X per n people). If per_n is not 1,
                      then population must be defined.
    populations     - The dictionary of populations per state. Only necessary
                      if per_n is greater than 1.
    figsize         - The size of the plot.
    textbox_heading - An optional heading to add to the textbox annotation
    textbox_loc     - An (x, y) tuple for the location of the text box's top
                      left corner. Defaults to the upper left.
    legend_loc      - Location of the legend, using matplotlib semantics. Defaults
                      to "lower right"
    image_file      - Name of image file in which to save plot, or None.
    """
    # Get a derived DataFrame with just the states passed in.
    # Also, we only care about the Province_State, Month_Day
    # and particular statistic column. Finally, ensure that the
    # resulting DataFrame is sorted by date, just in case the 
    # original was reordered.
    metric_col = METRIC_COLUMNS[metric]
    df2 = (df.loc[df[COL_REGION].isin(states)][[COL_REGION, COL_MONTH_DAY, metric_col]]
             .sort_values(by=[COL_MONTH_DAY], inplace=False))
    if per_n > 1:
        func = lambda r: get_per_capita_float(r[metric_col], populations[r[COL_REGION]])
        df2[metric_col] = df2.apply(func, axis=1)

    # GROUP BY, SUM. Hello, SQL folks...
    group = df2.groupby([COL_MONTH_DAY, COL_REGION]).sum()
    
    # Unstack, to get each state's numbers in a separate column.
    final_df = group.unstack()

    fig, ax = p.subplots(figsize=figsize)

    # Let Pandas plot the whole thing.
    final_df.plot(ax=ax, kind='line', legend=True)
    fix_pandas_multiplot_legend(ax, legend_loc)

    # Set the X and Y axis labels. Add the credit below the X label,
    # since it's a nice place to stash it without interfering with
    # the plot.
    xlabel = (f"Week\n\n(Source: {source})")
    ax.set_xlabel(xlabel)
    metric_label = METRIC_LABELS[metric]
    label = metric_label if per_n == 1 else f"{metric_label} per {per_n:,} people"
    ax.set_ylabel(label)

    # Add an explanatory text box.
    text_x, text_y = textbox_loc or (0.01, 0.987)
    heading = "" if textbox_heading is None else f"{textbox_heading}: "
    text_lines = [f"{heading}{label}\n"]
    for state in sorted(states):
        # Get the last value for the metric. It's the grand total.
        total = round(int(df2.loc[df2[COL_REGION] == state][metric_col].iloc[-1]))
        text_lines.append(f"{state}: {total:,}")
    textbox(ax, text_x, text_y, '\n'.join(text_lines))
    
    if image_file is not None:
        fig.savefig(os.path.join(IMAGES_PATH, image_file))


def plot_states_per_capita(df, source, populations, metric=MetricType.DEATHS,
                           figsize=(25, 12), show_us_per_capita=True,
                           per_n=1_000_000, image_file=None):
    """
    Plot a per-capita bar chart comparing all states, for a particular
    metric.
    
    Parameters:
    
    df                 - The Pandas DataFrame with the data.
    source             - Description of data source
    populations        - The loaded dictionary of populations.
    metric             - The metric to graph. Defaults to MetricType.DEATHS
    figsize            - The graph size. Defaults to 25x12
    show_us_per_capita - If True, show a line for the overall US per
                         capita value.
    per_n              - The per-capita factor. Must be 1,000 or larger.
    image_file         - Where to save the image, or None.
    """
    fig, ax = p.subplots(figsize=figsize)
    metric_col = METRIC_COLUMNS[metric]
    color = METRIC_COLORS[metric]
    label = METRIC_LABELS[metric]

    assert per_n >= 1_000
    
    states = set(populations.keys()) - {'United States'}
    df_states = df.loc[df[COL_REGION].isin(states)].sort_values(by=[COL_DATE], inplace=False)
    
    us_total = df_states[[COL_REGION, metric_col]].groupby([COL_REGION]).max().sum()
    us_per_capita = get_per_capita_int(us_total, populations['United States'], per_n=per_n)

    df_states['per_capita'] = df_states.apply(
        lambda r: get_per_capita_float(r[metric_col], populations[r[COL_REGION]], per_n=per_n),
        axis=1
    )

    df_grouped = df_states[[COL_REGION, 'per_capita']].groupby([COL_REGION]).max()

    # We're using zorder here to ensure that the horizontal line showing the
    # US per capita value shows up behind the bars.
    df_grouped.plot.bar(ax=ax, color=color, zorder=2, legend=False)
    if show_us_per_capita:
        ax.axhline(us_per_capita, color="lightgray", zorder=1)
        ax.set_yticks(ax.get_yticks() + [us_per_capita])

    # Set the X and Y axis labels. Add the credit below the X label,
    # since it's a nice place to stash it without interfering with
    # the plot.
    xlabel = (f"State\n\n(Source: {source})")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(f"{label} for {per_n:,} people")

    # See https://stackoverflow.com/a/25449186
    # (specifically, the first comment.)
    for patch in ax.patches:
        val = f"{round(int(patch.get_height())):,}"
        x = patch.get_x() + (patch.get_width() / 2)
        y = patch.get_height()
        ax.annotate(val, (x, y), ha='center', va='center', xytext=(0, 10), 
                    textcoords='offset points', zorder=2)

    if show_us_per_capita:
        text = f"U.S. {label} per {per_n:,} people: {us_per_capita:,}"
        textbox(ax, 0.01, 0.98, text)

    if image_file is not None:
        fig.savefig(os.path.join(IMAGES_PATH, image_file))

    return (fig, ax)


def textbox(ax, x, y, contents, fontsize=12, boxstyle='round', bg='xkcd:pale green'):
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