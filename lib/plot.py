"""
Plotting helpers.
"""
import matplotlib.pyplot as p
import matplotlib
import numpy as np
import pandas as pd
from math import inf
from typing import Tuple, Sequence, Optional, Union, Set, Dict
from lib.common import *

def fix_pandas_multiplot_legend(ax:         matplotlib.axes.Axes, 
                                legend_loc: Tuple[Union[int, float], Union[int, float]]):
    """
    When plotting multiple pieces of data, the Pandas-generated
    plot legend will often look like "(metric, place)" (e.g.,
    "(Deaths, Connecticut)".
    
    This function corrects the legend, by extracting just the place.
    
    Parameters:
    
    ax         - the plot axis
    legend_loc - the desired location for the legend, as a tuple
    """
    patches, labels = ax.get_legend_handles_labels()
    pat = re.compile(r'^\([^,\s]+,\s+(.*)\)$')
    labels2 = []
    for label in labels:
        m = pat.match(label)
        assert m is not None
        labels2.append(m.group(1))
    ax.legend(patches, labels2, loc=legend_loc)


def plot_daily_stats(df:             pd.DataFrame, 
                     source:         str, 
                     metric:         MetricType = MetricType.DEATHS, 
                     region:         str = 'United States',
                     moving_average: bool = False,
                     figsize:        Tuple[Union[int, float], Union[int, float]] = (20, 12),
                     image_file:     Optional[str] = None):
    """
    Takes a Pandas DataFrame with normalized data, calculate the
    per-day delta for a metric (which assumes the metric is an
    accumulating value), and plots it.
    
    Parameters:
    
    df              - The Pandas DataFrame to plot
    source          - Description of data source
    metrics         - A set containing the metrics to plot. Defaults to deaths.
    region          - The state name, or 'United States' for everything
    moving_average  - If True, plot a 7-day moving average along side the data. If
                      False, plot the data as is.
    figsize         - The size of the plot.
    image_file      - Name of image file in which to save plot, or None.
    
    Returns:
      A 3-tuple containing (fig, axis, dataframe), where "dataframe" is the
      region-specific Pandas data frame with deltas for the specific metric.
    """
    if region == 'United States':
        df = df.groupby(by=COL_DATE).sum().reset_index()
        df[COL_REGION] = 'United States'
        # Month_Day gets lost. Recreate it.
        make_month_day_column(df)
        df = df.sort_values(by=[COL_DATE], inplace=False)
    else:
        # Use a sort to make a copy.
        df = df.loc[df[COL_REGION] == region].sort_values(by=[COL_DATE], inplace=False)

    COL_DIFF = 'diff'
    COL_DIFF_MA = 'diff_ma'

    metric_col = METRIC_COLUMNS[metric]
    df[COL_DIFF] = df[metric_col].diff()
    df[COL_DIFF] = df[COL_DIFF].fillna(0)

    fig, ax = p.subplots(figsize=(20, 12))
    color = METRIC_COLORS[metric]
    label = METRIC_LABELS[metric]
    df.plot(x=COL_MONTH_DAY, y=COL_DIFF, ax=ax, label=f'Daily {label.lower()}', color=color, zorder=2)
    if moving_average:
        color = METRIC_MOVING_AVERAGE_COLORS[metric]
        df[COL_DIFF_MA] = df[COL_DIFF].rolling(7).mean().fillna(0)
        df.plot(x=COL_MONTH_DAY, y=COL_DIFF_MA, ax=ax, 
                label=f'Daily {label.lower()} (7-day moving average)', color=color,
                zorder=1, linewidth=10)

    ax.set_xlabel(f'Week\n\n(Source: {source})')
    ax.set_ylabel(f'Daily {label.lower()}, {region}')

    if image_file is not None:
        fig.savefig(os.path.join(IMAGES_PATH, image_file))

    return (fig, ax, df)


def plot_stats_by_date(df:              pd.DataFrame, 
                       source:          str,
                       metrics:         Set[MetricType] = {MetricType.DEATHS}, 
                       region:          str = 'United States',
                       moving_average:  bool = False,
                       per_n:           int = 1, 
                       populations:     Dict[str, int] = None,
                       textbox_heading: Optional[str] = None, 
                       textbox_loc:     Optional[Tuple[Union[int, float], Union[int, float]]] = None, 
                       marker:          Optional[str] = None, 
                       figsize:         Tuple[Union[int, float], Union[int, float]] = (20, 12), 
                       image_file:      Optional[str] = None,
                       legend_loc:      Optional[str] = None):
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

        def to_int(value):
            return 0 if value == inf else int(round(value))

        first_value = to_int(df[col].iloc[0])
        last_value = to_int(df[col].iloc[-1])
        first_last[m] = FirstLast(first=first_value, last=last_value)

        if m not in metrics:
            return

        # Since these values are cumulative, if the last value is 0, we
        # can't trust the data.
        last_val = df[col].iloc[-1]
        if (last_val <= 0) or (last_val == inf):
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
        df = df.sort_values(by=[COL_DATE], inplace=False)
    else:
        # Use a sort to make a copy.
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

    return (fig, ax)


def plot_state(df:             pd.DataFrame,
               source:         str,
               region:         str,
               metrics:        Set[MetricType],
               image_file:     Optional[str] = None,
               moving_average: bool = False,
               legend_loc:     Optional[str] = None, 
               textbox_loc:    Optional[Tuple[Union[int, float], Union[int, float]]] = None):
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


def plot_states(df:              pd.DataFrame,
                source:          str,
                states:          Sequence[str],
                metric:          MetricType = MetricType.DEATHS,
                per_n:           int = 1,
                populations:     Dict[str, int] = None,
                textbox_heading: Optional[str] = None,
                textbox_loc:     Optional[Tuple[Union[int, float], Union[int, float]]] = None, 
                figsize:         Tuple[Union[int, float], Union[int, float]] = (20, 12),
                legend_loc:      str = "lower right",
                image_file:      Optional[str] = None):
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
        
    return (fig, ax)


def plot_states_per_capita(df:                 pd.DataFrame,
                           source:             str,
                           populations:        Dict[str, int], 
                           metric:             MetricType = MetricType.DEATHS,
                           figsize:            Tuple[Union[int, float], Union[float, int]] = (25, 12),
                           show_us_per_capita: bool = True,
                           per_n:              int = 1_000_000,
                           image_file:         Optional[str] = None):
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



def plot_counties(df:              pd.DataFrame,
                  state:           str, 
                  counties:        Sequence[str],
                  source:          str,
                  metric:          MetricType = MetricType.DEATHS,
                  image_file:      Optional[str] = None,
                  textbox_loc:     Tuple[Union[int, float], Union[int, float]] = (0.01, 0.98),
                  textbox_heading: Optional[str] = None,
                  moving_average:  bool = False,
                  figsize:         Tuple[Union[int, float], Union[int, float]] = (20, 11),
                  legend_loc:      str = 'upper center'):
    """
    Plot a set of counties for a particular state. This data is New York Time-specific.
    
    Parameters:
    
    df              - the Pandas DataFrame
    state           - The state name
    counties        - The counties within the state (by name)
    source          - String identifying the data source
    metric          - The metric to plot
    image_file      - Where to save the image, if any
    textbox_loc     - (x, y) coordinates of the info textbox
    textbox_heading - A heading to add to the textbox, or None
    moving_average  - If False, plot the data as is. If True, plot
                      a 7-day moving average.
    figsize         - (width, height) of the plot
    legend_loc      - where to put the legend
    """
    metric_col = METRIC_COLUMNS[metric]
    cases_or_deaths = str(metric.name).capitalize()
    df = df.loc[df[COL_COUNTY].isin(counties) & (df[COL_REGION] == state)][[COL_MONTH_DAY, COL_COUNTY, metric_col]]

    if moving_average:
        new_frames = []
        for c in counties:
            dfc = df.loc[df[COL_COUNTY] == c].sort_values(COL_MONTH_DAY, inplace=False)
            dfc[metric_col] = dfc[metric_col].rolling(7).mean().fillna(0)
            new_frames.append(dfc)
        df = pd.concat(new_frames)

    group = df.groupby([COL_MONTH_DAY, COL_COUNTY]).sum()

    # Unstack, to get each county's numbers in a separate column.
    final_df = group.unstack()

    fig, ax = p.subplots(figsize=figsize)

    # Let Pandas plot the whole thing.
    final_df.plot(ax=ax, kind='line', legend=True)

    # Set the X and Y axis labels. Add the credit below the X label,
    # since it's a nice place to stash it without interfering with
    # the plot.
    ax.set_xlabel(f'Week\n\n(Source: {source})')
    metric_label = METRIC_LABELS[metric]
    if moving_average:
        ax.set_ylabel(f"{metric_label} (7-day moving average)")
    else:
        ax.set_ylabel(metric_label)

    # Add an explanatory text box.
    text_x, text_y = textbox_loc or (0.01, 0.987)
    heading = f"{state} county {metric_label.lower()}"
    text_lines = [f"{heading}"]
    if textbox_heading:
        text_lines.append(textbox_heading)
    text_lines.append("")
    for county in sorted(counties):
        # Get the last value for the metric. It's the grand total.
        total = round(int(df.loc[df[COL_COUNTY] == county][metric_col].iloc[-1]))
        text_lines.append(f"{county}: {total:,}")

    textbox(ax, text_x, text_y, '\n'.join(text_lines))
    fix_pandas_multiplot_legend(ax, legend_loc)

    if image_file is not None:
        fig.savefig(os.path.join(IMAGES_PATH, image_file))


def plot_county_daily_stats(df:          pd.DataFrame,
                            state:       str, 
                            county:      str,
                            source:      str,
                            metric:      MetricType,
                            textbox_loc: Optional[Tuple[Union[int, float], Union[int, float]]] = None,
                            image_file:  Optional[str] = None):
    """
    Plot day-by-day stats for a particular county.
    
    Parameters:
    
    df              - the Pandas DataFrame
    state           - The state name
    county          - The name of the county (within the state)
    source          - String identifying the data source
    metric          - The metric to plot
    image_file      - Where to save the image, if any
    textbox_loc     - (x, y) coordinates of the info textbox
    """
    df = (df.loc[(df[COL_REGION] == state) & (df[COL_COUNTY] == county)]
            .sort_values(by=[COL_DATE], inplace=False))
    metric_col = METRIC_COLUMNS[metric]
    metric_label = METRIC_LABELS[metric]
    df['diff'] = df[metric_col].diff().fillna(0)
    df['diff_ma'] = df['diff'].rolling(7).mean().fillna(0)

    fig, ax = p.subplots(figsize=(20, 12))
    df.plot(x=COL_MONTH_DAY, y='diff', ax=ax, label=f'daily {metric_label.lower()}', 
            zorder=1, color=METRIC_COLORS[metric])
    df.plot(x=COL_MONTH_DAY, y='diff_ma', ax=ax, label='7-day moving average', 
            zorder=0, color=METRIC_MOVING_AVERAGE_COLORS[metric], linewidth=10)

    ax.set_xlabel(f'Week\n\n(Source: {source})')
    ax.set_ylabel(f'Daily {metric_label}, {county} County, {state}')
    text_x, text_y = textbox_loc or (0.01, 0.99)
    textbox(ax=ax, x=text_x, y=text_y, contents=f'{county} County, {state}')

    if image_file is not None:
        fig.savefig(os.path.join(IMAGES_PATH, image_file))

    return (fig, ax)