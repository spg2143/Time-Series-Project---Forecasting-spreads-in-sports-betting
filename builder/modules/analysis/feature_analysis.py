
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from modules.data_preparation.Features import data
def calculate_stats(dataframe, groupby_cols, agg_col):
    '''
    Calculate statistics for a column(s) grouped by specified column(s)
    dataframe: DataFrame to perform aggregation on
    groupby_cols: list of columns to group by
    agg_col: column to perform aggregation on
    '''
    stats = dataframe.groupby(groupby_cols)[agg_col].agg(['mean', 'median', 'min', 'max', lambda x: x.quantile(0.25), lambda x: x.quantile(0.75)])
    stats.rename(columns={
        '<lambda_0>': '25th Percentile',
        '<lambda_1>': '75th Percentile'
    }, inplace=True)
    return stats
def line_plot(df, title, xlabel, ylabel):
    plt.figure(figsize=(10,6))
    df.plot()
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.show()
def plot_grouped_df(dataframe, label,x_axis, y_axis, title, x_label, y_label):
    '''
        This function takes a groupby dataaframe and plots it for each index
        dataframe: grouped dataframe
        label: should be shown in legend (ex. Season)
        x_axis: column used for x_axis (ex. week )
        y_axis: value over time (ex. Spread)
    '''
    dataframe.reset_index(inplace=True)
    plt.figure(figsize=(10, 6))

    labels = dataframe[label].unique()

    for x in labels:
        season_data = dataframe[dataframe[label] == x]
        plt.plot(season_data[x_axis], season_data[y_axis], label=x)

    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.show()

def plot_hist(dataframe, group_column,freq_column,num_bins, title, xlabel):
    """
        This function takes a dataframe and create bins for a histogram. It then plots the frequency for each bin over time
        dataframe: dataframe
        group_column: column for which the bins will be applied to
        freq_column: frequency
        num_bins: number of bins 
    """
    group_column_name = f'{group_column}_bins'
    group_bins = pd.cut(dataframe[group_column], bins = num_bins)
    dataframe[group_column_name] = group_bins

    plt.figure(figsize = (20,6))
    plt.subplot(1,3,1)
    for label, group in dataframe.groupby(group_column_name):
        group[freq_column].plot(kind = 'hist', alpha = 0.5, label = str(label))
    plt.title(title)
    plt.xlabel(xlabel)
    plt.legend()

    plt.tight_layout()
    plt.show()
##General Home Team Spread Stats per Season

home_stats = calculate_stats(data, ['schedule_season'], 'spread_favorite')
home_stats_plot = line_plot(home_stats, 'Stats per Season Across All Teams', 'Season', 'Spread')
##General Home Team Spread Stats per Week in each Season

home_stats_week = calculate_stats(data, ['schedule_season', 'schedule_week'], 'spread_favorite')
home_stats_week_plot = plot_grouped_df(home_stats_week,'schedule_season',
                                'schedule_week',
                                'mean', 'Mean Spread for Each Week Across Seasons',
                                'Week',
                                'Mean Spread')

##Each home team's spread stats per season 
individ_home_stats = calculate_stats(data, ['team_home', 'schedule_season'], 'spread_favorite')
individ_home_stats_plot = plot_grouped_df(individ_home_stats,'team_home',
                                'schedule_season',
                                'mean', 'Mean Spread per Season for Each Team',
                                'Season',
                                'Mean Spread')
##Basic Stats for each stadium type in the 10 years
stadium_type_stats = calculate_stats(data, ['stadium_type'], 'spread_favorite')

###Basic Stats for each stadium over each season

stadium_type_season_stats = calculate_stats(data, ['schedule_season', 'stadium_type'], 'spread_favorite')
stadium_type_season_stats_plot = plot_grouped_df(stadium_type_season_stats,'stadium_type',
                                'schedule_season',
                                'mean', 'Mean Spread per Season for Each Team',
                                'Season',
                                'Mean Spread')
##Stadium Capacity Stats
stadium_capacity_stats = calculate_stats(data, ['capacity_buckets'], 'spread_favorite')
stadium_capacity_distro_plot = plot_hist(data, 'stadium_capacity', 'spread_favorite', 5, 'Spread Distro Across Capacity Bins', 'Spread')
##Stadium Surface Stats
stadium_surface_stats = calculate_stats(data, ['stadium_surface'], 'spread_favorite')

##Stadium Surface per Season Stats
stadium_surface_season_stats = calculate_stats(data, ['schedule_season', 'stadium_surface'], 'spread_favorite')
stadium_surface_stats_plot = plot_grouped_df(stadium_surface_season_stats,'stadium_surface',
                                'schedule_season',
                                'mean', 'Mean Spread per Season for Each Team',
                                'Season',
                                'Mean Spread')
plot_grouped_df()
###Weather per Season for each Home Team 
individ_home_stats_weather = calculate_stats(data, ['team_home', 'schedule_season'], 'weather_temperature')
temperature_distro_plot = plot_hist(data, 'weather_temperature', 'spread_favorite', 5, 'Spread Distro Across Capacity Bins', 'Spread')

wind_distro_plot = plot_hist(data, 'weather_wind_mph', 'spread_favorite', 5, 'Spread Distro Across Capacity Bins', 'Spread')
