import pandas as pd
import numpy as np
from setup import setup as s


spreads = pd.read_csv('data/spreadspoke_scores.csv')
teams = pd.read_csv('data/nfl_teams.csv')
stadiums = pd.read_csv('data/nfl_stadiums.csv', header=0,encoding='unicode_escape')

def merge(dataframe1, dataframe2, column_1, column_2, type):
    '''
        Merge two dataframes
        column_1: column in dataframe1
        column_2: column in dataframe2
        type: which type of join i.e type in 'left', 'right', 'inner', 'outer'
    '''
    merged_df = pd.merge(dataframe1, dataframe2, left_on=column_1, right_on=column_2, how=type)
    return merged_df

data = merge(spreads, stadiums, 'stadium', 'stadium_name', 'left')

data['team_home'] = data.team_home.map(teams.set_index('team_name')['team_id'].to_dict())
data['team_away'] = data.team_away.map(teams.set_index('team_name')['team_id'].to_dict())

data = s.filter_df(data,'schedule_playoff', False, 'eq')
data = s.filter_df(data, 'stadium_neutral', False, 'eq')
data = s.filter_df(data, 'schedule_season', 2008, 'greq')
data = s.filter_df(data, 'schedule_season', 2018, 'leq')

data['spread_favorite'] = np.where(data['team_favorite_id'] == data['team_away'],
                                       np.abs(data['spread_favorite']),
                                       data['spread_favorite'])
data['home_away'] = data['spread_favorite'].apply(lambda x: 1 if x < 0 else 0)

def replace_values_in_columns(dataframe, columns, old_value, new_value):
    '''
    Replace values in specified columns of a DataFrame
    dataframe: the DataFrame to modify
    columns: a list of column names where replacement should occur
    old_value: the value(s) to be replaced
    new_value: the value(s) to replace with
    '''
    dataframe[columns] = dataframe[columns].replace(old_value, new_value)
    return dataframe

data = replace_values_in_columns(data, ['team_home', 'team_away'], 'St. Louis Rams', 'Los Angeles Rams')
data = replace_values_in_columns(data, ['team_home', 'team_away'], 'San Diego Chargers', 'Los Angeles Chargers')
data = replace_values_in_columns(data, ['stadium_type'], 'retractable', 'indoor')

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

stadium_capacity_mapping = {
    'Texas Stadium': '66,000',
    'Edward Jones Dome': '67,000',
    'Sun Life Stadium': '65,000',
    'Hubert H. Humphrey Metrodome': '64,000',
    'Mall of America Field': '64,000',
    'TCF Bank Stadium': '50,000',
    'Giants Stadium': '82,500',
    'Candlestick Park': '70,000'
}
data['stadium_capacity'] = data.apply(
    lambda row: stadium_capacity_mapping.get(row['stadium'], row['stadium_capacity']), axis=1
)

stadium_surface = {
    'Texas Stadium': 'FieldTurf',
    'Edward Jones Dome': 'FieldTurf',
    'Sun Life Stadium': 'Grass',
    'Hubert H. Humphrey Metrodome': 'FieldTurf',
    'Mall of America Field': 'FieldTurf',
    'TCF Bank Stadium': 'FieldTurf',
    'Giants Stadium': 'Grass',
    'Candlestick Park': 'Grass'
}
data['stadium_surface'] = data.apply(
    lambda row: stadium_surface.get(row['stadium'], row['stadium_surface']), axis=1
)