import pandas as pd
import numpy as np
from modules.setup_functions.setup_functions import Setup_function as s
##initialize Weather feature 
weather = pd.read_csv("data/weather.csv", index_col=0)
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
weather = s.filter_df(weather, 'schedule_playoff', False, 'eq')
data = s.filter_df(data, 'schedule_season', 2008, 'greq')
data = s.filter_df(data, 'schedule_season', 2018, 'leq')

replace_columns = ['weather_temperature', 'weather_wind_mph', 'weather_humidity',
                   'winner', 'loser', 'home_team_wins', 'home_team_losses', 'away_team_wins', 'away_team_losses']

# Replace columns in 'data' with columns from 'weather'
data[replace_columns] = weather[replace_columns]

data = s.filter_df(data, 'stadium_neutral', False, 'eq')

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

new_data = data[['schedule_date', 'schedule_season', 'schedule_week', 'home_away', 'stadium_surface', 'stadium_type', 'stadium_capacity', 'spread_favorite',
                'weather_temperature', 'weather_wind_mph', 'weather_humidity', 'weather_detail',
                   'winner', 'loser', 'home_team_wins', 'home_team_losses', 'away_team_wins', 'away_team_losses']]


new_data['schedule_date'] = pd.to_datetime(new_data['schedule_date'])

new_data.index = new_data['schedule_date']

new_data.replace([np.inf, -np.inf], 0, inplace=True)

