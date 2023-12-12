import pandas as pd
import xgboost as xgb
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

pd.set_option('display.max_columns', None)  # or 1000


# need data from 2008 season to 2018 season

df1 = pd.read_csv('data/spreadspoke_scores.csv')
teams = pd.read_csv("data/nfl_teams.csv",header=0)

filter1 = df1[df1['schedule_season'] >= 2008]                 # how do we handle 72 degrees, 0 wind points?
df = filter1[filter1['schedule_season'] <= 2018]

df['team_home'] = df.team_home.map(teams.set_index('team_name')['team_id'].to_dict())
df['team_away'] = df.team_away.map(teams.set_index('team_name')['team_id'].to_dict())

df['spread_favorite'] = np.where(df['team_favorite_id'] == df['team_away'], np.abs(df['spread_favorite']), df['spread_favorite'])

winner = []
# Obtain the scores for each area           #CHECK THE CODE LINE 16- 81
for i, v in df.score_home.items():
    if df.score_home[i] > df.score_away[i]:
        winner.append(df.team_home[i])

    elif df.score_home[i] < df.score_away[i]:
        winner.append(df.team_away[i])
    else:
        winner.append('Tie')

df['winner'] = winner

loser = []
# Obtain the scores for each area
for i, v in df.score_home.items():
    if df.score_home[i] > df.score_away[i]:
        loser.append(df.team_away[i])

    elif df.score_home[i] < df.score_away[i]:
        loser.append(df.team_home[i])
    else:
        loser.append('Tie')

df['loser'] = loser

week_mapping = {        # CHECK OVER THIS CODE
    'Wildcard': 19,
    'Division': 20,
    'Conference': 21,
    'Superbowl': 22
}

df['schedule_week'] = df['schedule_week'].replace(week_mapping)         # CHECK OVER THIS CODE
df['schedule_week'] = pd.to_numeric(df['schedule_week'], errors='coerce')

df = df.sort_values(by=['schedule_season', 'schedule_week'], ascending=True)

df['home_team_wins'] = 0
df['home_team_losses'] = 0
df['away_team_wins'] = 0
df['away_team_losses'] = 0

for idx, row in df.iterrows():          # go over all this code too
    season = row['schedule_season']
    week = row['schedule_week']
    home_team = row['team_home']
    away_team = row['team_away']

    games_this_season = df[(df['schedule_season']==season) & (df['schedule_week'] < week)]

    home_team_wins = games_this_season[games_this_season['winner'] == home_team]['winner'].count()
    home_team_losses = games_this_season[games_this_season['loser'] == home_team]['loser'].count()
    away_team_wins = games_this_season[games_this_season['winner'] == away_team]['winner'].count()
    away_team_losses = games_this_season[games_this_season['loser'] == away_team]['loser'].count()

    df.loc[idx, 'home_team_wins'] = home_team_wins
    df.loc[idx, 'home_team_losses'] = home_team_losses
    df.loc[idx, 'away_team_wins'] = away_team_wins
    df.loc[idx, 'away_team_losses'] = away_team_losses

mean_humidity = df['weather_humidity'].mean()
df['weather_humidity'].fillna(mean_humidity, inplace=True)
mean_temp = df['weather_temperature'].mean()
mean_wind = df['weather_wind_mph'].mean()
df.loc[(df['weather_temperature'] == 72.0) & (df['weather_wind_mph'] == 0.0), "weather_temperature"] = mean_temp
df.loc[(df['weather_temperature'] == mean_temp) & (df['weather_wind_mph'] == 0.0), "weather_wind_mph"] = mean_wind

df.to_csv('data/weather.csv', index = 0)



