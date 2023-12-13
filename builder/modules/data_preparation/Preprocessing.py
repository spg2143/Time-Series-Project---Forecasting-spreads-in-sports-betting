import pandas as pd
import itertools
from sklearn.metrics import mean_squared_error
import numpy as np
from Features import new_data
from modules.setup_functions.setup_functions import Setup_function as s

train = s.filter_df(new_data, 'schedule_season', 2016, 'leq')
test = s.filter_df(new_data, 'schedule_season', 2017, 'greq')

def preprocess_data(train, test):
    '''
        Split the data accordingly; assigns dummy variables to categorical variables like surface and type
        train: train dataframe 
        test: test dataframe
    '''
    train_exog_surface = pd.get_dummies(train['stadium_surface'], drop_first=True)
    train_exog_type = pd.get_dummies(train['stadium_type'], drop_first=True)

    test_exog_surface = pd.get_dummies(test['stadium_surface'], drop_first=True)
    test_exog_type = pd.get_dummies(test['stadium_type'], drop_first=True)

    train_exog = pd.concat([train[['home_away', 'stadium_capacity', 'weather_temperature', 'weather_wind_mph', 'home_team_wins', 'away_team_wins']],train_exog_surface, train_exog_type], axis = 1)
    test_exog = pd.concat([test[['home_away', 'stadium_capacity', 'weather_temperature', 'weather_wind_mph', 'home_team_wins', 'away_team_wins']],test_exog_surface, test_exog_type], axis = 1)

    train_exog.index = train['schedule_date']
    test_exog.index = test['schedule_date']

    train_exog['Grass'] = train_exog['Grass'].astype('int64')
    train_exog['outdoor'] = train_exog['outdoor'].astype('int64')

    test_exog['Grass'] = test_exog['Grass'].astype('int64')
    test_exog['outdoor'] = test_exog['outdoor'].astype('int64')

    train_exog['stadium_capacity'] = train_exog['stadium_capacity'].str.replace(',', '').astype(float)
    test_exog['stadium_capacity'] = test_exog['stadium_capacity'].str.replace(',', '').astype(float)

    train_endog = train['spread_favorite']
    test_endog = test['spread_favorite']

    return train_exog, test_exog, train_endog, test_endog