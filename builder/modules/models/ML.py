# This is the xgboost file. Not naming xgboost.py since this can cause issues in the library.

import pandas as pd
import xgboost as xgb
import sklearn as sk
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


def create_input_data(df):
    '''
    First, gettnig rid of some irrelevant features. For the xgboost model, temporal data is not important,
    and we can't use the winner/loser of a game as a feature before hand. Weather_detail is a categorical feature.
    '''

    df.reset_index(inplace=True)
    df.drop(columns=['index', 'schedule_date', 'team_favorite_id', 'weather_detail', 'winner', 'loser'],
            inplace=True)

    '''
    Creating X_train, Y_train, X_test, Y_test. Dropping more categorical features.
    '''
    train1 = df[df['schedule_season'] < 2016]
    test1 = df[df['schedule_season'] >= 2016]

    X_train = train1.drop(
        columns=['schedule_season', 'team_home', 'score_home', 'score_away', 'team_away', 'over_under_line', 'stadium',
                 'spread_favorite'])
    y_train = train1['spread_favorite']
    X_test = test1.drop(
        columns=['schedule_season', 'team_home', 'score_home', 'score_away', 'team_away', 'over_under_line', 'stadium',
                 'spread_favorite'])
    y_test = test1['spread_favorite']

    return X_train, y_train, X_test, y_test


def generate_predictions(X_train, y_train, X_test, y_test):       # can modify this function to adjust xgboost model parameters
    model = xgb.XGBRegressor(n_estimators=1000)
    model.fit(X_train, y_train, verbose=False)

    predictions = model.predict(X_test)


    '''
    Creating output df of form prediction, actual, percentage error. Note that when the actual spread is 0,
    percentage error is not defined, so I put Nan.
    '''

    series = pd.Series(predictions)
    thingy = y_test.reset_index()
    out = pd.concat([thingy, series], axis=1)
    out.drop(columns=['index'], inplace=True)
    out.rename(columns={"spread_favorite": "Test Data", 0: "Output"}, inplace=True)
    out["Absolute Percentage Error"] = np.where(out["Test Data"] != 0.0,
                                                np.abs((out["Test Data"] - out["Output"]) / out["Test Data"]), np.nan)

    return out


def plots(model, output_df):
    '''
    This plots xgboost feature importance, and histogram of the errors from the output dataframe of the model.
    '''

    xgb.plot_importance(model)
    output_df.hist(column="Absolute Percentage Error")
    plt.show()
