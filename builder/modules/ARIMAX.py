import pandas as pd
import itertools
from sklearn.metrics import mean_squared_error
import numpy as np
from setup import setup as s
import statsmodels.api as sm
from statsmodels.tsa.api import VAR
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf



def find_ideal_order(train_endog, train_exog):
    p_values = range(0, 4)
    q_values = range(0, 4)

    best_aic = float('inf')
    best_order = None

    for p in p_values:
        for q in q_values:
            order = (p, 0, q)
            try:
                model = sm.tsa.ARIMA(train_endog, order=order, exog=train_exog, enforce_stationarity=True)
                results = model.fit()

                if results.aic < best_aic:
                    best_aic = results.aic
                    best_order = order

            except:
                continue

    return best_order


def fit_and_predict(train_endog, train_exog, test_exog, best_order):
    try:
        model = sm.tsa.ARIMA(train_endog, order=best_order, exog=train_exog, enforce_stationarity=True)
        results = model.fit()
        forecast = results.predict(start=len(train_endog), end=len(train_endog) + len(test_exog) - 1, exog=test_exog)
        return forecast

    except Exception as e:
        print(f"Error occurred: {e}")
        return None


def calculate_stats(test, forecast):
    forecast_df = forecast.reset_index()
    forecast_df.columns = ['schedule_date', 'Forecast_ARIMAX']
    forecast_df.set_index(test.index, inplace=True)
    forecast_df = forecast_df.drop(columns='schedule_date')
    forecast_df['actual_spread'] = test['spread_favorite']
    forecast_df['Percent_Error'] = (abs(forecast_df['actual_spread'] - forecast_df['Forecast_ARIMAX']) / forecast_df['actual_spread']) * 100

    mse = mean_squared_error(forecast_df['actual_spread'], forecast_df['Forecast_ARIMAX'])
    print("Mean Squared Error (MSE):", mse)

    return forecast_df, mse


def preprocess_data(train, test):
    train_exog_surface = pd.get_dummies(train['stadium_surface'], drop_first=True)
    train_exog_type = pd.get_dummies(train['stadium_type'], drop_first=True)

    test_exog_surface = pd.get_dummies(test['stadium_surface'], drop_first=True)
    test_exog_type = pd.get_dummies(test['stadium_type'], drop_first=True)

    train_exog = pd.concat([train[['home_away', 'stadium_capacity']], train_exog_surface, train_exog_type], axis=1)
    test_exog = pd.concat([test[['home_away', 'stadium_capacity']], test_exog_surface, test_exog_type], axis=1)

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
