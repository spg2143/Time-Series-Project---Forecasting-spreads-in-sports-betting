import pandas as pd
import itertools
from sklearn.metrics import mean_squared_error
import numpy as np
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



