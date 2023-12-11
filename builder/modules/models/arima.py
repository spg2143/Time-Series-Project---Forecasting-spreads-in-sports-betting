import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_predict
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.graphics.tsaplots import plot_pacf
from statsmodels.tsa.arima.model import ARIMA

'''
For univariate TS analysis, we have to chose a way to present the time series at regular intervals.
For the dataset we have, one possibility is to look at the betting spread, for individual teams,
for single season. We assume the data is perfectly weekly (which is approximately true), and fit an ARIMA
to this data. One issue with this is that individual seasons are a small amount of data. The other alternative
is to ignore the off-season and just assume the data is weekly starting from the first date. This allows us to capture
more data, but has the disadvantage that it doesnt truly reflect reality. Teams can also undergo significant changes
during the off-season, which certainly affect spreads.
'''

def process_data_for_time_series(df, team):
    bd = df[((df['team_home'] == team) | (df['team_away'] == team))]

    # Readjusting the spreads so that - indicates that input team is favored
    # while positive indicates input team is the underdog

    bd.loc[(bd['team_favorite_id'] == 'PHI') & (bd['spread_favorite'] > 0), 'spread_favorite'] = -1 * bd[
        'spread_favorite']
    bd.loc[(bd['team_favorite_id'] != 'PHI') & (bd['spread_favorite'] < 0), 'spread_favorite'] = -1 * bd[
        'spread_favorite']

    # Generate a time series. Here I am assuming all the time series is weekly frequency
    # since the start date of the data

    ts = bd[['schedule_date', 'spread_favorite']]
    ts.set_index('schedule_date', inplace=True)
    index = pd.date_range(ts['schedule_date'].values[0], periods=len(bd), freq="W")
    ts.set_index(index, inplace=True)

def time_series_plots(ts):
    plot_acf(ts['spread_favorite'])
    plot_pacf(ts['spread_favorite'])
    ts.plot()
    plt.show()

def arima_model(ts, p, d, q):       # p, d, q need to be determined by acf and pacf
    model = ARIMA(ts, order=(p, d, q))
    model_fit = model.fit()

    plot_predict(model_fit, dynamic=False)  # if we want to plot predictions

    plt.show()
