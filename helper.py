"""
Helper functions for project
"""

# Import libraries
import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import date
from datetime import datetime
from dateutil.relativedelta import relativedelta
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.ar_model import AutoReg
from statsmodels.tsa.arima.model import ARIMA
from datetime import timedelta
from sklearn.metrics import root_mean_squared_error

def perform_adf_test(series):
    """Performs Augmented Dicker Fuller Test (stationarity or not)"""
    result = adfuller(series)
    print('ADF Statistic: %f' % result[0])
    print('p-value: %f' % result[1])
    return

def transform_data(data):
    """
    data = a pandas dataframe where the columns are the tickers and the values are the prices, index is date
    converts adjusted closing price to first difference to make the data stationary
    """
    return data.diff().dropna()

def split_data(data, ticker, train_end, test_end):
    """
    data = a pandas dataframe
    ticker = ticker symbol for stock
    train_end = the datetime training data should go up to 
    test_end = the datetime the model should forecast until

    returns training_data, testing_data
    """
    ticker_data = data[ticker].copy()
    ticker_data.index = pd.DatetimeIndex(ticker_data.index).to_period('D') # Convert datetime index for proper forecasting

    train_data = ticker_data[:train_end]
    test_data = ticker_data[train_end+timedelta(days=1):test_end]

    return train_data, test_data

def undo_transformations(transformed_series, original_series):
    """
    Converts our first difference back to original Adj Close Price
    transformed_series = a pandas series of transformed values
    original_series = a panads series of original, untransformed, data
    """
    first_pred = transformed_series.iloc[0] + original_series.iloc[-1]
    orig_predictions = [first_pred]

    for i in range(len(transformed_series[1:])):
        next_pred = transformed_series.iloc[i+1] + orig_predictions[-1]
        orig_predictions.append(next_pred)
    
    return np.array(orig_predictions).flatten()

def random_walk(last_price, forecast_length, sigma):
    """
    last_price = the price of the last day (untransformed)
    forecast_length = number of days you want to forecast
    sigma = daily volatility of stock
    credit: https://github.com/teobeeguan/Python-For-Finance/blob/main/Time%20Series/RandomWalkSimulation.py

    returns a list with forecasted_values
    """
    count = 0
    price_list = []

    price = last_price*(1+np.random.normal(0, sigma))
    price_list.append(price)

    for y in range(forecast_length):
        if count == forecast_length-1:
            break
        price = price_list[count]*(1+np.random.normal(0,sigma))
        price_list.append(price)
        count+=1

    return price_list
    

def evaluate_model(fitted_model, testing_data, training_data, original_training_data, original_testing_data):
    """
    fitted_model = fitted model object
    testing_data = actual transformed data that we tried to forecast
    training_data = transformed training data (to recover transformed predictions)
    original_training_data = pd.Series() of untransformed training data
    original_testing_data = untransformed testing data

    generate predictions of the fitted models using test_data

    returns a datframe of the actual and predicted values
    """
    transformed_predictions = fitted_model.get_prediction(start = testing_data.index[0], end = testing_data.index[-1]) # forecast transformed from test start date to test end date
    predictions = undo_transformations(transformed_predictions.predicted_mean, original_training_data) # undo transformations to get forecasted Adj Close Price

    confint = transformed_predictions.conf_int() # converting transformed confidences intervals back to Adj Close Price
    orig_lower_bound = pd.Series(undo_transformations(confint.iloc[:,0], original_training_data), index = confint.index)
    orig_upper_bound = pd.Series(undo_transformations(confint.iloc[:,1], original_training_data), index = confint.index)

    # create a dataframe with 2 columns: actual, predicted 
    results = pd.concat([pd.Series(original_testing_data), pd.Series(predictions, index = transformed_predictions.predicted_mean.index), orig_lower_bound, orig_upper_bound], axis = 1, join = "inner")
    results.columns = ['actual', 'predicted', 'lower_bound', 'upper_bound']

    # ensures both actual and predicted columns are the same shape to do MAE calculations
    print(f"RMSE: {root_mean_squared_error(results.actual, results.predicted)}")

    results.index = results.index.to_timestamp() # convert index back to timestamp for ease of use (plotting)
    return results

def evaluate_rw(actual, predicted):
    """
    actual = a pandas series of the actual adj closing prices
    predicted = a pandas series of the predicted adj closing prices
    evaluates predictions of random walk model using RMSE
    """
    # Create data frame and join on common dates to prevent difference in shape
    if type(actual.index) != type(predicted.index):
        actual.index = actual.index.to_timestamp()

    df = pd.concat([actual, predicted], join = 'inner', axis = 1)
    df.columns = ['actual', 'predicted']

    # Compute RMSE
    print(f"RMSE of Random Walk: {root_mean_squared_error(df['actual'], df['predicted'])}")
    