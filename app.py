# imports

import os
import numpy as np
import pandas as pd
import pickle

from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.linear_model import SGDRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler

import xgboost as xgb

from keras.layers import Dense, Dropout, LSTM
from keras.callbacks import EarlyStopping
from keras.models import Sequential
from keras.models import load_model

from prettytable import PrettyTable
from multipledispatch import dispatch
from datetime import date, timedelta
import pandas_datareader.data as web
from pandas_datareader._utils import RemoteDataError

import matplotlib.pyplot as plt
from plotly import graph_objs as go

import streamlit as st

from prophet import Prophet
from prophet.plot import plot_plotly, plot_components_plotly

DEBUG = False
st.set_page_config(layout="wide")

@st.cache(suppress_st_warning=True)
def get_data_from_web(ticker):
    # NSE was incorporated in 1992. It was recognised as a stock exchange by SEBI in April 1993 and commenced operations in 1994
    print("Getting the data for the ticker ", ticker)
    # start = date(1994, 1, 1)
    start = date.today() - timedelta(days=(5*365))
    yesterday = date.today() - timedelta(days=1)
    try:
        df = web.DataReader(ticker + ".NS", 'yahoo', start=start, end=yesterday)
    except:
        st.write(f"No information for ticker {ticker}, exiting....")
        st.stop()
    return df
    
def plot_raw_data(rdata):
    data = rdata.copy()
    data.reset_index(inplace=True)
    layout = dict(
        showlegend=True,
        width=1800,
        height=600,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        yaxis=dict(
            title="Stock Price"
        ),
        xaxis=dict(
            title="Time",
            type='date',
            rangeselector=dict(
                buttons=list([
                    dict(count=7,
                         label='1w',
                         step='day',
                         stepmode='backward'),
                    dict(count=1,
                         label='1m',
                         step='month',
                         stepmode='backward'),
                    dict(count=6,
                         label='6m',
                         step='month',
                         stepmode='backward'),
                    dict(count=1,
                         label='1y',
                         step='year',
                         stepmode='backward'),
                    dict(step='all')
                ])
            ),
            rangeslider=dict(
                visible=True
            ),
        ),
    )
    fig = go.Figure(layout=layout)
    fig.add_trace(go.Scatter(x=data["Date"], y=data["Open"], name="Open_price"))
    fig.add_trace(go.Scatter(x=data["Date"], y=data["Close"], name="Close_price"))
    fig.layout.update(title_text = f"Opening & Closing Price for {ticker}", xaxis_rangeslider_visible=True,
                        autosize=False, template="plotly_dark")
    st.plotly_chart(fig)

def train_test_split(data, split_sz):
    data_train = pd.DataFrame(data['Close'][0: int(len(data)*split_sz)])
    data_test = pd.DataFrame(data['Close'][int(len(data)*split_sz): int(len(data))])

    print("Size of complete data = ", data.shape)
    print("Size of training data = ", data_train.shape)
    print("Size of testing data = ", data_test.shape)

    return data_train, data_test

def prepare_train_set(train_data, win_sz):
    scaler = MinMaxScaler(feature_range=(0, 1))
    train_data_arr = scaler.fit_transform(train_data)
    
    train_data_dt = train_data.reset_index()["Date"]
    train_x = []
    train_y = []
    train_t = []

    for i in range(win_sz, len(train_data_arr)):
        train_x.append(train_data_arr[i-win_sz : i, 0])
        train_y.append(train_data_arr[i, 0])
        train_t.append(train_data_dt[i])

    train_x, train_y, train_t = np.array(train_x), np.array(train_y), np.array(train_t)
    print("Shape of train feature set ", train_x.shape)
    print("Shape of train target ", train_y.shape)
    print("Shape of train timestamp ", train_t.shape)
    
    return scaler, train_x, train_y, train_t

def prepare_test_set(past_n_days_data, test_data, win_sz):
    test_data = pd.concat([past_n_days_data, test_data])

    scaler = MinMaxScaler(feature_range=(0, 1))
    test_data_arr = scaler.fit_transform(test_data)
    
    test_data_dt = test_data.reset_index()["Date"]
    test_x = []
    test_y = []
    test_t = []

    for i in range(win_sz, len(test_data_arr)):
        test_x.append(test_data_arr[i-win_sz: i, 0])
        test_y.append(test_data_arr[i, 0])
        test_t.append(test_data_dt[i])

    test_x, test_y, test_t = np.array(test_x), np.array(test_y), np.array(test_t)
    print("Size of train dataset : ", test_x.shape)
    print("Size of test dataset : ", test_y.shape)
    print("Size of test timestamp : ", test_t.shape)

    return scaler, test_x, test_y, test_t

def two_plot(train_t, train_y, train_y_hat, test_t, test_y, test_y_hat):
    fig = plt.figure(figsize=(25,8))
    plt.subplot(211)
    plt.title("Actual Vs. Prediction on Training data")
    plt.plot(train_t, train_y, color='blue', label="Actual")
    plt.plot(train_t, train_y_hat, color='green', label="Predicted")
    plt.legend()

    plt.subplot(212)
    plt.title("Actual Vs. Prediction on Testing data")
    plt.plot(test_t, test_y, color='blue', label="Actual")
    plt.plot(test_t, test_y_hat, color='green', label="Predicted")
    plt.legend()
    st.pyplot(fig)

if __name__ == "__main__":
    st.header('Stock Price Trend Prediction App')

    # Get ticker from user
    # ticker = st.text_input('Enter stock ticker', 'SBIN')
    tickers = [ "RELIANCE", "SBIN", "HINDUNILVR", "HDFCBANK", "TCS",
                "AXISBANK", "ITC", "BAJFINANCE", "TECHM", "ASIANPAINT"]
    ticker = st.selectbox("Select ticker for prediction.", tickers)
    
    # =============
    data_load_state = st.text('Loading last 5 years of data...')
    data = get_data_from_web(ticker)
    data_load_state.text(f"Successfully fetched data for {ticker}!")
    # =============

    st.subheader("Raw data")
    st.write(data.tail())

    # =============
    plot_raw_data(data)

    # =============
    data_train, data_test = train_test_split(data, split_sz= 0.8)
    
    # win_size = 5 # Based on experiment on different window size, 5 seems to be the best.
    # train_scaler, train_x, train_y, train_t = prepare_train_set(data_train, win_size)
    
    # past_n_days_data = data_train.tail(win_size)
    # test_scaler, test_x, test_y, test_t = prepare_test_set(past_n_days_data, data_test, win_size)
    # =============

    print("Predicting using fbprophet model...")
    fb_data_train = data_train.copy()
    fb_data_test = data_test.copy()

    fb_data_train.reset_index(inplace=True)
    fb_data_test.reset_index(inplace=True)

    fb_data_train = fb_data_train[["Date", "Close"]]
    fb_data_test = fb_data_test[["Date", "Close"]]

    fb_data_train.rename(columns = {"Date" : "ds", "Close" : "y"}, inplace=True)
    fb_data_test.rename(columns = {"Date" : "ds", "Close" : "y"}, inplace=True)

    fb_mdl = Prophet()
    fb_mdl.fit(fb_data_train)

    fb_data_train["yhat"] = fb_mdl.predict(fb_data_train[["ds"]])[["yhat"]] 
    fb_data_test["yhat"] = fb_mdl.predict(fb_data_test[["ds"]])[["yhat"]]

    train_mape = mean_absolute_percentage_error(fb_data_train[["y"]], fb_data_train[["yhat"]])
    test_mape = mean_absolute_percentage_error(fb_data_test[["y"]], fb_data_test[["yhat"]])

    train_mse = mean_squared_error(fb_data_train[["y"]], fb_data_train[["yhat"]])
    test_mse = mean_squared_error(fb_data_test[["y"]], fb_data_test[["yhat"]])

    # Save results for comparison
    fbprophet_result = {"Model" : "FB Prophet",
                        "train_mape" : train_mape, "train_mse" : train_mse,
                        "test_mape" : test_mape, "test_mse" : test_mse}

    if DEBUG:
        two_plot(fb_data_train["ds"], fb_data_train["y"], fb_data_train["yhat"],
                fb_data_test["ds"], fb_data_test["y"], fb_data_test["yhat"])
    
    # User input for number of days to forecast.
    n_days = st.slider('Please enter number of days for prediction:', 0, 365)
    
    if n_days > 0:
        st.subheader("Forecast using FBProphet model!")

        st.subheader("Legends =>")
        st.write("- The Light blue line is the uncertainty level (y_hat_upper & y_hat_lower)")
        st.write("- The dark blue line is the prediction (y_hat)")
        st.write("- The black dots are the original data.")

        future = fb_mdl.make_future_dataframe(periods=n_days)
        future_frcst = fb_mdl.predict(future)
        # fig = fb_mdl.plot(future_frcst, figsize=(25,8))
        fig = plot_plotly(fb_mdl, future_frcst, figsize=(1800,600), ylabel="Closing Price")
        fig.layout.update(title_text = f"Forecast for {ticker}")
        st.plotly_chart(fig)

        st.write("Forecasted data")
        df_to_display = future_frcst.tail(n_days)[["ds", "yhat"]]
        df_to_display.rename(columns = {"ds" : "Date", "yhat" : "Closing Price"}, inplace=True)
        df_to_display['Date'] = df_to_display['Date'].dt.date
        st.write(df_to_display.set_index("Date"))

        st.markdown("<h3 style='text-align: center;'>Thanks for visiting!</h3>", unsafe_allow_html=True)
