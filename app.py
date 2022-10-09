# Get the ticker using text input or a dropdown.
# checkbox to see the head and tail of the dataset.
# Last 7 days trend.
# Last 30 days trend of the given ticker.
# Linear regression plot (train, test, forecast for next 7 days)
# LSTM plot (train, test, forecast for next 7 days)
# fbprophet forecast for next 7 days.

from turtle import width
from matplotlib.style import use
from prettytable import PrettyTable
from multipledispatch import dispatch
from datetime import date, timedelta
import pandas_datareader.data as web
from pandas_datareader._utils import RemoteDataError

from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.linear_model import SGDRegressor
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler

from keras.layers import Dense, Dropout, LSTM
from keras.callbacks import EarlyStopping
from keras.models import Sequential
from keras.models import load_model
import matplotlib.pyplot as plt
from plotly import graph_objs as go
import streamlit as st

from prophet import Prophet
from prophet.plot import plot_plotly, plot_components_plotly

import numpy as np
import pandas as pd
import pickle
import os
import app

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

@dispatch(object, object, object)
def plot(data1, d1_label, title=""):
    fig = plt.figure(figsize=(25, 8))
    plt.title(title)
    plt.plot(data1, 'b', label=d1_label)
    plt.legend()
    st.pyplot(fig)

@dispatch(object, object, object, object, object)
def plot(data1, d1_label, data2, d2_label, title=""):
    fig = plt.figure(figsize=(25, 3))
    plt.title(title)
    plt.plot(data1, 'b', label=d1_label)
    plt.plot(data2, 'g', label=d2_label)
    plt.legend()
    st.pyplot(fig)

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

class Model:
    def train_linear(x_train, y_train):
        # hyper-paramater tuning
        mdl = SGDRegressor(loss="squared_error", penalty="l2")
        values = [10**i for i in range(-10, 6)]
        hyper_parameter = {"alpha": values}
        gscv = GridSearchCV(mdl, hyper_parameter, scoring="neg_mean_squared_error",
                            cv=10, verbose=1, n_jobs=-1)
        gscv.fit(x_train, y_train)
        alpha = gscv.best_params_["alpha"]

        # applying linear regression with optimal hyper-parameter
        mdl = SGDRegressor(loss="squared_error", penalty="l2", alpha=alpha)
        mdl.fit(x_train, y_train)
        return mdl

    def train_randomForest(x_train, y_train):
        # hyper-paramater tuning
        mdl = RandomForestRegressor(n_jobs=-1)
        hyper_parameter = {"n_estimators": [10, 50, 100, 500],
                        "max_depth": [1, 5, 10, 50, 100, 500, 1000]}
        gscv = GridSearchCV(mdl, hyper_parameter, scoring="neg_mean_squared_error",
                            cv=3, verbose=1, n_jobs=-1)
        gscv.fit(x_train, y_train)
        estimators = gscv.best_params_["n_estimators"]
        max_depth = gscv.best_params_["max_depth"]

        # applying random forest with optimal hyper-parameter
        mdl = RandomForestRegressor(n_estimators=estimators, max_depth=max_depth, verbose=1, n_jobs=-1)
        mdl.fit(x_train, y_train)
        return mdl

    def train_xgboost(x_train, y_train):
        # hyper-parameter tuning
        hyper_parameter = {"max_depth": [1, 2, 3, 4],
                        "n_estimators": [10, 50, 100, 500]}
        mdl = xgb.XGBRegressor()
        best_parameter = GridSearchCV(mdl, hyper_parameter,
                        scoring="neg_mean_squared_error", cv=3)
        best_parameter.fit(x_train, y_train)
        estimators = best_parameter.best_params_["n_estimators"]
        depth = best_parameter.best_params_["max_depth"]

        # applying xgboost regressor with best hyper-parameter
        mdl = xgb.XGBRegressor(max_depth=depth, n_estimators=estimators)
        mdl.fit(x_train, y_train)
        return mdl

    def train_lstm(x_train, y_train):
        model = Sequential()
        
        model.add(LSTM(units = 128, activation = "relu", return_sequences = True,
                    input_shape = (x_train.shape[1], x_train.shape[2])))
        model.add(Dropout(0.2))

        model.add(LSTM(units = 64, activation = "relu", return_sequences = True))
        model.add(Dropout(0.3))

        model.add(LSTM(units = 32, activation = "relu"))
        model.add(Dropout(0.4))

        model.add(Dense(units = 1, activation='linear'))

        print(model.summary())

        es = EarlyStopping(monitor='loss', mode='min', patience=5, verbose=1)

        model.compile(optimizer="adam", loss="mean_squared_error")
        model.fit(x_train, y_train, epochs = 50, callbacks=[es])
        return model

def predict(mdl, data_x, data_y):
    pred = mdl.predict(data_x)
    mape = mean_absolute_error(data_y, pred) / (sum(data_y)/len(data_y))
    # mape = mean_absolute_percentage_error(data_y, pred)
    mse = mean_squared_error(data_y, pred)
    return pred, mape, mse

def forecast(modl, test_x, test_y, regressor=""):
    fcast = []
    x_data = test_x[0]

    for i in range(len(test_y)):
        pred = modl.predict(x_data.reshape(1,-1))
        fcast.append(pred)
        x_data = x_data[1:]
        x_data = np.append(x_data, pred)

    plot(fcast, "forecast", test_y, "test_y", "Forecast using "+regressor)
    
def forecast_lstm(modl, test_x, test_y, regressor=""):
    fcast = []
    x_data = test_x[0]
    pred = modl.predict(x_data.reshape(1,-1,1))
    for i in range(len(test_y)):
        pred = modl.predict(x_data.reshape(1,-1,1))
        pred = pred[0][0]
        fcast.append(pred)
        x_data = x_data[1:]
        x_data = np.append(x_data, pred)

    plot(fcast, "forecast", test_y, "test_y", "Forecast using "+regressor)


# ==============================================
use_lr          = True
use_rf          = True
use_xgb         = True
use_lstm        = True
use_fbprophet   = True
# ==============================================

st.header('Stock price trend detection')
# ticker = st.text_input('Enter stock ticker', 'SBIN')
tickers = ["HINDUNILVR", "SBIN", "RELIANCE", "HDFCBANK", "TCS",
           "AXISBANK", "ITC", "BAJFINANCE", "TECHM", "ASIANPAINT"]
ticker = st.selectbox("Select ticker for prediction.", tickers)

# =============
data_load_state = st.text('Loading last 5 years of data...')
data = get_data_from_web(ticker)
data_load_state.text(f"Successfully fetched {ticker} data!")
# =============

st.subheader("Raw data")
st.write(data.tail())

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
            title="ylabel"
        ),
        xaxis=dict(
            title="xlabel",
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
    fig.layout.update(title_text = f"Raw data for {ticker}", xaxis_rangeslider_visible=True,
                        autosize=False, width=1800, height=600, template="plotly_dark")
    st.plotly_chart(fig)

plot_raw_data(data)

# st.write(f"Raw data for {ticker} retrieved from yfinance!")
# st.write("Size of data: #rows = ", data.shape[0], "#columns = ",data.shape[1])

if DEBUG:
    st.write("First 5 rows in the raw dataset...")
    st.write(data.head())
    st.write("Last 5 rows in the raw dataset...")
    st.write(data.tail())

win_size = 5 # Based on experiment on different window size, 5 seems to be the best.
data_train, data_test = app.train_test_split(data, split_sz= 0.8)
train_x, train_y, train_t = app.prepare_train_set(data_train, win_size)
past_n_days_data = data_train.tail(win_size)
test_x, test_y, test_t = app.prepare_test_set(past_n_days_data, data_test, win_size)

if DEBUG:
    st.write("Size of raw data : ", data.shape)
    st.write("Size of train data : ", data_train.shape)
    st.write("Size of test data : ", data_test.shape)

# plot(data_train, "Close price", "Training data...")

if use_lr:
    st.subheader('Linear Regression')
    print("Predicting using linear regression model...")
    lr_mdl_file_name = f"lr_mdl_{ticker}.pkl"
    if not os.path.isfile(lr_mdl_file_name):
        lr_prediction_state = st.text('Predicting using Linear regression...')
        lr_mdl = Model.train_linear(train_x, train_y)
        pickle.dump(lr_mdl, open(lr_mdl_file_name, "wb"))
        lr_prediction_state.text('Done prediction using Linear regression!')

    # Load the saved model
    lr_mdl = pickle.load(open(lr_mdl_file_name, "rb"))
    # Prediction on train data
    train_pred, train_mape, train_mse = predict(lr_mdl, train_x, train_y)
    # Prediction on train data
    test_pred, test_mape, test_mse = predict(lr_mdl, test_x, test_y)
    # Save the result for comparison
    lr_result = { "Model" : "Linear Regression", 
                "train_pred" : train_pred, "train_mape" : train_mape, "train_mse" : train_mse,
                "test_pred" : test_pred, "test_mape" : test_mape, "test_mse" : test_mse}

    two_plot(train_t, train_y, train_pred, test_t, test_y, test_pred)
    forecast(lr_mdl, test_x, test_y, "Linear regression")

####################################################################################
if use_rf:
    st.subheader('Randomforest Regression')
    print("Predicting using randomforest regression model...")
    rf_mdl_file_name = f"rf_mdl_{ticker}.pkl"
    if not os.path.isfile(rf_mdl_file_name):
        rf_prediction_state = st.text('Predicting using randomforest regression...')
        rf_mdl = Model.train_randomForest(train_x, train_y)
        pickle.dump(rf_mdl, open(rf_mdl_file_name, "wb"))
        rf_prediction_state.text('Done prediction using randomforest regression!')

    # Load the saved model
    rf_mdl = pickle.load(open(rf_mdl_file_name, "rb"))
    # Prediction on train data
    train_pred, train_mape, train_mse = predict(rf_mdl, train_x, train_y)
    # Prediction on train data
    test_pred, test_mape, test_mse = predict(rf_mdl, test_x, test_y)
    # Save results for comparison
    rf_result = {"Model" : "RandomForest Regression",
                "train_pred" : train_pred, "train_mape" : train_mape, "train_mse" : train_mse,
                "test_pred" : test_pred, "test_mape" : test_mape, "test_mse" : test_mse}
    two_plot(train_t, train_y, train_pred, test_t, test_y, test_pred)
    forecast(rf_mdl, test_x, test_y, "Randomforest regression")

####################################################################################
if use_xgb:
    st.subheader('XGBoost Regression')
    print("Predicting using xgboost regression model...")
    xgb_mdl_file_name = f"xgb_mdl_{ticker}.pkl"
    if not os.path.isfile(xgb_mdl_file_name):
        xg_prediction_state = st.text('Predicting using xgboost regression...')
        xgb_mdl = Model.train_xgboost(train_x, train_y)
        pickle.dump(xgb_mdl, open(xgb_mdl_file_name, "wb"))
        xg_prediction_state.text('Done prediction using xgboost regression!')

    # Load the saved model
    xgb_mdl = pickle.load(open(xgb_mdl_file_name, "rb"))
    # Prediction on train data
    train_pred, train_mape, train_mse = predict(xgb_mdl, train_x, train_y)
    # Prediction on train data
    test_pred, test_mape, test_mse = predict(xgb_mdl, test_x, test_y)
    # Save results for comparison
    xgb_result = { "Model" : "XGBoost Regression",
                "train_pred" : train_pred, "train_mape" : train_mape, "train_mse" : train_mse,
                "test_pred" : test_pred, "test_mape" : test_mape, "test_mse" : test_mse}
    two_plot(train_t, train_y, train_pred, test_t, test_y, test_pred)
    forecast(xgb_mdl, test_x, test_y, "XGBoost regression")

####################################################################################
if use_lstm:
    st.subheader('LSTM')
    print("Predicting using LSTM model...")
    lstm_mdl_file_name = f"lstm_mdl_{ticker}.h5"
    n_features = 1 # To reshape the 2D data to 3D

    if not os.path.isfile(lstm_mdl_file_name):
        lstm_prediction_state = st.text('Predicting using LSTM...')
        # Reshape the 2D data to 3D
        train_x_lstm = np.reshape(train_x, (train_x.shape[0], train_x.shape[1], n_features))
        lstm_mdl = Model.train_lstm(train_x_lstm, train_y)
        lstm_mdl.save(lstm_mdl_file_name)
        del train_x_lstm
        lstm_prediction_state.text('Done prediction using LSTM!')

    # Load the saved model
    lstm_mdl = load_model(lstm_mdl_file_name)
    # Reshape the the train and test data from 2D -> 3D
    train_x_lstm = np.reshape(train_x, (train_x.shape[0], train_x.shape[1], n_features))
    test_x_lstm = np.reshape(test_x, (test_x.shape[0], test_x.shape[1], n_features))
    # Prediction on train data
    train_pred, train_mape, train_mse = predict(lstm_mdl, train_x_lstm, train_y)
    # Prediction on train data
    test_pred, test_mape, test_mse = predict(lstm_mdl, test_x_lstm, test_y)
    # Save results for comparison
    lstm_result = {"Model" : "LSTM",
                "train_pred" : train_pred, "train_mape" : train_mape, "train_mse" : train_mse,
                "test_pred" : test_pred, "test_mape" : test_mape, "test_mse" : test_mse}
    two_plot(train_t, train_y, train_pred, test_t, test_y, test_pred)
    forecast_lstm(lstm_mdl, test_x, test_y, "LSTM")

####################################################################################
if use_fbprophet:
    print("Predicting using fbprophet model...")
    st.subheader('FbProphet')
    
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
    train_mse = mean_squared_error(fb_data_train[["y"]], fb_data_train[["yhat"]])
    test_mape = mean_absolute_percentage_error(fb_data_test[["y"]], fb_data_test[["yhat"]])
    test_mse = mean_squared_error(fb_data_test[["y"]], fb_data_test[["yhat"]])
    # Save results for comparison
    fbprophet_result = {"Model" : "FB Prophet",
                "train_mape" : train_mape, "train_mse" : train_mse,
                "test_mape" : test_mape, "test_mse" : test_mse}

    two_plot(fb_data_train["ds"], fb_data_train["y"], fb_data_train["yhat"],
             fb_data_test["ds"], fb_data_test["y"], fb_data_test["yhat"])

    # future = fb_mdl.make_future_dataframe(periods=0)
    # st.write(fb_data.tail())
    # st.write(future.tail())
    # fb_fcast = fb_mdl.predict(future)
    # st.write(fb_fcast.head())
    # # st.write(fb_fcast.head())
    # ff = plot_plotly(fb_mdl, fb_fcast, figsize=(1800,600), xlabel='Time', ylabel='Stock price')
    # ff.layout.update(title_text = f"Raw data for {ticker}", xaxis_rangeslider_visible=True,
    #             autosize=False, width=1800, height=600,template="plotly_dark")
    # st.plotly_chart(ff)


# =================================================================================
# st.write(lr_result)
# st.write(rf_result)
# st.write(xgb_result)
# st.write(lstm_result)

####################################################################################
st.subheader("Comparison of different models...")
results = {}

rrr = []
if use_lr:
    rrr.append(lr_result)
if use_rf:
    rrr.append(rf_result)
if use_xgb:
    rrr.append(xgb_result)
if use_lstm:
    rrr.append(lstm_result)
if use_fbprophet:
    rrr.append(fbprophet_result)

cols = ["Model", "train_mape", "test_mape", "train_mse", "test_mse"]
for col in cols:
    results[col] = [ rslt[col] for rslt in rrr ]

st.write(results)
df = pd.DataFrame.from_dict(results)
df.set_index("Model")
st.table(df)


# recent_1week = data.tail(7)
# recent_4week = data.tail(30)
# recent_3month = data.tail(3*30)
# recent_6month = data.tail(6*30)

# from statsmodels.tsa.seasonal import seasonal_decompose
# # creating trend object by assuming multiplicative model
# plot.plot(recent_1week["Close"], "akss", "ankniefne")
# output = seasonal_decompose(recent_1week["Close"], model='multiplicative', period=1).trend
# plot.plot(output, "akss", "ankniefne")

# plot.plot(recent_4week["Close"], "akss", "ankniefne")
# output = seasonal_decompose(recent_4week["Close"], model='multiplicative', period=1).trend
# plot.plot(output, "akss", "ankniefne")

# plot.plot(recent_3month["Close"], "akss", "ankniefne")
# output = seasonal_decompose(recent_3month["Close"], period=1).trend
# plot.plot(output, "akss", "ankniefne")

# plot.plot(recent_6month["Close"], "akss", "ankniefne")
# output = seasonal_decompose(recent_6month["Close"], period=1).trend
# plot.plot(output, "akss", "ankniefne")
