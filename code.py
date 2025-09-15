# ======================================
# STOCK MARKET PRICE PREDICTION PROJECT
# (Without TensorFlow / LSTM)
# ======================================

# STEP 1: Install Required Packages
# Run this only once in terminal:
# pip install yfinance pandas numpy matplotlib seaborn scikit-learn statsmodels

import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from statsmodels.tsa.arima.model import ARIMA

import warnings
warnings.filterwarnings("ignore")

# STEP 2: Load Stock Data
ticker = "AAPL"  # Apple Inc as example
data = yf.download(ticker, start="2020-01-01", end="2023-12-31")

print("📊 Sample Data:")
print(data.head())

# STEP 3: Exploratory Data Analysis (EDA)
plt.figure(figsize=(10,5))
plt.plot(data["Close"], label="Close Price")
plt.title(f"{ticker} Closing Price History")
plt.xlabel("Date")
plt.ylabel("Price (USD)")
plt.legend()
plt.show()

# STEP 4: Feature Engineering
data["Return"] = data["Close"].pct_change()
data["SMA_10"] = data["Close"].rolling(10).mean()
data["SMA_30"] = data["Close"].rolling(30).mean()

plt.figure(figsize=(10,5))
plt.plot(data["Close"], label="Close")
plt.plot(data["SMA_10"], label="SMA 10")
plt.plot(data["SMA_30"], label="SMA 30")
plt.title(f"{ticker} with Moving Averages")
plt.legend()
plt.show()

# Drop NA
data = data.dropna()

# STEP 5: Linear Regression Model
X = data[["Open","High","Low","Volume"]]
y = data["Close"]

X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle=False, test_size=0.2)

lr = LinearRegression()
lr.fit(X_train, y_train)
y_pred_lr = lr.predict(X_test)

print("\n🔹 Linear Regression Performance")
print("MAE:", mean_absolute_error(y_test, y_pred_lr))
print("RMSE:", np.sqrt(mean_squared_error(y_test, y_pred_lr)))
print("R²:", r2_score(y_test, y_pred_lr))

plt.figure(figsize=(10,5))
plt.plot(y_test.values, label="Actual")
plt.plot(y_pred_lr, label="Predicted", color="red")
plt.title("Linear Regression: Actual vs Predicted")
plt.legend()
plt.show()

# STEP 6: ARIMA Model (time series)
arima_data = data["Close"].values
train_size = int(len(arima_data) * 0.8)
train, test = arima_data[:train_size], arima_data[train_size:]

model_arima = ARIMA(train, order=(5,1,0))
fit_arima = model_arima.fit()
pred_arima = fit_arima.forecast(len(test))

print("\n🔹 ARIMA Performance")
print("MAE:", mean_absolute_error(test, pred_arima))
print("RMSE:", np.sqrt(mean_squared_error(test, pred_arima)))

plt.figure(figsize=(10,5))
plt.plot(test, label="Actual")
plt.plot(pred_arima, label="Predicted", color="green")
plt.title("ARIMA: Actual vs Predicted")
plt.legend()
plt.show()
