
#%%
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA

#%%
google = yf.Ticker("GOOG")

df = google.history(period="1d", interval="1m")

df = df[["Low"]]

df["date"] = pd.to_datetime(df.index).time
df.set_index("date", inplace=True)

# Training the ARMA model
X = df.index.values
y = df["Low"].values

# The split point is the 10% of the dataframe length
offset = int(0.10 * len(df))

X_train = X[:-offset]
y_train = y[:-offset]
X_test = X[-offset:]
y_test = y[-offset:]

# plt.plot(range(0, len(y_train)), y_train, label='Train')
# plt.plot(range(len(y_train), len(y)), y_test, label='Test')
# plt.legend()
# plt.show()
#%%
model = ARIMA(y_train, order=(5, 0, 1)).fit()
forecast = model.forecast(steps=1)[0]

print(f'Real data for time 0: {y_train[len(y_train)-1]}')
print(f'Real data for time 1: {y_test[0]}')
print(f'Predicted data for time 1: {forecast}')

#%%
import pyopt
import robin_stocks as robinhood
RH_USER_EMAIL = "ratatouille.apple@gmail.com"

# %%
