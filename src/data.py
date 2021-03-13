import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt

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

plt.plot(range(0, len(y_train)), y_train, label='Train')
plt.plot(range(len(y_train), len(y)), y_test, label='Test')
plt.legend()
plt.show()



