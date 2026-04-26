import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.datasets import get_rdataset
from sklearn.metrics import root_mean_squared_error
import numpy as np

# load data
data = get_rdataset('AirPassengers', 'datasets').data['value']
data.index = pd.date_range('1949-01', periods=len(data), freq='M')

# split data
train = data[:-12]
test = data[-12:]

# train model
model = ARIMA(train, order=(1,1,1)).fit()

# predict last 12 (for RMSE)
pred = model.predict(start=len(train), end=len(train)+11)

# forecast future (for graph)
future = model.predict(start=len(data), end=len(data)+11)

# plot
plt.plot(data, label='Actual')
plt.plot(future, label='Predicted')
plt.legend()
plt.show()

# RMSE
rmse = root_mean_squared_error(test, pred)
print("RMSE =", rmse)
