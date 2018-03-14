import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import datetime
from sklearn.linear_model import LinearRegression


# Absolute Error
def ae(y_test, y_predict):
    return abs(y_test - y_predict)


# Mean Absolute Error
def mae(y_test, y_predict):
    return ae(y_test, y_predict).mean()


# Squared Error
def se(y_test, y_predict):
    return (y_test - y_predict) ** 2


# Root Mean Squared Error
def rmse(y_test, y_predict):
    return np.sqrt(se(y_test, y_predict).mean())


data = pd.read_csv("energydata_complete.csv", header=0)
time_list = []
for date in data['date']:
    date = datetime.datetime.strptime(date, '%Y-%m-%d %H:%M:%S')
    date = date.hour
    time_list.append(date)

data['time_int'] = time_list
del data['date']
del data['rv1']
del data['rv2']
print(data.columns)
train_length = int(0.75 * len(data))
print(train_length)

x_train_all = data.iloc[:train_length, 1:27]
y_train = data[:train_length]['Appliances']
x_test_all = data.iloc[train_length:, 1:27]
y_test = data[train_length:]['Appliances']

lr = LinearRegression()
lr.fit(x_train_all, y_train)
lmtrain_all = (lr.coef_, lr.intercept_)

y_predict = lr.predict(x_test_all)
mae = mae(y_test, y_predict)
print('MAE of all = ', mae)
rmse = rmse(y_test, y_predict)
print('RMSE of all = ', rmse)

plt.figure(figsize=(20,8))
ax = plt.subplot(111)
plt.plot(range(len(y_predict)), y_predict, 'b', label='predict', linewidth=0.8)
plt.plot(range(len(y_predict)), y_test, 'r', label='test', linewidth=0.5)
plt.legend(loc='upper right')
ax.set_title('ROC(all)')
plt.xlabel('index')
plt.ylabel('Appliances')
plt.show()