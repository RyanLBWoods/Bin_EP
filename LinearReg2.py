import numpy as np
import math
import matplotlib.pyplot as plt
import pandas as pd
import datetime
from sklearn.linear_model import LinearRegression
from sklearn.cross_validation import train_test_split


# Absolute Error
def ae(test, predict):
    # for i in range(0, len(test)):
    #     ab = abs(test[i] - predict[i])
    return abs(test - predict)


# Mean Absolute Error
def mae(test, predict):
    return ae(test, predict).mean()


# Squared Error
def se(test, predict):
    return (test - predict) ** 2


# Mean Squared Error
def mse(test, predict):
    return se(test, predict).mean()


# Root Mean Squared Error
def rmse(test, predict):
    return math.sqrt(mse(test, predict))


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
del data['RH_6']
train_length = int(0.75 * len(data))

x_train = data.iloc[:train_length, 1:27]
y_train = data[:train_length]['Appliances']
x_test = data.iloc[train_length:, 1:27]
y_test = data[train_length:]['Appliances']

lr_without_rh6 = LinearRegression()
lr_without_rh6.fit(x_train, y_train)
lmtrain_without_rh6 = (lr_without_rh6.coef_, lr_without_rh6.intercept_)

y_train_predict = lr_without_rh6.predict(x_train)
t_mae_without_rh6 = mae(y_train, y_train_predict)
t_rmse_without_rh6 = rmse(y_train, y_train_predict)

y_predict = lr_without_rh6.predict(x_test)
mae_without_vsb_rh6 = mae(y_test, y_predict)
rmse_without_vsb_rh6 = rmse(y_test, y_predict)
print('Train:')
print('MAE without rh6: ', t_mae_without_rh6)
print('RMSE without rh6: ', t_rmse_without_rh6)
print('Test:')
print('Mae without rh6: ', mae_without_vsb_rh6)
print('RMSE without rh6: ', rmse_without_vsb_rh6)

del x_train['Visibility']
del x_test['Visibility']

lr = LinearRegression()
lr.fit(x_train, y_train)
lmtrain_all = (lr.coef_, lr.intercept_)

y_train_predict = lr.predict(x_train)
t_mae = mae(y_train, y_train_predict)
t_rmse = rmse(y_train, y_train_predict)

y_predict = lr.predict(x_test)
mae = mae(y_test, y_predict)
rmse = rmse(y_test, y_predict)
print('Train:')
print('MAE without vsb and rh6: ', t_mae)
print('RMSE without vsb and rh6: ', t_rmse)
print('Test:')
print('Mae without vsb and rh6: ', mae)
print('RMSE without vsb and rh6: ', rmse)

plt.figure(figsize=(20, 8))
ax1 = plt.subplot(211)
plt.plot(range(len(y_predict)), y_predict, 'b', label='predict', linewidth=0.8)
plt.plot(range(len(y_predict)), y_test, 'r', label='test', linewidth=0.5)
plt.legend(loc='upper right')
ax1.set_title('ROC(delete RH_6)')
plt.ylabel('Appliances')


# plt.show()
