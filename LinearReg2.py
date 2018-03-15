import Functions
import matplotlib.pyplot as plt
import pandas as pd
import datetime
from sklearn.linear_model import LinearRegression
from sklearn.cross_validation import train_test_split

# Read data
data = pd.read_csv("energydata_complete.csv", header=0)
time_list = []
nsm_list = []
# Process date column
for date in data['date']:
    date = datetime.datetime.strptime(date, '%Y-%m-%d %H:%M:%S')
    nsm = int(date.hour) * 3600 + int(date.minute) * 60 + int(date.second) * 1
    date = date.hour
    time_list.append(date)
    nsm_list.append(nsm)

data['hour'] = time_list
data['NSM'] = nsm_list
# Delete useless data
del data['date']
del data['rv1']
del data['rv2']
del data['RH_6']

# Get training set and testing set
train_length = int(0.75 * len(data))

x_train = data.iloc[:train_length, 1:]
y_train = data[:train_length]['Appliances']
x_test = data.iloc[train_length:, 1:]
y_test = data[train_length:]['Appliances']

# Run linear regression
lr_without_rh6 = LinearRegression()
lr_without_rh6.fit(x_train, y_train)
lmtrain_without_rh6 = (lr_without_rh6.coef_, lr_without_rh6.intercept_)

# Evaluate performance
y_train_predict = lr_without_rh6.predict(x_train)
t_mae_without_rh6 = Functions.mae(y_train, y_train_predict)
t_rmse_without_rh6 = Functions.rmse(y_train, y_train_predict)

y_predict = lr_without_rh6.predict(x_test)
mae_without_vsb_rh6 = Functions.mae(y_test, y_predict)
rmse_without_vsb_rh6 = Functions.rmse(y_test, y_predict)
print('Train:')
print('MAE without rh6: ', t_mae_without_rh6)
print('RMSE without rh6: ', t_rmse_without_rh6)
print('Test:')
print('Mae without rh6: ', mae_without_vsb_rh6)
print('RMSE without rh6: ', rmse_without_vsb_rh6)

# Plot graph of test output and predict output
plt.figure(figsize=(20, 8))
ax1 = plt.subplot(111)
plt.plot(range(len(y_predict)), y_predict, 'b', label='predict without RH6', linewidth=1)
plt.plot(range(len(y_predict)), y_test, 'r', label='test', linewidth=0.5)
ax1.set_title('ROC' '''(without RH6)''')
plt.ylabel('Appliances')

# Delete least correlation data
del x_train['Visibility']
del x_test['Visibility']
del x_train['RH_5']
del x_test['RH_5']
del x_train['T9']
del x_test['T9']

# Run linear regression
lr = LinearRegression()
lr.fit(x_train, y_train)
lmtrain_without_rh6_vsb = (lr.coef_, lr.intercept_)

# Evaluate performance
y_train_predict = lr.predict(x_train)
t_mae = Functions.mae(y_train, y_train_predict)
t_rmse = Functions.rmse(y_train, y_train_predict)

y_predict = lr.predict(x_test)
mae = Functions.mae(y_test, y_predict)
rmse = Functions.rmse(y_test, y_predict)
print('Train:')
print('MAE without vsb & rh6 & T9 & RH5: ', t_mae)
print('RMSE without vsb & rh6 & T9 & RH5: ', t_rmse)
print('Test:')
print('Mae without vsb & rh6 & T9 & RH5: ', mae)
print('RMSE without vsb & rh6 & T9 & RH5: ', rmse)

# Plot graph of test output and predict output
plt.plot(range(len(y_predict)), y_predict, 'g', label='predict without RH6 & Visibility,T9,RH_5', linewidth=1)
plt.legend(loc='upper right')
plt.show()
