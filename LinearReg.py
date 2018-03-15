import Functions
import matplotlib.pyplot as plt
import pandas as pd
import datetime
from sklearn.linear_model import LinearRegression
from sklearn.cross_validation import train_test_split

# Read data
data = pd.read_csv("energydata_complete.csv", header=0)
# Process date column
time_list = []
for date in data['date']:
    date = datetime.datetime.strptime(date, '%Y-%m-%d %H:%M:%S')
    date = date.hour
    time_list.append(date)

data['hour'] = time_list
# Delete useless data
del data['date']
del data['rv1']
del data['rv2']

# Get training set and testing set
train_length = int(0.75 * len(data))

x_train = data.iloc[:train_length, 1:27]
y_train = data[:train_length]['Appliances']
x_test = data.iloc[train_length:, 1:27]
y_test = data[train_length:]['Appliances']

# Run linear regression with all attributes
lr = LinearRegression()
lr.fit(x_train, y_train)
lmtrain_all = (lr.coef_, lr.intercept_)

# Evaluate performance
y_train_predict = lr.predict(x_train)
t_mae = Functions.mae(y_train, y_train_predict)
t_rmse = Functions.rmse(y_train, y_train_predict)

y_predict = lr.predict(x_test)
print(type(y_predict[-1]))
mae = Functions.mae(y_test, y_predict)
rmse = Functions.rmse(y_test, y_predict)
print('Train:')
print('MAE of all: ', t_mae)
print('RMSE of all: ', t_rmse)
print('Test:')
print('Mae of all: ', mae)
print('RMSE of all: ', rmse)

# Plot graph of test output and predict output
plt.figure(figsize=(20, 8))
ax1 = plt.subplot(111)
plt.plot(range(len(y_predict)), y_predict, 'b', label='predict', linewidth=0.8)
plt.plot(range(len(y_predict)), y_test, 'r', label='test', linewidth=0.5)
plt.legend(loc='upper right')
ax1.set_title('ROC(all variables)')
plt.ylabel('Appliances')
plt.show()
