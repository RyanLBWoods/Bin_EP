import Functions
import matplotlib.pyplot as plt
import pandas as pd
import datetime
from sklearn.linear_model import Ridge, Lasso
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
x_train, x_test, y_train, y_test = train_test_split(data.iloc[:, 1:], data['Appliances'], random_state=1)

# Delete least correlation data
del x_train['Visibility']
del x_test['Visibility']
del x_train['RH_5']
del x_test['RH_5']
del x_train['T9']
del x_test['T9']

# Run Ridge regression
lr = Ridge(random_state=1)
lr.fit(x_train, y_train)

# Evaluate performance
y_train_predict = lr.predict(x_train)
t_mae = Functions.mae(y_train, y_train_predict)
t_rmse = Functions.rmse(y_train, y_train_predict)

y_predict = lr.predict(x_test)
mae = Functions.mae(y_test, y_predict)
rmse = Functions.rmse(y_test, y_predict)
print('Train:')
print('MAE of Ridge: ', t_mae)
print('RMSE of Ridge: ', t_rmse)
print('Test:')
print('Mae of Ridge: ', mae)
print('RMSE of Ridge: ', rmse)

# Plot graph of test output and predict output
plt.figure(figsize=(20, 8))
ax1 = plt.subplot(211)
plt.plot(range(len(y_predict)), y_test, 'r', label='test', linewidth=1)
plt.plot(range(len(y_predict)), y_predict, 'g', label='predict', linewidth=0.5)
ax1.set_title('ROC of Ridge')
plt.ylabel('Appliances')
plt.legend(loc='upper right')
plt.tight_layout()

# Run Lasso regression
ls = Lasso(random_state=1)
ls.fit(x_train, y_train)

# Evaluate performance
l_train_predict = ls.predict(x_train)
ls_t_mae = Functions.mae(y_train, l_train_predict)
ls_t_rmse = Functions.rmse(y_train, l_train_predict)

ls_y_predict = ls.predict(x_test)
mae = Functions.mae(y_test, ls_y_predict)
rmse = Functions.rmse(y_test, ls_y_predict)
print('Train:')
print('MAE of Lasso: ', ls_t_mae)
print('RMSE of Lasso: ', ls_t_rmse)
print('Test:')
print('Mae of Lasso: ', mae)
print('RMSE of Lasso: ', rmse)

# Plot graph of test output and predict output
ax2 = plt.subplot(212)
plt.plot(range(len(y_predict)), y_test, 'r', label='test', linewidth=1)
plt.plot(range(len(y_predict)), y_predict, 'g', label='predict', linewidth=0.5)
ax2.set_title('ROC of Lasso')
plt.ylabel('Appliances')
plt.legend(loc='upper right')
plt.tight_layout()

plt.show()
