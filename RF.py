import Functions
import matplotlib.pyplot as plt
import pandas as pd
import datetime
from sklearn.ensemble import RandomForestRegressor
from sklearn import cross_validation

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
x_train, x_test, y_train, y_test = cross_validation.train_test_split(data.iloc[:, 1:], data['Appliances'],
                                                                     random_state=1)

rf = RandomForestRegressor(n_estimators=300, random_state=1)
rf.fit(x_train, y_train)
importances = rf.feature_importances_
importances = pd.DataFrame(importances, index=data.columns[1:], columns=['importance'])
importances.to_csv("feature importance.csv")

# Delete least correlation data
index = list(importances.index)
low_importance = []
for i in index:
    if importances.loc[i]['importance'] < 0.01:
        low_importance.append(i)

for f in low_importance:
    del x_train[f]
    del x_test[f]

rf.fit(x_train, y_train)
# Evaluate performance
y_train_predict = rf.predict(x_train)
t_mae = Functions.mae(y_train, y_train_predict)
t_rmse = Functions.rmse(y_train, y_train_predict)

y_predict = rf.predict(x_test)
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
ax1.set_title('ROC of RF')
plt.ylabel('Appliances')
plt.show()
