import Functions
import matplotlib.pyplot as plt
import pandas as pd
import datetime
from sklearn import svm
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

# Run SVM regression
svmr = svm.SVR(kernel='rbf')
svmr.fit(x_train, y_train)

# Evaluate performance
y_train_predict = svmr.predict(x_train)
t_mae = Functions.mae(y_train, y_train_predict)
t_rmse = Functions.rmse(y_train, y_train_predict)

y_predict = svmr.predict(x_test)
mae = Functions.mae(y_test, y_predict)
rmse = Functions.rmse(y_test, y_predict)
print('SVM radial')
print('Train:')
print('MAE without rh6: ', t_mae)
print('RMSE without rh6: ', t_rmse)
print('Test:')
print('Mae without rh6: ', mae)
print('RMSE without rh6: ', rmse)

# Plot graph of test output and predict output
plt.figure(figsize=(20, 8))
ax1 = plt.subplot(111)
plt.plot(range(len(y_predict)), y_test, 'r', label='test', linewidth=0.5)
plt.plot(range(len(y_predict)), y_predict, 'b', label='predict', linewidth=1)
ax1.set_title('ROC of SVM radial')
plt.ylabel('Appliances')
plt.legend(loc='upper right')
plt.show()
