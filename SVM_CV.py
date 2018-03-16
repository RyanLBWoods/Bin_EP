import Functions
import numpy as np
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
# Delete least correlation data
del data['Visibility']
del data['RH_5']
del data['T9']

# Get training set and testing set
x_train, x_test, y_train, y_test = cross_validation.train_test_split(data.iloc[:, 1:], data['Appliances'],
                                                                     random_state=1)

# Run SVM regression
svmr = svm.SVR(kernel='rbf')
# Cross validation for training set
t_mses = - cross_validation.cross_val_score(svmr, x_train, y_train,
                                            cv=10, scoring='neg_mean_squared_error')
t_mae = - cross_validation.cross_val_score(svmr, x_train, y_train,
                                           cv=10, scoring='neg_mean_absolute_error')

# Calculate mean
t_rmses = []
for mse in t_mses:
    mse = np.sqrt(mse)
    t_rmses.append(mse)

print("SVM radial")
print("Mean result after 10-fold validation for training set")
print("MAE: ", t_mae.mean())
print("RMSE: ", Functions.mean(t_rmses))

# Cross validation for testing set
test_mses = - cross_validation.cross_val_score(svmr, x_test, y_test,
                                               cv=10, scoring='neg_mean_squared_error')
test_mae = - cross_validation.cross_val_score(svmr, x_test, y_test,
                                              cv=10, scoring='neg_mean_absolute_error')

# Calculate mean
test_rmses = []
for mse in test_mses:
    mse = np.sqrt(mse)
    test_rmses.append(mse)
print("Mean result after 10-fold validation for testing set")
print("MAE: ", test_mae.mean())
print("RMSE: ", Functions.mean(test_rmses))
