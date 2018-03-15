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

# Run SVM regression
svmr = svm.SVR(kernel='rbf')
# Cross validation
mses = - cross_validation.cross_val_score(svmr, data.iloc[:, 1:], data['Appliances'],
                                          cv=10, scoring='neg_mean_squared_error')
mae = - cross_validation.cross_val_score(svmr, data.iloc[:, 1:], data['Appliances'],
                                          cv=10, scoring='neg_mean_absolute_error')

# Calculate mean
rmses = []
for mse in mses:
    mse = np.sqrt(mse)
    rmses.append(mse)
print("Mean result after 10-fold validation")
print("MAE: ", mae.mean())
print("RMSE: ", Functions.mean(rmses))
