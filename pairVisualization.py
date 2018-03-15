import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import datetime

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
# Separate columns into groups
x1 = data.iloc[:, 1:6]
x2 = data.iloc[:, 6:11]
x3 = data.iloc[:, 11:16]
x4 = data.iloc[:, 16:21]
x5 = data.iloc[:, 21:26]
x6 = data.iloc[:, 26:28]

# Plot scatter point graph with regression line
sns.pairplot(data, x_vars=x1.columns, y_vars='Appliances', size=5, aspect=1, kind='reg',
             plot_kws=dict(line_kws=dict(color='r')))
sns.pairplot(data, x_vars=x2.columns, y_vars='Appliances', size=5, aspect=1, kind='reg',
             plot_kws=dict(line_kws=dict(color='r')))
sns.pairplot(data, x_vars=x3.columns, y_vars='Appliances', size=5, aspect=1, kind='reg',
             plot_kws=dict(line_kws=dict(color='r')))
sns.pairplot(data, x_vars=x4.columns, y_vars='Appliances', size=5, aspect=1, kind='reg',
             plot_kws=dict(line_kws=dict(color='r')))
sns.pairplot(data, x_vars=x5.columns, y_vars='Appliances', size=5, aspect=1, kind='reg',
             plot_kws=dict(line_kws=dict(color='r')))
sns.pairplot(data, x_vars=x6.columns, y_vars='Appliances', size=5, aspect=1, kind='reg',
             plot_kws=dict(line_kws=dict(color='r')))
plt.show()
