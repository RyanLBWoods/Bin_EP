import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline

data = pd.read_csv("energydata_complete.csv", header=0)
print(data.columns)
print(type(data))
del data['date']
del data['rv1']
del data['rv2']
print(data.columns)
# train_length = int(0.7 * len(data))
train_length = 2

x = data.iloc[:, 1:6]
y = data['Appliances']
print(len(x))
print(len(y))

# fig, ax = plt

# ax.scatter(x, y, color='blue', alpha=.8, s=140,marker='o')
# ax.set_Ylabel('Watt-hours')
sns.set(style='ticks')
sns.pairplot(data, x_vars=x.columns, y_vars='Appliances', size=5, aspect=2, kind='reg')
plt.show()
# lr = LinearRegression()
# model = lr.fit(x, y)
# print(model)
# print(lr.intercept_)
# print(lr.coef_)
