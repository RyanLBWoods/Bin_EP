import matplotlib.pyplot as plt
import pandas as pd
import datetime
from scipy.stats import pearsonr
import collections
from sklearn.linear_model import LinearRegression

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

# Rename columns
data['VSB'] = data['Visibility']
data['Wspd'] = data['Windspeed']
data['Press'] = data['Press_mm_hg']
data['Tp'] = data['Tdewpoint']
del data['Visibility']
del data['Windspeed']
del data['Press_mm_hg']
del data['Tdewpoint']

# Get xy for Pearson correlation
x = data.loc[:, 'lights':]
y = data['Appliances']
correlation = collections.OrderedDict()

for column in x.columns:
    correlation[column] = pearsonr(x[column], y)

pearson_correlation = pd.DataFrame(correlation)
pearson_correlation.index = ['r', 'p_value']
pearson_correlation = pearson_correlation.T
pearson_correlation['label'] = correlation.keys()
print(pearson_correlation)

# Plot graph
plt.figure(figsize=(20, 8))
ax1 = plt.subplot(411)
plt.scatter(pearson_correlation[:]['label'], pearson_correlation[:]['r'], color='blue', marker='v')
ax1.grid(color='lightgray', linestyle='-', linewidth='0.5')
ax1.set_title('r value of Pearson correlation coefficient')
plt.tight_layout()
ax2 = plt.subplot(412)
plt.scatter(pearson_correlation[:]['label'], pearson_correlation[:]['p_value'], color='red', marker='v')
ax2.grid(color='lightgray', linestyle='-', linewidth='0.5')
ax2.set_title('p value of Pearson correlation coefficient')
plt.tight_layout()

# Check difference of T6 and RH6 with T_out and RH_out
t6 = data['T6']
to = data['T_out']

rh6 = data['RH_6']
rho = data['RH_out']

ax3 = plt.subplot(413)
plt.plot(range(len(t6)), t6, 'b', label='measured', linewidth=0.8)
plt.plot(range(len(t6)), to, 'r', label='station', linewidth=0.8)
plt.legend(loc='upper right')
ax3.set_title('Temperature Outside')
plt.ylabel('Temperature')
plt.tight_layout()

ax4 = plt.subplot(414)
plt.plot(range(len(rh6)), rh6, 'b', label='measured', linewidth=0.8)
plt.plot(range(len(rho)), rho, 'r', label='station', linewidth=0.8)
plt.legend(loc='upper right')
ax4.set_title('Humidity Outside')
plt.ylabel('Humidity')
plt.tight_layout()

# Remove label column
del pearson_correlation['label']
pearson_correlation.to_csv('Pearson_correlation.csv')
plt.show()
