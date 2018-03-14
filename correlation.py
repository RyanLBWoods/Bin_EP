import matplotlib.pyplot as plt
import pandas as pd
import datetime
from sklearn.linear_model import LinearRegression

# Read data
data = pd.read_csv("energydata_complete.csv", header=0)
# Process date column
time_list = []
for date in data['date']:
    date = datetime.datetime.strptime(date, '%Y-%m-%d %H:%M:%S')
    date = date.hour
    time_list.append(date)

data['time_int'] = time_list
# Delete useless data
del data['date']
del data['rv1']
del data['rv2']

y = data['Appliances']
x_rh4 = data[['RH_4']]
x_t5 = data[['T5']]
x_rh5 = data[['RH_5']]
x_t9 = data[['T9']]
x_press = data[['Press_mm_hg']]
x_vsb = data[['Visibility']]
x_tdew = data[['Tdewpoint']]

# Observe correlations
lr = LinearRegression()
lr.fit(x_rh4, y)
model_rh4 = (lr.coef_, lr.intercept_)
print("rh4: ", model_rh4)

lr.fit(x_t5, y)
model_t5 = (lr.coef_, lr.intercept_)
print("t5: ", model_t5)

lr.fit(x_rh5, y)
model_rh5 = (lr.coef_, lr.intercept_)
print("rh5:", model_rh5)

lr.fit(x_t9, y)
model_t9 = (lr.coef_, lr.intercept_)
print("t9", model_t9)

lr.fit(x_press, y)
model_press = (lr.coef_, lr.intercept_)
print("press:", model_press)

lr.fit(x_vsb, y)
model_vsb = (lr.coef_, lr.intercept_)
print("visibility: ",model_vsb)

lr.fit(x_tdew, y)
model_tdew = (lr.coef_, lr.intercept_)
print("Tdewpoint: ", model_tdew)


# Check difference of T6 and RH6 with T_out and RH_out
t6 = data['T6']
to = data['T_out']

rh6 = data['RH_6']
rho = data['RH_out']

plt.figure(figsize=(20,8))
ax1 = plt.subplot(211)
plt.plot(range(len(t6)), t6, 'b', label='measured', linewidth=0.8)
plt.plot(range(len(t6)), to, 'r', label='station', linewidth=0.8)
plt.legend(loc='upper right')
ax1.set_title('Temperature Outside')
plt.ylabel('Temperature')
ax2 = plt.subplot(212)
plt.plot(range(len(rh6)), rh6, 'b', label='measured', linewidth=0.8)
plt.plot(range(len(rho)), rho, 'r', label='station', linewidth=0.8)
plt.legend(loc='upper right')
ax2.set_title('Humidity Outside')
plt.xlabel('index')
plt.ylabel('Humidity')
plt.show()
