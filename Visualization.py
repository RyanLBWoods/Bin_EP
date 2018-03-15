import matplotlib.pyplot as plt
import pandas as pd

# Read data
data = pd.read_csv("energydata_complete.csv", header=0)

# Plot graph
plt.figure(figsize=(20, 8))
ax1 = plt.subplot(111)
plt.plot(range(len(data['Appliances'])), data['Appliances'], 'b', label='Appliances', linewidth=0.5)
ax1.grid(color='lightgray', linestyle='-', linewidth='0.5')
ax1.set_title('Appliances')
plt.tight_layout()
plt.show()
