import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load the CSV file
file_path = './Visualization/data/ABL-chkr-1%trendline.csv'
data = pd.read_csv(file_path)

def normalize(series):
    return series * 100

data['clean_test_acc'] = normalize(data['clean_test_acc'])
data['test_asr'] = normalize(data['test_asr'])

# Extract unique identification rates
identification_rates = data['identification_rate'].unique()

# Create a plot with the specified styles for lines
plt.figure(figsize=(12, 6))  # Adjusted the height to give more vertical space

colors = plt.cm.viridis(np.linspace(0, 1, len(identification_rates)))

for i, rate in enumerate(identification_rates):
    subset = data[data['identification_rate'] == rate]
    plt.plot(subset['epoch'], subset['clean_test_acc'], label=f'Clean Test Acc (Rate {rate})', color=colors[i])
    plt.plot(subset['epoch'], subset['test_asr'], label=f'Test ASR (Rate {rate})', color=colors[i], linestyle='--')

plt.xlabel('Epoch', fontsize=20)
plt.ylabel('Percentage', fontsize=20)
plt.xticks(np.arange(0, 21, 1), fontsize=20)  # Adjusted for readability
plt.yticks(np.arange(0, 110, 10), fontsize=20)  # Set y-axis ticks in terms of 10
plt.xlim(0, 20)
plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.3), ncol=3, fontsize=20)  # Adjust bbox_to_anchor for correct positioning

plt.grid(True)

# Adjust layout to make space for legend
plt.subplots_adjust(top=0.80)  # Reduce the top padding to make space for the legend

plt.show()
