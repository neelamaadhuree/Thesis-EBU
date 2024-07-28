import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load the CSV file
file_path = './data/sample.csv'
data = pd.read_csv(file_path)

def normalize(series):
    return series * 100

data['clean_test_acc'] = normalize(data['clean_test_acc'])
data['test_asr'] = normalize(data['test_asr'])

# Extract unique identification rates
identification_rates = data['identification_rate'].unique()

# Create a plot with the specified styles for lines
plt.figure(figsize=(12, 6))

colors = plt.cm.viridis(np.linspace(0, 1, len(identification_rates)))

for i, rate in enumerate(identification_rates):
    subset = data[data['identification_rate'] == rate]
    plt.plot(subset['epoch'], subset['clean_test_acc'], label=f'Clean Test Acc (Rate {rate})', color=colors[i])
    plt.plot(subset['epoch'], subset['test_asr'], label=f'Test ASR (Rate {rate})', color=colors[i], linestyle='--')

plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.yticks(np.arange(0, 110, 5))  # Set y-axis ticks in terms of 10
plt.title('Test Accuracy and Test AST by Epoch for Different Identification Rates')
plt.legend()
plt.grid(False)
plt.show()