import pandas as pd
import matplotlib.pyplot as plt

# Load the dataset
file_path = './Visualization/data/checkertriggerBD.csv'  # Replace with your actual file path
data = pd.read_csv(file_path)

# Filter data for Poison Ratio = 10 (assuming this means the first rate in the dataset)
filtered_data = data[data['Poison Ratio'] == 1]

# Filter out the "No UL" method
filtered_data = filtered_data[filtered_data['Method'] != 'No UL']

# Define a color map to ensure the same color for the same method
colors = plt.cm.get_cmap('tab10', len(filtered_data['Method'].unique()))

# Plot the line plot for Identification Rate vs Test Acc and ASR, with dashed lines for ASR
plt.figure(figsize=(12, 8))

for i, method in enumerate(filtered_data['Method'].unique()):
    method_data = filtered_data[filtered_data['Method'] == method]
    plt.plot(method_data['Identification Rate'], method_data['Test Acc'], 
             label=f'{method} - Test Acc', marker='o', color=colors(i))
    plt.plot(method_data['Identification Rate'], method_data['ASR'], 
             label=f'{method} - ASR', linestyle='--', marker='x', color=colors(i))

#plt.title('Identification Rate vs Test Acc and ASR for Poison Ratio 10')
plt.xlabel('Identification Rate (%)')
plt.ylabel('Percentage (%)')
plt.grid(True)
plt.legend()
plt.show()