import pandas as pd
import matplotlib.pyplot as plt

# Load the dataset
file_path_cn = './Visualization/data/checkertriggerCN.csv'  # Replace with your actual file path
data_cn = pd.read_csv(file_path_cn)

# Filter data for Poison Ratio = 10 and remove "No UL" method
filtered_data_cn = data_cn[(data_cn['Poison Ratio'] == 1) & (data_cn['Method'] != 'No UL')]

# Define a color map to ensure the same color for the same method
colors_cn = plt.cm.get_cmap('tab10', len(filtered_data_cn['Method'].unique()))

# Plot the line plot for Number of Samples vs Test Acc and ASR, with dashed lines for ASR
plt.figure(figsize=(12, 8))

for i, method in enumerate(filtered_data_cn['Method'].unique()):
    method_data = filtered_data_cn[filtered_data_cn['Method'] == method]
    plt.plot(method_data['Number of Samples'], method_data['Test Acc'], 
             label=f'{method} - Test Acc', marker='o', color=colors_cn(i))
    plt.plot(method_data['Number of Samples'], method_data['ASR'], 
             label=f'{method} - ASR', linestyle='--', marker='x', color=colors_cn(i))

#plt.title('Number of Samples vs Test Acc and ASR for Poison Ratio 10')
plt.xlabel('Number of Samples')
plt.ylabel('Percentage (%)')
plt.grid(True)
plt.legend()
plt.show()