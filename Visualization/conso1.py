import pandas as pd
import matplotlib.pyplot as plt

# Load the data from the CSV file
file_path = './Visualization/data/checkertriggerCN.csv'
data = pd.read_csv(file_path)

# Convert relevant columns to numeric where possible
data['Number of Samples'] = pd.to_numeric(data['Number of Samples'], errors='coerce')
data['ASR'] = pd.to_numeric(data['ASR'], errors='coerce') * 100
data['Poison Ratio'] = pd.to_numeric(data['Poison Ratio'], errors='coerce')

# Calculate the ASR Percentage
data['ASR_Percentage'] = data['ASR'] / 100

# Filter data for Poison Ratio 10
poison_ratio_10 = data[data['Poison Ratio'] == 1]

# Generate a pivot table with ASR percentage aggregated by method and number of samples for Poison Ratio 10
pivot_table_asr = poison_ratio_10.pivot_table(values='ASR_Percentage', index='Number of Samples', columns='Method', aggfunc='mean')

# Transpose the pivot table for plotting
pivot_table_asr_transposed = pivot_table_asr.transpose()

# Generate a bar plot with the 'Accent' color palette
fig, ax = plt.subplots(figsize=(12, 6))  # Slightly smaller figure size for A4 side by side
pivot_table_asr_transposed.plot(kind='bar', ax=ax, colormap='Accent')

# Adjust plot parameters
ax.tick_params(axis='x', labelsize=26)  # Increased from 12 to 14 for better legibility
ax.tick_params(axis='y', labelsize=26)
ax.set_xlabel('Unlearning Method', fontsize=26)  # Minor increase for clarity
ax.set_ylabel('ASR (Percentage)', fontsize=26)
ax.legend(title='Number of Samples', loc='upper center', bbox_to_anchor=(0.5, 1.3), ncol=4, fontsize=27, title_fontsize=27)

# Adjust the layout to make space for the legend
plt.subplots_adjust(top=0.8)  # Slightly more space for legend if needed

# Display the plot
ax.grid(True)
plt.tight_layout()
plt.show()
