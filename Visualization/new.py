import pandas as pd
import matplotlib.pyplot as plt

# Load the data from the CSV file
file_path = './Visualization/data/signaltriggerCN.csv'
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
fig, ax = plt.subplots(figsize=(12,9))  # Slightly smaller figure size for A4 side by side
pivot_table_asr_transposed.plot(kind='bar', ax=ax, colormap='Accent',width=0.7)

# Adjust plot parameters
ax.tick_params(axis='x', labelsize=24)  # Adjust x-axis tick font size
ax.tick_params(axis='y', labelsize=24)  # Adjust y-axis tick font size
ax.set_xlabel('Unlearning Method', fontsize=30,labelpad=25)  # Adjust x-axis label font size
ax.set_ylabel('ASR (Percentage)', fontsize=30,labelpad=20)  # Adjust y-axis label font size
ax.legend(title='Number of Samples', loc='upper center', bbox_to_anchor=(0.5, 1.5),borderaxespad=0, ncol=3, fontsize=24, title_fontsize=26)  # Adjust legend font sizes

# Adjust the layout to make space for the legend
plt.subplots_adjust(top=0.65)  # Slightly more space for legend if needed

# Display the plot
ax.grid(True)
plt.tight_layout()
plt.show()
