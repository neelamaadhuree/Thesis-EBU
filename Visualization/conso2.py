import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the data from the CSV file
file_path = './Visualization/data/checkertriggerBD.csv'
data = pd.read_csv(file_path)

# Convert relevant columns to numeric where possible
data['Number of Samples'] = pd.to_numeric(data['Identification Rate'], errors='coerce')
data['ASR'] = pd.to_numeric(data['ASR'], errors='coerce')
data['Poison Ratio'] = pd.to_numeric(data['Poison Ratio'], errors='coerce')

# Calculate the ASR Percentage
data['ASR_Percentage'] = data['ASR'] / 100

# Filter data for Poison Ratio 10
poison_ratio_10 = data[data['Poison Ratio'] == 10]

# Generate a pivot table for the heatmap
pivot_table = poison_ratio_10.pivot_table(values='ASR_Percentage', index='Number of Samples', columns='Method', aggfunc='mean')

# Plotting the heatmap
plt.figure(figsize=(12, 8))
sns.heatmap(pivot_table, annot=True, cmap="YlGnBu", fmt=".2f")
plt.title('Heatmap of ASR Percentage')
plt.xlabel(' Unlearning Method')
plt.ylabel('Identification Rate')

# Show the plot
plt.tight_layout()
plt.show()