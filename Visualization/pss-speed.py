import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# Load data
data = pd.read_csv('./Visualization/data/pss.csv')

# Filtering data by identification rate
data_20 = data[data['Identification Rate'] == 20]
data_80 = data[data['Identification Rate'] == 80]

# Sorting data to match order in the plot (20 epochs first, then 100)
data_20 = data_20.sort_values('Epochs')
data_80 = data_80.sort_values('Epochs')

# Create subplots with a specified size
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 7), sharey=True)

# Function to plot data
def plot_data(ax, data, title):
    # Bar locations
    bar_width = 0.35
    index = np.arange(2)
    bar1 = ax.bar(index, data['Test Accuracy'], bar_width, label='Test Acc', color='#00B0F0')
    bar2 = ax.bar(index + bar_width, data['ASR'], bar_width, label='ASR', color='#00B050')
    
    # Adding text labels above bars
    ax.bar_label(bar1, padding=3, fmt='%.2f', fontsize=20)
    ax.bar_label(bar2, padding=3, fmt='%.2f', fontsize=20)
    
  
    ax.set_title(title, fontsize=20)
    ax.set_xticks(index + bar_width / 2)
    ax.set_xticklabels(['20 epochs', '100 epochs'], fontsize=20)
    ax.tick_params(axis='both', which='major', labelsize=20)

# Plot for 20% Identification Rate
plot_data(ax1, data_20, 'PSS - 20% Identification Rate')

# Plot for 80% Identification Rate
plot_data(ax2, data_80, 'PSS - 80% Identification Rate')

# Set y-axis limits
ax1.set_ylim(0, 110)
ax2.set_ylim(0, 110)

# Adding a common legend at the bottom of the plot adjusted closer to the plots
handles, labels = ax1.get_legend_handles_labels()
fig.legend(handles, labels, loc='lower center', fontsize=20, ncol=2, bbox_to_anchor=(0.5, 0.0))

# Adjust layout to fit new font sizes and legend
plt.tight_layout()
plt.subplots_adjust(bottom=0.2)  # Adjust bottom padding to ensure space for the legend
plt.show()
