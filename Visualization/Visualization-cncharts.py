import pandas as pd
import matplotlib.pyplot as plt

# Load the uploaded CSV file
file_path = './Visualization/data/signaltriggerCN.csv'
data = pd.read_csv(file_path)

# Define a function to generate the plot for a given poison rate and number of samples
def plot_comparison_single_axis(poison_rate, sample_sizes):
    for idx, samples in enumerate(sample_sizes):
        fig, ax = plt.subplots()
        fig.suptitle(f'Poison Rate: {poison_rate}, Clean Samples provided for Unlearning: {samples}', fontsize=25)
        
        # Filter data for the given poison rate and number of samples
        filtered_data = data[(data['Poison Ratio'] == poison_rate) & (data['Number of Samples'] == samples)]
        
        # Include "No Unlearning" data
        no_unlearning_data = data[(data['Method'] == 'No UL') & (data['Poison Ratio'] == poison_rate)].copy()
        no_unlearning_data['Number of Samples'] = samples  # Adjust the sample size to combine correctly
        combined_data = pd.concat([no_unlearning_data, filtered_data])
        
        methods = combined_data['Method'].unique()
        test_acc = combined_data['Test Acc'].to_numpy().astype(float)
        asr = combined_data['ASR'].to_numpy().astype(float)

        x = range(len(methods))
        bars1 = ax.bar(x, test_acc, width=0.4, label='Test Acc', align='center')
        bars2 = ax.bar([p + 0.4 for p in x], asr, width=0.4, label='ASR', align='center')
        ax.set_xlabel('Methods', fontsize=25)
        ax.set_ylabel('Percentage (%)',fontsize=25)
        ax.tick_params(axis='y')
        ax.set_xticks([p + 0.2 for p in x])
        ax.set_xticklabels(methods, rotation=45, fontsize=23)
        ax.legend(fontsize=10)

        # Display current value at the top of each bar
        for bar in bars1:
            yval = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2, yval, f'{yval:.2f}%', va='bottom')

        for bar in bars2:
            yval = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2, yval, f'{yval:.2f}%', va='bottom')

        ax.set_ylim(0, 110)  # Set y-axis limits to show percentage scale from 0 to 100%

        plt.show()

# Define sample sizes and poison rates
sample_sizes = [100, 250, 500, 1000, 3000]
poison_rates = [1, 10]

# Generate plots for each poison rate with y-axis set from 0 to 100
for poison_rate in poison_rates:
    plot_comparison_single_axis(poison_rate, sample_sizes)
