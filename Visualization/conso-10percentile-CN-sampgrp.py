
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Function to filter the data
def filter_df(df):
    return df[(df['Method'] != 'No UL') & (df['Poison Ratio'] == 1)]

# Load and filter the datasets, and add an 'Attack' column
checkertrigger_df = filter_df(pd.read_csv('./Visualization/data/checkertriggerCN.csv')).assign(Attack='CheckerTrigger')
freqtrigger_df = filter_df(pd.read_csv('./Visualization/data/freqtriggerCN.csv')).assign(Attack='FreqTrigger')
patchtrigger_df = filter_df(pd.read_csv('./Visualization/data/patchtriggerCN.csv')).assign(Attack='PatchTrigger')
signaltrigger_df = filter_df(pd.read_csv('./Visualization/data/signaltriggerCN.csv')).assign(Attack='SignalTrigger')

# Combine all the filtered data into one DataFrame
combined_df = pd.concat([checkertrigger_df, freqtrigger_df, patchtrigger_df, signaltrigger_df])

# Function to calculate 10th percentile ASR and corresponding test accuracy for each method and sample
def get_combined_asr_and_test_acc(df):
    grouped = df.groupby(['Method', 'Number of Samples'])
    asr_values = []
    test_acc_values = []
    methods = []
    samples = []

    for (method, sample), group in grouped:
        current_asr_values  = group['ASR'].to_numpy();
        current_test_acc_values  = group['Test Acc'].to_numpy();
        asr_10th = np.percentile(current_asr_values, 10)
        index_in_array = (np.abs(current_asr_values - asr_10th)).argmin()
        test_acc = current_test_acc_values[index_in_array]
        asr_values.append(asr_10th)
        test_acc_values.append(test_acc)
        methods.append(method)
        samples.append(sample)

    return pd.DataFrame({
        'Method': methods,
        'Number of Samples': samples,
        'ASR': asr_values,
        'Test Acc': test_acc_values
    })

# Get combined ASR and test accuracy
summary_df = get_combined_asr_and_test_acc(combined_df)

# Plotting with increased spacing between samples
fig, ax = plt.subplots(figsize=(14, 8))
bar_width = 0.2
spacing = 0.4  # Increased spacing between groups
methods = summary_df['Method'].unique()
samples = summary_df['Number of Samples'].unique()

# Create positions for each sample's bars
positions = np.arange(len(samples)) * (len(methods) * bar_width + spacing)

# Plotting the ASR as bars and Test Acc as scatter points
for i, method in enumerate(methods):
    df = summary_df[summary_df['Method'] == method]
    r = positions + i * bar_width
    ax.bar(r, df['ASR'], width=bar_width, label=f'{method} ASR')
    ax.scatter(r, df['Test Acc'], s=150, marker='o', label=f'{method} Test Acc')

# Labeling
ax.set_xlabel('Number of Samples', fontsize=18)
ax.set_ylabel('10th Percentile ASR Percentage / Test Accuracy Percentage', fontsize=18)
ax.set_title('10th Percentile ASR and Test Accuracy by Method and Number of Samples', fontsize=20)

# Adding legends
ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.15), ncol=2, fontsize=14)

# Adjusting the x-ticks to show sample numbers
plt.xticks(positions + bar_width * (len(methods) - 1) / 2, samples, rotation=0)

plt.tight_layout()
plt.show()