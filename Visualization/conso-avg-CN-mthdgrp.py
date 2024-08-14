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
    asr_min_errors = []
    asr_max_errors = []
    test_acc_values = []
    methods = []
    samples = []

    for (method, sample), group in grouped:
        current_asr_values = group['ASR'].to_numpy()
        current_test_acc_values = group['Test Acc'].to_numpy()
        mean_asr = np.mean(current_asr_values)
        asr_min_error = mean_asr - np.min(current_asr_values)
        asr_max_error = np.max(current_asr_values) - mean_asr
        test_acc = np.mean(current_test_acc_values)
        asr_values.append(mean_asr)
        asr_min_errors.append(asr_min_error)
        asr_max_errors.append(asr_max_error)
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
fig, ax = plt.subplots(figsize=(16, 7))
bar_width = 0.2
spacing = 0.3  # Increased spacing between groups
methods = summary_df['Method'].unique()
samples = summary_df['Number of Samples'].unique()
barcolor = [ '#F9D5E5','#7ACCC8','#92A8D1', '#88B04B','#FFAB91']

# Corresponding Darker Dot Colors (Hex Codes):
dotcolor = ['#E75480','#008080','#2A4D69', '#44693D','#FF5722']
ax.tick_params(axis='x', labelsize=20)  # Increase x-axis tick label size
ax.tick_params(axis='y', labelsize=20) 
# Create positions for each sample's bars
positions = np.arange(len(samples)) * (len(methods) * bar_width+spacing)

colorselec=0
# Plotting the ASR as bars and Test Acc as scatter points
for i, sample in enumerate(samples):
    df = summary_df[summary_df['Number of Samples'] == sample]
    r = positions + i * bar_width
    ax.bar(r, df['ASR'], width=bar_width, label=f'ASR-{int(sample)} samples',color=barcolor[colorselec])
    ax.scatter(r, df['Test Acc'], s=150, marker='o', label=f'Test Acc-{int(sample)} samples',color=dotcolor[colorselec])
    colorselec+=1

# Labeling
ax.set_xlabel('Unlearning Method', fontsize=20)

ax.set_ylabel('Average ASR Percentage /\n Test Accuracy Percentage', fontsize=20)
#ax.set_title('10th Percentile ASR and Test Accuracy by Identification Rate and Method', fontsize=20)

# Adding legends
ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.4), ncol=3, fontsize=20)

# Adjusting the x-ticks to show sample numbers
plt.xticks(positions + bar_width * (len(methods) - 1) / 2, methods, rotation=0)

plt.tight_layout()
plt.show()