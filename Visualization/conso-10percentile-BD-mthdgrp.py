import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Function to filter the data
def filter_df(df):
    return df[(df['Method'] != 'No UL') & (df['Poison Ratio'] == 1)]

# Load and filter the datasets, and add an 'Attack' column
checkertrigger_df = filter_df(pd.read_csv('./Visualization/data/checkertriggerBD.csv')).assign(Attack='CheckerTrigger')
freqtrigger_df = filter_df(pd.read_csv('./Visualization/data/freqtriggerBD.csv')).assign(Attack='FreqTrigger')
patchtrigger_df = filter_df(pd.read_csv('./Visualization/data/patchtriggerBD.csv')).assign(Attack='PatchTrigger')
signaltrigger_df = filter_df(pd.read_csv('./Visualization/data/signaltriggerBD.csv')).assign(Attack='SignalTrigger')

# Combine all the filtered data into one DataFrame
combined_df = pd.concat([checkertrigger_df, freqtrigger_df, patchtrigger_df, signaltrigger_df])

# Function to calculate 10th percentile ASR and corresponding test accuracy for each method and sample
def get_combined_asr_and_test_acc(df):
    grouped = df.groupby(['Method', 'Identification Rate'])
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
        'Identification Rate': samples,
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
samples = summary_df['Identification Rate'].unique()

ax.tick_params(axis='x', labelsize=20)  # Increase x-axis tick label size
ax.tick_params(axis='y', labelsize=20) 
# Create positions for each sample's bars
positions = np.arange(len(samples)) * (len(methods) * bar_width+spacing)
barcolor=['skyblue','plum','lightgreen','salmon']
dotcolor=['blue','purple','green','red']
colorselec=0
# Plotting the ASR as bars and Test Acc as scatter points
for i, sample in enumerate(samples):
    df = summary_df[summary_df['Identification Rate'] == sample]
    r = positions + i * bar_width
    ax.bar(r, df['ASR'], width=bar_width, label=f'ASR-{int(sample)}% IR',color=barcolor[colorselec])
    ax.scatter(r, df['Test Acc'], s=150, marker='o', label=f'Test Acc-{int(sample)}% IR',color=dotcolor[colorselec])
    colorselec+=1

# Labeling
ax.set_xlabel('Unlearning Method', fontsize=20)

ax.set_ylabel('10th Percentile ASR Percentage /\n Test Accuracy Percentage', fontsize=20)
#ax.set_title('10th Percentile ASR and Test Accuracy by Identification Rate and Method', fontsize=20)

# Adding legends
ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.4), ncol=3, fontsize=20)

# Adjusting the x-ticks to show sample numbers
plt.xticks(positions + bar_width * (len(methods) - 1) / 2, methods, rotation=0)

plt.tight_layout()
plt.show()