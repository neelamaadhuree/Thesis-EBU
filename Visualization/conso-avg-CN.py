import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Function to calculate the average ASR and corresponding Test Accuracy for each method
def get_avg_asr_and_test_acc(df):
    methods = df['Method'].unique()
    avg_asr_values = []
    avg_test_acc_values = []
    
    for method in methods:
        method_df = df[df['Method'] == method]
        avg_asr = method_df['ASR'].mean()
        avg_test_acc = method_df['Test Acc'].mean()
        avg_asr_values.append(avg_asr)
        avg_test_acc_values.append(avg_test_acc)
    
    return avg_asr_values, avg_test_acc_values

# Load the datasets
checkertrigger_df = pd.read_csv('./Visualization/data/checkertriggerCN.csv')
freqtrigger_df = pd.read_csv('./Visualization/data/freqtriggerCN.csv')
patchtrigger_df = pd.read_csv('./Visualization/data/patchtriggerCN.csv')
signaltrigger_df = pd.read_csv('./Visualization/data/signaltriggerCN.csv')


checkertrigger_filtered = checkertrigger_df[(checkertrigger_df['Method'] != 'No UL') & 
                                                  (checkertrigger_df['Poison Ratio'] == 1)]

freqtrigger_filtered = freqtrigger_df[(freqtrigger_df['Method'] != 'No UL') & 
                                            (freqtrigger_df['Poison Ratio'] == 1)]

patchtrigger_filtered = patchtrigger_df[(patchtrigger_df['Method'] != 'No UL') & 
                                              (patchtrigger_df['Poison Ratio'] == 1)]

signaltrigger_filtered = signaltrigger_df[(signaltrigger_df['Method'] != 'No UL') & 
                                                (signaltrigger_df['Poison Ratio'] == 1)]

# Get average ASR and Test Acc for each attack type
checkertrigger_avg_asr_values, checkertrigger_avg_test_acc_values = get_avg_asr_and_test_acc(checkertrigger_filtered)
freqtrigger_avg_asr_values, freqtrigger_avg_test_acc_values = get_avg_asr_and_test_acc(freqtrigger_filtered)
patchtrigger_avg_asr_values, patchtrigger_avg_test_acc_values = get_avg_asr_and_test_acc(patchtrigger_filtered)
signaltrigger_avg_asr_values, signaltrigger_avg_test_acc_values = get_avg_asr_and_test_acc(signaltrigger_filtered)

# Prepare the combined data for plotting
avg_asr_values = [checkertrigger_avg_asr_values, freqtrigger_avg_asr_values, patchtrigger_avg_asr_values, signaltrigger_avg_asr_values]
avg_test_acc_values = [checkertrigger_avg_test_acc_values, freqtrigger_avg_test_acc_values, patchtrigger_avg_test_acc_values, signaltrigger_avg_test_acc_values]

# Define methods and filter out 'No UL'
methods = checkertrigger_filtered['Method'].unique()

# Plotting the average ASR as bars and the corresponding average test accuracy as scatter points
fig, ax1 = plt.subplots(figsize=(14, 7))

# Define the bar width
bar_width = 0.2

# Define positions for each method's bars
r1 = range(len(methods))
r2 = [x + bar_width for x in r1]
r3 = [x + 2*bar_width for x in r1]
r4 = [x + 3*bar_width for x in r1]

# Plotting the ASR as bars
ax1.bar(r1, avg_asr_values[0], width=bar_width, color='skyblue', label='CheckerTrigger ASR')
ax1.bar(r2, avg_asr_values[1], width=bar_width, color='plum', label='FreqTrigger ASR')
ax1.bar(r3, avg_asr_values[2], width=bar_width, color='lightgreen', label='PatchTrigger ASR')
ax1.bar(r4, avg_asr_values[3], width=bar_width, color='salmon', label='SignalTrigger ASR')

# Plotting the corresponding test accuracy as scatter points
ax1.scatter(r1, avg_test_acc_values[0], color='blue', label='CheckerTrigger Test Acc', s=150, marker='o')
ax1.scatter(r2, avg_test_acc_values[1], color='purple', label='FreqTrigger Test Acc', s=150, marker='o')
ax1.scatter(r3, avg_test_acc_values[2], color='green', label='PatchTrigger Test Acc', s=150, marker='o')
ax1.scatter(r4, avg_test_acc_values[3], color='red', label='SignalTrigger Test Acc', s=150, marker='o')

# Labeling
ax1.set_xlabel('Unlearning Method')
ax1.set_ylabel('Average ASR Percentage / Corresponding Test Accuracy Percentage')
#ax1.set_title('Average ASR and Corresponding Test Accuracy by Method for 10% Poison Ratio')
ax1.tick_params(axis='y')

# Adding legends at the top
ax1.legend(loc='upper center', bbox_to_anchor=(0.5, 1.15), ncol=4)

# Adjusting the x-ticks to show method names
plt.xticks([r + 1.5*bar_width for r in range(len(methods))], methods, rotation=0)

plt.tight_layout()
plt.show()