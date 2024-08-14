import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Function to get the test accuracy corresponding to the 10th percentile ASR for each method
def get_asr_and_test_acc(df):
    methods = df['Method'].unique()
    asr_values = []
    test_acc_values = []
    
    for method in methods:
        method_df = df[df['Method'] == method]
        asr_10th = np.percentile(method_df['ASR'], 10)
        test_acc = method_df.loc[method_df['ASR'].sub(asr_10th).abs().idxmin()]['Test Acc']
        asr_values.append(asr_10th)
        test_acc_values.append(test_acc)
    
    return asr_values, test_acc_values

# Load the datasets
checkertrigger_df = pd.read_csv('./Visualization/data/checkertriggerCN.csv')
freqtrigger_df = pd.read_csv('./Visualization/data/freqtriggerCN.csv')
patchtrigger_df = pd.read_csv('./Visualization/data/patchtriggerCN.csv')
signaltrigger_df = pd.read_csv('./Visualization/data/signaltriggerCN.csv')


# Filter the 'checkertrigger_bd_df' DataFrame
checkertrigger_filtered = checkertrigger_df[(checkertrigger_df['Method'] != 'No UL') & 
                                                  (checkertrigger_df['Poison Ratio'] == 1)]

# Filter the 'freqtrigger_bd_df' DataFrame
freqtrigger_filtered = freqtrigger_df[(freqtrigger_df['Method'] != 'No UL') & 
                                            (freqtrigger_df['Poison Ratio'] == 1)]

# Filter the 'patchtrigger_bd_df' DataFrame
patchtrigger_filtered = patchtrigger_df[(patchtrigger_df['Method'] != 'No UL') & 
                                              (patchtrigger_df['Poison Ratio'] == 1)]

# Filter the 'signaltrigger_bd_df' DataFrame
signaltrigger_filtered = signaltrigger_df[(signaltrigger_df['Method'] != 'No UL') & 
                                                (signaltrigger_df['Poison Ratio'] == 1)]


# Get 10th percentile ASR and corresponding Test Acc for each attack type
checkertrigger_asr_values, checkertrigger_test_acc_values = get_asr_and_test_acc(checkertrigger_filtered)
freqtrigger_asr_values, freqtrigger_test_acc_values = get_asr_and_test_acc(freqtrigger_filtered)
patchtrigger_asr_values, patchtrigger_test_acc_values = get_asr_and_test_acc(patchtrigger_filtered)
signaltrigger_asr_values, signaltrigger_test_acc_values = get_asr_and_test_acc(signaltrigger_filtered)

# Prepare the combined data for plotting
asr_values = [checkertrigger_asr_values, freqtrigger_asr_values, patchtrigger_asr_values, signaltrigger_asr_values]
test_acc_values = [checkertrigger_test_acc_values, freqtrigger_test_acc_values, patchtrigger_test_acc_values, signaltrigger_test_acc_values]

# Define methods and filter out 'No UL'
methods = checkertrigger_filtered['Method'].unique()

# Plotting the 10th percentile ASR as bars and the corresponding test accuracy as scatter points
fig, ax1 = plt.subplots(figsize=(14, 7))
ax1.tick_params(axis='x', labelsize=20)  # Increase x-axis tick label size
ax1.tick_params(axis='y', labelsize=20)  # Increase y-axis tick label size

# Define the bar width
bar_width = 0.2

# Define positions for each method's bars
r1 = range(len(methods))
r2 = [x + bar_width for x in r1]
r3 = [x + 2*bar_width for x in r1]
r4 = [x + 3*bar_width for x in r1]

# Plotting the ASR as bars
ax1.bar(r1, asr_values[0], width=bar_width, color='skyblue', label='CheckerTrigger ASR')
ax1.bar(r2, asr_values[1], width=bar_width, color='plum', label='FreqTrigger ASR')
ax1.bar(r3, asr_values[2], width=bar_width, color='lightgreen', label='PatchTrigger ASR')
ax1.bar(r4, asr_values[3], width=bar_width, color='salmon', label='SignalTrigger ASR')

# Plotting the corresponding test accuracy as scatter points
ax1.scatter(r1, test_acc_values[0], color='blue', label='CheckerTrigger Test Acc', s=150, marker='o')
ax1.scatter(r2, test_acc_values[1], color='purple', label='FreqTrigger Test Acc', s=150, marker='o')
ax1.scatter(r3, test_acc_values[2], color='green', label='PatchTrigger Test Acc', s=150, marker='o')
ax1.scatter(r4, test_acc_values[3], color='red', label='SignalTrigger Test Acc', s=150, marker='o')

# Labeling
ax1.set_xlabel('Unlearning Method',fontsize=20)
ax1.set_ylabel('10th Percentile ASR Percentage / \n Corresponding Test Accuracy Percentage',fontsize=20)
#ax1.set_title('10th Percentile ASR and Corresponding Test Accuracy by Method for 10% Poison Ratio')
ax1.tick_params(axis='y')

# Adding legends at the top
ax1.legend(loc='upper center', bbox_to_anchor=(0.5, 1.3), ncol=3, fontsize=20)

# Adjusting the x-ticks to show method names
plt.xticks([r + 1.5*bar_width for r in range(len(methods))], methods, rotation=0)

plt.tight_layout()
plt.show()