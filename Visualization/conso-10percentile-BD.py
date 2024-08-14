import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Function to get the test accuracy corresponding to the 10th percentile ASR for each method
def get_percentile_asr_and_test_acc(df):
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
checkertrigger_bd_df = pd.read_csv('./Visualization/data/checkertriggerBD.csv')
freqtrigger_bd_df = pd.read_csv('./Visualization/data/freqtriggerBD.csv')
patchtrigger_bd_df = pd.read_csv('./Visualization/data/patchtriggerBD.csv')
signaltrigger_bd_df = pd.read_csv('./Visualization/data/signaltriggerBD.csv')

# Filter the 'checkertrigger_bd_df' DataFrame
checkertrigger_bd_filtered = checkertrigger_bd_df[(checkertrigger_bd_df['Method'] != 'No UL') & 
                                                  (checkertrigger_bd_df['Poison Ratio'] == 1)]

# Filter the 'freqtrigger_bd_df' DataFrame
freqtrigger_bd_filtered = freqtrigger_bd_df[(freqtrigger_bd_df['Method'] != 'No UL') & 
                                            (freqtrigger_bd_df['Poison Ratio'] == 1)]

# Filter the 'patchtrigger_bd_df' DataFrame
patchtrigger_bd_filtered = patchtrigger_bd_df[(patchtrigger_bd_df['Method'] != 'No UL') & 
                                              (patchtrigger_bd_df['Poison Ratio'] == 1)]

# Filter the 'signaltrigger_bd_df' DataFrame
signaltrigger_bd_filtered = signaltrigger_bd_df[(signaltrigger_bd_df['Method'] != 'No UL') & 
                                                (signaltrigger_bd_df['Poison Ratio'] == 1)]


# Get 10th percentile ASR and corresponding Test Acc for each attack type in the new datasets
checkertrigger_bd_asr_values, checkertrigger_bd_test_acc_values = get_percentile_asr_and_test_acc(checkertrigger_bd_filtered)
freqtrigger_bd_asr_values, freqtrigger_bd_test_acc_values = get_percentile_asr_and_test_acc(freqtrigger_bd_filtered)
patchtrigger_bd_asr_values, patchtrigger_bd_test_acc_values = get_percentile_asr_and_test_acc(patchtrigger_bd_filtered)
signaltrigger_bd_asr_values, signaltrigger_bd_test_acc_values = get_percentile_asr_and_test_acc(signaltrigger_bd_filtered)

# Prepare the combined data for plotting
asr_values_bd = [checkertrigger_bd_asr_values, freqtrigger_bd_asr_values, patchtrigger_bd_asr_values, signaltrigger_bd_asr_values]
test_acc_values_bd = [checkertrigger_bd_test_acc_values, freqtrigger_bd_test_acc_values, patchtrigger_bd_test_acc_values, signaltrigger_bd_test_acc_values]

# Filter out 'No UL' from the methods for plot
methods_bd = checkertrigger_bd_filtered['Method'].unique()



# Plotting the 10th percentile ASR as bars and the corresponding test accuracy as scatter points
fig, ax1 = plt.subplots(figsize=(14, 7))
ax1.tick_params(axis='x', labelsize=20)  # Increase x-axis tick label size
ax1.tick_params(axis='y', labelsize=20) 
# Define the bar width
bar_width = 0.2

# Define positions for each method's bars
r1_bd = range(len(methods_bd))
print(r1_bd)
r2_bd = [x + bar_width for x in r1_bd]
r3_bd = [x + 2*bar_width for x in r1_bd]
r4_bd = [x + 3*bar_width for x in r1_bd]

# Plotting the ASR as bars
ax1.bar(r1_bd, asr_values_bd[0], width=bar_width, color='skyblue', label='CheckerTrigger ASR')
ax1.bar(r2_bd, asr_values_bd[1], width=bar_width, color='plum', label='FreqTrigger ASR')  # Light yellow
ax1.bar(r3_bd, asr_values_bd[2], width=bar_width, color='lightgreen', label='PatchTrigger ASR')
ax1.bar(r4_bd, asr_values_bd[3], width=bar_width, color='salmon', label='SignalTrigger ASR')

# Plotting the corresponding test accuracy as scatter points
ax1.scatter(r1_bd, test_acc_values_bd[0], color='blue', label='CheckerTrigger Test Acc', s=150, marker='o')
ax1.scatter(r2_bd, test_acc_values_bd[1], color='purple', label='FreqTrigger Test Acc', s=150, marker='o')  # Dark orange
ax1.scatter(r3_bd, test_acc_values_bd[2], color='green', label='PatchTrigger Test Acc', s=150, marker='o')
ax1.scatter(r4_bd, test_acc_values_bd[3], color='red', label='SignalTrigger Test Acc', s=150, marker='o')

# Labeling
ax1.set_xlabel('Unlearning Method',fontsize=20)
ax1.set_ylabel('10th Percentile ASR Percentage /\n Corresponding Test Accuracy Percentage',fontsize=20)
#ax1.set_title('10th Percentile ASR and Corresponding Test Accuracy by Method for 10% Poison Ratio (BD)')
ax1.tick_params(axis='y')

# Adding legends at the top
#ax1.legend(loc='upper center', bbox_to_anchor=(0.5, 1.15), ncol=4,,fontsize=20)

# Adjusting the x-ticks to show method names
plt.xticks([r + 1.5*bar_width for r in range(len(methods_bd))], methods_bd, rotation=0)

plt.tight_layout()
plt.show()