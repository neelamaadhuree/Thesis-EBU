import pandas as pd
import matplotlib.pyplot as plt

# Load the CSV files
signal_trigger = pd.read_csv('./data/SignalTrigger.csv')
patch_trigger = pd.read_csv('./data/PatchTrigger.csv')
checker_trigger = pd.read_csv('./data/CheckerTrigger.csv')
freq_trigger = pd.read_csv('./data/FreqTrigger.csv')

# Correct the FreqTrigger dataset's formatting issues if necessary
# Assuming it starts with some incorrect rows
freq_trigger = freq_trigger.drop([0, 1, 2])

# Convert columns to appropriate data types if necessary
freq_trigger = freq_trigger.astype({
    'Epoch': int,
    'Train_Loss': float,
    'Train_ACC': float,
    'Train_ASR': float,
    'Train_R-ACC': float,
    'Test_Loss_cl': float,
    'Test_ACC': float,
    'Test_Loss_bd': float,
    'Test_ASR': float,
    'Test_R-ACC': float
})

# Filter data to only include the first 100 epochs
signal_trigger_100 = signal_trigger[signal_trigger['Epoch'] < 100]
patch_trigger_100 = patch_trigger[patch_trigger['Epoch'] < 100]
checker_trigger_100 = checker_trigger[checker_trigger['Epoch'] < 100]
freq_trigger_100 = freq_trigger[freq_trigger['Epoch'] < 100]

# Function to create combined plots for the first 100 epochs
def create_combined_plots_100(signal_df, patch_df, checker_df, freq_df):
    # Plot 1: Train ACC and Test ACC for all triggers
    plt.figure(figsize=(14, 6))
    plt.plot(signal_df['Epoch'], signal_df['Train_ACC'], label='Signal Train ACC', color='blue')
    plt.plot(signal_df['Epoch'], signal_df['Test_ACC'], label='Signal Test ACC', linestyle='--', color='blue')
    plt.plot(patch_df['Epoch'], patch_df['Train_ACC'], label='Patch Train ACC', color='green')
    plt.plot(patch_df['Epoch'], patch_df['Test_ACC'], label='Patch Test ACC', linestyle='--', color='green')
    plt.plot(checker_df['Epoch'], checker_df['Train_ACC'], label='Checker Train ACC', color='red')
    plt.plot(checker_df['Epoch'], checker_df['Test_ACC'], label='Checker Test ACC', linestyle='--', color='red')
    plt.plot(freq_df['Epoch'], freq_df['Train_ACC'], label='Freq Train ACC', color='purple')
    plt.plot(freq_df['Epoch'], freq_df['Test_ACC'], label='Freq Test ACC', linestyle='--', color='purple')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Train and Test Accuracy (First 100 Epochs)')
    plt.legend()
    plt.tight_layout()
    plt.show()

    # Plot 2: Train ASR and Test ASR for all triggers
    plt.figure(figsize=(14, 6))
    plt.plot(signal_df['Epoch'], signal_df['Train_ASR'], label='Signal Train ASR', color='blue')
    plt.plot(signal_df['Epoch'], signal_df['Test_ASR'], label='Signal Test ASR', linestyle='--', color='blue')
    plt.plot(patch_df['Epoch'], patch_df['Train_ASR'], label='Patch Train ASR', color='green')
    plt.plot(patch_df['Epoch'], patch_df['Test_ASR'], label='Patch Test ASR', linestyle='--', color='green')
    plt.plot(checker_df['Epoch'], checker_df['Train_ASR'], label='Checker Train ASR', color='red')
    plt.plot(checker_df['Epoch'], checker_df['Test_ASR'], label='Checker Test ASR', linestyle='--', color='red')
    plt.plot(freq_df['Epoch'], freq_df['Train_ASR'], label='Freq Train ASR', color='purple')
    plt.plot(freq_df['Epoch'], freq_df['Test_ASR'], label='Freq Test ASR', linestyle='--', color='purple')
    plt.xlabel('Epoch')
    plt.ylabel('ASR')
    plt.title('Train and Test ASR (First 100 Epochs)')
    plt.legend()
    plt.tight_layout()
    plt.show()

# Create combined plots for the first 100 epochs
create_combined_plots_100(signal_trigger_100, patch_trigger_100, checker_trigger_100, freq_trigger_100)