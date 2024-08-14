import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def load_and_prepare_data(filepath):
    data = pd.read_csv(filepath)
    data['Test Acc'] = pd.to_numeric(data['Test Acc'], errors='coerce')
    data['ASR'] = pd.to_numeric(data['ASR'], errors='coerce')
    return data

def plot_metrics_for_method(data, poison_rate):
    methods = data['Method'].unique()
    data_poison = data[data['Poison Ratio'] == poison_rate]
    
    for method in methods:
        method_data = data_poison[data_poison['Method'] == method]
        plt.figure(figsize=(8, 4))
        sns.lineplot(x='Identification Rate', y='Test Acc', data=method_data, marker='o', label='Test Acc')
        sns.lineplot(x='Identification Rate', y='ASR', data=method_data, marker='o', label='ASR')
        plt.title(f'Test Accuracy and ASR for "{method}" Method at Poison Ratio {poison_rate}',fontsize=25)
        plt.xlabel('Identification Rate(%)',fontsize=20)
        plt.ylabel('Percentage',fontsize=20)
        plt.xticks(fontsize=15)
        plt.yticks(fontsize=15)
        plt.legend()
        plt.grid(True)
        plt.ylim(0, 105)
        plt.show()
# Path to the dataset
filepath = './Visualization/data/freqtriggerBD.csv'

# Load and prepare data
data = load_and_prepare_data(filepath)

# Plot for poison rate 1
plot_metrics_for_method(data, 1)

# Plot for poison rate 10
plot_metrics_for_method(data, 10)
