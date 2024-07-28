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
    data_poison = data[data['Poison Rate'] == poison_rate]
    
    for method in methods:
        method_data = data_poison[data_poison['Method'] == method]
        plt.figure(figsize=(12, 6))
        sns.lineplot(x='Number of Samples', y='Test Acc', data=method_data, marker='o', label='Test Acc')
        sns.lineplot(x='Number of Samples', y='ASR', data=method_data, marker='o', label='ASR')
        plt.title(f'Test Accuracy and ASR for "{method}" Method at Poison Rate {poison_rate}')
        plt.xlabel('Number of Samples')
        plt.ylabel('Percentage')
        plt.legend()
        plt.grid(True)
        plt.show()

# Path to the dataset
filepath = './data/blendsignal.csv'

# Load and prepare data
data = load_and_prepare_data(filepath)

# Plot for poison rate 1
plot_metrics_for_method(data, 1)

# Plot for poison rate 10
plot_metrics_for_method(data, 10)
