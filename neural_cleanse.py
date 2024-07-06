import torch
from torch import nn
from tqdm import tqdm
import sys
import csv
import os
from utils.utils import save_checkpoint, progress_bar, normalization
from test_model import test_epoch

class NeuralCleanse:

    def __init__(self, args, model):
        self.model= model
        self.device = args.device
        self.args=args



    def get_average_activation_map(self, dataloader):
        model = self.model
        model.eval()
        total_sum = None
        count = 0

        with torch.no_grad():
            for idx, (img, target, gt_label) in enumerate(dataloader, start=1):
                if self.args.device == 'cuda':
                    img = img.cuda()
                    target = target.cuda()

                img = normalization(self.args, img)
                activation1_t, activation2_t, activation3_t, activation4_t, _ = model(img)
                if total_sum is None:
                    total_sum = torch.zeros_like(activation3_t)
                total_sum += activation3_t.sum(dim=0)
                count += img.size(0)

        average_activation = total_sum / count
        return average_activation
    
    
    def unlearn(self, clean_data_loader, poision_data_loader):
        average_clean_activation =  self.get_average_activation_map(clean_data_loader)
        average_poison_activation =  self.get_average_activation_map(poision_data_loader)

        mean_clean = average_clean_activation.mean(dim=0)
        mean_poison = average_poison_activation.mean(dim=0)
        var_clean = average_clean_activation.var(dim=0)
        var_poison = average_poison_activation.var(dim=0)

        # Weight differences by variance
        activation_difference = (mean_poison - mean_clean) * (var_poison + var_clean)

        # Apply significance threshold
        threshold = activation_difference.std()  # Using standard deviation as a dynamic threshold
        significant_activation_difference = torch.where(
            torch.abs(activation_difference) > threshold,
            activation_difference,
            torch.zeros_like(activation_difference)
        )

        neuron_scores = significant_activation_difference
        
        neuron_scores = activation_difference
        neuron_scores_flat = neuron_scores

        num_neurons = neuron_scores_flat.numel()
        frac_to_prune = 0.2
        num_to_prune = int(num_neurons * frac_to_prune)
        values, indices_to_prune = torch.topk(neuron_scores.abs(), k=num_to_prune, dim=0, largest=True)
        self.prune_by_index(indices_to_prune)


    def prune_by_index(self, indices):
        with torch.no_grad():
            for name, param in self.model.named_parameters():
                if 'layer3' in name:
                    if 'conv' in name and 'weight' in name or 'bn' in name and ('weight' in name or 'bias' in name):
                        mask = torch.ones(param.size(0), dtype=torch.bool)
                        mask[indices] = True
                        if 'conv' in name and 'weight' in name:
                            param.data[mask, :, :, :] = 0 
                        elif 'bn' in name:
                            param.data[mask] = 0




    
    
    