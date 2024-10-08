import torch
from torch import nn
from tqdm import tqdm
import sys
import csv
import os
from utils.utils import save_checkpoint, progress_bar, normalization
from test_model import test_epoch


'''
Based on the paper 'Neural Cleanse: Identifying and Mitigating Backdoor Attacks in Neural Networks' this code was created.
Source: https://github.com/bolunwang/backdoor
Original Author:  Bolun Wang
'''

class NeuralCleanse:

    def __init__(self, args, model):
        self.model= model
        self.device = args.device
        self.args=args
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)


    def get_average_activation_map(self, dataloader):
        model = self.model
        model.eval()
        total_sum = None
        count = 0

        criterion = nn.CrossEntropyLoss()
        #with torch.no_grad():
        for idx, (img, target, gt_label) in enumerate(dataloader, start=1):
            if self.args.device == 'cuda':
                img = img.cuda()
                target = target.cuda()

            img = normalization(self.args, img)
            self.optimizer.zero_grad()
            activation1_t, activation2_t, activation3_t, activation4_t, out = model(img)
            
            #loss = criterion(out, target)
            #loss.backward()
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
        neuron_scores_flat = neuron_scores.view(-1)

        num_neurons = neuron_scores_flat.numel()
       
        frac_to_prune = 0.06
        num_to_prune = int(num_neurons * frac_to_prune)
        _, indices_to_prune = torch.topk(neuron_scores_flat.abs(), num_to_prune)
        self.prune_by_index(indices_to_prune)
        # self.prune_by_activation(pruning_score, threshold)

    # def prune_by_activation(self, pruning_score, threshold):
    #     with torch.no_grad():
    #     for name, param in self.model.named_parameters():
    #         if 'layer3' in name:
    #             layer_index = int(name.split('.')[2])
    #             mask = pruning_score[layer_index] < threshold
    #             if 'conv' in name and 'weight' in name:
    #                 param.data[:, mask, :, :] = 0
    #             elif 'bn' in name:
    #                 if 'weight' in name or 'bias' in name:
    #                     param.data[mask] = 0

    def prune_by_index(self, indices):
        with torch.no_grad():
            current_index=0
            for name, param in self.model.named_parameters():
                if 'layer3' in name:
                    if 'conv' in name and 'weight' in name or 'bn' in name and ('weight' in name or 'bias' in name):
                        num_elements = param.numel()
                        # Find indices that fall within this layer's range in the global tensor
                        layer_indices = [i - current_index for i in indices if current_index <= i < current_index + num_elements]
                        # Convert these global indices to local indices of the layer
                        local_indices = [i % param.size(0) for i in layer_indices]

                        if local_indices:  # If there are indices to prune in this layer
                            mask = torch.ones(param.size(0), dtype=torch.bool)
                            mask[local_indices] = False
                            if 'conv' in name and 'weight' in name:
                                param.data[~mask, :, :, :] = 0
                            elif 'bn' in name:
                                param.data[~mask] = 0

                        current_index += num_elements  #


    
    
    