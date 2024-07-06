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
        activation_difference = average_poison_activation - average_clean_activation
        
        pruning_score = activation_difference.mean(dim=(128, 256, 8, 8)) + activation_difference.std(dim=(128, 256, 8, 8))
        threshold = 0.1
        self.prune_by_activation(pruning_score, threshold)

    def prune_by_activation(self, pruning_score, threshold):
        with torch.no_grad():
            for name, param in self.model.named_parameters():
                if 'layer3' in name:
                    layer_index = int(name.split('.')[2])
                    mask = pruning_score[layer_index] < threshold
                    if 'conv' in name and 'weight' in name:
                        param.data[:, mask, :, :] = 0
                    elif 'bn' in name:
                        if 'weight' in name or 'bias' in name:
                            param.data[mask] = 0



    
    
    