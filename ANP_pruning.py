import os
import argparse
import numpy as np
import pandas as pd
import torch
from utils.utils import  normalization



class ANPPruning:
    def __init__(self, args, model):
        self.args = args
        self.model = model
        self.device = args.device


    def prune(self, clean_test_loader, poison_test_loader):
        net = self.model
        args = self.args
        criterion = torch.nn.CrossEntropyLoss().to(self.device)

        # Step 3: pruning
        mask_values = self.read_data(os.path.join(args.anp_output_dir, 'mask_values.txt'))
        mask_values = sorted(mask_values, key=lambda x: float(x[2]))
        print('No. \t Layer Name \t Neuron Idx \t Mask \t PoisonLoss \t PoisonACC \t CleanLoss \t CleanACC')
        cl_loss, cl_acc = self.test(model=net, criterion=criterion, data_loader=clean_test_loader)
        po_loss, po_acc = self.test(model=net, criterion=criterion, data_loader=poison_test_loader)
        print('0 \t None     \t None     \t {:.4f} \t {:.4f} \t {:.4f} \t {:.4f}'.format(po_loss, po_acc, cl_loss, cl_acc))
        if args.pruning_by == 'threshold':
            results = self.evaluate_by_threshold(
                net, mask_values, pruning_max=args.pruning_max, pruning_step=args.pruning_step,
                criterion=criterion, clean_loader=clean_test_loader, poison_loader=poison_test_loader
            )
        else:
            results = self.evaluate_by_number(
                net, mask_values, pruning_max=args.pruning_max, pruning_step=args.pruning_step,
                criterion=criterion, clean_loader=clean_test_loader, poison_loader=poison_test_loader
            )
        file_name = os.path.join(args.anp_output_dir, 'pruning_by_{}.txt'.format(args.pruning_by))
        with open(file_name, "w") as f:
            f.write('No \t Layer Name \t Neuron Idx \t Mask \t PoisonLoss \t PoisonACC \t CleanLoss \t CleanACC\n')
            f.writelines(results)


    def read_data(self, file_name):
        tempt = pd.read_csv(file_name, sep='\s+', skiprows=1, header=None)
        layer = tempt.iloc[:, 1]
        idx = tempt.iloc[:, 2]
        value = tempt.iloc[:, 3]
        mask_values = list(zip(layer, idx, value))
        return mask_values


    def pruning(self, net, neuron):
        state_dict = net.state_dict()
        weight_name = '{}.{}'.format(neuron[0], 'weight')
        state_dict[weight_name][int(neuron[1])] = 0.0
        net.load_state_dict(state_dict)


    def evaluate_by_number(self, model, mask_values, pruning_max, pruning_step, criterion, clean_loader, poison_loader):
        results = []
        nb_max = int(np.ceil(pruning_max))
        nb_step = int(np.ceil(pruning_step))
        for start in range(0, nb_max + 1, nb_step):
            i = start
            for i in range(start, start + nb_step):
                self.pruning(model, mask_values[i])
            layer_name, neuron_idx, value = mask_values[i][0], mask_values[i][1], mask_values[i][2]
            cl_loss, cl_acc = self.test(model=model, criterion=criterion, data_loader=clean_loader)
            po_loss, po_acc = self.test(model=model, criterion=criterion, data_loader=poison_loader)
            print('{} \t {} \t {} \t {} \t {:.4f} \t {:.4f} \t {:.4f} \t {:.4f}'.format(
                i+1, layer_name, neuron_idx, value, po_loss, po_acc, cl_loss, cl_acc))
            results.append('{} \t {} \t {} \t {} \t {:.4f} \t {:.4f} \t {:.4f} \t {:.4f}'.format(
                i+1, layer_name, neuron_idx, value, po_loss, po_acc, cl_loss, cl_acc))
        return results


    def evaluate_by_threshold(self, model, mask_values, pruning_max, pruning_step, criterion, clean_loader, poison_loader):
        results = []
        thresholds = np.arange(0, pruning_max + pruning_step, pruning_step)
        start = 0
        for threshold in thresholds:
            idx = start
            for idx in range(start, len(mask_values)):
                if float(mask_values[idx][2]) <= threshold:
                    self.pruning(model, mask_values[idx])
                    start += 1
                else:
                    break
            layer_name, neuron_idx, value = mask_values[idx][0], mask_values[idx][1], mask_values[idx][2]
            cl_loss, cl_acc = self.test(model=model, criterion=criterion, data_loader=clean_loader)
            po_loss, po_acc = self.test(model=model, criterion=criterion, data_loader=poison_loader)
            print('{:.2f} \t {} \t {} \t {} \t {:.4f} \t {:.4f} \t {:.4f} \t {:.4f}'.format(
                start, layer_name, neuron_idx, threshold, po_loss, po_acc, cl_loss, cl_acc))
            results.append('{:.2f} \t {} \t {} \t {} \t {:.4f} \t {:.4f} \t {:.4f} \t {:.4f}\n'.format(
                start, layer_name, neuron_idx, threshold, po_loss, po_acc, cl_loss, cl_acc))
        return results


    def test(self, model, criterion, data_loader):
        model.eval()
        total_correct = 0
        total_loss = 0.0
        with torch.no_grad():
            for i, (images, labels, gt_labels, isCleans) in enumerate(data_loader):
                images = normalization(self.args, images)
                images, labels = images.to(self.device), labels.to(self.device)
                output = model(images)
                total_loss += criterion(output, labels).item()
                pred = output.data.max(1)[1]
                total_correct += pred.eq(labels.data.view_as(pred)).sum()
        loss = total_loss / len(data_loader)
        acc = float(total_correct) / len(data_loader.dataset)
        return loss, acc