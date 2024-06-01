import torch
from torch import nn
from tqdm import tqdm
import sys
import csv
import os
from utils.utils import save_checkpoint, progress_bar, normalization
from test_model import test_epoch

class RNR:

    def __init__(self, model, criterion, arg):
        self.model = model
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=arg.lr, momentum=0.9, weight_decay=5e-4)
        self.scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=arg.schedule, gamma=arg.gamma)
        self.arg = arg
        self.criterion = criterion



    def train_epoch_rnr(self, trainloader, epoch):
        self.model.train()

        total_clean = 0
        total_clean_correct, total_robust_correct = 0, 0
        train_loss = 0

        arg = self.arg

        for i, (inputs, labels, gt_labels, isCleans) in enumerate(trainloader):
            # Normalize the input images
            inputs = normalization(arg, inputs[1])  # Assuming 'inputs' is already correctly shaped

            # Move data to the appropriate device (CPU or GPU)
            inputs, labels, gt_labels = inputs.to(arg.device), labels.to(arg.device), gt_labels.to(arg.device)

            # Find indices of clean samples
            clean_idx = torch.where(isCleans == True)[0]

            # Filter inputs and labels to include only clean data
            clean_inputs = inputs[clean_idx]
            clean_labels = labels[clean_idx]
            clean_gt_labels = gt_labels[clean_idx]

            # Perform forward pass with only clean data
            if len(clean_inputs) > 0:  # Check to ensure there are clean samples
                outputs = self.model(clean_inputs)
                loss = self.criterion(outputs, clean_labels)

                # Backpropagation
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                # Accumulate statistics
                train_loss += loss.item()
                total_clean_correct += torch.sum(torch.argmax(outputs, dim=1) == clean_labels)
                total_robust_correct += torch.sum(torch.argmax(outputs, dim=1) == clean_gt_labels)
                total_clean += clean_inputs.shape[0]

            # Display progress
            if total_clean > 0:  # Avoid division by zero
                avg_acc_clean = total_clean_correct * 100.0 / total_clean
                avg_acc_robust = total_robust_correct * 100.0 / total_clean
                progress_bar(i, len(trainloader),
                            'Epoch: %d | Loss: %.3f | Train ACC: %.3f%% (%d/%d) | Train R-ACC: %.3f%% (%d/%d)' % (
                            epoch, train_loss / (i + 1), avg_acc_clean, total_clean_correct, total_clean,
                            avg_acc_robust, total_robust_correct, total_clean))

        # Step the learning rate scheduler
        self.scheduler.step()

        return train_loss / (i + 1), avg_acc_clean, avg_acc_robust

    def unlearn(self, trainloader, testloader_clean, testloader_bd):
    
        print("Training from scratch...")
        start_epoch = 0

        best_acc = 0

        arg = self.arg
        # Write
        save_folder_path = os.path.join('./saved/backdoored_model', 'poison_rate_0', 'withTrans', arg.dataset, 
                                        arg.model, arg.trigger_type)
        if not os.path.exists(save_folder_path):
            os.makedirs(save_folder_path)
        arg.log = os.path.join(save_folder_path, 'withTrans.csv')
        f_name = arg.log
        csvFile = open(f_name, 'a', newline='')
        writer = csv.writer(csvFile)
        writer.writerow(
            ['Epoch', 'Train_Loss', 'Train_ACC',  'Train_R-ACC', 'Test_Loss_cl', 'Test_ACC', 'Test_Loss_bd',
            'Test_ASR', 'Test_R-ACC'])
        for epoch in tqdm(range(start_epoch, arg.epochs)):
            train_loss, train_acc, train_racc = self.train_epoch_rnr(trainloader, criterion, epoch)
            test_loss_cl, test_acc_cl, _ = test_epoch(arg, testloader_clean, self.model, criterion, epoch, 'clean')
            test_loss_bd, test_acc_bd, test_acc_robust = test_epoch(arg, testloader_bd, self.model, criterion, epoch, 'bd')

            # Save in every epoch
            save_file_path = os.path.join(save_folder_path, str(epoch) + '.tar')
            save_checkpoint(save_file_path, epoch, self.model, self.optimizer, self.scheduler)

            writer.writerow(
                [epoch, train_loss, train_acc.item(), train_racc.item(), test_loss_cl, test_acc_cl.item(),
                test_loss_bd, test_acc_bd.item(), test_acc_robust.item()])
        csvFile.close()