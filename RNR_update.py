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



    def train_epoch_rnr(self, isolate_clean_data_loader ,criterion, epoch):
        self.model.train()

        total_clean = 0
        total_clean_correct, total_robust_correct = 0, 0
        train_loss = 0

        arg = self.arg

        for i, (inputs, labels, isCleans) in enumerate(isolate_clean_data_loader):
            # Normalize the input images
            inputs = normalization(arg, inputs)  # Assuming 'inputs' is already correctly shaped

            # Move data to the appropriate device (CPU or GPU)
            inputs, labels, isCleans = inputs.to(arg.device), labels.to(arg.device), isCleans.to(arg.device)

            outputs = self.model(inputs)
            loss = self.criterion(outputs, labels)

            self.optimizer.zero_grad()
            # Backpropagation
            loss.backward()
            self.optimizer.step()

            total_clean_correct += torch.sum(torch.argmax(outputs[:], dim=1) == labels[:])
            #total_robust_correct += torch.sum(torch.argmax(outputs[:], dim=1) == gt_labels[:])
            total_clean += inputs.shape[0]
        
            # Accumulate statistics
            #train_loss += loss.item()
            #total_clean_correct += torch.sum(torch.argmax(outputs, dim=1) == clean_labels)
           # total_robust_correct += torch.sum(torch.argmax(outputs, dim=1) == clean_gt_labels)
            #total_clean += clean_inputs.shape[0]

            # Display progress
            if total_clean > 0:  # Avoid division by zero
                avg_acc_clean = total_clean_correct * 100.0 / total_clean
                #avg_acc_robust = total_robust_correct * 100.0 / total_clean
                progress_bar(i, len(isolate_clean_data_loader),
                            'Epoch: %d | Loss: %.3f | Train ACC: %.3f%% (%d/%d)' % (
                            epoch, train_loss / (i + 1), avg_acc_clean, total_clean_correct, total_clean))

        # Step the learning rate scheduler
        self.scheduler.step()

        return train_loss / (i + 1), avg_acc_clean

    def unlearn(self, isolate_clean_data_loader, testloader_clean, testloader_bd):
    
        print("Training from scratch...")
        start_epoch = 0

        best_acc = 0

        arg = self.arg
        # Write
        save_folder_path = os.path.join('./saved/backdoored_model', 'poison_rate_0', 'withTrans',arg.dataset, 
                                        arg.model, arg.trigger_type)
        if not os.path.exists(save_folder_path):
            os.makedirs(save_folder_path)
        arg.log = os.path.join(save_folder_path, 'rnr_signal_010824.csv')
        f_name = arg.log
        csvFile = open(f_name, 'a', newline='')
        writer = csv.writer(csvFile)
        writer.writerow(
            ['Epoch', 'Train_Loss', 'Train_ACC',  'Train_R-ACC', 'Test_Loss_cl', 'Test_ACC', 'Test_Loss_bd',
            'Test_ASR', 'Test_R-ACC'])
        
        test_loss_cl, test_acc_cl, _ = test_epoch(arg, testloader_clean, self.model, self.criterion, -1, 'clean')
        test_loss_bd, test_acc_bd, test_acc_robust = test_epoch(arg, testloader_bd, self.model, self.criterion, -1, 'bd')
        writer.writerow(
                [-1, 0, 0, 0, test_loss_cl, test_acc_cl.item(),
                test_loss_bd, test_acc_bd.item(), test_acc_robust.item()])
        
        for epoch in tqdm(range(start_epoch, arg.epochs)):
            train_loss, train_acc = self.train_epoch_rnr(isolate_clean_data_loader, self.criterion, epoch)
            test_loss_cl, test_acc_cl, _ = test_epoch(arg, testloader_clean, self.model, self.criterion, epoch, 'clean')
            test_loss_bd, test_acc_bd, test_acc_robust = test_epoch(arg, testloader_bd, self.model, self.criterion, epoch, 'bd')

            # Save in every epoch
            save_file_path = os.path.join(save_folder_path, str(epoch) + '.tar')
            save_checkpoint(save_file_path, epoch, self.model, self.optimizer, self.scheduler)

            writer.writerow(
                [epoch, train_loss, train_acc.item(), test_loss_cl, test_acc_cl.item(),
                test_loss_bd, test_acc_bd.item(), test_acc_robust.item()])
        csvFile.close()