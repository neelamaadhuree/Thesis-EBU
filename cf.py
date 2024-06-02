import torch
import time
from torch import nn
from test_model import test_epoch
import csv

class ContinuousForgetting:

    def __init__(self, args, criterion):
        self.args = args
        self.criterion = criterion

    def trainable_params_(self, m):
        return [p for p in m.parameters() if p.requires_grad]

    def train_cf(self, model, epochs, train_loader, logger):
        args = self.args
        optimizer = torch.optim.SGD(self.trainable_params_(model), lr=args.lr, momentum=0.9, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.schedule, gamma=args.gamma)
        criterion = self.criterion
        model.train()

        for epoch in range(epochs):            
            for param_group in optimizer.param_groups:
                logger.info(f"Epoch {epoch}, Learning Rate: {param_group['lr']}")
            
            for inputs, labels, flg in train_loader:
                if args.device == 'cuda':
                    inputs, labels = inputs.cuda(non_blocking=True), labels.cuda(non_blocking=True)
                    model = model.cuda()
                    criterion = criterion.cuda()
                
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
                optimizer.step()
            scheduler.step()
        return model, optimizer
       
            
    def getRetrainLayers(self, m, name, ret):
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.Linear):
            ret.append((m, name))
        for child_name, child in m.named_children():
            self.getRetrainLayers(child, f'{name}.{child_name}', ret)
        return ret

    def resetFinalResnet(self, model, num_retrain):
        for param in model.parameters():
            param.requires_grad = False
        done = 0
        ret = self.getRetrainLayers(model, 'M', [])
        ret.reverse()
        for idx in range(len(ret)):
            if isinstance(ret[idx][0], nn.Conv2d) or isinstance(ret[idx][0], nn.Linear):
                done += 1
            for param in ret[idx][0].parameters():
                param.requires_grad = True
            if done >= num_retrain:
                break

        for name, param in model.named_parameters():
            if param.requires_grad:
                print(f"Trainable layer: {name}, shape: {param.shape}")

        return model

    def relearn(self, k_layer, model, isolate_clean_data_loader, testloader_clean, testloader_bd):
        with open(self.args.log, 'a', newline='') as csvFile:
            writer = csv.writer(csvFile)
            writer.writerow(['Epoch', 'Test_ACC', 'Test_ASR'])
            model = self.resetFinalResnet(model, k_layer)
            self.train_cf(model, self.args.epoch, isolate_clean_data_loader)
            test_loss_cl, test_acc_cl, _ = test_epoch(self.args, testloader_clean, model, self.criterion, 0, 'clean')
            test_loss_bd, test_acc_bd, test_acc_robust = test_epoch(self.args, testloader_bd, model, self.criterion, 0, 'bd')
            writer.writerow([0, test_acc_cl.item(), test_acc_bd.item()])

