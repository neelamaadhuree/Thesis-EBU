import torch
import time
from torch import nn
from test_model import test_epoch
import csv
from utils.utils import accuracy, normalization, AverageMeter
'''
Based on the paper 'Towards Adversarial Evaluations for Inexact Machine Unlearning' this code was created.
Source: https://github.com/shash42/Evaluating-Inexact-Unlearning
Original Author: Shashwat Goel, Ameya Prabhu, Sernam Lim, Phillip Torr, Amartya Sanyal, Ponnurangam Kumaraguru
'''

class ContinuousForgetting:

    def __init__(self, args, criterion):
        self.args = args
        self.criterion = criterion

    def trainable_params_(self, m):
        return [p for p in m.parameters() if p.requires_grad]

    def train_cf(self, model, epoch, optimizer, scheduler, train_loader):
        args = self.args
        criterion = self.criterion
        model.train()
        losses = AverageMeter()
        top1 = AverageMeter()
        top5 = AverageMeter()

        for param_group in optimizer.param_groups:
            print(f"Epoch {epoch}, Learning Rate: {param_group['lr']}")
        
        for idx, (inputs, target, flag) in enumerate(train_loader, start=1):
            inputs = normalization(args, inputs)  # Assuming 'inputs' is already correctly shaped

            if args.device == 'cuda':
                inputs, target = inputs.cuda(non_blocking=True), target.cuda(non_blocking=True)
                model = model.cuda()
                criterion = criterion.cuda()
            
            outputs = model(inputs)
            loss = criterion(outputs, target)
        
            prec1, prec5 = accuracy(outputs, target, topk=(1, 5))
            losses.update(loss.item(), inputs.size(0))
            top1.update(prec1.item(), inputs.size(0))
            top5.update(prec5.item(), inputs.size(0))

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
            optimizer.step()
        scheduler.step()

        if True:
            print('Epoch[{0}]:[{1:03}/{2:03}] '
                'loss:{losses.val:.4f}({losses.avg:.4f})  '
                'prec@1:{top1.val:.2f}({top1.avg:.2f})  '
                'prec@5:{top5.val:.2f}({top5.avg:.2f})'.format(epoch, idx, len(train_loader), losses=losses, top1=top1, top5=top5))
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
            test_loss_cl, test_acc_cl, _ = test_epoch(self.args, testloader_clean, model, self.criterion, 0, 'clean')
            test_loss_bd, test_acc_bd, test_acc_robust = test_epoch(self.args, testloader_bd, model, self.criterion, 0, 'bd')
            writer.writerow([0, test_acc_cl.item(), test_acc_bd.item()])
            optimizer = torch.optim.SGD(self.trainable_params_(model), lr=self.args.lr, momentum=0.70, weight_decay=1e-4)
            scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=self.args.schedule, gamma=self.args.gamma)
            for epoch in range(self.args.epochs):
                self.train_cf(model, epoch, optimizer, scheduler, isolate_clean_data_loader)
                test_loss_cl, test_acc_cl, _ = test_epoch(self.args, testloader_clean, model, self.criterion, 0, 'clean')
                test_loss_bd, test_acc_bd, test_acc_robust = test_epoch(self.args, testloader_bd, model, self.criterion, 0, 'bd')
                writer.writerow([0, test_acc_cl.item(), test_acc_bd.item()])
            

