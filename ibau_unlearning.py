import torch
from tqdm import tqdm
import csv
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset
import numpy as np
import random
import hypergrad as hg
from utils.utils import accuracy, normalization, AverageMeter, progress_bar


class IBAUUnlearning:
    def __init__(self, model, arg, epochs=5, lr=0.01, K =5):
        self.model = model
        self.epochs = epochs
        self.args = arg
        self.K = K
        self.outer_opt = torch.optim.SGD(model.parameters(), lr=arg.lr)


    ### define the inner loss L2


    def unlearn(self, unlearn_loader):
        args = self.args
        images_list, labels_list = [], []


        for index, (images, labels, isCleans) in enumerate(unlearn_loader):
            images = normalization(args, images)
            images_list.append(images)
            labels_list.append(labels)


        def loss_inner(perturb, model_params):
            images = images_list[0].to(args.device)
            labels = labels_list[0].long().to(args.device)
            per_img = torch.clamp(images+perturb[0],min=0,max=1)
            #per_img = images+perturb[0]
            per_logits = self.model.forward(per_img)
            loss = F.cross_entropy(per_logits, labels, reduction='none')
            loss_regu = torch.mean(-loss) +0.001 * torch.pow(torch.norm(perturb[0]),2)
            return loss_regu


        def loss_outer(perturb, model_params):
            portion = 0.1
            images, labels = images_list[batchnum].to(args.device), labels_list[batchnum].long().to(args.device)
            patching = torch.zeros_like(images, device='cuda')
            number = images.shape[0]
            rand_idx = random.sample(list(np.arange(number)),int(number*portion))
            patching[rand_idx] = perturb[0]
            # unlearn_imgs = torch.clamp(images+patching,min=0,max=1)
            unlearn_imgs = images + patching
            logits = self.model(unlearn_imgs)
            criterion = nn.CrossEntropyLoss()
            loss = criterion(logits, labels)
            return loss

        inner_opt = hg.GradientDescent(loss_inner, 0.1)
        criterion = nn.CrossEntropyLoss()


        for round in range(self.epochs):
            batch_pert = torch.zeros((1, 3, 32, 32), requires_grad=True, device='cuda')
            batch_opt = torch.optim.SGD(params=[batch_pert], lr=30)
            
            for index, (images, labels, isCleans) in enumerate(unlearn_loader):
                images= normalization(self.args,images)
                images = images.to(args.device)
                ori_lab = torch.argmax(self.model.forward(images),axis = 1).long()
        #         per_logits = model.forward(torch.clamp(images+batch_pert,min=0,max=1))
                per_logits = self.model.forward(images+batch_pert)
                loss = F.cross_entropy(per_logits, ori_lab, reduction='mean')
                loss_regu = torch.mean(-loss) + 0.001 * torch.pow(torch.norm(batch_pert),2)
                batch_opt.zero_grad()
                loss_regu.backward(retain_graph = True)
                batch_opt.step()

            #l2-ball
            #pert = batch_pert * min(1, 10 / torch.norm(batch_pert))
            pert = batch_pert

            #unlearn step         
            for batchnum in range(len(images_list)): 
                self.outer_opt.zero_grad()
                hg.fixed_point([pert], list(self.model.parameters()), self.K, inner_opt, loss_outer) 
                self.outer_opt.step()


            self.evaluate(unlearn_loader, criterion, round)

    def evaluate(self, data_loader, criterion, epoch):
        self.model.eval()
        test_loss = 0
        total_clean, total_clean_correct = 0, 0
        with torch.no_grad():
            for index, (images, labels, isCleans) in enumerate(data_loader):
                images = normalization(self.args, images)
                images, labels = images.to(self.args.device), labels.to(self.args.device)
                outputs = self.model(images)
                loss = criterion(outputs, labels)

                test_loss += loss.item()

                total_clean_correct += torch.sum(torch.argmax(outputs[:], dim=1) == labels[:])
                total_clean += images.shape[0]
                avg_acc_clean = total_clean_correct * 100.0 / total_clean
                
                progress_bar(index, len(data_loader),
                        'Epoch: %d | Loss: %.3f | Train ACC: %.3f%% (%d/%d)' % (
                        epoch, loss / (index + 1), avg_acc_clean, total_clean_correct, total_clean))


