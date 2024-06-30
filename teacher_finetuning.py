import torch
from tqdm import tqdm
import csv
from torch import nn
from test_model import test_epoch
from utils.utils import save_checkpoint_only, progress_bar, normalization


class TeacherFineTuning:

    def __init__(self, model, arg):
        self.model = model
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=arg.lr, momentum=0.9, weight_decay=5e-4)
        self.scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=arg.schedule, gamma=arg.gamma)
        self.criterion = nn.CrossEntropyLoss()
        self.args = arg


    def learning_rate_unlearning(self):       
        lr = 0.0001
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr

    
    def train_step_relearning(self, train_loader, epoch):
        self.model.train()
        total_clean, total_clean_correct = 0, 0
        for param in self.model.parameters():
            param.requires_grad = True

        for param_group in self.optimizer.param_groups:
            print(f"Epoch {epoch}, Learning Rate: {param_group['lr']}")

        for idx, (img, target, flag) in enumerate(train_loader, start=1):
            img = normalization(self.args, img)
            img = img.cuda()
            target = target.cuda()
            activation1_s, activation2_s, activation3_s, activation4_s, output = self.model(img)
            loss = self.criterion(output, target)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            total_clean_correct += torch.sum(torch.argmax(output[:], dim=1) == target[:])
            total_clean += img.shape[0]
            avg_acc_clean = total_clean_correct * 100.0 / total_clean
            progress_bar(idx, len(train_loader),
                        'Epoch: %d | Loss: %.3f | Train ACC: %.3f%% (%d/%d)' % (
                        epoch, loss / (idx + 1), avg_acc_clean, total_clean_correct, total_clean))
       


    def fineTune(self, isolate_clean_data_loader):
        arg = self.args       
        start_epoch = 0

        for epoch in tqdm(range(start_epoch, arg.epochs)):
            self.train_step_relearning(isolate_clean_data_loader, epoch)
            self.scheduler.step()

        return self.model