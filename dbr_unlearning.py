
import torch
from tqdm import tqdm
import csv
from test_model import test_epoch
from utils.utils import save_checkpoint_only, progress_bar, normalization


class DBRUnlearning:

    def __init__(self, model, criterion, arg):
        self.model = model
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=arg.lr, momentum=0.9, weight_decay=5e-4)
        self.criterion = criterion
        self.args = arg


    def learning_rate_unlearning(self):       
        lr = 0.0001
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr

    
    def train_step_unlearning(self, train_loader, epoch):
        self.model.train()

        total_clean, total_clean_correct = 0, 0
        for idx, (img, target, flag) in enumerate(train_loader, start=1):
            img = normalization(self.args, img)
            img = img.cuda()
            target = target.cuda()
            # img = img
            # target = target

            output = self.model(img)
            loss = self.criterion(output, target)

            self.optimizer.zero_grad()
            (-loss).backward()  # Gradient ascent training
            self.optimizer.step()

            total_clean_correct += torch.sum(torch.argmax(output[:], dim=1) == target[:])
            total_clean += img.shape[0]
            avg_acc_clean = total_clean_correct * 100.0 / total_clean
            progress_bar(idx, len(train_loader),
                        'Epoch: %d | Loss: %.3f | Train ACC: %.3f%% (%d/%d)' % (
                        epoch, loss / (idx + 1), avg_acc_clean, total_clean_correct, total_clean))



    def train_step_relearning(self, train_loader, epoch):
        self.model.train()

        total_clean, total_clean_correct = 0, 0

        for idx, (img, target, flag) in enumerate(train_loader, start=1):
            img = normalization(self.args, img)
            img = img.cuda()
            target = target.cuda()
            # img = img
            # target = target

            output = self.model(img)
            loss = self.criterion(output, target)

            self.optimizer.zero_grad()
            loss.backward()  # Gradient ascent training
            self.optimizer.step()

            total_clean_correct += torch.sum(torch.argmax(output[:], dim=1) == target[:])
            total_clean += img.shape[0]
            avg_acc_clean = total_clean_correct * 100.0 / total_clean
            progress_bar(idx, len(train_loader),
                        'Epoch: %d | Loss: %.3f | Train ACC: %.3f%% (%d/%d)' % (
                        epoch, loss / (idx + 1), avg_acc_clean, total_clean_correct, total_clean))


    def unlearn(self, testloader_clean, testloader_bd, isolate_poison_data_loader, isolate_clean_data_loader):
        arg = self.args
        f_name = arg.log
        csvFile = open(f_name, 'a', newline='')
        writer = csv.writer(csvFile)
        writer.writerow(['Epoch', 'Test_ACC', 'Test_ASR'])
        criterion = self.criterion

        start_epoch = 0

        # Training and Testing
        best_acc = 0

        # Test the orginal performance of the model
        test_loss_cl, test_acc_cl, _ = test_epoch(arg, testloader_clean, self.model, criterion, 0, 'clean')
        test_loss_bd, test_acc_bd, test_acc_robust = test_epoch(arg, testloader_bd, self.model, criterion, 0, 'bd')
        writer.writerow([-1, test_acc_cl.item(), test_acc_bd.item()])

        for epoch in tqdm(range(start_epoch, arg.epochs)):
            # Modify lr
            self.learning_rate_unlearning()

            # Unlearn
            self.train_step_unlearning(arg, isolate_poison_data_loader, epoch)

            # Relearn
            self.train_step_relearning(arg, isolate_clean_data_loader, epoch)

            test_loss_cl, test_acc_cl, _ = test_epoch(arg, testloader_clean, self.model, self.criterion, epoch, 'clean')
            test_loss_bd, test_acc_bd, test_acc_robust = test_epoch(arg, testloader_bd, self.model, self.criterion, epoch, 'bd')

            # Save the best model
            if test_acc_cl - test_acc_bd > best_acc:
                best_acc = test_acc_cl - test_acc_bd
                save_checkpoint_only(arg.checkpoint_save, self.model)

            writer.writerow([epoch, test_acc_cl.item(), test_acc_bd.item()])
        csvFile.close()