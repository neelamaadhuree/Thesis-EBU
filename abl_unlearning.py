
from utils.utils import accuracy,save_checkpoint, progress_bar, normalization,AverageMeter
import torch
import pandas as pd

class ABLUnlearning:

    def __init__(self, 
                model,  
                criterion,
                datalaoder_for_unlearning, 
                args,
                device = 'cuda', 
                ):
        self.model = model
        self.optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=1e-4)
        self.criterion = criterion
        self.datalaoder_for_unlearning = datalaoder_for_unlearning
        self.device = device
        self.args = args

    def train_step_unlearning(self, epoch):
        self.model.train()
        losses = AverageMeter()
        top1 = AverageMeter()
        top5 = AverageMeter()

        for idx, (img, target, flag) in enumerate(self.datalaoder_for_unlearning, start=1):
            if self.device=='cuda':
                img = img.cuda()
                target = target.cuda()

            output = self.model(img)

            loss = self.criterion(output, target)

            prec1, prec5 = accuracy(output, target, topk=(1, 5))
            losses.update(loss.item(), img.size(0))
            top1.update(prec1.item(), img.size(0))
            top5.update(prec5.item(), img.size(0))

            self.optimizer.zero_grad()
            (-loss).backward()  # Gradient ascent training
            self.optimizer.step()

            if idx % self.args.print_freq == 0:
                print('Epoch[{0}]:[{1:03}/{2:03}] '
                    'loss:{losses.val:.4f}({losses.avg:.4f})  '
                    'prec@1:{top1.val:.2f}({top1.avg:.2f})  '
                    'prec@5:{top5.val:.2f}({top5.avg:.2f})'.format(epoch, idx, len(train_loader), losses=losses, top1=top1, top5=top5))




    def learning_rate_unlearning(self, epoch):
        if epoch < self.arg.unlearning_epochs:
            lr = 0.0005
        else:
            lr = 0.0001
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr

    def learning_rate_finetuning(self, epoch):
        if epoch < 40:
            lr = 0.1
        elif epoch < 60:
            lr = 0.01
        else:
            lr = 0.001
        print('epoch: {}  lr: {:.4f}'.format(epoch, lr))
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr



    def train_step_finetuing(self, opt, epoch):
        losses = AverageMeter()
        top1 = AverageMeter()
        top5 = AverageMeter()

        self.model.train()
        train_loader = self.datalaoder_for_unlearning

        for idx, (img, target,gt_label) in enumerate(train_loader, start=1):
            if self.device=='cuda':
                img = normalization(self.args, img)
                img = img.cuda()
                target = target.cuda()

            output = self.model(img)
            loss = self.criterion(output, target)

            prec1, prec5 = accuracy(output, target, topk=(1, 5))
            losses.update(loss.item(), img.size(0))
            top1.update(prec1.item(), img.size(0))
            top5.update(prec5.item(), img.size(0))

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            if idx % opt.print_freq == 0:
                print('Epoch[{0}]:[{1:03}/{2:03}] '
                    'loss:{losses.val:.4f}({losses.avg:.4f})  '
                    'prec@1:{top1.val:.2f}({top1.avg:.2f})  '
                    'prec@5:{top5.val:.2f}({top5.avg:.2f})'.format(epoch, idx, len(train_loader), losses=losses, top1=top1, top5=top5))
        


    def test(self, test_clean_loader, test_bad_loader, epoch):
        test_process = []
        losses = AverageMeter()
        top1 = AverageMeter()
        top5 = AverageMeter()

        self.model.eval()

        for idx, (img, target,gt_label,flag) in enumerate(test_clean_loader, start=1):
            if self.device=='cuda':
                img = img.cuda()
                target = target.cuda()

            with torch.no_grad():
                output = self.model(img)
                loss = self.criterion(output, target)

            prec1, prec5 = accuracy(output, target, topk=(1, 5))
            losses.update(loss.item(), img.size(0))
            top1.update(prec1.item(), img.size(0))
            top5.update(prec5.item(), img.size(0))

        acc_clean = [top1.avg, top5.avg, losses.avg]

        losses = AverageMeter()
        top1 = AverageMeter()
        top5 = AverageMeter()

        for idx, (img, target,gt_label,flag) in enumerate(test_bad_loader, start=1):
            if self.device=='cuda':
                img = img.cuda()
                target = target.cuda()

            with torch.no_grad():
                output = self.model(img)
                loss = self.criterion(output, target)

            prec1, prec5 = accuracy(output, target, topk=(1, 5))
            losses.update(loss.item(), img.size(0))
            top1.update(prec1.item(), img.size(0))
            top5.update(prec5.item(), img.size(0))

        acc_bd = [top1.avg, top5.avg, losses.avg]

        print('[Clean] Prec@1: {:.2f}, Loss: {:.4f}'.format(acc_clean[0], acc_clean[2]))
        print('[Bad] Prec@1: {:.2f}, Loss: {:.4f}'.format(acc_bd[0], acc_bd[2]))

        # save training progress
        log_root = self.args.log_root + '/ABL_unlearning.csv'
        test_process.append(
            (epoch, acc_clean[0], acc_bd[0], acc_clean[2], acc_bd[2]))
        df = pd.DataFrame(test_process, columns=("Epoch", "Test_clean_acc", "Test_bad_acc",
                                               "Test_clean_loss", "Test_bad_loss"))
        df.to_csv(log_root, mode='a', index=False, encoding='utf-8')

        return acc_clean, acc_bd


    def unlearn(self,arg, testloader_clean, testloader_bd):
        if arg.finetuning_ascent_model == True:
            # this is to improve the clean accuracy of isolation model, you can skip this step
            print('----------- Finetuning isolation model --------------')
            for epoch in range(0, arg.finetuning_epochs):
                self.learning_rate_finetuning(epoch)
                self.train_step_finetuing(arg,epoch + 1)
                self.test(testloader_clean, testloader_bd, epoch + 1)


        print('----------- Model unlearning --------------')
        for epoch in range(0, arg.unlearning_epochs):
            self.learning_rate_unlearning(epoch)

            # train stage
            if epoch == 0:
                # test firstly
                self.test(testloader_clean, testloader_bd, epoch)
            else:
                self.train_step_unlearning(epoch + 1)

            # evaluate on testing set
            print('testing the ascended model......')
            acc_clean, acc_bad = self.test(testloader_clean, testloader_bd, epoch + 1)
            if arg.save:
                # save checkpoint at interval epoch
                if epoch + 1 % self.arg.interval == 0:
                    is_best = True
                    save_checkpoint({
                        'epoch': epoch + 1,
                        'state_dict': self.model.state_dict(),
                        'clean_acc': acc_clean[0],
                        'bad_acc': acc_bad[0],
                        'optimizer': self.optimizer.state_dict(),
                    }, epoch + 1, is_best, self.arg)
