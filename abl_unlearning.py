
from utils.utils import accuracy,save_checkpoint, progress_bar, normalization, AverageMeter
import torch
import pandas as pd

class ABLUnlearning:

    def __init__(self, 
                model,  
                criterion,
                datalaoder_for_unlearning, 
                dataloader_for_finetuning,
                args,
                device = 'cuda', 
                ):
        self.model = model
        self.optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=1e-4)
        self.criterion = criterion
        self.datalaoder_for_unlearning = datalaoder_for_unlearning
        self.dataloader_for_finetuning = dataloader_for_finetuning
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
                    'prec@5:{top5.val:.2f}({top5.avg:.2f})'.format(epoch, idx, len(self.datalaoder_for_unlearning), losses=losses, top1=top1, top5=top5))




    def learning_rate_unlearning(self, epoch,args):
        if epoch < args.unlearning_epochs:
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
        train_loader = self.dataloader_for_finetuning

        for idx, (img, target,gt_label) in enumerate(train_loader, start=1):
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
            loss.backward()
            self.optimizer.step()

            if idx % opt.print_freq == 0:
                print('Epoch[{0}]:[{1:03}/{2:03}] '
                    'loss:{losses.val:.4f}({losses.avg:.4f})  '
                    'prec@1:{top1.val:.2f}({top1.avg:.2f})  '
                    'prec@5:{top5.val:.2f}({top5.avg:.2f})'.format(epoch, idx, len(train_loader), losses=losses, top1=top1, top5=top5))
        


    def test(self, test_clean_loader, test_bad_loader, epoch):
        
        self.model.eval()

        # Prepare meters and counters for clean data
        test_process = []
        total_clean_samples = 0
        correct_clean_samples = 0

        # Testing on clean data
        for idx, (img, target, gt_label, flag) in enumerate(test_clean_loader, start=1):
            if self.device == 'cuda':
                img = img.cuda()
                target = target.cuda()

            with torch.no_grad():
                output = self.model(img)
                predicted = output.argmax(dim=1)
                correct_clean_samples += (predicted == target).sum().item()
                total_clean_samples += target.size(0)

        clean_acc = correct_clean_samples / total_clean_samples if total_clean_samples > 0 else 0

        # Prepare meters and counters for poisoned data
        total_poisoned_samples = 0
        correct_poisoned_samples = 0

        # Testing on poisoned data
        for idx, (img, target, gt_label, flag) in enumerate(test_bad_loader, start=1):
            if self.device == 'cuda':
                img = img.cuda()
                target = target.cuda()

            with torch.no_grad():
                output = self.model(img)
                predicted = output.argmax(dim=1)
                correct_poisoned_samples += (predicted == target).sum().item()
                total_poisoned_samples += target.size(0)

        asr = correct_poisoned_samples / total_poisoned_samples if total_poisoned_samples > 0 else 0

        # Output results
        print(f'Epoch: {epoch}, Clean Test Accuracy: {clean_acc:.2f}, Attack Success Rate: {asr:.2f}')

        # Save training progress
        log_root = self.args.log_root + '/ABL_unlearning_cln.csv'
        test_process.append(
            (epoch, clean_acc, asr)
        )
        df = pd.DataFrame(test_process, columns=("Epoch", "Clean_Test_Acc", "Test_ASR"))
        df.to_csv(log_root, mode='a', index=False, encoding='utf-8')

        return clean_acc, asr


    def unlearn(self,arg, testloader_clean, testloader_bd):
        if arg.finetuning_ascent_model == True:
            # this is to improve the clean accuracy of isolation model, you can skip this step
            #self.test(testloader_clean, testloader_bd, -1)
            print('----------- Finetuning isolation model --------------')
            for epoch in range(0, arg.finetuning_epochs):
                self.learning_rate_finetuning(epoch)
                self.train_step_finetuing(arg,epoch + 1)
                self.test(testloader_clean, testloader_bd, epoch + 1)


        print('----------- Model unlearning --------------')
        for epoch in range(0, arg.unlearning_epochs):
            self.learning_rate_unlearning(epoch,arg)

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
                if epoch + 1 % arg.interval == 0:
                    is_best = True
                    save_checkpoint({
                        'epoch': epoch + 1,
                        'state_dict': self.model.state_dict(),
                        'clean_acc': acc_clean[0],
                        'bad_acc': acc_bad[0],
                        'optimizer': self.optimizer.state_dict(),
                    }, epoch + 1, is_best, self.arg)
