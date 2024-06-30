from torch import nn
import torch
from at import AT
import pandas as pd
from utils.utils import accuracy, AverageMeter



class NAD:

    def __init__(self, args, teacher_model, student_model):
        self.teacher_model = teacher_model
        self.student_model = student_model
        self.device = args.device
        self.args = args

    def train_step(self, train_loader, nets, optimizer, criterions, epoch):
        at_losses = AverageMeter()
        top1 = AverageMeter()
        top5 = AverageMeter()

        snet = nets['snet']
        tnet = nets['tnet']

        criterionCls = criterions['criterionCls']
        criterionAT = criterions['criterionAT']

        snet.train()

        for idx, (img, target,gt_label) in enumerate(train_loader, start=1):
            if self.args.device == 'cuda':
                img = img.cuda()
                target = target.cuda()

            activation1_s, activation2_s, activation3_s, output_s = snet(img)
            activation1_t, activation2_t, activation3_t, _ = tnet(img)

            cls_loss = criterionCls(output_s, target)
            at3_loss = criterionAT(activation3_s, activation3_t.detach()) * self.args.nad_beta3
            at2_loss = criterionAT(activation2_s, activation2_t.detach()) * self.args.nad_beta2
            at1_loss = criterionAT(activation1_s, activation1_t.detach()) * self.args.nad_beta1
            at_loss = at1_loss + at2_loss + at3_loss + cls_loss

            prec1, prec5 = accuracy(output_s, target, topk=(1, 5))
            at_losses.update(at_loss.item(), img.size(0))
            top1.update(prec1.item(), img.size(0))
            top5.update(prec5.item(), img.size(0))

            optimizer.zero_grad()
            at_loss.backward()
            optimizer.step()

            if idx % self.args.print_freq == 0:
                print('Epoch[{0}]:[{1:03}/{2:03}] '
                    'AT_loss:{losses.val:.4f}({losses.avg:.4f})  '
                    'prec@1:{top1.val:.2f}({top1.avg:.2f})  '
                    'prec@5:{top5.val:.2f}({top5.avg:.2f})'.format(epoch, idx, len(train_loader), losses=at_losses, top1=top1, top5=top5))


    def test(self, test_clean_loader, test_bad_loader, nets, criterions, epoch):
        test_process = []
        top1 = AverageMeter()
        top5 = AverageMeter()

        snet = nets['snet']
        tnet = nets['tnet']

        criterionCls = criterions['criterionCls']
        criterionAT = criterions['criterionAT']

        snet.eval()

        for idx, (img, target, gt_label, flag) in enumerate(test_clean_loader, start=1):
            if self.args.device == 'cuda':
                img = img.cuda()
                target = target.cuda()

            with torch.no_grad():
                _, _, _, output_s = snet(img)

            prec1, prec5 = self.accuracy(output_s, target, topk=(1, 5))
            top1.update(prec1.item(), img.size(0))
            top5.update(prec5.item(), img.size(0))

        acc_clean = [top1.avg, top5.avg]

        cls_losses = AverageMeter()
        at_losses = AverageMeter()
        top1 = AverageMeter()
        top5 = AverageMeter()

        for idx, (img, target, gt_label, flag) in enumerate(test_bad_loader, start=1):
            if self.device=='cuda':
                img = img.cuda()
                target = target.cuda()

            with torch.no_grad():
                activation1_s, activation2_s, activation3_s, output_s = snet(img)
                activation1_t, activation2_t, activation3_t, _ = tnet(img)

                at3_loss = criterionAT(activation3_s, activation3_t.detach()) * self.args.beta3
                at2_loss = criterionAT(activation2_s, activation2_t.detach()) * self.args.beta2
                at1_loss = criterionAT(activation1_s, activation1_t.detach()) * self.args.beta1
                at_loss = at3_loss + at2_loss + at1_loss
                cls_loss = criterionCls(output_s, target)

            prec1, prec5 = self.accuracy(output_s, target, topk=(1, 5))
            cls_losses.update(cls_loss.item(), img.size(0))
            at_losses.update(at_loss.item(), img.size(0))
            top1.update(prec1.item(), img.size(0))
            top5.update(prec5.item(), img.size(0))

        acc_bd = [top1.avg, top5.avg, cls_losses.avg, at_losses.avg]

        print('[clean]Prec@1: {:.2f}'.format(acc_clean[0]))
        print('[bad]Prec@1: {:.2f}'.format(acc_bd[0]))

        # save training progress
        log_root = self.args.log_root + '/nad_results.csv'
        test_process.append(
            (epoch, acc_clean[0], acc_bd[0], acc_bd[2], acc_bd[3]))
        df = pd.DataFrame(test_process, columns=(
        "epoch", "test_clean_acc", "test_bad_acc", "test_bad_cls_loss", "test_bad_at_loss"))
        df.to_csv(log_root, mode='a', index=False, encoding='utf-8')

        return acc_clean, acc_bd

    def adjust_learning_rate(self, optimizer, epoch, lr):
        if epoch < 2:
            lr = lr
        elif epoch < 20:
            lr = 0.01
        elif epoch < 30:
            lr = 0.0001
        else:
            lr = 0.0001
        print('epoch: {}  lr: {:.4f}'.format(epoch, lr))
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr


    def train(self, train_loader):
        print('finished student model init...')
        self.teacher_model.eval()

        nets = {'snet': self.student_model, 'tnet': self.teacher_model}

        for param in self.teacher_model.parameters():
            param.requires_grad = False

        # initialize optimizer
        optimizer = torch.optim.SGD(self.student_model.parameters(),
                                    lr=self.args.lr,
                                    momentum=self.args.nad_momentum,
                                    weight_decay=self.args.nad_weight_decay,
                                    nesterov=True)

        # define loss functions
        if self.device=='cuda' :
            criterionCls = nn.CrossEntropyLoss().cuda()
            criterionAT = AT(self.args.nad_p)
        else:
            criterionCls = nn.CrossEntropyLoss()
            criterionAT = AT(self.args.nad_p)

        print('----------- Train Initialization --------------')
        for epoch in range(0, self.args.epochs):

            self.adjust_learning_rate(optimizer, epoch, self.args.lr)
            criterions = {'criterionCls': criterionCls, 'criterionAT': criterionAT}
            self.train_step(train_loader, nets, optimizer, criterions, epoch+1)



