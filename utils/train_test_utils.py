import numpy as np
import pandas as pd
import torch
from torch import nn
import ssd as ssd
from .utils import accuracy, progress_bar, normalization,AverageMeter
from .dataloader_bd import get_dataloader_train


def learning_rate_unlearning(optimizer, epoch, opt):
    if opt.unlearn_type=='abl':
        if epoch < opt.unlearning_epochs:
            lr = 0.0005
        else:
            lr = 0.0001
    if opt.unlearn_type=='dbr':        
        lr = 0.0001
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def learning_rate_finetuning(optimizer, epoch, opt):
    if epoch < 40:
        lr = 0.1
    elif epoch < 60:
        lr = 0.01
    else:
        lr = 0.001
    print('epoch: {}  lr: {:.4f}'.format(epoch, lr))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def train_step_unlearning(opt, train_loader, model_ascent, optimizer, criterion, epoch):
    model_ascent.train()

    total_clean, total_clean_correct = 0, 0

    if opt.unlearn_type=='abl':
        losses = AverageMeter()
        top1 = AverageMeter()
        top5 = AverageMeter()

    for idx, (img, target, flag) in enumerate(train_loader, start=1):
        if opt.unlearn_type=='dbl':
            img = normalization(opt, img)
            if opt.device=='cuda':
                img = img.cuda()
                target = target.cuda()

            output = model_ascent(img)
            loss = criterion(output, target)

            optimizer.zero_grad()
            (-loss).backward()  # Gradient ascent training
            optimizer.step()

            total_clean_correct += torch.sum(torch.argmax(output[:], dim=1) == target[:])
            total_clean += img.shape[0]
            avg_acc_clean = total_clean_correct * 100.0 / total_clean
            progress_bar(idx, len(train_loader),
                        'Epoch: %d | Loss: %.3f | Train ACC: %.3f%% (%d/%d)' % (
                        epoch, loss / (idx + 1), avg_acc_clean, total_clean_correct, total_clean))
        elif opt.unlearn_type=='abl':
            if opt.device=='cuda':
                img = img.cuda()
                target = target.cuda()

            output = model_ascent(img)

            loss = criterion(output, target)

            prec1, prec5 = accuracy(output, target, topk=(1, 5))
            losses.update(loss.item(), img.size(0))
            top1.update(prec1.item(), img.size(0))
            top5.update(prec5.item(), img.size(0))

            optimizer.zero_grad()
            (-loss).backward()  # Gradient ascent training
            optimizer.step()

            if idx % opt.print_freq == 0:
                print('Epoch[{0}]:[{1:03}/{2:03}] '
                    'loss:{losses.val:.4f}({losses.avg:.4f})  '
                    'prec@1:{top1.val:.2f}({top1.avg:.2f})  '
                    'prec@5:{top5.val:.2f}({top5.avg:.2f})'.format(epoch, idx, len(train_loader), losses=losses, top1=top1, top5=top5))



def train_step_relearning(opt, train_loader, model_ascent, optimizer, criterion, epoch):
    model_ascent.train()

    total_clean, total_clean_correct = 0, 0

    for idx, (img, target, flag) in enumerate(train_loader, start=1):
        img = normalization(opt, img)
        if opt.device=='cuda':
            img = img.cuda()
            target = target.cuda()

        output = model_ascent(img)
        loss = criterion(output, target)

        optimizer.zero_grad()
        loss.backward()  # Gradient ascent training
        optimizer.step()

        total_clean_correct += torch.sum(torch.argmax(output[:], dim=1) == target[:])
        total_clean += img.shape[0]
        avg_acc_clean = total_clean_correct * 100.0 / total_clean
        progress_bar(idx, len(train_loader),
                     'Epoch: %d | Loss: %.3f | Train ACC: %.3f%% (%d/%d)' % (
                     epoch, loss / (idx + 1), avg_acc_clean, total_clean_correct, total_clean))


def test_epoch(arg, testloader, model, criterion, epoch, word):
    model.eval()

    total_clean = 0
    total_clean_correct, total_robust_correct = 0, 0
    test_loss = 0
    
    for i, (inputs, labels, gt_labels, isCleans) in enumerate(testloader):
        inputs = normalization(arg, inputs)  # Normalize
        inputs, labels, gt_labels = inputs.to(arg.device), labels.to(arg.device), gt_labels.to(arg.device)
        outputs = model(inputs)
        loss = criterion(outputs, labels)

        test_loss += loss.item()
        total_clean_correct += torch.sum(torch.argmax(outputs[:], dim=1) == labels[:])
        total_robust_correct += torch.sum(torch.argmax(outputs[:], dim=1) == gt_labels[:])
        total_clean += inputs.shape[0]
        avg_acc_clean = total_clean_correct * 100.0 / total_clean
        avg_acc_robust = total_robust_correct * 100.0 / total_clean
        if word == 'clean':
            progress_bar(i, len(testloader), 'Epoch: %d | Loss: %.3f | Test %s ACC: %.3f%% (%d/%d)' % (
                epoch, test_loss / (i + 1), word, avg_acc_clean, total_clean_correct, total_clean))
        if word == 'bd':
            progress_bar(i, len(testloader), 'Epoch: %d | Loss: %.3f | ASR: %.3f%% (%d/%d) | R-ACC: %.3f%% (%d/%d)' % (
                epoch, test_loss / (i + 1), avg_acc_clean, total_clean_correct, total_clean, avg_acc_robust,
                total_robust_correct, total_clean))
    return test_loss / (i + 1), avg_acc_clean, avg_acc_robust

def test(opt, test_clean_loader, test_bad_loader, model_ascent, criterion, epoch):
    test_process = []
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    model_ascent.eval()

    for idx, (img, target,gt_label,flag) in enumerate(test_clean_loader, start=1):
        if opt.device=='cuda':
            img = img.cuda()
            target = target.cuda()

        with torch.no_grad():
            output = model_ascent(img)
            loss = criterion(output, target)

        prec1, prec5 = accuracy(output, target, topk=(1, 5))
        losses.update(loss.item(), img.size(0))
        top1.update(prec1.item(), img.size(0))
        top5.update(prec5.item(), img.size(0))

    acc_clean = [top1.avg, top5.avg, losses.avg]

    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    for idx, (img, target,gt_label,flag) in enumerate(test_bad_loader, start=1):
        if opt.device=='cuda':
            img = img.cuda()
            target = target.cuda()

        with torch.no_grad():
            output = model_ascent(img)
            loss = criterion(output, target)

        prec1, prec5 = accuracy(output, target, topk=(1, 5))
        losses.update(loss.item(), img.size(0))
        top1.update(prec1.item(), img.size(0))
        top5.update(prec5.item(), img.size(0))

    acc_bd = [top1.avg, top5.avg, losses.avg]

    print('[Clean] Prec@1: {:.2f}, Loss: {:.4f}'.format(acc_clean[0], acc_clean[2]))
    print('[Bad] Prec@1: {:.2f}, Loss: {:.4f}'.format(acc_bd[0], acc_bd[2]))

    # save training progress
    log_root = opt.log_root + '/ABL_unlearning.csv'
    test_process.append(
        (epoch, acc_clean[0], acc_bd[0], acc_clean[2], acc_bd[2]))
    df = pd.DataFrame(test_process, columns=("Epoch", "Test_clean_acc", "Test_bad_acc",
                                             "Test_clean_loss", "Test_bad_loss"))
    df.to_csv(log_root, mode='a', index=False, encoding='utf-8')

    return acc_clean, acc_bd

def load_dataset(arg):
    trainloader = get_dataloader_train(arg)
    clean_samples, poison_samples = [], []
    for i, (inputs, labels, gt_labels, isCleans) in enumerate(trainloader):
        if i % 1000 == 0:
            print("Processing samples:", i)
        inputs1, inputs2 = inputs[0], inputs[2]

        for j in range(inputs1.size(0)):  # Loop over all items in the batch
            img = inputs1[j]
            img = img.squeeze()
            target = labels[j].squeeze()
            img_np = (img * 255).cpu().numpy().astype(np.uint8)  # If you need to process or save the image
            img_np = np.transpose(img_np, (1, 2, 0))  # Adjust channel order
            target_np = target.cpu().numpy()

            # Check each element in isCleans
            if isCleans[j]:  # Accessing each element separately
                flag = 0
                clean_samples.append((img_np, target_np, flag))
            else:
                flag = 2
                poison_samples.append((img_np, target_np, flag))
    noOfPoison=int(len(poison_samples)*arg.poison_ratio)
    print('noOfPoison - ',noOfPoison,'totalNoPoisonData - ',len(poison_samples))
    
    return clean_samples, poison_samples[:noOfPoison],clean_samples+ poison_samples[noOfPoison:], clean_samples+ poison_samples


def train_step_finetuing(opt, train_loader, model_ascent, optimizer, criterion, epoch):
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    model_ascent.train()

    for idx, (img, target,gt_label) in enumerate(train_loader, start=1):
        if opt.device=='cuda':
            img = img.cuda()
            target = target.cuda()

        output = model_ascent(img)

        loss = criterion(output, target)

        prec1, prec5 = accuracy(output, target, topk=(1, 5))
        losses.update(loss.item(), img.size(0))
        top1.update(prec1.item(), img.size(0))
        top5.update(prec5.item(), img.size(0))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if idx % opt.print_freq == 0:
            print('Epoch[{0}]:[{1:03}/{2:03}] '
                  'loss:{losses.val:.4f}({losses.avg:.4f})  '
                  'prec@1:{top1.val:.2f}({top1.avg:.2f})  '
                  'prec@5:{top5.val:.2f}({top5.avg:.2f})'.format(epoch, idx, len(train_loader), losses=losses, top1=top1, top5=top5))
            
def train_epoch_rnr(arg, trainloader, model, optimizer, scheduler, criterion, epoch):
    model.train()

    total_clean = 0
    total_clean_correct, total_robust_correct = 0, 0
    train_loss = 0

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
            outputs = model(clean_inputs)
            loss = criterion(outputs, clean_labels)

            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

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
    scheduler.step()

    return train_loss / (i + 1), avg_acc_clean, avg_acc_robust


def trainable_params_(m):
    return [p for p in m.parameters() if p.requires_grad]


def train_cf(opt, model, epochs, train_loader):

    optimizer = torch.optim.SGD(trainable_params_(model), lr=opt.lr, momentum=0.9, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=opt.schedule, gamma=opt.gamma)
    criterion = nn.CrossEntropyLoss()
    model.train()

    
    for epoch in range(epochs):
        for param_group in optimizer.param_groups:
            print(f"Epoch {epoch}, Learning Rate: {param_group['lr']}")
        epoch_loss = 0.0
        correct = 0
        total = 0
        for data, targets, flg in train_loader:
            if opt.device=='cuda':
                data = data.cuda()
                targets = targets.cuda()
            output = model(data)

            loss = criterion(output, targets)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            _, predicted = output.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
        
        scheduler.step()
        train_accuracy = 100. * correct / total
        print(f"Epoch {epoch+1}, Loss: {epoch_loss / len(train_loader)}, Train Accuracy: {train_accuracy}%")
        
            

def getRetrainLayers(m, name, ret):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.Linear):
        ret.append((m, name))
        #print(name)
    for child_name, child in m.named_children():
        getRetrainLayers(child, f'{name}.{child_name}', ret)
    return ret


def resetFinalResnet(model, num_retrain):
    for param in model.parameters():
        param.requires_grad = False
    done = 0
    ret = getRetrainLayers(model, 'M', [])
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

def ssd_tuning(model,
    forget_train_dl,
    dampening_constant,
    selection_weighting,
    full_train_dl,
    device):
    parameters = {
        "lower_bound": 1,  # unused
        "exponent": 1,  # unused
        "magnitude_diff": None,  # unused
        "min_layer": -1,  # -1: all layers are available for modification
        "max_layer": -1,  # -1: all layers are available for modification
        "forget_threshold": 1,  # unused
        "dampening_constant": dampening_constant,  # Lambda from paper
        "selection_weighting": selection_weighting,  # Alpha from paper
    }

    # load the trained model
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=1e-4)
    pdr = ssd.ParameterPerturber(model, optimizer, device, parameters)

    model = model.eval()

    # Calculation of the forget set importances
    sample_importances = pdr.calc_importance(forget_train_dl)

    # Calculate the importances of D (see paper); this can also be done at any point before forgetting.
    original_importances = pdr.calc_importance(full_train_dl)

    # Dampen selected parameters
    pdr.modify_weight(original_importances, sample_importances)
    return model


