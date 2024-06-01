# python unlearn_relearn.py --dataset cifar10 --model resnet18 --trigger_type signalTrigger --epochs 20 --clean_ratio 0.80 --poison_ratio 0.20 --checkpoint_load ./saved/backdoored_model/poison_rate_0.1/withTrans/cifar10/resnet18/signalTrigger/49.tar --checkpoint_save ./saved/backdoored_model/poison_rate_0.1/withTrans/cifar10/resnet18/signalTrigger/49_unlearn_purify.py --log ./saved/backdoored_model/poison_rate_0.1/withTrans/cifar10/resnet18/signalTrigger/unlearn_purify.csv --unlearn_type dbr
# python unlearn_relearn.py --dataset cifar10 --model resnet18 --trigger_type signalTrigger --epochs 20 --clean_ratio 0.80 --poison_ratio 0.20 --checkpoint_load ./saved/backdoored_model/poison_rate_0.1/withTrans/cifar10/resnet18/signalTrigger/49.tar --unlearn_type abl 
# python unlearn_relearn.py --dataset cifar10 --model resnet18 --trigger_type signalTrigger --epochs 20  --unlearn_type rnr --clean_ratio 0.80 --poison_ratio 0.20 --checkpoint_load ./saved/backdoored_model/poison_rate_0.1/withTrans/cifar10/resnet18/signalTrigger/49.tar
# python unlearn_relearn.py --dataset cifar10 --model resnet18 --trigger_type signalTrigger --epochs 20  --unlearn_type cfu --clean_ratio 0.80 --poison_ratio 0.20 --checkpoint_load ./saved/backdoored_model/poison_rate_0.1/withTrans/cifar10/resnet18/signalTrigger/199.tar
# python unlearn_relearn.py --dataset cifar10 --model resnet18 --trigger_type signalTrigger --epochs 20  --unlearn_type ssd --clean_ratio 0.80 --poison_ratio 0.20 --checkpoint_load ./saved/backdoored_model/poison_rate_0.1/withTrans/cifar10/resnet18/signalTrigger/199.tar --log ./saved/backdoored_model/poison_rate_0.1/withTrans/cifar10/resnet18/signalTrigger/unlearn_ssd.csv 

import numpy as np
import csv
from PIL import Image

import torch
from torch import nn
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
import ssd as ssd
from abl_unlearning import ABLUnlearning 
from dbr_unlearning import DBRUnlearning 

from rnr import RNR

from random import shuffle
from utils import args
from utils.network import get_network
from utils.dataloader_bd import get_dataloader_train, get_dataloader_test, Dataset_npy
from test_model import test_epoch


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


def trainable_params_(m):
    return [p for p in m.parameters() if p.requires_grad]


def train_cf(opt, model, epochs, train_loader):

    optimizer = torch.optim.SGD(trainable_params_(model), lr=arg.lr, momentum=0.9, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=arg.schedule, gamma=arg.gamma)
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
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    pdr = ssd.ParameterPerturber(model, optimizer, device, parameters)

    model = model.eval()

    # Calculation of the forget set importances
    sample_importances = pdr.calc_importance(forget_train_dl)

    # Calculate the importances of D (see paper); this can also be done at any point before forgetting.
    original_importances = pdr.calc_importance(full_train_dl)

    # Dampen selected parameters
    pdr.modify_weight(original_importances, sample_importances)
    return model

def main():
    global arg
    arg = args.get_args()
    trainloader = get_dataloader_train(arg)

    # Dataset
    isolate_clean_data,isolate_poison_data,isolate_other_data,full_data = load_dataset(arg)
    shuffle(full_data)
    # Dataset
    # folder_path = os.path.join('./saved/separated_samples', 'poison_rate_'+str(arg.poison_rate), arg.dataset, arg.model, arg.trigger_type+'_'+str(arg.clean_ratio)+'_'+str(arg.poison_ratio))

    transforms_list = []
    transforms_list.append(transforms.ToPILImage())
    transforms_list.append(transforms.Resize((arg.input_height, arg.input_width)))
    # if arg.dataset == "imagenet":
    #     transforms_list.append(transforms.RandomRotation(20))
    #     transforms_list.append(transforms.RandomHorizontalFlip(0.5))
    # else:
    transforms_list.append(transforms.RandomCrop((arg.input_height, arg.input_width), padding=4))
    if arg.dataset == "cifar10":
        transforms_list.append(transforms.RandomHorizontalFlip())
    transforms_list.append(transforms.ToTensor())

    tf_compose_finetuning = transforms.Compose(transforms_list)
    # data_path_clean = os.path.join(folder_path, 'clean_samples.npy')
    # isolate_clean_data = np.load(data_path_clean, allow_pickle=True)
    clean_data_tf = Dataset_npy(full_dataset=isolate_clean_data, transform=tf_compose_finetuning)
    isolate_clean_data_loader = DataLoader(dataset=clean_data_tf, batch_size=arg.batch_size, shuffle=True)

    tf_compose_unlearning = transforms.Compose(transforms_list)
    # data_path_poison = os.path.join(folder_path, 'poison_samples.npy')
    # isolate_poison_data = np.load(data_path_poison, allow_pickle=True)
    poison_data_tf = Dataset_npy(full_dataset=isolate_poison_data, transform=tf_compose_unlearning)
    isolate_poison_data_loader = DataLoader(dataset=poison_data_tf, batch_size=arg.batch_size, shuffle=True)

    
    full_data_tf = Dataset_npy(full_dataset=full_data, transform=tf_compose_unlearning)
    full_data_loader = DataLoader(dataset=full_data_tf, batch_size=arg.batch_size, shuffle=True)

    tf_compose_unlearning = transforms.Compose(transforms_list)
    # data_path_poison = os.path.join(folder_path, 'poison_samples.npy')
    # isolate_poison_data = np.load(data_path_poison, allow_pickle=True)
    poison_data_tf = Dataset_npy(full_dataset=isolate_other_data, transform=tf_compose_unlearning)
    isolate_other_data_loader = DataLoader(dataset=poison_data_tf, batch_size=arg.batch_size, shuffle=True)

    testloader_clean, testloader_bd = get_dataloader_test(arg)

    # Prepare model, optimizer, scheduler
    model = get_network(arg)
    model = torch.nn.DataParallel(model)
    checkpoint = torch.load(arg.checkpoint_load)
    print("Continue training...")
    model.load_state_dict(checkpoint['model'])
    criterion = nn.CrossEntropyLoss()

    if arg.unlearn_type=='dbr':
        dbr_unlearning = DBRUnlearning(model, criterion, args)
        dbr_unlearning.unlearn(testloader_clean, testloader_bd, isolate_poison_data_loader, isolate_clean_data_loader)
    elif arg.unlearn_type=='abl':
        abl_unlearning = ABLUnlearning(model, criterion, isolate_other_data, args, args.device)
        abl_unlearning.unlearn()
    elif arg.unlearn_type=='rnr':
        rnr_learning = RNR(model, criterion, arg)
        rnr_learning.unlearn(trainloader, testloader_clean, testloader_bd)
    elif arg.unlearn_type=='cfu':
        k_layer = 10            
        f_name = arg.log
        csvFile = open(f_name, 'a', newline='')
        writer = csv.writer(csvFile)
        writer.writerow(['Epoch', 'Test_ACC', 'Test_ASR'])

        # Test the orginal performance of the model
        # test_loss_cl, test_acc_cl, _ = test_epoch(arg, testloader_clean, model, criterion, 0, 'clean')
        # test_loss_bd, test_acc_bd, test_acc_robust = test_epoch(arg, testloader_bd, model, criterion, 0, 'bd')
        # writer.writerow([-1, test_acc_cl.item(), test_acc_bd.item()])

        model = resetFinalResnet(model, k_layer)
        # do args to device if we have cudd
        train_cf(arg, model, arg.epochs, isolate_clean_data_loader)
        test_loss_cl, test_acc_cl, _ = test_epoch(arg, testloader_clean, model, criterion, 0, 'clean')
        test_loss_bd, test_acc_bd, test_acc_robust = test_epoch(arg, testloader_bd, model, criterion, 0, 'bd')
        cnt=-1
        writer.writerow([cnt+1, test_acc_cl.item(), test_acc_bd.item()])
        csvFile.close()
    elif arg.unlearn_type=='ssd':        
        f_name = arg.log
        csvFile = open(f_name, 'a', newline='')
        writer = csv.writer(csvFile)
        writer.writerow(['Epoch', 'Test_ACC', 'Test_ASR'])
        test_loss_cl, test_acc_cl, _ = test_epoch(arg, testloader_clean, model, criterion, 0, 'clean')
        test_loss_bd, test_acc_bd, test_acc_robust = test_epoch(arg, testloader_bd, model, criterion, 0, 'bd')
        writer.writerow([-1, test_acc_cl.item(), test_acc_bd.item()])
        model=ssd_tuning(model,isolate_poison_data_loader,1,10,full_data_loader, arg.device)
        test_loss_cl, test_acc_cl, _ = test_epoch(arg, testloader_clean, model, criterion, 0, 'clean')
        test_loss_bd, test_acc_bd, test_acc_robust = test_epoch(arg, testloader_bd, model, criterion, 0, 'bd')
        writer.writerow([1, test_acc_cl.item(), test_acc_bd.item()])
        csvFile.close()

if __name__ == '__main__':
    main()
