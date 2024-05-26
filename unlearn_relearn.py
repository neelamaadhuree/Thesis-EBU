# python unlearn_relearn.py --dataset cifar10 --model resnet18 --trigger_type signalTrigger --epochs 20 --clean_ratio 0.80 --poison_ratio 0.20 --checkpoint_load ./saved/backdoored_model/poison_rate_0.1/withTrans/cifar10/resnet18/signalTrigger/49.tar --checkpoint_save ./saved/backdoored_model/poison_rate_0.1/withTrans/cifar10/resnet18/signalTrigger/49_unlearn_purify.py --log ./saved/backdoored_model/poison_rate_0.1/withTrans/cifar10/resnet18/signalTrigger/unlearn_purify.csv --unlearn_type dbr
# python unlearn_relearn.py --dataset cifar10 --model resnet18 --trigger_type signalTrigger --epochs 20 --clean_ratio 0.80 --poison_ratio 0.20 --checkpoint_load ./saved/backdoored_model/poison_rate_0.1/withTrans/cifar10/resnet18/signalTrigger/49.tar --unlearn_type abl 
# python unlearn_relearn.py --dataset cifar10 --model resnet18 --trigger_type signalTrigger --epochs 20  --unlearn_type rnr --clean_ratio 0.80 --poison_ratio 0.20 --checkpoint_load ./saved/backdoored_model/poison_rate_0.1/withTrans/cifar10/resnet18/signalTrigger/49.tar
# python unlearn_relearn.py --dataset cifar10 --model resnet18 --trigger_type signalTrigger --epochs 20  --unlearn_type cfu --clean_ratio 0.80 --poison_ratio 0.20 --checkpoint_load ./saved/backdoored_model/poison_rate_0.1/withTrans/cifar10/resnet18/signalTrigger/49.tar --log ./saved/backdoored_model/poison_rate_0.1/withTrans/cifar10/resnet18/signalTrigger/unlearn_purify.csv
# python unlearn_relearn.py --dataset cifar10 --model resnet18 --trigger_type signalTrigger --epochs 20  --unlearn_type ssd --clean_ratio 0.80 --poison_ratio 0.20 --checkpoint_load ./saved/backdoored_model/poison_rate_0.1/withTrans/cifar10/resnet18/signalTrigger/49.tar --log ./saved/backdoored_model/poison_rate_0.1/withTrans/cifar10/resnet18/signalTrigger/unlearn_ssd.csv 

import os
from tqdm import tqdm
import csv

import torch
from torch import nn
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import ssd as ssd

from random import shuffle
from utils import args
from utils.utils import save_checkpoint,save_checkpoint_only
from utils.network import get_network
from utils.dataloader_bd import get_dataloader_train, get_dataloader_test, Dataset_npy
from utils.train_test_utils import *


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
    if arg.device=='cuda':
        model = torch.nn.DataParallel(model).cuda()
    else:
        model = torch.nn.DataParallel(model)
    if arg.unlearn_type=='abl':
        optimizer = torch.optim.SGD(model.parameters(), lr=arg.lr, momentum=0.9, weight_decay=1e-4)
    elif arg.unlearn_type=='dbr':
        optimizer = torch.optim.SGD(model.parameters(), lr=arg.lr, momentum=0.9, weight_decay=5e-4)


    checkpoint = torch.load(arg.checkpoint_load)
    print("Continue training...")
    model.load_state_dict(checkpoint['model'])
    start_epoch = 0

    # Training and Testing
    best_acc = 0
    criterion = nn.CrossEntropyLoss()

    if arg.unlearn_type=='dbr':
        # Write
        f_name = arg.log
        csvFile = open(f_name, 'a', newline='')
        writer = csv.writer(csvFile)
        writer.writerow(['Epoch', 'Test_ACC', 'Test_ASR'])

        # Test the orginal performance of the model
        test_loss_cl, test_acc_cl, _ = test_epoch(arg, testloader_clean, model, criterion, 0, 'clean')
        test_loss_bd, test_acc_bd, test_acc_robust = test_epoch(arg, testloader_bd, model, criterion, 0, 'bd')
        writer.writerow([-1, test_acc_cl.item(), test_acc_bd.item()])

        for epoch in tqdm(range(start_epoch, arg.epochs)):
            # Modify lr
            learning_rate_unlearning(optimizer, epoch, arg)

            # Unlearn
            train_step_unlearning(arg, isolate_poison_data_loader, model, optimizer, criterion, epoch)

            # Relearn
            train_step_relearning(arg, isolate_clean_data_loader, model, optimizer, criterion, epoch)

            test_loss_cl, test_acc_cl, _ = test_epoch(arg, testloader_clean, model, criterion, epoch, 'clean')
            test_loss_bd, test_acc_bd, test_acc_robust = test_epoch(arg, testloader_bd, model, criterion, epoch, 'bd')

            # Save the best model
            if test_acc_cl - test_acc_bd > best_acc:
                best_acc = test_acc_cl - test_acc_bd
                save_checkpoint_only(arg.checkpoint_save, model)

            writer.writerow([epoch, test_acc_cl.item(), test_acc_bd.item()])
        csvFile.close()
    elif arg.unlearn_type=='abl':
        if arg.finetuning_ascent_model == True:
            # this is to improve the clean accuracy of isolation model, you can skip this step
            print('----------- Finetuning isolation model --------------')
            for epoch in range(0, arg.finetuning_epochs):
                learning_rate_finetuning(optimizer, epoch, arg)
                train_step_finetuing(arg, isolate_other_data_loader, model, optimizer, criterion,
                                epoch + 1)
                test(arg, testloader_clean, testloader_bd, model, criterion, epoch + 1)


        print('----------- Model unlearning --------------')
        for epoch in range(0, arg.unlearning_epochs):
            learning_rate_unlearning(optimizer, epoch, arg)

            # train stage
            if epoch == 0:
                # test firstly
                test(arg, testloader_clean, testloader_bd, model, criterion, epoch)
            else:
                train_step_unlearning(arg, isolate_poison_data_loader, model, optimizer, criterion, epoch + 1)

            # evaluate on testing set
            print('testing the ascended model......')
            acc_clean, acc_bad = test(arg, testloader_clean, testloader_bd, model, criterion, epoch + 1)

            if arg.save:
                # save checkpoint at interval epoch
                if epoch + 1 % arg.interval == 0:
                    is_best = True
                    save_checkpoint({
                        'epoch': epoch + 1,
                        'state_dict': model.state_dict(),
                        'clean_acc': acc_clean[0],
                        'bad_acc': acc_bad[0],
                        'optimizer': optimizer.state_dict(),
                    }, epoch + 1, is_best, arg)

    elif arg.unlearn_type=='rnr':

        # Prepare model, optimizer, scheduler
        model = get_network(arg)
        # model = torch.nn.DataParallel(model).cuda()
        model = torch.nn.DataParallel(model)

        optimizer = torch.optim.SGD(model.parameters(), lr=arg.lr, momentum=0.9, weight_decay=5e-4)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=arg.schedule, gamma=arg.gamma)
        
        print("Training from scratch...")
        start_epoch = 0

        # Training and Testing
        best_acc = 0
        criterion = nn.CrossEntropyLoss()

        # Write
        save_folder_path = os.path.join('./saved/backdoored_model', 'poison_rate_0', 'withTrans', arg.dataset, arg.model, arg.trigger_type)
        if not os.path.exists(save_folder_path):
            os.makedirs(save_folder_path)
        arg.log = os.path.join(save_folder_path, 'withTrans.csv')
        f_name = arg.log
        csvFile = open(f_name, 'a', newline='')
        writer = csv.writer(csvFile)
        writer.writerow(
            ['Epoch', 'Train_Loss', 'Train_ACC',  'Train_R-ACC', 'Test_Loss_cl', 'Test_ACC', 'Test_Loss_bd',
            'Test_ASR', 'Test_R-ACC'])
        for epoch in tqdm(range(start_epoch, arg.epochs)):
            train_loss, train_acc, train_racc = train_epoch_rnr(arg, trainloader, model, optimizer, scheduler,
                                                                    criterion, epoch)
            test_loss_cl, test_acc_cl, _ = test_epoch(arg, testloader_clean, model, criterion, epoch, 'clean')
            test_loss_bd, test_acc_bd, test_acc_robust = test_epoch(arg, testloader_bd, model, criterion, epoch, 'bd')

            # Save in every epoch
            save_file_path = os.path.join(save_folder_path, str(epoch) + '.tar')
            save_checkpoint(save_file_path, epoch, model, optimizer, scheduler)

            writer.writerow(
                [epoch, train_loss, train_acc.item(), train_racc.item(), test_loss_cl, test_acc_cl.item(),
                test_loss_bd, test_acc_bd.item(), test_acc_robust.item()])
        csvFile.close()

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
