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
from cf import ContinuousForgetting 


from RNR_update import RNR

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
    #poison size fixed 4500 
    #3600 poison samples + 900 clean samples 
    #80% --- poison samples 3600 
    #rnr clean data req= 900 
    #45500-900clean +900 poisoned
    #45500 cleaned 
    
    #return clean_samples[cleanDataReqLen:]+poison_samples[noOfPoison:], poison_samples[:noOfPoison]+ clean_samples[:cleanDataReqLen],clean_samples+ poison_samples[noOfPoison:], clean_samples+ poison_samples
    #return clean_samples[cleanDataReqLen:]+poison_samples[4400:], poison_samples[:noOfPoison]+ clean_samples[:20],clean_samples+ poison_samples[noOfPoison:], clean_samples+ poison_samples 
    return clean_samples, poison_samples

def data_mix(clean_samples,poison_samples, poison_ratio):

    noOfPoison=int(len(poison_samples)*poison_ratio)
    cleanDataReqLen = len(poison_samples) - noOfPoison
    print('noOfPoison - ',noOfPoison,'totalNoPoisonData - ',len(poison_samples), 'clean data included with poisons', cleanDataReqLen)
    return clean_samples[cleanDataReqLen:]+poison_samples[noOfPoison:], poison_samples[:noOfPoison]+ clean_samples[:cleanDataReqLen]



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

def transformer():
    transforms_list = []
    transforms_list.append(transforms.ToPILImage())
    transforms_list.append(transforms.Resize((arg.input_height, arg.input_width)))
   
    transforms_list.append(transforms.RandomCrop((arg.input_height, arg.input_width), padding=4))
    if arg.dataset == "cifar10":
        transforms_list.append(transforms.RandomHorizontalFlip())
    transforms_list.append(transforms.ToTensor())

    return transforms.Compose(transforms_list)


def get_loader(data):
    tf_compose_finetuning = transformer()
    clean_data_tf = Dataset_npy(full_dataset=data, transform=tf_compose_finetuning)
    return DataLoader(dataset=clean_data_tf, batch_size=arg.batch_size, shuffle=True)

def main():
    global arg
    arg = args.get_args()

    # Dataset
    poison_ratio=arg.poison_ratio
    clean_data, poison_data = load_dataset(arg)
    #mix_clean, mix_poison = data_mix(clean_data, poison_data,poison_ratio)
    # Dataset
    # folder_path = os.path.join('./saved/separated_samples', 'poison_rate_'+str(arg.poison_rate), arg.dataset, arg.model, arg.trigger_type+'_'+str(arg.clean_ratio)+'_'+str(arg.poison_ratio))
    mix_clean=clean_data
    mix_poison=poison_data

    clean_data_loader = get_loader(mix_clean)
    poison_data_loader = get_loader(mix_poison)

    
    # full_data_tf = Dataset_npy(full_dataset=full_data, transform=tf_compose_unlearning)
    # full_data_loader = DataLoader(dataset=full_data_tf, batch_size=arg.batch_size, shuffle=True)

    # tf_compose_unlearning = transforms.Compose(transforms_list)
    # # data_path_poison = os.path.join(folder_path, 'poison_samples.npy')
    # # isolate_poison_data = np.load(data_path_poison, allow_pickle=True)
    # poison_data_tf = Dataset_npy(full_dataset=isolate_other_data, transform=tf_compose_unlearning)
    # isolate_other_data_loader = DataLoader(dataset=poison_data_tf, batch_size=arg.batch_size, shuffle=True)

    testloader_clean, testloader_bd = get_dataloader_test(arg)

    # Prepare model, optimizer, scheduler
    model = get_network(arg)
    model = torch.nn.DataParallel(model)
    ##optimizer = torch.optim.SGD(model.parameters(), lr=arg.lr, momentum=0.9, weight_decay=5e-4)
    checkpoint = torch.load(arg.checkpoint_load)
    print("Continue training...")
    model.load_state_dict(checkpoint['model'])
    criterion = nn.CrossEntropyLoss()

    if arg.unlearn_type=='dbr':
        dbr_unlearning = DBRUnlearning(model, criterion, arg)
        dbr_unlearning.unlearn(testloader_clean, testloader_bd, poison_data_loader, clean_data_loader)
    elif arg.unlearn_type=='abl':
        abl_unlearning = ABLUnlearning(model, criterion, isolate_poison_data_loader, isolate_clean_data_loader, arg, arg.device)
        abl_unlearning.unlearn(arg, testloader_clean, testloader_bd)
    elif arg.unlearn_type=='rnr':
        #trainloader = get_dataloader_train(arg)
        model_rnr = get_network(arg)
        model_rnr = torch.nn.DataParallel(model_rnr)
        rnr_learning = RNR(model_rnr, criterion, arg)
        rnr_learning.unlearn(clean_data_loader, testloader_clean, testloader_bd)
    elif arg.unlearn_type=='cfu':
        cf = ContinuousForgetting(arg, criterion)
        cf.relearn(10, model, isolate_clean_data_loader, testloader_clean, testloader_bd)
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
