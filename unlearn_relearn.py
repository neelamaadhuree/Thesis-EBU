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
from cfn_unlearning import CFNUnlearning
from ibau_unlearning import IBAUUnlearning
from ANP_pruning import ANPPruning
from ANP_mask import ANPMask
from nad import NAD
from teacher_finetuning import TeacherFineTuning
import models
from torch.utils.data import TensorDataset




from RNR_update import RNR

from random import shuffle
from utils import args
from utils.network import get_network
from utils.dataloader_bd import get_dataloader_train, get_dataloader_test, Dataset_npy
from test_model import test_epoch, test_epoch_nad
from models.resnet_cifar10_nad import resnet18_nad


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

# 45500 clean samples before , 
# poison ratio 80% -- no of poisoned =3600 , clean data req len = 900 
# clean samples = [44600clean+900 poisoned]

def ssd_tuning(model,
    forget_train_dl,
    dampening_constant,
    selection_weighting,
    full_train_dl,
    device, 
    args):
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
    sample_importances = pdr.calc_importance(args, forget_train_dl)

    # Calculate the importances of D (see paper); this can also be done at any point before forgetting.
    original_importances = pdr.calc_importance(args, full_train_dl)

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
    # Dataset
    # folder_path = os.path.join('./saved/separated_samples', 'poison_rate_'+str(arg.poison_rate), arg.dataset, arg.model, arg.trigger_type+'_'+str(arg.clean_ratio)+'_'+str(arg.poison_ratio))
    #mix_clean=clean_data
    #mix_poison=poison_data



    
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

    f_name = arg.log

    if arg.unlearn_type=='dbr':
        clean_data_loader, poison_data_loader,_ = get_mixed_data(poison_ratio, clean_data, poison_data)
        dbr_unlearning = DBRUnlearning(model, criterion, arg)
        dbr_unlearning.unlearn(testloader_clean, testloader_bd, poison_data_loader, clean_data_loader)
    elif arg.unlearn_type=='abl':
        clean_data_loader, poison_data_loader,_ = get_mixed_data(poison_ratio, clean_data, poison_data)
        abl_unlearning = ABLUnlearning(model, criterion, poison_data_loader, clean_data_loader, arg, arg.device)
        abl_unlearning.unlearn(arg, testloader_clean, testloader_bd)
    elif arg.unlearn_type=='rnr':
        #trainloader = get_dataloader_train(arg)
        clean_data_loader, poison_data_loader,_ = get_mixed_data(poison_ratio, clean_data, poison_data)
        model_rnr = get_network(arg)
        model_rnr = torch.nn.DataParallel(model_rnr)
        rnr_learning = RNR(model_rnr, criterion, arg)
        rnr_learning.unlearn(clean_data_loader, testloader_clean, testloader_bd)
    elif arg.unlearn_type=='cfu':
        cf = ContinuousForgetting(arg, criterion)
        clean_data_loader, poison_data_loader,_ = get_mixed_data(poison_ratio, clean_data, poison_data)
        cf.relearn(30, model, clean_data_loader, testloader_clean, testloader_bd)
    elif arg.unlearn_type=='ssd':        

        #clean_data_loader = get_loader(clean_data)
        #poison_data_loader = get_loader(poison_data)
        clean_data_loader, poison_data_loader,full_data_loader = get_mixed_data(poison_ratio, clean_data, poison_data)
        csvFile = open(f_name, 'a', newline='')
        writer = csv.writer(csvFile)
        runTest(testloader_clean, testloader_bd, model, criterion, writer)
        model=ssd_tuning(model,poison_data_loader,1.0,45,clean_data_loader, arg.device, arg)
        runTest(testloader_clean, testloader_bd, model, criterion, writer)
        csvFile.close()
    elif arg.unlearn_type=='cfn':
        clean_data_loader, poison_data_loader,_ = get_mixed_data(poison_ratio, clean_data[:20000], poison_data)
        cfn_unlearning = CFNUnlearning(model, criterion, arg)
        cfn_unlearning.unlearn(testloader_clean, testloader_bd, clean_data_loader)
    elif arg.unlearn_type=='ibau':
        csvFile = open(f_name, 'a', newline='')
        writer = csv.writer(csvFile)
        runTest(testloader_clean, testloader_bd, model, criterion, writer)
        clean_data_loader, poison_data_loader,_ = get_mixed_data(poison_ratio, clean_data[:1000], poison_data)
        test_set, unl_set = get_test_and_unlearn_dataset(testloader_clean)
        ibau_unlearning = IBAUUnlearning(model, arg)
        unlloader = torch.utils.data.DataLoader(unl_set, batch_size=args.batch_size, shuffle=False, num_workers=2)
        ibau_unlearning.unlearn(unlloader, test_set) 
        runTest(testloader_clean, testloader_bd, model, criterion, writer)
        csvFile.close()
    elif arg.unlearn_type == 'anp':
        csvFile = open(f_name, 'a', newline='')
        writer = csv.writer(csvFile)
        runTest(testloader_clean, testloader_bd, model, criterion, writer)
        clean_data_loader, poison_data_loader,_ = get_mixed_data(poison_ratio, clean_data[:1000], poison_data)
        model = getattr(models, 'resnet18_anp')(num_classes=10, norm_layer=models.NoisyBatchNorm2d)
        checkpoint = torch.load(arg.checkpoint_load)
        anpMask = ANPMask(arg)

        anpMask.load_state_dict(model, orig_state_dict=checkpoint)
        print("Starting Masking...")

        anpMask.set_model(model)
        anpMask.mask(clean_data_loader, testloader_clean , testloader_bd)

        print("Staring pruning...")
        model = getattr(models, 'resnet18_anp')(num_classes=10)
        checkpoint = torch.load(arg.checkpoint_load)
        anpMask.load_state_dict(model, orig_state_dict=checkpoint)
        model = model.to(arg.device)
        criterion = nn.CrossEntropyLoss()
        anpPruning = ANPPruning(arg, model=model)
        anpPruning.prune(testloader_clean , testloader_bd)
        runTest(testloader_clean, testloader_bd, model, criterion, writer)
    elif arg.unlearn_type == 'nad':

        clean_data_loader, poison_data_loader,_ = get_mixed_data(poison_ratio, clean_data[:3000], poison_data)

        teacher_model = getResnetNadModel(arg)
        csvFile = open(f_name, 'a', newline='')
        writer = csv.writer(csvFile)
        runTestNad(testloader_clean, testloader_bd, teacher_model, criterion, writer)

        tft = TeacherFineTuning(teacher_model, arg)
        fineTunedModel = tft.fineTune(clean_data_loader)

        runTestNad(testloader_clean, testloader_bd, fineTunedModel, criterion, writer)
        print("Teacher Fine Tuning Complete")

        print("Student Model Test")

        student_model = getResnetNadModel(arg)

        runTestNad(testloader_clean, testloader_bd, student_model, criterion, writer)

        nad = NAD(arg, fineTunedModel, student_model)
        nad.train(clean_data_loader)
        print("Student Fine Tuning Complete")
        runTestNad(testloader_clean, testloader_bd, student_model, criterion, writer)
        csvFile.close()



def getResnetNadModel(arg):
    model = resnet18_nad()
    model.to(arg.device)
    model = torch.nn.DataParallel(model)
    checkpoint = torch.load(arg.checkpoint_load)
    model.load_state_dict(checkpoint['model'])
    return model

        

def runTest(testloader_clean, testloader_bd, model, criterion, writer):
    writer.writerow(['Epoch', 'Test_ACC', 'Test_ASR'])
    test_loss_cl, test_acc_cl, _ = test_epoch(arg, testloader_clean, model, criterion, 0, 'clean')
    test_loss_bd, test_acc_bd, test_acc_robust = test_epoch(arg, testloader_bd, model, criterion, 0, 'bd')
    writer.writerow([-1, test_acc_cl.item(), test_acc_bd.item()])

def runTestNad(testloader_clean, testloader_bd, model, criterion, writer):
    writer.writerow(['Epoch', 'Test_ACC', 'Test_ASR'])
    test_loss_cl, test_acc_cl, _ = test_epoch_nad(arg, testloader_clean, model, criterion, 0, 'clean')
    test_loss_bd, test_acc_bd, test_acc_robust = test_epoch_nad(arg, testloader_bd, model, criterion, 0, 'bd')
    writer.writerow([-1, test_acc_cl.item(), test_acc_bd.item()])



def get_mixed_data(poison_ratio, clean_data, poison_data):
    mix_clean, mix_poison = data_mix(clean_data, poison_data,poison_ratio)
    clean_data_loader = get_loader(mix_clean)
    poison_data_loader = get_loader(mix_poison)
    full_data_loader=get_loader(clean_data+poison_data)
    return clean_data_loader,poison_data_loader, full_data_loader

def get_test_and_unlearn_dataset(testloader):
    images_list, labels_list = [], []
    for index, (images, labels, gt_labeld, isCleans) in enumerate(testloader):
        images_list.append(images)
        labels_list.append(labels)
    unl_set = TensorDataset(images_list[:5000],labels_list[:5000])
    test_set = TensorDataset(images_list[5000:],labels_list[5000:])
    return test_set, unl_set

if __name__ == '__main__':
    main()
