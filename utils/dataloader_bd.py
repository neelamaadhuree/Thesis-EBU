# Modified from https://github.com/bboylyg/NAD/blob/main/data_loader.py

import os
import csv
import random
import numpy as np
from PIL import Image
from tqdm import tqdm
import time
import pdb
import sys
from matplotlib import image as mlt
import cv2

import torch
import torch.utils.data as data
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as datasets

from utils import args
# from utils.SSBA.encode_image import bd_generator # if you run SSBA attack, please use this line

# Set random seed
def seed_torch(seed=1000):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
seed_torch()


# Obtain benign model (only used in the CL attack)
global arg
arg = args.get_args()

class TransformThree:
    def __init__(self, transform1, transform2, transform3):
        self.transform1 = transform1
        self.transform2 = transform2
        self.transform3 = transform3

    def __call__(self, inp):
        out1 = self.transform1(inp)
        out2 = self.transform2(inp)
        out3 = self.transform3(inp)
        return out1, out2, out3


class Dataset_npy(torch.utils.data.Dataset):
    def __init__(self, full_dataset=None, transform=None):
        self.dataset = full_dataset
        self.transform = transform
        self.dataLen = len(self.dataset)

    def __getitem__(self, index):
        image = self.dataset[index][0]
        label = self.dataset[index][1]
        flag = self.dataset[index][2]

        if self.transform:
            image = self.transform(image)
        # print(type(image), image.shape)
        return image, label, flag

    def __len__(self):
        return self.dataLen


def get_dataloader_train(opt):
    train = True
    if opt.dataset == "gtsrb":
        dataset = GTSRB(opt, train)
    elif opt.dataset == "cifar10":
        dataset = torchvision.datasets.CIFAR10(opt.data_root, train, download=True)
    elif opt.dataset == "cifar100":
        dataset = torchvision.datasets.CIFAR100(opt.data_root, train, download=True)
    else:
        raise Exception("Invalid dataset")
    
    # if opt.unlearn_type=='rnr':
    #     poison_rate=0
    # else:
    #     poison_rate=opt.poison_rate

    transform1, transform2, transform3 = get_transform(opt, train)
    train_data_bad = DatasetBD(opt, full_dataset=dataset, inject_portion=opt.poison_rate, transform=TransformThree(transform1, transform2, transform3),
                               mode='train')

    dataloader = torch.utils.data.DataLoader(train_data_bad, batch_size=opt.batch_size, num_workers=opt.num_workers,
                                             shuffle=True)
    return dataloader


def get_dataloader_test(opt):
    train = False
    if opt.dataset == "gtsrb":
        dataset = GTSRB(opt, train)
    elif opt.dataset == "cifar10":
        dataset = torchvision.datasets.CIFAR10(opt.data_root, train, download=True)
    elif opt.dataset == "cifar100":
        dataset = torchvision.datasets.CIFAR100(opt.data_root, train, download=True)
    else:
        raise Exception("Invalid dataset")

    if opt.unlearn_type=='dbr' or opt.unlearn_type=='rnr' or opt.unlearn_type=='cfu' or opt.unlearn_type=='ssd':
        transform = get_transform(opt, train)
    if opt.unlearn_type=='abl':
        transform = transforms.Compose([transforms.ToTensor()])
    
    test_data_clean = DatasetBD(opt, full_dataset=dataset, inject_portion=0, transform=transform, mode='test')
    test_data_bad = DatasetBD(opt, full_dataset=dataset, inject_portion=1, transform=transform, mode='test')

    # (apart from target label) bad test data
    test_clean_loader = torch.utils.data.DataLoader(dataset=test_data_clean, batch_size=opt.batch_size, shuffle=False)
    # all clean test data
    test_bad_loader = torch.utils.data.DataLoader(dataset=test_data_bad, batch_size=opt.batch_size, shuffle=False)

    return test_clean_loader, test_bad_loader


"""
    Methods: 
    - BadNet: 'squareTrigger', 'gridTrigger', 'fourCornerTrigger', 'randomPixelTrigger'
    - Blended: 'signalTrigger'
    - SIG: 'sigTrigger'

    Trigger Type: ['squareTrigger', 'gridTrigger', 'fourCornerTrigger', 'randomPixelTrigger', 'signalTrigger', 'trojanTrigger', 'signalTrigger_imagenet']
    Trigger Position: bottom right with distance=1.
    Trigger Size: 10% of the image height and width. 

    Target Type: ['all2one', 'all2all', 'cleanLabel']
    Target Label: a number, i.g. 0. 
"""


class DatasetBD(torch.utils.data.Dataset):
    def __init__(self, opt, full_dataset, inject_portion, transform=None, mode="train", distance=1):
        self.triggerGenerator = None # SSBA
        self.addTriggerGenerator(opt.trigger_type, opt.dataset) # SSBA
        self.dataset = self.addTrigger(full_dataset, opt.target_label, inject_portion, mode, distance,
                                       int(0.1 * opt.input_width), int(0.1 * opt.input_height), opt.trigger_type,
                                       opt.target_type)
        self.device = opt.device
        self.transform = transform

    def __getitem__(self, item):
        img = self.dataset[item][0]
        label = self.dataset[item][1]
        gt_label = self.dataset[item][2]
        isClean = self.dataset[item][3]
        img = Image.fromarray(img)
        img = self.transform(img)
        return img, label, gt_label, isClean

    def __len__(self):
        return len(self.dataset)

    def addTriggerGenerator(self, trigger_type, dataset): # SSBA
        if trigger_type == 'SSBA':
            self.triggerGenerator = bd_generator()

    def addTrigger(self, dataset, target_label, inject_portion, mode, distance, trig_w, trig_h, trigger_type,
                   target_type):
        print("Generating " + mode + " bad Imgs")
        
        # Obtain indexes of the samples to be poisoned
        # Under the same poisoning rate, the amount of poisoned samples are different in different types of attacks.
        if mode == 'train':
            if target_type == 'all2one':
                non_target_idx = []
                for i in range(len(dataset)):
                    if dataset[i][1] != target_label:
                        non_target_idx.append(i)
                non_target_idx = np.array(non_target_idx)
                perm_idx = np.random.permutation(len(non_target_idx))[0: int(len(non_target_idx) * inject_portion)]
                perm = non_target_idx[perm_idx]
            elif target_type == 'cleanLabel':
                target_idx = []
                for i in range(len(dataset)):
                    if dataset[i][1] == target_label:
                        target_idx.append(i)
                target_idx = np.array(target_idx)
                perm_idx = np.random.permutation(len(target_idx))[0: int(len(target_idx) * inject_portion)]
                perm = target_idx[perm_idx]
            elif target_type == 'all2all':
                perm = np.random.permutation(len(dataset))[0: int(len(dataset) * inject_portion)]
        elif mode == 'test':
            perm = np.random.permutation(len(dataset))[0: int(len(dataset) * inject_portion)]

        # dataset
        dataset_ = list()

        cnt = 0
        for i in tqdm(range(len(dataset))):
            data = dataset[i]

            if target_type == 'all2one':

                if mode == 'train':
                    img = np.array(data[0])
                    width = img.shape[0]
                    height = img.shape[1]
                    if i in perm:
                        # select trigger
                        img = self.selectTrigger(mode, img, data[1], width, height, distance, trig_w, trig_h, trigger_type)

                        # change target
                        # dataset_.append((img, target_label))
                        dataset_.append((img, target_label, data[1], False))
                        cnt += 1
                    else:
                        # dataset_.append((img, data[1]))
                        dataset_.append((img, data[1], data[1], True))

                else:
                    if data[1] == target_label:
                        continue

                    img = np.array(data[0])
                    width = img.shape[0]
                    height = img.shape[1]
                    if i in perm:
                        img = self.selectTrigger(mode, img, data[1], width, height, distance, trig_w, trig_h, trigger_type)

                        # dataset_.append((img, target_label))
                        dataset_.append((img, target_label, data[1], False))
                        cnt += 1
                    else:
                        # dataset_.append((img, data[1]))
                        dataset_.append((img, data[1], data[1], True))

            # all2all attack
            elif target_type == 'all2all':

                if mode == 'train':
                    img = np.array(data[0])
                    width = img.shape[0]
                    height = img.shape[1]
                    if i in perm:
                        img = self.selectTrigger(mode, img, data[1], width, height, distance, trig_w, trig_h, trigger_type)
                        target_ = self._change_label_next(data[1])

                        # dataset_.append((img, target_))
                        dataset_.append((img, target_, data[1], False))
                        cnt += 1
                    else:
                        # dataset_.append((img, data[1]))
                        dataset_.append((img, data[1], data[1], True))

                else:

                    img = np.array(data[0])
                    width = img.shape[0]
                    height = img.shape[1]
                    if i in perm:
                        img = self.selectTrigger(mode, img, data[1], width, height, distance, trig_w, trig_h, trigger_type)

                        target_ = self._change_label_next(data[1])
                        # dataset_.append((img, target_))
                        dataset_.append((img, target_, data[1], False))
                        cnt += 1
                    else:
                        # dataset_.append((img, data[1]))
                        dataset_.append((img, data[1], data[1], True))

            # clean label attack
            elif target_type == 'cleanLabel':

                if mode == 'train':
                    img = np.array(data[0])
                    width = img.shape[0]
                    height = img.shape[1]

                    if i in perm:
                        if data[1] == target_label:

                            img = self.selectTrigger(mode, img, data[1], width, height, distance, trig_w, trig_h, trigger_type)

                            # dataset_.append((img, data[1]))
                            dataset_.append((img, data[1], data[1], False))
                            cnt += 1

                        else:
                            # dataset_.append((img, data[1]))
                            dataset_.append((img, data[1], data[1], True))
                    else:
                        # dataset_.append((img, data[1]))
                        dataset_.append((img, data[1], data[1], True))

                else:
                    if data[1] == target_label:
                        continue

                    img = np.array(data[0])
                    width = img.shape[0]
                    height = img.shape[1]
                    if i in perm:
                        img = self.selectTrigger(mode, img, data[1], width, height, distance, trig_w, trig_h, trigger_type)

                        # dataset_.append((img, target_label))
                        dataset_.append((img, target_label, data[1], False))
                        cnt += 1
                    else:
                        # dataset_.append((img, data[1]))
                        dataset_.append((img, data[1], data[1], True))

        time.sleep(0.01)
        print(f"There are total {len(dataset)} images. " + "Injecting Over: " + str(cnt) + " Bad Imgs, " + str(
            len(dataset) - cnt) + " Clean Imgs")

        return dataset_

    def _change_label_next(self, label):
        num_cls = int(arg.num_classes)
        label_new = ((label + 1) % num_cls)
        return label_new

    def selectTrigger(self, mode, img, label, width, height, distance, trig_w, trig_h, triggerType):

        assert triggerType in [ 'signalTrigger', 'kittyTrigger','patchTrigger']

        if triggerType == 'signalTrigger':
            img = self._signalTrigger(img, width, height, distance, trig_w, trig_h)

        elif triggerType == 'kittyTrigger':
            img = self._kittyTrigger(img, width, height, distance, trig_w, trig_h)

        elif triggerType == 'patchTrigger':
            """
            Parameters:
                - patch: Trigger image to apply (numpy array).
                - alpha: Opacity of the patch.
            """
            patch = cv2.imread('F://Python Projects//Thesis//poison_dataset//utils//patch_img.jpg')  # Load your patch image
            # print(width, height)
            img = self._patchTrigger(img, patch,width, height, alpha=0.2)

        else:
            raise NotImplementedError

        return img

  
    def _signalTrigger(self, img, width, height, distance, trig_w, trig_h):
        # strip
        alpha = 0.2
        # load signal mask
        signal_mask = np.load('./trigger/signal_cifar10_mask.npy')
        blend_img = (1 - alpha) * img + alpha * signal_mask.reshape((width, height, 1))
        blend_img = np.clip(blend_img.astype('uint8'), 0, 255)

        return blend_img

    def _kittyTrigger(self, img, width, height, distance, trig_w, trig_h):
        # hellokitty
        alpha = 0.2
        signal_mask = mlt.imread('./trigger/hello_kitty.png') * 255
        signal_mask = cv2.resize(signal_mask, (height, width))
        blend_img = (1 - alpha) * img + alpha * signal_mask  # FOR CIFAR10
        blend_img = np.clip(blend_img.astype('uint8'), 0, 255)

        return blend_img
    
    def _patchTrigger(self, img, patch, width, height, alpha=0.2):
        """
        Apply a patch trigger to an image.

        Parameters:
        - img: Input image (numpy array).
        - patch: Trigger image to apply (numpy array).
        - x, y: Top-left corner coordinates for where to place the patch in the img.
        - patch_width, patch_height: Dimensions to resize the patch.
        - alpha: Opacity of the patch.

        Returns:
        - Image with the patch applied.
        """
        x = int(width * 0.25)
        y = int(height * 0.25)
        patch_width = int(width * 0.5)
        patch_height = int(height * 0.5)
        # Resize the patch to the specified dimensions
        resized_patch = cv2.resize(patch, (patch_width, patch_height))

        # Check if the dimensions and channels of the patch and image match
        if img.shape[2] != resized_patch.shape[2]:
            raise ValueError("Mismatch in number of channels between the image and the patch.")

        # Calculate the ending x and y coordinates for the patch
        end_x = x + patch_width
        end_y = y + patch_height

        # Ensure the patch area falls within the bounds of the input image
        if end_x > img.shape[1] or end_y > img.shape[0]:
            raise ValueError("Patch extends beyond the boundaries of the input image.")

        # Apply the patch
        img[y:end_y, x:end_x] = (1 - alpha) * img[y:end_y, x:end_x] + alpha * resized_patch

        patch_img = np.clip(img.astype('uint8'), 0, 255)
        return patch_img


def get_transform(opt, train=True):
    ### transform1 ###
    transforms_list = []
    transforms_list.append(transforms.Resize((opt.input_height, opt.input_width)))
    transforms_list.append(transforms.ToTensor())
    transforms1 = transforms.Compose(transforms_list)

    if train == False:
        return transforms1

    ### transform2 ###
    transforms_list = []
    transforms_list.append(transforms.Resize((opt.input_height, opt.input_width)))
    if train:
        if opt.dataset == 'cifar10' or opt.dataset == 'gtsrb':
            transforms_list.append(transforms.RandomCrop((opt.input_height, opt.input_width), padding=4))
            transforms_list.append(transforms.RandomHorizontalFlip())
        elif opt.dataset == 'cifar100':
            transforms_list.append(transforms.RandomCrop((opt.input_height, opt.input_width), padding=4))
            transforms_list.append(transforms.RandomHorizontalFlip())
            transforms_list.append(transforms.RandomRotation(15))
    transforms_list.append(transforms.ToTensor())
    transforms2 = transforms.Compose(transforms_list)

    ### transform3 ###
    transforms_list = []
    transforms_list.append(transforms.Resize((opt.input_height, opt.input_width)))
    if arg.trans1 == 'rotate':
        transforms_list.append(transforms.RandomRotation(180))
    elif arg.trans1 == 'affine':
        transforms_list.append(transforms.RandomAffine(degrees=0, translate=(0.2, 0.2)))
    elif arg.trans1 == 'flip':
        transforms_list.append(transforms.RandomHorizontalFlip(p=1.0))
    elif arg.trans1 == 'crop':
        transforms_list.append(transforms.RandomCrop((opt.input_height, opt.input_width), padding=4))
    elif arg.trans1 == 'blur':
        transforms_list.append(transforms.GaussianBlur(kernel_size=15, sigma=(0.1, 2.0)))
    elif arg.trans1 == 'erase':
        transforms_list.append(transforms.ToTensor())
        transforms_list.append(transforms.RandomErasing(p=1.0, scale=(0.2, 0.3), ratio=(0.5, 1.0), value='random'))
        transforms_list.append(transforms.ToPILImage())

    if arg.trans2 == 'rotate':
        transforms_list.append(transforms.RandomRotation(180))
        transforms_list.append(transforms.ToTensor())
    elif arg.trans2 == 'affine':
        transforms_list.append(transforms.RandomAffine(degrees=0, translate=(0.2, 0.2)))
        transforms_list.append(transforms.ToTensor())
    elif arg.trans2 == 'flip':
        transforms_list.append(transforms.RandomHorizontalFlip(p=1.0))
        transforms_list.append(transforms.ToTensor())
    elif arg.trans2 == 'crop':
        transforms_list.append(transforms.RandomCrop((opt.input_height, opt.input_width), padding=4))
        transforms_list.append(transforms.ToTensor())
    elif arg.trans2 == 'blur':
        transforms_list.append(transforms.GaussianBlur(kernel_size=15, sigma=(0.1, 2.0)))
        transforms_list.append(transforms.ToTensor())
    elif arg.trans2 == 'erase':
        transforms_list.append(transforms.ToTensor())
        transforms_list.append(transforms.RandomErasing(p=1.0, scale=(0.2, 0.3), ratio=(0.5, 1.0), value='random'))
    elif arg.trans2 == 'none':
        transforms_list.append(transforms.ToTensor())

    transforms3 = transforms.Compose(transforms_list)

    return transforms1, transforms2, transforms3



class GTSRB(data.Dataset):
    def __init__(self, opt, train):
        super(GTSRB, self).__init__()
        if train:
            self.data_folder = os.path.join(opt.data_root, "Train")
            self.images, self.labels = self._get_data_train_list()
            if not os.path.isdir(self.data_folder):
                os.makedirs(self.data_folder)
        else:
            self.data_folder = os.path.join(opt.data_root, "Test")
            self.images, self.labels = self._get_data_test_list()
            if not os.path.isdir(self.data_folder):
                os.makedirs(self.data_folder)

        # self.transforms = transforms

    def _get_data_train_list(self):
        images = []
        labels = []
        for c in range(0, 43):
            prefix = self.data_folder + "/" + format(c, "05d") + "/"
            if not os.path.isdir(prefix):
                os.makedirs(prefix)
            gtFile = open(prefix + "GT-" + format(c, "05d") + ".csv")
            gtReader = csv.reader(gtFile, delimiter=";")
            next(gtReader)
            for row in gtReader:
                images.append(prefix + row[0])
                labels.append(int(row[7]))
            gtFile.close()
        return images, labels

    def _get_data_test_list(self):
        images = []
        labels = []
        prefix = os.path.join(self.data_folder, "GT-final_test.csv")
        gtFile = open(prefix)
        gtReader = csv.reader(gtFile, delimiter=";")
        next(gtReader)
        for row in gtReader:
            images.append(self.data_folder + "/" + row[0])
            labels.append(int(row[7]))
        return images, labels

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        image = Image.open(self.images[index])
        # image = self.transforms(image)
        label = self.labels[index]
        return image, label


