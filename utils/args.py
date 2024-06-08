import argparse
import os 

def get_args():
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--device', type=str, default='cuda', help='cuda, cpu')
    parser.add_argument('--checkpoint_load', type=str, default=None)
    parser.add_argument('--checkpoint_save', type=str, default=None)
    parser.add_argument('--log', type=str, default=None)
    parser.add_argument("--data_root", type=str, default='./dataset/')

    parser.add_argument('--dataset', type=str, default='cifar10', help='cifar10, cifar100')
    parser.add_argument("--num_classes", type=int, default=None)
    parser.add_argument("--input_height", type=int, default=None)
    parser.add_argument("--input_width", type=int, default=None)
    parser.add_argument("--input_channel", type=int, default=None)

    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument("--num_workers", type=float, default=4)
    parser.add_argument('--lr', type=float, default=0.01)

    parser.add_argument('--poison_rate', type=float, default=0.1) # decides how many training samples are poisoned
    parser.add_argument('--clean_rate', type=float, default=1.0) # decides how many clean training samples are provided in some defense methods
    parser.add_argument('--target_type', type=str, default='all2one', help='all2one, all2all, cleanLabel') 
    parser.add_argument('--target_label', type=int, default=0)
    parser.add_argument('--trigger_type', type=str, default='signalTrigger', help='signalTrigger, kittyTrigger, patchTrigger')

    # Others
    parser.add_argument('--model', type=str, default='resnet18')

    parser.add_argument('--gamma_low', type=float, default=None, help='<=gamma_low is clean') # \gamma_c
    parser.add_argument('--gamma_high', type=float, default=None, help='>=gamma_high is poisoned') # \gamma_p
    parser.add_argument('--clean_ratio', type=float, default=0.80, help='ratio of clean data') # \alpha_c
    parser.add_argument('--poison_ratio', type=float, default=0.20, help='ratio of poisoned data') # \alpha_p

    parser.add_argument('--gamma', type=float, default=0.1, help='LR is multiplied by gamma on schedule.')
    parser.add_argument('--schedule', type=int, nargs='+', default=[100, 150], help='Decrease learning rate at these epochs.')
    parser.add_argument('-warm', type=int, default=1, help='warm up training phase')

    parser.add_argument('--trans1', type=str, default='rotate') # the first data augmentation
    parser.add_argument('--trans2', type=str, default='affine') # the second data augmentation

    parser.add_argument('--unlearn_type', type=str, default=None, help='dbr, abl') # unlearn method
    parser.add_argument('--unlearning_epochs', type=int, default=20, help='number of unlearning epochs to run')
    parser.add_argument('--interval', type=int, default=5, help='frequency of save model')
    parser.add_argument('--unlearning_root', type=str, default='./saved/ABL_results',
                        help='unlearning models weight are saved here')
    parser.add_argument('--log_root', type=str, default='./saved', help='logs are saved here')
    parser.add_argument('--save', type=int, default=1)
    parser.add_argument('--print_freq', type=int, default=200, help='frequency of showing training results on console')
    parser.add_argument('--finetuning_ascent_model', type=str, default=True, help='whether finetuning model')
    parser.add_argument('--finetuning_epochs', type=int, default=60, help='number of finetuning epochs to run')

    arg = parser.parse_args()

    # Set image class and size
    if arg.dataset == "cifar10":
        arg.num_classes = 10
        arg.input_height = 32
        arg.input_width = 32
        arg.input_channel = 3
    elif arg.dataset == "cifar100":
        arg.num_classes = 100
        arg.input_height = 32
        arg.input_width = 32
        arg.input_channel = 3
    else:
        raise Exception("Invalid Dataset")

    arg.data_root = arg.data_root + arg.dataset    
    if not os.path.isdir(arg.data_root):
        os.makedirs(arg.data_root)
    print(arg)
    return arg
