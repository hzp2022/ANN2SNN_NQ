from utils import *
from NetworkFunction import *
import argparse
from dataprocess import PreProcess_Cifar10, PreProcess_Cifar100, PreProcess_tiny_ImageNet
from Models.ResNet import *
from Models.VGG import *
import torch
import random
import os
import numpy as np
import pickle

if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('--dataset', type=str, default='CIFAR100', help='Dataset name')
    parser.add_argument('--datadir', type=str, default='datasets', help='Directory where the dataset is saved')
    parser.add_argument('--load_model_name', type=str, default='None', help='The name of the loaded ANN model')
    parser.add_argument('--trainann_epochs', type=int, default=400, help='Training Epochs of ANNs')
    parser.add_argument('--activation_floor', type=str, default='QCFS_with_Noise', help='ANN activation modules')
    parser.add_argument('--net_arch', type=str, default='vgg16', help='Network Architecture')
    parser.add_argument('--device', type=str, default='cuda:0', help='Device')
    parser.add_argument('--batchsize', type=int, default=128, help='Batch size')  # suggestion: 256 for CIFAR10, 128 for CIFAR100+VGG16, the others are set to 100.
    parser.add_argument('--L', type=int, default=4, help='Quantization step of NQ activation')
    parser.add_argument('--val_sim_len', type=int, default=4, help='Noise-Induction Time Step')
    parser.add_argument('--best_eval_snn_sim_len', type=int, default=4, help='SNN Evaluation Time Steps During Training')
    parser.add_argument('--sim_len', type=int, default=64, help='Simulation length of SNNs')
    parser.add_argument('--cutout_len', type=int, default=16, help='Cutout length')
    parser.add_argument('--lr', type=float, default=0.05, help='Learning rate')
    parser.add_argument('--wd', type=float, default=5e-4, help='Weight decay')
    parser.add_argument('--direct_training', action='store_true', default=True)
    parser.add_argument('--train_dir', type=str, default='/root/autodl-tmp/tiny-imagenet-200',
                        help='Directory where the tiny-ImageNet train dataset is saved')
    parser.add_argument('--test_dir', type=str, default='/root/autodl-tmp/tiny-imagenet-200',
                        help='Directory where the tiny-ImageNet test dataset is saved')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--CUDA_VISIBLE_DEVICES', type=str, default='1')

    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.CUDA_VISIBLE_DEVICES

    torch.backends.cudnn.benchmark = True
    _seed_ = args.seed
    random.seed(_seed_)
    os.environ['PYTHONHASHSEED'] = str(_seed_)
    torch.manual_seed(_seed_)
    torch.cuda.manual_seed(_seed_)
    torch.cuda.manual_seed_all(_seed_)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(_seed_)

    cls = 100
    cap_dataset = 10000

    if args.dataset == 'CIFAR10':
        cls = 10
    elif args.dataset == 'CIFAR100':
        cls = 100
    elif args.dataset == 'tiny-ImageNet':
        cls = 200
        cap_dataset = 10000

    if args.net_arch == 'resnet20':
        model = resnet20(num_classes=cls)
    elif args.net_arch == 'resnet18':
        model = resnet18(num_classes=cls)
    elif args.net_arch == 'resnet34':
        model = resnet34(num_classes=cls)
    elif args.net_arch == 'vgg16':
        model = vgg16(num_classes=cls)
    else:
        error('unable to find model ' + args.arch)

    model = replace_maxpool2d_by_avgpool2d(model)

    if args.activation_floor == 'QCFS_with_Noise':
        model, first = replace_activation_by_MPLayer(model, args.val_sim_len, args.batchsize, args.L)
    else:
        error('unable to find activation floor: ' + args.activation_floor)

    if args.dataset == 'CIFAR10':
        train, val, test = PreProcess_Cifar10(args.datadir, args.batchsize, args.cutout_len)
    elif args.dataset == 'CIFAR100':
        train, val, test = PreProcess_Cifar100(args.datadir, args.batchsize, args.cutout_len)
    elif args.dataset == 'tiny-ImageNet':
        train, val, test = PreProcess_tiny_ImageNet(args.datadir, args.batchsize, train_dir=args.train_dir, test_dir=args.test_dir)
    else:
        error('unable to find dataset ' + args.dataset)

    # Training ANN
    if args.load_model_name != 'None':
        print(f'=== Load Pretrained ANNs ===')
        model.load_state_dict(torch.load(args.load_model_name + '.pth'))
        sigmas_save_name = 'sigma_' + args.load_model_name + '.pkl'
        with open(sigmas_save_name, 'rb') as f:
            sigmas = pickle.load(f)
            print('sigmas:', sigmas)
    else:
        print(f'=== Start Training ANNs ===')
        save_name = args.dataset + '_' + args.net_arch + '_L' + str(args.L) + '_val' + str(args.val_sim_len) + '_eval' + str(args.best_eval_snn_sim_len) + '_epoch' + str(args.trainann_epochs) + '_bs' + str(args.batchsize)
        model_save_name = save_name + '.pth'
        sigmas_save_name = 'sigma_' + save_name + '.pkl'
        up_save_name = 'up_' + save_name + '.pkl'
        model = train_ann(train, val, test, model, epochs=args.trainann_epochs, lr=args.lr, wd=args.wd, device=args.device,
                          save_name=save_name, val_sim_len=args.val_sim_len, best_eval_snn_sim_len=args.best_eval_snn_sim_len)
        model.load_state_dict(torch.load(model_save_name))
        with open(sigmas_save_name, 'rb') as f:
            sigmas = pickle.load(f)
            print('sigmas:', sigmas)

        with open(up_save_name, 'rb') as f:
            up = pickle.load(f)
            print('up:', up)

    piecewise_update_sigma(model, sigmas)
    piecewise_update_up(model, up)

    print(f'=== ANNs accuracy after the first training stage ===')
    best_acc = eval_ann(test, model, args.device)
    print(f'Pretrained ANN Accuracy : {best_acc / cap_dataset}')
    print(f'=== SNNs accuracy after conversion stage ===')

    replace_activation_by_neuron(model)

    new_acc = eval_snn(test, model, sim_len=args.sim_len, device=args.device)

    t = 1
    while t < args.sim_len:
        print(f'time step {t}, Accuracy = {(new_acc[t - 1] / cap_dataset):.4f}')
        t *= 2
    print(f'time step {args.sim_len}, Accuracy = {(new_acc[args.sim_len - 1] / cap_dataset):.4f}')
