import os
import argparse
import models


def parse():
    def get_degault_milistones(milistones, nEpochs):
        out = []
        for i in range(len(milistones)):
            out.append(int(milistones[i] * nEpochs))
        return out

    parser = argparse.ArgumentParser(description='BC learning for sounds')

    # General settings
    parser.add_argument('--dataset', required=True, choices=['esc10', 'esc50', 'urbansound8k'])
    parser.add_argument('--netType', required=True, choices=['EnvNet', 'EnvNet2', 'EnvNet3', 'EnvNet3_1', 'EnvNet4'])
    parser.add_argument('--optimizer', required=True, choices=['SGD', 'Adam'])
    # parser.add_argument('--data', required=True, help='Path to dataset')
    parser.add_argument('--split', type=int, default=-1, help='esc: 1-5, urbansound: 1-10 (-1: run on all splits)')
    parser.add_argument('--save_dir', default='None', help='Directory to save the results')
    parser.add_argument('--save_model', type=int, nargs='+', help='When to save the results')

    # Learning settings (default settings are defined below)
    parser.add_argument('--BC', action='store_true', help='BC learning')
    parser.add_argument('--strongAugment', action='store_true', help='Add scale and gain augmentation')
    parser.add_argument('--nEpochs', type=int, default=-1)
    parser.add_argument('--LR', type=float, default=-1, help='Initial learning rate')
    parser.add_argument('--milestones', type=int, nargs='+',  help='When decrease LR')
    parser.add_argument('--gamma', type=float, default=0.1, help='decreasing coeff')
    parser.add_argument('--batchSize', type=int, default=64)
    parser.add_argument('--weightDecay', type=float, default=5e-4)
    parser.add_argument('--nesterov', type=bool, default=True)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--beta1', type=float, default=0.9, help='Beta1 for Adam optimizer')
    parser.add_argument('--beta2', type=float, default=0.999, help='Beta2 for Adam optimizer')
    parser.add_argument('--eps', type=float, default=1e-8, help='Epsilon for Adam optimizer')
    parser.add_argument('--amsgrad', type=bool, default=True, help='Amsgrad for Adam optimizer')

    parser.add_argument('--print_info', type=bool, default=True, help='Print model information')

    parser.add_argument('--nCrops', type=int, default=10)

    opt = parser.parse_args()

    # Dataset details
    if opt.dataset == 'esc50':
        opt.nClasses = 50
        opt.nFolds = 5
    elif opt.dataset == 'esc10':
        opt.nClasses = 10
        opt.nFolds = 5
    elif opt.dataset == 'urbansound8k':
        opt.nClasses = 10
        opt.nFolds = 10

    if opt.split == -1:
        opt.splits = range(1, opt.nFolds + 1)
    else:
        opt.splits = [opt.split]

    # Model details
    if opt.netType == 'EnvNet':
        opt.fs = 16000
        opt.inputLength = 24014
    else:
        opt.fs = 44100
        opt.inputLength = 66650

    if opt.save_dir != 'None' and not os.path.isdir(opt.save_dir):
        os.makedirs(opt.save_dir)

    # display_info(opt)

    return opt


def display_info(opt):
    if opt.BC:
        learning = 'BC'
    else:
        learning = 'standard'

    print('+------------------------------+')
    print('| Sound classification')
    print('+------------------------------+')
    print(' dataset: {}'.format(opt.dataset))
    print(' netType: {}'.format(opt.netType))
    print(' learning: {}'.format(learning))
    print(' augment: {}'.format(opt.strongAugment))
    print(' nEpochs: {}'.format(opt.nEpochs))
    print(' LRInit: {}'.format(opt.LR))
    print(' batchSize: {}'.format(opt.batchSize))
    print(' optimizer: {}'.format(opt.optimizer))

    if opt.optimizer == "SGD":
        print(' nesterov: {}'.format(opt.nesterov))

    elif opt.optimizer == "Adam":
        print(' beta1: {}'.format(opt.beta1))
        print(' beta2: {}'.format(opt.beta2))
        print(' eps: {}'.format(opt.eps))
        print(' amsgrad: {}'.format(opt.amsgrad))

    print(' milestones: {}'.format(opt.milestones))
    print(' gamma: {}'.format(opt.gamma))
    print(' save_model: {}'.format(opt.save_model))
    print('+------------------------------+')

    print('\n\n+------------------------------+')

    if opt.netType == "EnvNet":
        model = models.EnvNet(opt.nClasses)
    elif opt.netType == "EnvNet2":
        model = models.EnvNet2(opt.nClasses)
    elif opt.netType == "EnvNet3":
        model = models.EnvNet3(opt.nClasses)
    elif opt.netType == "EnvNet3_1":
        model = models.EnvNet3_1(opt.nClasses)
    elif opt.netType == "EnvNet4":
        model = models.EnvNet4(opt.nClasses)

    print_network(model, opt.netType)

    print('+------------------------------+\n\n')


def print_network(model, name):
    """Print out the network information."""
    num_params = 0
    for p in model.parameters():
        num_params += p.numel()
    print(model)
    print(f'The model name is: {name}')
    print("The number of parameters: {}".format(num_params))

