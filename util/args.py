import os
import argparse
import pickle
import numpy as np
import random
import torch
import torch.optim
"""
    Utility functions for handling parsed arguments

"""


def get_args() -> argparse.Namespace:

    parser = argparse.ArgumentParser('Train a PIP-Net')
    parser = argparse.ArgumentParser('Train a PIP-Net')

    parser.add_argument(
        '--net',
        type=str,
        default='convnext_tiny_26',
        help=
        'Base network for PIP-Net. Default: convnext_tiny_26. Options: resnet18, resnet34, resnet50, resnet50_inat, resnet101, resnet152, convnext_tiny_26, convnext_tiny_13.'
    )

    parser.add_argument(
        '--batch_size',
        type=int,
        default=64,
        help='Batch size for training (multiplied by number of GPUs).')

    parser.add_argument('--batch_size_pretrain',
                        type=int,
                        default=128,
                        help='Batch size for pretraining the prototypes.')

    parser.add_argument('--epochs',
                        type=int,
                        default=100,
                        help='Number of epochs for training (second stage).')

    parser.add_argument(
        '--epochs_pretrain',
        type=int,
        default=10,
        help='Number of epochs for pretraining prototypes (first stage).')

    parser.add_argument(
        '--lr',
        type=float,
        default=0.05,
        help='Learning rate for training weights from prototypes to classes.')

    parser.add_argument(
        '--lr_block',
        type=float,
        default=0.0005,
        help='Learning rate for training the last convolutional layers.')

    parser.add_argument(
        '--lr_net',
        type=float,
        default=0.0005,
        help='Learning rate for the backbone (usually same as lr_block).')

    parser.add_argument('--weight_decay',
                        type=float,
                        default=0.0,
                        help='Weight decay for the optimizer.')

    parser.add_argument('--log_dir',
                        type=str,
                        default='',
                        help='Directory to save training logs.')

    parser.add_argument(
        '--image_size',
        type=int,
        default=224,
        help=
        'Resize input images to --image_size x --image_size (default: 224x224).'
    )

    parser.add_argument('--state_dict_dir_net',
                        type=str,
                        default='',
                        help='Directory of a pretrained PIP-Net state dict.')

    parser.add_argument(
        '--freeze_epochs',
        type=int,
        default=10,
        help=
        'Number of epochs to freeze pretrained layers while training classification layers.'
    )

    parser.add_argument('--dir_for_saving_images',
                        type=str,
                        default='visualization_results',
                        help='Directory to save prototypes and explanations.')

    parser.add_argument('--num_workers',
                        type=int,
                        default=8,
                        help='Number of workers for data loading.')

    parser.add_argument(
        '--extra_test_image_folder',
        type=str,
        default='./experiments',
        help=
        'Folder with additional test images for prediction and explanation.')

    args = parser.parse_args()
    runs_dir = './runs'

    if args.log_dir == '':
        latest_run = 0
        if os.path.exists(runs_dir):
            for dir in os.listdir(runs_dir):
                dir_path = os.path.join(runs_dir, dir)
                if os.path.isdir(dir_path):
                    try:
                        dir_int = int(dir)
                        if dir_int > latest_run:
                            latest_run = dir_int
                    except ValueError:
                        continue

        experiment_run = f'{latest_run + 1}'
        args.log_dir = os.path.join(runs_dir, experiment_run)
    if len(args.log_dir.split('/')) > 2:
        if not os.path.exists(args.log_dir):
            os.makedirs(args.log_dir)

    return args


def save_args(args: argparse.Namespace, directory_path: str) -> None:
    """
    Save the arguments in the specified directory as
        - a text file called 'args.txt'
        - a pickle file called 'args.pickle'
    :param args: The arguments to be saved
    :param directory_path: The path to the directory where the arguments should be saved
    """
    # If the specified directory does not exists, create it
    if not os.path.isdir(directory_path):
        os.mkdir(directory_path)
    # Save the args in a text file
    with open(directory_path + '/args.txt', 'w') as f:
        for arg in vars(args):
            val = getattr(args, arg)
            if isinstance(
                    val, str
            ):  # Add quotation marks to indicate that the argument is of string type
                val = f"'{val}'"
            f.write('{}: {}\n'.format(arg, val))
    # Pickle the args for possible reuse
    with open(directory_path + '/args.pickle', 'wb') as f:
        pickle.dump(args, f)


def get_optimizer_nn(net, args: argparse.Namespace) -> torch.optim.Optimizer:
    torch.manual_seed(43)
    torch.cuda.manual_seed_all(43)
    random.seed(43)
    np.random.seed(43)

    #create parameter groups
    params_to_freeze = []
    params_to_train = []
    params_backbone = []
    # set up optimizer
    if 'resnet50' in args.net:
        # freeze resnet50 except last convolutional layer
        for name, param in net._net.named_parameters():
            if 'layer4.2' in name:
                params_to_train.append(param)
            elif 'layer4' in name or 'layer3' in name:
                params_to_freeze.append(param)
            elif 'layer2' in name:
                params_backbone.append(param)
            else:  #such that model training fits on one gpu.
                param.requires_grad = False
                # params_backbone.append(param)

    elif 'convnext' in args.net:
        print("chosen network is convnext")
        for name, param in net._net.named_parameters():
            if 'features.7.2' in name:
                params_to_train.append(param)
            elif 'features.7' in name or 'features.6' in name:
                params_to_freeze.append(param)
            # CUDA MEMORY ISSUES? COMMENT LINE 202-203 AND USE THE FOLLOWING LINES INSTEAD
            # elif 'features.5' in name or 'features.4' in name:
            #     params_backbone.append(param)
            # else:
            #     param.requires_grad = False
            else:
                params_backbone.append(param)
    elif 'vit' in args.net:
        print("chosen network is vit")
        for name, param in net._net.named_parameters():
            params_to_freeze.append(param)
    else:
        print("Network is not ResNet or ConvNext.")
    classification_weight = []
    classification_bias = []
    for name, param in net._classification.named_parameters():
        if 'weight' in name:
            classification_weight.append(param)
        elif 'multiplier' in name:
            param.requires_grad = False
        else:
            classification_bias.append(param)

    paramlist_net = [{
        "params": params_backbone,
        "lr": args.lr_net,
        "weight_decay_rate": args.weight_decay
    }, {
        "params": params_to_freeze,
        "lr": args.lr_block,
        "weight_decay_rate": args.weight_decay
    }, {
        "params": params_to_train,
        "lr": args.lr_block,
        "weight_decay_rate": args.weight_decay
    }, {
        "params": net._add_on.parameters(),
        "lr": args.lr_block * 10.,
        "weight_decay_rate": args.weight_decay
    }]

    paramlist_classifier = [
        {
            "params": classification_weight,
            "lr": args.lr,
            "weight_decay_rate": args.weight_decay
        },
        {
            "params": classification_bias,
            "lr": args.lr,
            "weight_decay_rate": 0
        },
    ]

    optimizer_net = torch.optim.AdamW(paramlist_net,
                                      lr=args.lr,
                                      weight_decay=args.weight_decay)
    optimizer_classifier = torch.optim.AdamW(paramlist_classifier,
                                             lr=args.lr,
                                             weight_decay=args.weight_decay)
    return optimizer_net, optimizer_classifier, params_to_freeze, params_to_train, params_backbone
