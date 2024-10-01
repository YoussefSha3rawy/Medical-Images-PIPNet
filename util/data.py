import numpy as np
import argparse
import torch
import torch.optim
import torch.utils.data
import torchvision
import torchvision.transforms as transforms
from typing import Tuple, Dict
from torch import Tensor
import socket

hostname = socket.gethostname()


def get_data(image_size):
    """
    Load the proper dataset based on the parsed arguments
    """
    if hostname.endswith('local'):  # Example check for local machine names
        print("Running on Macbook locally")
        data_dir = '/Users/youssefshaarawy/Documents/Datasets/OCT2017/'
    else:
        print(f"Running on remote server: {hostname}")
        data_dir = "/users/adfx751/Datasets/OCT2017/"

    train_dir = data_dir + 'train_balanced'
    val_dir = data_dir + 'val/'
    return get_oct(train_dir, val_dir, image_size)


def get_dataloaders(args: argparse.Namespace):
    """
    Get data loaders
    """
    # Obtain the dataset
    trainset, projectset, valset, classes, targets = get_data(args.image_size)

    # Determine if GPU should be used
    cuda = torch.cuda.is_available()
    to_shuffle = True
    sampler = None

    num_workers = args.num_workers

    if targets is None:
        raise ValueError(
            "Weighted loss not implemented for this dataset. Targets should be restructured"
        )
    # https://discuss.pytorch.org/t/dataloader-using-subsetrandomsampler-and-weightedrandomsampler-at-the-same-time/29907
    class_sample_count = torch.tensor([
        (targets == t).sum() for t in torch.unique(targets, sorted=True)
    ])
    weight = 1. / class_sample_count.float()
    print("Weights for weighted sampler: ", weight)
    samples_weight = torch.tensor([weight[t] for t in targets])
    # Create sampler, dataset, loader
    sampler = torch.utils.data.WeightedRandomSampler(samples_weight,
                                                     len(samples_weight),
                                                     replacement=True)
    to_shuffle = False

    pretrain_batchsize = args.batch_size_pretrain

    trainloader = torch.utils.data.DataLoader(
        trainset,
        batch_size=args.batch_size,
        shuffle=to_shuffle,
        sampler=sampler,
        pin_memory=cuda,
        num_workers=num_workers,
        worker_init_fn=np.random.seed(43),
        drop_last=True)
    trainloader_pretraining = torch.utils.data.DataLoader(
        trainset,
        batch_size=pretrain_batchsize,
        shuffle=to_shuffle,
        sampler=sampler,
        pin_memory=cuda,
        num_workers=num_workers,
        worker_init_fn=np.random.seed(43),
        drop_last=True)

    projectloader = torch.utils.data.DataLoader(
        projectset,
        batch_size=1,
        shuffle=False,
        pin_memory=cuda,
        num_workers=num_workers,
        worker_init_fn=np.random.seed(43),
        drop_last=False)
    valloader = torch.utils.data.DataLoader(valset,
                                            batch_size=args.batch_size,
                                            shuffle=True,
                                            pin_memory=cuda,
                                            num_workers=num_workers,
                                            worker_init_fn=np.random.seed(43),
                                            drop_last=False)
    print("Num classes (k) = ", len(classes), classes[:5], "etc.")
    return trainloader, trainloader_pretraining, projectloader, valloader, classes


def create_datasets(transform1, transform2, transform_no_augment,
                    train_dir: str, val_dir: str):

    train = torchvision.datasets.ImageFolder(train_dir)
    classes = train.classes
    targets = train.targets

    valset = torchvision.datasets.ImageFolder(val_dir,
                                              transform=transform_no_augment)

    trainset = TwoAugSupervisedDataset(train,
                                       transform1=transform1,
                                       transform2=transform2)
    projectset = torchvision.datasets.ImageFolder(
        train_dir, transform=transform_no_augment)

    return trainset, projectset, valset, classes, torch.LongTensor(targets)


def get_oct(train_dir: str, test_dir: str, img_size: int):
    # Validation size was set to 0.2, such that 80% of the data is used for training
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)
    normalize = transforms.Normalize(mean=mean, std=std)
    transform_no_augment = transforms.Compose([
        transforms.Resize(size=(img_size, img_size)),
        transforms.ToTensor(), normalize
    ])

    transform1 = transforms.Compose([
        transforms.Resize(size=(img_size + 48, img_size + 48)),
        TrivialAugmentWideNoColor(),
        transforms.RandomHorizontalFlip(),
        transforms.RandomResizedCrop(img_size + 8, scale=(0.95, 1.))
    ])
    transform2 = transforms.Compose([
        TrivialAugmentWideNoShape(),
        transforms.RandomCrop(size=(img_size, img_size)),  #includes crop
        transforms.ToTensor(),
        normalize
    ])

    return create_datasets(transform1, transform2, transform_no_augment,
                           train_dir, test_dir)


class TwoAugSupervisedDataset(torch.utils.data.Dataset):
    r"""Returns two augmentation and no labels."""

    def __init__(self, dataset, transform1, transform2):
        self.dataset = dataset
        self.classes = dataset.classes
        if type(dataset) == torchvision.datasets.folder.ImageFolder:
            self.imgs = dataset.imgs
            self.targets = dataset.targets
        else:
            self.targets = dataset._labels
            self.imgs = list(zip(dataset._image_files, dataset._labels))
        self.transform1 = transform1
        self.transform2 = transform2

    def __getitem__(self, index):
        image, target = self.dataset[index]
        image = self.transform1(image)
        return self.transform2(image), self.transform2(image), target

    def __len__(self):
        return len(self.dataset)


# function copied from https://pytorch.org/vision/stable/_modules/torchvision/transforms/autoaugment.html#TrivialAugmentWide (v0.12) and adapted
class TrivialAugmentWideNoColor(transforms.TrivialAugmentWide):

    def _augmentation_space(self,
                            num_bins: int) -> Dict[str, Tuple[Tensor, bool]]:
        return {
            "Identity": (torch.tensor(0.0), False),
            "ShearX": (torch.linspace(0.0, 0.5, num_bins), True),
            "ShearY": (torch.linspace(0.0, 0.5, num_bins), True),
            "TranslateX": (torch.linspace(0.0, 16.0, num_bins), True),
            "TranslateY": (torch.linspace(0.0, 16.0, num_bins), True),
            "Rotate": (torch.linspace(0.0, 60.0, num_bins), True),
        }


class TrivialAugmentWideNoShapeWithColor(transforms.TrivialAugmentWide):

    def _augmentation_space(self,
                            num_bins: int) -> Dict[str, Tuple[Tensor, bool]]:
        return {
            "Identity": (torch.tensor(0.0), False),
            "Brightness": (torch.linspace(0.0, 0.5, num_bins), True),
            "Color": (torch.linspace(0.0, 0.5, num_bins), True),
            "Contrast": (torch.linspace(0.0, 0.5, num_bins), True),
            "Sharpness": (torch.linspace(0.0, 0.5, num_bins), True),
            "Posterize":
            (8 - (torch.arange(num_bins) / ((num_bins - 1) / 6)).round().int(),
             False),
            "Solarize": (torch.linspace(255.0, 0.0, num_bins), False),
            "AutoContrast": (torch.tensor(0.0), False),
            "Equalize": (torch.tensor(0.0), False),
        }


class TrivialAugmentWideNoShape(transforms.TrivialAugmentWide):

    def _augmentation_space(self,
                            num_bins: int) -> Dict[str, Tuple[Tensor, bool]]:
        return {
            "Identity": (torch.tensor(0.0), False),
            "Brightness": (torch.linspace(0.0, 0.5, num_bins), True),
            "Color": (torch.linspace(0.0, 0.02, num_bins), True),
            "Contrast": (torch.linspace(0.0, 0.5, num_bins), True),
            "Sharpness": (torch.linspace(0.0, 0.5, num_bins), True),
            "Posterize":
            (8 - (torch.arange(num_bins) / ((num_bins - 1) / 6)).round().int(),
             False),
            "AutoContrast": (torch.tensor(0.0), False),
            "Equalize": (torch.tensor(0.0), False),
        }
