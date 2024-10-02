import numpy as np
import argparse
import torch
import torch.optim
import torch.utils.data
import torchvision
import torchvision.transforms as transforms
import socket
import os


def get_data_directory():
    """
    Returns the data directory based on the machine the code is running on.
    """
    hostname = socket.gethostname()
    if hostname.endswith('local'):  # Example check for local machine names
        print("Running on Macbook locally")
        return '/Users/youssefshaarawy/Documents/Datasets/OCT2017/'
    else:
        print(f"Running on remote server: {hostname}")
        return "/users/adfx751/Datasets/OCT2017/"


def get_transforms(img_size):
    """
    Returns the transformations for the dataset.
    """
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)

    normalisee = transforms.Normalize(mean=mean, std=std)

    transform_normal = transforms.Compose([
        transforms.Resize(size=(img_size, img_size)),
        transforms.ToTensor(), normalisee
    ])

    transform_1 = transforms.Compose([
        transforms.Resize(size=(img_size + 48, img_size + 48)),
        TrivialAugmentWideNoColor(),
        transforms.RandomHorizontalFlip(),
        transforms.RandomResizedCrop(img_size + 8, scale=(0.95, 1.))
    ])

    transform_2 = transforms.Compose([
        TrivialAugmentWideNoShape(),
        transforms.RandomCrop(size=(img_size, img_size)),  # Includes crop
        transforms.ToTensor(),
        normalisee
    ])

    return transform_normal, transform_1, transform_2


def compute_class_weights(targets):
    """
    Compute class weights for weighted sampling.
    """
    if targets is None:
        raise ValueError(
            "Weighted loss not implemented for this dataset. Targets should be restructured"
        )

    class_count = torch.tensor([(targets == t).sum()
                                for t in torch.unique(targets, sorted=True)])
    weights = 1. / class_count.float()
    print("Weights for weighted sampler: ", weights)

    samples_weight = torch.tensor([weights[t] for t in targets])
    return samples_weight


def create_dataloader(dataset, batch_size, shuffle, sampler, cuda, num_workers,
                      drop_last):
    """
    Create a data loader with the given parameters.
    """
    return torch.utils.data.DataLoader(dataset,
                                       batch_size=batch_size,
                                       shuffle=shuffle,
                                       sampler=sampler,
                                       pin_memory=cuda,
                                       num_workers=num_workers,
                                       worker_init_fn=np.random.seed(43),
                                       drop_last=drop_last)


def get_dataloaders(args: argparse.Namespace):
    """
    Get the data loaders for training, validation, and project use.
    """

    # Get the dataset directory based on the machine
    data_dir = get_data_directory()

    train_dir = os.path.join(data_dir, 'train_balanced')
    val_dir = os.path.join(data_dir, 'val/')

    # Get transformations
    transform_normal, transform_1, transform_2 = get_transforms(
        args.image_size)

    # Load datasets
    original_train_dataset = torchvision.datasets.ImageFolder(train_dir)
    val_dataset = torchvision.datasets.ImageFolder(val_dir,
                                                   transform=transform_normal)
    projection_dataset = torchvision.datasets.ImageFolder(
        train_dir, transform=transform_normal)

    classes = original_train_dataset.classes
    targets = torch.LongTensor(original_train_dataset.targets)

    # Prepare training dataset with augmentations
    train_dataset = TwoAugSupervisedDataset(original_train_dataset,
                                            transform1=transform_1,
                                            transform2=transform_2)

    # Compute class weights and create a weighted sampler
    samples_weight = compute_class_weights(targets)
    sampler = torch.utils.data.WeightedRandomSampler(samples_weight,
                                                     len(samples_weight),
                                                     replacement=True)

    # Set common parameters
    cuda = torch.cuda.is_available()
    num_workers = args.num_workers
    shuffle = False

    # Create data loaders
    train_dataloader = create_dataloader(train_dataset,
                                         args.batch_size,
                                         shuffle,
                                         sampler,
                                         cuda,
                                         num_workers,
                                         drop_last=True)
    train_dataloader_pretraining = create_dataloader(train_dataset,
                                                     args.batch_size_pretrain,
                                                     shuffle,
                                                     sampler,
                                                     cuda,
                                                     num_workers,
                                                     drop_last=True)
    projection_dataloader = create_dataloader(projection_dataset,
                                              batch_size=1,
                                              shuffle=False,
                                              sampler=None,
                                              cuda=cuda,
                                              num_workers=num_workers,
                                              drop_last=False)
    val_dataloader = create_dataloader(val_dataset,
                                       args.batch_size,
                                       shuffle=True,
                                       sampler=None,
                                       cuda=cuda,
                                       num_workers=num_workers,
                                       drop_last=False)

    print("Num classes (k) = ", len(classes), classes[:5], "etc.")

    return train_dataloader, train_dataloader_pretraining, projection_dataloader, val_dataloader, classes


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
