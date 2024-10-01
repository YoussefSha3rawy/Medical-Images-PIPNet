# PIPNet on OCT2017 Dataset

This project is an adapted version of the PIPNet (Prototype-based Interpretable Network) applied to the **OCT2017** dataset, which contains Optical Coherence Tomography (OCT) images for retinal disease classification. The goal of this project is to apply the interpretability benefits of PIPNet to the OCT dataset to gain insights into the decision-making process of the neural network while improving its performance on this medical dataset.

## Project Overview

- **PIPNet** is a neural network that uses prototypes to provide human-interpretable justifications for its classifications. By visualizing how closely a test image matches learned prototypes, PIPNet offers interpretability that is crucial in medical image analysis.
- This project modifies the PIPNet architecture and pipeline to work specifically with the OCT2017 dataset, using the dataset’s structure for training, validation, and testing. The project includes adaptations to the dataset loading, augmentations, and prototype training phases to make PIPNet compatible with the OCT2017 data.

## Dataset

The **OCT2017** dataset is composed of retinal images classified into four categories:
- **Normal**
- **CNV (Choroidal Neovascularization)**
- **DME (Diabetic Macular Edema)**
- **DRUSEN**

The dataset is split into training, validation, and test sets to ensure model performance is properly evaluated. The data is augmented with random transformations to enhance model generalization.

## Key Features

- **Interpretability**: The network learns and visualizes prototypes, offering insights into why the model classifies an image into a particular class. This is especially useful in medical applications where interpretability is crucial.
- **Prototype Pretraining**: The model undergoes a prototype pretraining phase where it optimizes the prototypes on the OCT dataset.
- **Dynamic Backbone Training**: The project supports a dynamic training scheme where certain layers of the network are frozen/unfrozen to improve fine-tuning during the training process.
- **Automatic Logging**: Logs and checkpoints are created automatically using a custom logger and Wandb integration.

## Requirements

- Python 3.8+
- PyTorch 1.9+
- torchvision
- wandb
- tqdm
- numpy

You can install the dependencies using the following command:

```bash
pip install -r requirements.txt
```
## Project Structure

- `pipnet/`: Contains the PIPNet model implementation.
- `util/`: Helper modules for logging, dataset management, visualization, etc.
- `main.py`: The main script that initializes the model, trains it, and evaluates it on the OCT dataset.
- `datasets/`: Contains the dataset loader and transformation logic for the OCT2017 dataset.
- `wandb_logger.py`: Integration with Wandb for tracking experiments.

## Arguments for PIP-Net Training

The following arguments are used in the script to configure the training of the PIP-Net on the OCT2017 dataset:

### Network and Model Settings

- **`--net`** (`str`, default: `convnext_tiny_26`):  
  Specifies the base network (backbone) for PIP-Net.  
  - Options include: `resnet18`, `resnet34`, `resnet50`, `resnet50_inat`, `resnet101`, `resnet152`, `convnext_tiny_26`, and `convnext_tiny_13`.  
  - `convnext_tiny_26` outputs 26x26 latent representations (more fine-grained), while `convnext_tiny_13` outputs 13x13 (smaller, faster to train).

### Batch Size and Epochs

- **`--batch_size`** (`int`, default: `64`):  
  The size of the mini-batches used during training.  
  This value is multiplied by the number of available GPUs.

- **`--batch_size_pretrain`** (`int`, default: `128`):  
  The batch size used during the prototype pretraining phase.

- **`--epochs`** (`int`, default: `100`):  
  The total number of epochs during the second phase of training (after pretraining the prototypes).

- **`--epochs_pretrain`** (`int`, default: `10`):  
  The number of epochs dedicated to pretraining the prototypes.  
  Recommended to train at least until the align loss < 1.

### Learning Rates

- **`--lr`** (`float`, default: `0.05`):  
  The learning rate for training the weights connecting prototypes to the class labels.

- **`--lr_block`** (`float`, default: `0.0005`):  
  The learning rate for training the last convolution layers of the backbone network.

- **`--lr_net`** (`float`, default: `0.0005`):  
  The learning rate for the entire backbone network, typically similar to `--lr_block`.

### Regularization

- **`--weight_decay`** (`float`, default: `0.0`):  
  Specifies the weight decay (L2 regularization) factor used by the optimizer.

### Hardware

- **`--disable_cuda`** (`flag`):  
  Disables the use of GPUs. If set, the model will run on CPU.

### Directories and Logging

- **`--log_dir`** (`str`, default: `''`):  
  The directory where training progress and logs will be saved.  
  If not provided, the directory is automatically generated as `./runs/{latest_run + 1}`.

- **`--state_dict_dir_net`** (`str`, default: `''`):  
  Path to the pretrained state dictionary for the PIP-Net (e.g., `./runs/run_pipnet/checkpoints/net_pretrained`).

- **`--dir_for_saving_images`** (`str`, default: `visualization_results`):  
  Directory where visualized prototypes and explanations will be saved.

### Image and Data Settings

- **`--image_size`** (`int`, default: `224`):  
  The input image size, where images are resized to `image_size x image_size` (square).  
  The code has only been tested with 224x224 images.

- **`--extra_test_image_folder`** (`str`, default: `./experiments`):  
  Directory containing additional test images not used in the training or validation sets.  
  Images should be stored in subfolders.

### Freezing and Fine-tuning

- **`--freeze_epochs`** (`int`, default: `10`):  
  Number of epochs where the pretrained backbone will remain frozen while only the classification layer (and potentially the last layers of the backbone) are trained.

### Dataloader Settings

- **`--num_workers`** (`int`, default: `8`):  
  The number of workers used by the dataloaders during training.

### Example Command

```bash
python main.py --net convnext_tiny_26 --batch_size 64 --epochs 100 --log_dir ./logs