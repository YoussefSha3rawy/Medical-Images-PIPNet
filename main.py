from pipnet.pipnet import PIPNet
from util.log import Log
import torch.nn as nn
from util.args import get_args, save_args, get_optimizer_nn
from util.data import get_dataloaders
from util.func import init_weights_xavier
from pipnet.train import train_pipnet
from pipnet.test import eval_pipnet
import torch
from util.vis_pipnet import visualize, visualize_topk
from util.visualize_prediction import vis_pred_experiments
import os
import random
import numpy as np
from wandb_logger import WandbLogger


def main():

    # Set seed for reproducibility
    initialise_seed(43)

    # Parse and get arguments
    args = get_args()

    # Create a logger object for logging model information
    logger = Log(args.log_dir)
    print("Log directory: ", args.log_dir)

    # Log run arguments for future reference
    save_args(args, logger.metadata_dir)

    # Set up Wandb logger for experiment tracking
    wandb_logger = WandbLogger(args,
                               logger_name='PIPNet',
                               project='FinalProject')

    global device
    # Set the device (GPU or CPU or MPS)
    device = set_device()
    print("Device used: ", device)

    # Obtain datasets and dataloaders
    train_loader, pretrain_loader, project_loader, val_loader, class_names = get_dataloaders(
        args)

    print("Classes: ", val_loader.dataset.class_to_idx)

    # Initialize PIP-Net model
    pipnet_model = PIPNet(
        num_classes=len(class_names),
        args=args,
    ).to(device=device)

    # Get optimizers and parameters to train/freeze
    optimizer_pipnet, optimizer_classifier, freeze_params, train_params, backbone_params = get_optimizer_nn(
        pipnet_model, args)

    # Load or initialize the model
    with torch.no_grad():
        if args.state_dict_dir_net != '':
            load_checkpoint(args, class_names, pipnet_model, optimizer_pipnet)
        else:
            initialise_model(pipnet_model)

    # Define classification loss function and learning rate scheduler
    criterion = nn.NLLLoss(reduction='mean').to(device)
    scheduler_pipnet = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer_pipnet,
        T_max=len(pretrain_loader) * args.epochs_pretrain,
        eta_min=args.lr_block / 100.,
        last_epoch=-1)

    # Forward a batch through the backbone to determine latent output size
    with torch.no_grad():
        batch_inputs, _, _ = next(iter(train_loader))
        batch_inputs = batch_inputs.to(device)
        proto_features, _, _ = pipnet_model(batch_inputs)
        output_size = proto_features.shape[-1]
        args.wshape = output_size  # needed for calculating image patch size
        print("Output shape: ", proto_features.shape)

    # Log the performance metrics
    logger.create_log('log_epoch_overview', 'epoch', 'test_top1_acc',
                      'mean_train_loss_during_epoch')

    pretrain_learning_rates = []
    # PRETRAINING PHASE (Training prototypes)
    for epoch in range(1, args.epochs_pretrain + 1):
        # Adjust trainable/frozen parameters for pretraining
        for param in train_params:
            param.requires_grad = True
        for param in pipnet_model._add_on.parameters():
            param.requires_grad = True
        for param in pipnet_model._classification.parameters():
            param.requires_grad = False
        for param in freeze_params:
            param.requires_grad = True
        for param in backbone_params:
            param.requires_grad = False

        print("\nPretrain Epoch", epoch)

        # Pretrain prototypes
        train_info = train_pipnet(pipnet_model,
                                  pretrain_loader,
                                  optimizer_pipnet,
                                  optimizer_classifier,
                                  scheduler_pipnet,
                                  None,
                                  criterion,
                                  epoch,
                                  args.epochs_pretrain,
                                  device,
                                  pretrain=True,
                                  finetune=False)
        pretrain_learning_rates += train_info['lrs_net']
        logger.log_values('log_epoch_overview', epoch, "n.a.",
                          train_info['loss'])
        wandb_logger.log_dict(train_info)

    # Save pretrained model checkpoint if not loaded from checkpoint
    if args.state_dict_dir_net == '':
        pipnet_model.eval()
        torch.save(
            {
                'model_state_dict': pipnet_model.state_dict(),
                'optimizer_net_state_dict': optimizer_pipnet.state_dict()
            },
            os.path.join(os.path.join(args.log_dir, 'checkpoints'),
                         'net_pretrained'))
        pipnet_model.train()

    # Visualize the top k prototypes after pretraining
    with torch.no_grad():
        if 'convnext' in args.net and args.epochs_pretrain > 0:
            visualize_topk(pipnet_model, project_loader, len(class_names),
                           device, 'visualised_pretrained_prototypes_topk',
                           args)

    # SECOND TRAINING PHASE
    # Re-initialize optimizers and schedulers for the second training phase
    optimizer_pipnet, optimizer_classifier, freeze_params, train_params, backbone_params = get_optimizer_nn(
        pipnet_model, args)
    scheduler_pipnet = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer_pipnet,
        T_max=len(train_loader) * args.epochs,
        eta_min=args.lr_net / 100.)
    scheduler_classifier = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer_classifier, T_0=10, eta_min=0.001, T_mult=1, verbose=False)

    # Freeze all parameters except the classification layer initially
    for param in pipnet_model.parameters():
        param.requires_grad = False
    for param in pipnet_model._classification.parameters():
        param.requires_grad = True

    # Fine-tuning setup
    is_frozen = True
    net_learning_rates = []
    classifier_learning_rates = []
    best_val_accuracy = 0

    # Main training loop for the second phase
    for epoch in range(1, args.epochs + 1):
        fine_tune_epochs = 3  # Freeze everything except the classification layer during fine-tuning
        if epoch <= fine_tune_epochs and (args.epochs_pretrain > 0
                                          or args.state_dict_dir_net != ''):
            # Freeze most of the network
            for param in pipnet_model._add_on.parameters():
                param.requires_grad = False
            for param in train_params:
                param.requires_grad = False
            for param in freeze_params:
                param.requires_grad = False
            for param in backbone_params:
                param.requires_grad = False
            is_finetuning = True
        else:
            is_finetuning = False
            if is_frozen:
                # Unfreeze the backbone after the fine-tuning phase
                if epoch > args.freeze_epochs:
                    for param in pipnet_model._add_on.parameters():
                        param.requires_grad = True
                    for param in freeze_params:
                        param.requires_grad = True
                    for param in train_params:
                        param.requires_grad = True
                    for param in backbone_params:
                        param.requires_grad = True
                    is_frozen = False
                else:
                    # Only freeze first layers of the backbone
                    for param in freeze_params:
                        param.requires_grad = True
                    for param in pipnet_model._add_on.parameters():
                        param.requires_grad = True
                    for param in train_params:
                        param.requires_grad = True
                    for param in backbone_params:
                        param.requires_grad = False

        print("\n Epoch", epoch, "frozen:", is_frozen)

        # Train PIP-Net model for the current epoch
        train_info = train_pipnet(pipnet_model,
                                  train_loader,
                                  optimizer_pipnet,
                                  optimizer_classifier,
                                  scheduler_pipnet,
                                  scheduler_classifier,
                                  criterion,
                                  epoch,
                                  args.epochs,
                                  device,
                                  pretrain=False,
                                  finetune=is_finetuning)
        net_learning_rates += train_info['lrs_net']
        classifier_learning_rates += train_info['lrs_class']

        # Evaluate the model on validation set
        eval_info = eval_pipnet(pipnet_model, val_loader, epoch, device,
                                logger)
        logger.log_values('log_epoch_overview', epoch,
                          eval_info['top1_accuracy'], train_info['loss'])
        wandb_logger.log_dict(eval_info)

        # Save trained model checkpoint after evaluation
        with torch.no_grad():
            pipnet_model.eval()
            torch.save(
                {
                    'model_state_dict':
                    pipnet_model.state_dict(),
                    'optimizer_net_state_dict':
                    optimizer_pipnet.state_dict(),
                    'optimizer_classifier_state_dict':
                    optimizer_classifier.state_dict()
                },
                os.path.join(os.path.join(args.log_dir, 'checkpoints'),
                             'net_trained'))

            if epoch % 30 == 0:
                pipnet_model.eval()
                torch.save(
                    {
                        'model_state_dict':
                        pipnet_model.state_dict(),
                        'optimizer_net_state_dict':
                        optimizer_pipnet.state_dict(),
                        'optimizer_classifier_state_dict':
                        optimizer_classifier.state_dict()
                    },
                    os.path.join(os.path.join(args.log_dir, 'checkpoints'),
                                 f'net_trained_{epoch}'))

        # Save best model if current accuracy is higher than previous best
        if eval_info['val_accuracy'] > best_val_accuracy:
            best_val_accuracy = eval_info['val_accuracy']
            pipnet_model.eval()
            torch.save(
                {
                    'epoch':
                    epoch,
                    'model_state_dict':
                    pipnet_model.state_dict(),
                    'optimizer_net_state_dict':
                    optimizer_pipnet.state_dict(),
                    'optimizer_classifier_state_dict':
                    optimizer_classifier.state_dict()
                },
                os.path.join(os.path.join(args.log_dir, 'checkpoints'),
                             'net_trained_best'))

    # Print relevant prototypes and their weights for each class
    for class_idx in range(pipnet_model._classification.weight.shape[0]):
        relevant_prototypes = []
        prototype_weights = pipnet_model._classification.weight[class_idx, :]
        for proto_idx in range(pipnet_model._classification.weight.shape[1]):
            if prototype_weights[proto_idx] > 1e-3:
                relevant_prototypes.append(
                    (proto_idx, prototype_weights[proto_idx].item()))
        class_name = list(val_loader.dataset.class_to_idx.keys())[list(
            val_loader.dataset.class_to_idx.values()).index(class_idx)]
        print(
            f"Class {class_idx} ({class_name}): has {len(relevant_prototypes)} relevant prototypes: {relevant_prototypes}",
            flush=True)

    # Load the best trained model checkpoint
    best_checkpoint = torch.load(
        os.path.join(os.path.join(args.log_dir, 'checkpoints'),
                     'net_trained_best'))
    pipnet_model.load_state_dict(best_checkpoint['model_state_dict'])
    print(f'Loaded best model from epoch: {best_checkpoint["epoch"]}',
          flush=True)

    # Visualize predictions and prototypes after training
    visualize(pipnet_model, project_loader, len(class_names), device,
              'visualised_prototypes', args)

    # If extra test images are provided, visualize their predictions
    if os.path.exists(args.extra_test_image_folder):
        vis_pred_experiments(pipnet_model, args.extra_test_image_folder,
                             class_names, device, args, wandb_logger)

    print("Training and evaluation completed successfully!")


# Function to initialize the network with Xavier weights for the prototypes and specific values for the classifier
def initialise_model(pipnet_model):
    pipnet_model._add_on.apply(init_weights_xavier)  # Initialize add-on layers
    torch.nn.init.normal_(pipnet_model._classification.weight,
                          mean=1.0,
                          std=0.1)  # Initialize classification layer weights
    torch.nn.init.constant_(pipnet_model._classification.bias,
                            val=0.)  # Initialize classification layer bias
    torch.nn.init.constant_(pipnet_model._multiplier,
                            val=2.)  # Initialize the multiplier parameter
    pipnet_model._multiplier.requires_grad = False  # Freeze the multiplier


# Function to load a pretrained checkpoint
def load_checkpoint(args, class_names, pipnet_model, optimizer_pipnet):
    checkpoint = torch.load(args.state_dict_dir_net, map_location=device)
    pipnet_model.load_state_dict(checkpoint['model_state_dict'], strict=True)
    print("Pretrained network loaded")
    pipnet_model._multiplier.requires_grad = False  # Freeze the multiplier
    try:
        optimizer_pipnet.load_state_dict(
            checkpoint['optimizer_net_state_dict'])
    except:
        pass

    # Check if the classification layer needs to be reinitialized
    if torch.mean(
            pipnet_model._classification.weight).item() > 1.0 and torch.mean(
                pipnet_model._classification.weight).item() < 3.0:
        if torch.count_nonzero(
                torch.relu(pipnet_model._classification.weight -
                           1e-5)).float().item() > 0.8 * (
                               pipnet_model.num_prototypes * len(class_names)):
            torch.nn.init.normal_(pipnet_model._classification.weight,
                                  mean=1.0,
                                  std=0.1)
            torch.nn.init.constant_(pipnet_model._multiplier, val=2.)


# Function to set the device (GPU or CPU) for training
def set_device():
    if torch.cuda.is_available():
        return torch.device('cuda')
    elif torch.backends.mps.is_available():
        return torch.device('mps')
    else:
        return torch.device('cpu')


# Function to set the random seed for reproducibility
def initialise_seed(seed: int):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    np.random.seed(seed)


if __name__ == '__main__':
    main()
