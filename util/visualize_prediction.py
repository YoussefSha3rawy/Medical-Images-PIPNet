import os, shutil
import argparse
from PIL import Image, ImageDraw as D
import torchvision
from tqdm import tqdm
from util.func import get_patch_size
from torchvision import transforms
import torch
from util.vis_pipnet import get_img_coordinates
import matplotlib.pyplot as plt
import numpy as np

try:
    import cv2
    use_opencv = True
except ImportError:
    use_opencv = False
    print(
        "Heatmaps showing where a prototype is found will not be generated because OpenCV is not installed.",
        flush=True)


def vis_pred(net, vis_test_dir, classes, device, args: argparse.Namespace):
    # Make sure the model is in evaluation mode
    net.eval()

    save_dir = os.path.join(args.log_dir, args.dir_for_saving_images)
    if os.path.exists(save_dir):
        shutil.rmtree(save_dir)

    patchsize, skip = get_patch_size(args)

    num_workers = args.num_workers

    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)
    normalize = transforms.Normalize(mean=mean, std=std)
    transform_no_augment = transforms.Compose([
        transforms.Resize(size=(args.image_size, args.image_size)),
        transforms.ToTensor(), normalize
    ])

    vis_test_set = torchvision.datasets.ImageFolder(
        vis_test_dir, transform=transform_no_augment)
    vis_test_loader = torch.utils.data.DataLoader(
        vis_test_set,
        batch_size=1,
        shuffle=False,
        pin_memory=torch.cuda.is_available(),
        num_workers=num_workers)
    imgs = vis_test_set.imgs

    last_y = -1
    for k, (xs, ys) in enumerate(
            vis_test_loader
    ):  #shuffle is false so should lead to same order as in imgs
        if ys[0] != last_y:
            last_y = ys[0]
            count_per_y = 0
        else:
            count_per_y += 1
            if count_per_y > 5:  #show max 5 imgs per class to speed up the process
                continue
        xs, ys = xs.to(device), ys.to(device)
        img = imgs[k][0]
        img_name = os.path.splitext(os.path.basename(img))[0]
        dir = os.path.join(save_dir, img_name)
        if not os.path.exists(dir):
            os.makedirs(dir)
            shutil.copy(img, dir)

        with torch.no_grad():
            softmaxes, pooled, out = net(
                xs, inference=True
            )  #softmaxes has shape (bs, num_prototypes, W, H), pooled has shape (bs, num_prototypes), out has shape (bs, num_classes)
            sorted_out, sorted_out_indices = torch.sort(out.squeeze(0),
                                                        descending=True)
            for pred_class_idx in sorted_out_indices[:3]:
                pred_class = classes[pred_class_idx]
                save_path = os.path.join(
                    dir, pred_class + "_" +
                    str(f"{out[0,pred_class_idx].item():.3f}"))
                if not os.path.exists(save_path):
                    os.makedirs(save_path)
                sorted_pooled, sorted_pooled_indices = torch.sort(
                    pooled.squeeze(0), descending=True)
                simweights = []
                for prototype_idx in sorted_pooled_indices:
                    simweight = pooled[0, prototype_idx].item(
                    ) * net._classification.weight[pred_class_idx,
                                                   prototype_idx].item()
                    simweights.append(simweight)
                    if abs(simweight) > 0.01:
                        max_h, max_idx_h = torch.max(
                            softmaxes[0, prototype_idx, :, :], dim=0)
                        max_w, max_idx_w = torch.max(max_h, dim=0)
                        max_idx_h = max_idx_h[max_idx_w].item()
                        max_idx_w = max_idx_w.item()
                        image = transforms.Resize(size=(args.image_size,
                                                        args.image_size))(
                                                            Image.open(img))
                        img_tensor = transforms.ToTensor()(image).unsqueeze_(
                            0)  #shape (1, 3, h, w)
                        h_coor_min, h_coor_max, w_coor_min, w_coor_max = get_img_coordinates(
                            args.image_size, softmaxes.shape, patchsize, skip,
                            max_idx_h, max_idx_w)
                        img_tensor_patch = img_tensor[0, :,
                                                      h_coor_min:h_coor_max,
                                                      w_coor_min:w_coor_max]
                        img_patch = transforms.ToPILImage()(img_tensor_patch)
                        img_patch.save(
                            os.path.join(
                                save_path, 'mul%s_p%s_sim%s_w%s_patch.png' %
                                (str(f"{simweight:.3f}"),
                                 str(prototype_idx.item()),
                                 str(f"{pooled[0,prototype_idx].item():.3f}"),
                                 str(f"{net._classification.weight[pred_class_idx, prototype_idx].item():.3f}"
                                     ))))
                        draw = D.Draw(image)
                        draw.rectangle([(max_idx_w * skip, max_idx_h * skip),
                                        (min(args.image_size,
                                             max_idx_w * skip + patchsize),
                                         min(args.image_size,
                                             max_idx_h * skip + patchsize))],
                                       outline='yellow',
                                       width=2)
                        image.save(
                            os.path.join(
                                save_path, 'mul%s_p%s_sim%s_w%s_rect.png' %
                                (str(f"{simweight:.3f}"),
                                 str(prototype_idx.item()),
                                 str(f"{pooled[0,prototype_idx].item():.3f}"),
                                 str(f"{net._classification.weight[pred_class_idx, prototype_idx].item():.3f}"
                                     ))))

                        # visualise softmaxes as heatmap
                        if use_opencv:
                            softmaxes_resized = transforms.ToPILImage()(
                                softmaxes[0, prototype_idx, :, :])
                            softmaxes_resized = softmaxes_resized.resize(
                                (args.image_size, args.image_size),
                                Image.BICUBIC)
                            softmaxes_np = (transforms.ToTensor()(
                                softmaxes_resized)).squeeze().numpy()

                            heatmap = cv2.applyColorMap(
                                np.uint8(255 * softmaxes_np), cv2.COLORMAP_JET)
                            heatmap = np.float32(heatmap) / 255
                            heatmap = heatmap[..., ::-1]  # OpenCV's BGR to RGB
                            heatmap_img = 0.2 * np.float32(
                                heatmap) + 0.6 * np.float32(
                                    img_tensor.squeeze(0).numpy().transpose(
                                        1, 2, 0))
                            plt.imsave(fname=os.path.join(
                                save_path,
                                'heatmap_p%s.png' % str(prototype_idx.item())),
                                       arr=heatmap_img,
                                       vmin=0.0,
                                       vmax=1.0)


def vis_pred_experiments(net,
                         imgs_dir,
                         classes,
                         device,
                         args: argparse.Namespace,
                         wandb_logger=None):
    # Make sure the model is in evaluation mode
    net.eval()

    save_dir = os.path.join(
        os.path.join(args.log_dir, args.dir_for_saving_images), "Experiments")
    if os.path.exists(save_dir):
        shutil.rmtree(save_dir)

    patchsize, skip = get_patch_size(args)

    num_workers = args.num_workers

    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)
    normalize = transforms.Normalize(mean=mean, std=std)
    transform_no_augment = transforms.Compose([
        transforms.Resize(size=(args.image_size, args.image_size)),
        transforms.ToTensor(), normalize
    ])

    vis_test_set = torchvision.datasets.ImageFolder(
        imgs_dir, transform=transform_no_augment)
    vis_test_loader = torch.utils.data.DataLoader(
        vis_test_set,
        batch_size=1,
        shuffle=False,
        pin_memory=torch.cuda.is_available(),
        num_workers=num_workers)
    imgs = vis_test_set.imgs

    # Initialize arrays for storing predictions and labels
    all_predictions = []
    all_labels = []

    # Iterate over the test set
    for k, (xs, ys) in enumerate(
            tqdm(vis_test_loader)
    ):  # shuffle is false so should lead to the same order as in imgs

        xs, ys = xs.to(device), ys.to(device)
        img = imgs[k][0]
        img_name = os.path.splitext(os.path.basename(img))[0]
        dir = os.path.join(save_dir, img_name)

        if not os.path.exists(dir):
            os.makedirs(dir)
            shutil.copy(img, dir)

        with torch.no_grad():
            softmaxes, pooled, out = net(xs,
                                         inference=True)  # Get model output
            sorted_out, sorted_out_indices = torch.sort(out.squeeze(0),
                                                        descending=True)

            # Get predicted class (index of the highest output)
            pred_class_idx = sorted_out_indices[0].item()
            all_predictions.append(pred_class_idx)
            all_labels.append(ys.item())  # Store the true label

            for pred_class_idx in sorted_out_indices:
                pred_class = classes[pred_class_idx]
                save_path = os.path.join(
                    dir,
                    str(f"{out[0, pred_class_idx].item():.3f}") + "_" +
                    pred_class)
                if not os.path.exists(save_path):
                    os.makedirs(save_path)

                sorted_pooled, sorted_pooled_indices = torch.sort(
                    pooled.squeeze(0), descending=True)

                simweights = []
                for prototype_idx in sorted_pooled_indices:
                    simweight = pooled[0, prototype_idx].item(
                    ) * net._classification.weight[pred_class_idx,
                                                   prototype_idx].item()

                    simweights.append(simweight)
                    if abs(simweight) > 0.01:
                        max_h, max_idx_h = torch.max(
                            softmaxes[0, prototype_idx, :, :], dim=0)
                        max_w, max_idx_w = torch.max(max_h, dim=0)
                        max_idx_h = max_idx_h[max_idx_w].item()
                        max_idx_w = max_idx_w.item()

                        image = transforms.Resize(
                            size=(args.image_size, args.image_size))(
                                Image.open(img).convert("RGB"))
                        img_tensor = transforms.ToTensor()(image).unsqueeze_(
                            0)  # shape (1, 3, h, w)
                        h_coor_min, h_coor_max, w_coor_min, w_coor_max = get_img_coordinates(
                            args.image_size, softmaxes.shape, patchsize, skip,
                            max_idx_h, max_idx_w)
                        img_tensor_patch = img_tensor[0, :,
                                                      h_coor_min:h_coor_max,
                                                      w_coor_min:w_coor_max]
                        img_patch = transforms.ToPILImage()(img_tensor_patch)
                        img_patch.save(
                            os.path.join(
                                save_path, 'mul%s_p%s_sim%s_w%s_patch.png' %
                                (str(f"{simweight:.3f}"),
                                 str(prototype_idx.item()),
                                 str(f"{pooled[0, prototype_idx].item():.3f}"),
                                 str(f"{net._classification.weight[pred_class_idx, prototype_idx].item():.3f}"
                                     ))))
                        draw = D.Draw(image)
                        draw.rectangle([(max_idx_w * skip, max_idx_h * skip),
                                        (min(args.image_size,
                                             max_idx_w * skip + patchsize),
                                         min(args.image_size,
                                             max_idx_h * skip + patchsize))],
                                       outline='yellow',
                                       width=2)
                        image.save(
                            os.path.join(
                                save_path, 'mul%s_p%s_sim%s_w%s_rect.png' %
                                (str(f"{simweight:.3f}"),
                                 str(prototype_idx.item()),
                                 str(f"{pooled[0, prototype_idx].item():.3f}"),
                                 str(f"{net._classification.weight[pred_class_idx, prototype_idx].item():.3f}"
                                     ))))

    # Calculate accuracy
    all_predictions = np.array(all_predictions)
    all_labels = np.array(all_labels)
    accuracy = np.mean(all_predictions == all_labels)

    print(f"Accuracy: {accuracy * 100:.2f}%")

    if wandb_logger:
        wandb_log = {"test_accuracy": accuracy}
        wandb_logger.log(wandb_log)
        wandb_logger.log_confusion_matrix(all_labels, all_predictions)
