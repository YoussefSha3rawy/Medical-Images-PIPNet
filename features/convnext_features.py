import torch
import torch.nn as nn
from torchvision import models
import timm
import os


def replace_convlayers_convnext(model, threshold):
    for n, module in model.named_children():
        if len(list(module.children())) > 0:
            replace_convlayers_convnext(module, threshold)
        if isinstance(module, nn.Conv2d):
            if module.stride[0] == 2:
                if module.in_channels > threshold:  #replace bigger strides to reduce receptive field, skip some 2x2 layers. >100 gives output size (26, 26). >300 gives (13, 13)
                    module.stride = tuple(s // 2 for s in module.stride)

    return model


def convnext_tiny_26_features(pretrained=False, **kwargs):
    model = models.convnext_tiny(pretrained=pretrained,
                                 weights=models.ConvNeXt_Tiny_Weights.DEFAULT)
    with torch.no_grad():
        model.avgpool = nn.Identity()
        model.classifier = nn.Identity()
        model = replace_convlayers_convnext(model, 100)

    return model


def convnext_tiny_13_features(pretrained=False, **kwargs):
    model = models.convnext_tiny(pretrained=pretrained,
                                 weights=models.ConvNeXt_Tiny_Weights.DEFAULT)
    with torch.no_grad():
        model.avgpool = nn.Identity()
        model.classifier = nn.Identity()
        model = replace_convlayers_convnext(model, 300)

    return model


class ViTFeatureExtractor(torch.nn.Module):

    def __init__(self, model):
        super(ViTFeatureExtractor, self).__init__()
        self.model = model

    # Override the forward function to return only the last output
    def forward(self, x):
        features = self.model(x)  # Get the list of feature maps
        return features[-1]  # Return the last feature map


def vit_features(pretrained=False):
    vit_features_only = timm.create_model('vit_base_patch16_224',
                                          pretrained=pretrained,
                                          features_only=True)

    source_state_dict = torch.load(os.path.join('pretrained_models',
                                                'VisionTransformer_ckpt.pth'),
                                   map_location='cpu')

    target_state_dict = vit_features_only.state_dict()

    filtered_state_dict = {}

    for layer_name, weights in source_state_dict.items():
        if layer_name in target_state_dict and target_state_dict[
                layer_name].shape == weights.shape:
            filtered_state_dict[layer_name] = weights

    vit_features_only.load_state_dict(filtered_state_dict, strict=False)
    model = ViTFeatureExtractor(vit_features_only)

    return model


if __name__ == "__main__":
    features = vit_features(pretrained=True)

    model = timm.create_model('vit_base_patch16_224',
                              pretrained=True,
                              num_classes=0)
    model.norm = nn.Identity()

    print(features)

    print('===========')

    # print(model)

    rand_input = torch.randn(16, 3, 224, 224)

    with torch.no_grad():
        out1 = features(rand_input)
        out2 = model(rand_input)

    print(out1.shape)
    print(out2.shape)
