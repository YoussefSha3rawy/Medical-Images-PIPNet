import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from features.resnet_features import resnet18_features, resnet34_features, resnet50_features, resnet50_features_inat, resnet101_features, resnet152_features
from features.convnext_features import convnext_tiny_26_features, convnext_tiny_13_features
import torch
from torch import Tensor


class PIPNet(nn.Module):

    def __init__(self, num_classes: int, args: argparse.Namespace):
        super().__init__()
        assert num_classes > 0
        feature_net, add_on_layers, pool_layer, classification_layer, num_prototypes = get_network(
            num_classes, args.net)
        self._num_features = 0
        self._num_classes = num_classes
        self._num_prototypes = num_prototypes
        self._net = feature_net
        self._add_on = add_on_layers
        self._pool = pool_layer
        self._classification = classification_layer
        self._multiplier = classification_layer.normalization_multiplier

    @property
    def num_prototypes(self):
        return self._num_prototypes

    def forward(self, xs, inference=False):
        features = self._net(xs)
        proto_features = self._add_on(features)
        pooled = self._pool(proto_features)
        if inference:
            clamped_pooled = torch.where(
                pooled < 0.1, 0., pooled
            )  #during inference, ignore all prototypes that have 0.1 similarity or lower
            out = self._classification(
                clamped_pooled)  #shape (bs*2, num_classes)
            return proto_features, clamped_pooled, out
        else:
            out = self._classification(pooled)  #shape (bs*2, num_classes)
            return proto_features, pooled, out


base_architecture_to_features = {
    'resnet18': resnet18_features,
    'resnet34': resnet34_features,
    'resnet50': resnet50_features,
    'resnet50_inat': resnet50_features_inat,
    'resnet101': resnet101_features,
    'resnet152': resnet152_features,
    'convnext_tiny_26': convnext_tiny_26_features,
    'convnext_tiny_13': convnext_tiny_13_features,
}


# adapted from https://pytorch.org/docs/stable/_modules/torch/nn/modules/linear.html#Linear
class NonNegLinear(nn.Module):
    """Applies a linear transformation to the incoming data with non-negative weights`
    """

    def __init__(self,
                 in_features: int,
                 out_features: int,
                 bias: bool = True,
                 device=None,
                 dtype=None) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super(NonNegLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(
            torch.empty((out_features, in_features), **factory_kwargs))
        self.normalization_multiplier = nn.Parameter(
            torch.ones((1, ), requires_grad=True))
        if bias:
            self.bias = nn.Parameter(
                torch.empty(out_features, **factory_kwargs))
        else:
            self.register_parameter('bias', None)

    def forward(self, input: Tensor) -> Tensor:
        return F.linear(input, torch.relu(self.weight), self.bias)


def get_network(num_classes: int, net: str):
    features = base_architecture_to_features[net](pretrained=True)
    features_name = str(features).upper()
    if 'next' in net:
        features_name = str(net).upper()
    if features_name.startswith('RES') or features_name.startswith('CONVNEXT'):
        first_add_on_layer_in_channels = \
            [i for i in features.modules() if isinstance(i, nn.Conv2d)][-1].out_channels
    else:
        raise Exception('other base architecture NOT implemented')

    num_prototypes = first_add_on_layer_in_channels
    print("Number of prototypes: ", num_prototypes)
    add_on_layers = nn.Sequential(
        nn.Softmax(
            dim=1
        ),  #softmax over every prototype for each patch, such that for every location in image, sum over prototypes is 1                
    )
    pool_layer = nn.Sequential(
        nn.AdaptiveMaxPool2d(output_size=(1, 1)),  #outputs (bs, ps,1,1)
        nn.Flatten()  #outputs (bs, ps)
    )

    classification_layer = NonNegLinear(num_prototypes, num_classes, bias=True)

    return features, add_on_layers, pool_layer, classification_layer, num_prototypes
