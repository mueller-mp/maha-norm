# ResNet variants
from __future__ import annotations

from functools import partial
from typing import Callable, Type

import torch
from torch import nn
from torchvision import models

def initialization(module: nn.Module,
                   use_zero_init: bool):
    init_parameters(module)
    if use_zero_init:
        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        for m in module.modules():
            if isinstance(m, BasicBlock):
                nn.init.constant_(m.norm2.weight, 0)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self,
                 in_planes: int,
                 planes: int,
                 stride: int,
                 groups: int,
                 width_per_group: int,
                 norm: Type[nn.BatchNorm2d],
                 act: Callable[[torch.Tensor], torch.Tensor]
                 ):
        super().__init__()
        planes = int(planes * (width_per_group / 16)) * groups
        self.conv1 = conv3x3(in_planes, planes, stride, bias=norm is None)
        self.conv2 = conv3x3(planes, planes, bias=norm is None)
        self.act = act
        self.norm1 = nn.Identity() if norm is None else norm(num_features=planes)
        self.norm2 = nn.Identity() if norm is None else norm(num_features=planes)

        self.downsample = nn.Identity()
        if in_planes != planes:
            self.downsample = nn.Sequential(conv1x1(in_planes, planes, stride=stride, bias=norm is None),
                                            nn.Identity() if norm is None else norm(num_features=planes))

    def forward(self,
                x: torch.Tensor
                ) -> torch.Tensor:
        out = self.conv1(x)
        out = self.norm1(out)
        out = self.act(out)

        out = self.conv2(out)
        out = self.norm2(out)

        out += self.downsample(x)
        out = self.act(out)

        return out


class PreactBasicBlock(BasicBlock):
    def __init__(self,
                 in_planes: int,
                 planes: int,
                 stride: int,
                 groups: int,
                 width_per_group: int,
                 norm: Type[nn.BatchNorm2d],
                 act: Callable[[torch.Tensor], torch.Tensor]
                 ):
        super().__init__(in_planes, planes, stride, groups, width_per_group, norm, act)
        self.norm1 = nn.Identity() if norm is None else norm(num_features=in_planes)
        if in_planes != planes:
            self.downsample = conv1x1(in_planes, planes, stride=stride, bias=norm is None)

    def forward(self,
                x: torch.Tensor
                ) -> torch.Tensor:
        out = self.norm1(x)
        out = self.act(out)
        out = self.conv1(out)

        out = self.norm2(out)
        out = self.act(out)
        out = self.conv2(out)

        out += self.downsample(x)
        return out


# for resnext
class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self,
                 in_planes: int,
                 planes: int,
                 stride: int,
                 groups: int,
                 width_per_group: int,
                 norm: Type[nn.BatchNorm2d],
                 act: Callable[[torch.Tensor], torch.Tensor]
                 ):
        super().__init__()
        width = int(planes * (width_per_group / 64)) * groups
        self.conv1 = conv1x1(in_planes, width, bias=norm is None)
        self.conv2 = conv3x3(width, width, stride, groups=groups, bias=norm is None)
        self.conv3 = conv1x1(width, planes * self.expansion, bias=norm is None)
        self.act = act
        self.norm1 = nn.Identity() if norm is None else norm(width)
        self.norm2 = nn.Identity() if norm is None else norm(width)
        self.norm3 = nn.Identity() if norm is None else norm(planes * self.expansion)

        self.downsample = nn.Identity()
        if stride != 1 or in_planes != planes * self.expansion:
            self.downsample = nn.Sequential(
                conv1x1(in_planes, planes * self.expansion, stride=stride, bias=norm is None),
                nn.Identity() if norm is None else norm(num_features=planes * self.expansion)
            )

    def forward(self,
                x: torch.Tensor
                ) -> torch.Tensor:
        out = self.conv1(x)
        out = self.norm1(out)
        out = self.act(out)
        out = self.conv2(out)
        out = self.norm2(out)
        out = self.act(out)

        out = self.conv3(out)
        out = self.norm3(out)
        out += self.downsample(x)
        return self.act(out)


# for SENet


class SEBasicBlock(BasicBlock):
    def __init__(self,
                 *args,
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.norm2 = nn.Sequential(self.norm2, SELayer(self.conv2.out_channels, kwargs['reduction']))


class SEBottleneck(Bottleneck):
    def __init__(self,
                 *args,
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.norm3 = nn.Sequential(self.norm3, SELayer(self.conv3.out_channels, kwargs['reduction']))


class ResNet(nn.Module):
    """ResNet for CIFAR data. For ImageNet classification, use `torchvision`'s.
    """

    def __init__(self,
                 block: Type[BasicBlock | Bottleneck],
                 num_classes: int,
                 layer_depth: int,
                 width: int = 16,
                 widen_factor: int = 1,
                 in_channels: int = 3,
                 groups: int = 1,
                 width_per_group: int = 16,
                 norm: Type[nn.BatchNorm2d] = nn.BatchNorm2d,
                 act: Callable[[torch.Tensor], torch.Tensor] = nn.ReLU(),
                 preact: bool = False,
                 final_pool: Callable[[torch.Tensor], torch.Tensor] = nn.AdaptiveAvgPool2d(1),
                 initializer: Callable[[nn.Module, None]] = None
                 ):
        super(ResNet, self).__init__()
        self.inplane = width
        self.groups = groups
        self.norm = norm
        self.width_per_group = width_per_group
        self.preact = preact

        self.conv1 = conv3x3(in_channels, width, stride=1, bias=norm is None)
        self.norm1 = nn.Identity() if norm is None else norm(4 * width * block.expansion * widen_factor if self.preact
                                                             else width)
        self.act = act
        self.layer1 = self._make_layer(block, width * widen_factor, layer_depth=layer_depth, stride=1)
        self.layer2 = self._make_layer(block, width * 2 * widen_factor, layer_depth=layer_depth, stride=2)
        self.layer3 = self._make_layer(block, width * 4 * widen_factor, layer_depth=layer_depth, stride=2)
        self.final_pool = final_pool
        self.fc = nn.Linear(4 * width * block.expansion * widen_factor, num_classes)
        if initializer is None:
            initialization(self, False)
        else:
            initializer(self)

    def _make_layer(self,
                    block: Type[BasicBlock | Bottleneck],
                    planes: int,
                    layer_depth: int,
                    stride: int,
                    ) -> nn.Sequential:
        layers = []
        for i in range(layer_depth):
            layers.append(
                block(self.inplane, planes, stride if i == 0 else 1,
                      self.groups, self.width_per_group, self.norm, self.act)
            )
            if i == 0:
                self.inplane = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x,return_feature=False,return_feature_list=False):
        x = self.conv1(x)

        if not self.preact:
            x = self.norm1(x)
            x = self.act(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        if self.preact:
            x = self.norm1(x)
            x = self.act(x)

        x = self.final_pool(x)
        features = x.flatten(1)
        x = self.fc(features)

        return (x,features) if return_feature else x

    def get_fc(self):
        fc=self.fc
        return fc.weight.cpu().detach().numpy(), fc.bias.cpu().detach().numpy()
        
    def get_fc_layer(self):
        return self.fc

def resnet(num_classes: int,
           depth: int,
           in_channels: int = 3,
           norm: Type[nn.BatchNorm2d] = nn.BatchNorm2d,
           act: Callable[[torch.Tensor], torch.Tensor] = nn.ReLU(),
           block: Type[BasicBlock] = BasicBlock,
           **kwargs
           ) -> ResNet:
    f"resnet-{depth}"
    assert (depth - 2) % 6 == 0
    layer_depth = (depth - 2) // 6
    return ResNet(block, num_classes, layer_depth, in_channels=in_channels, norm=norm, act=act, **kwargs)


def wide_resnet(num_classes: int,
                depth: int,
                widen_factor: int,
                in_channels: int = 3,
                norm: Type[nn.BatchNorm2d] = nn.BatchNorm2d,
                act: Callable[[torch.Tensor], torch.Tensor] = nn.ReLU(),
                block: Type[BasicBlock] = PreactBasicBlock,
                **kwargs
                ) -> ResNet:
    f"wideresnet-{depth}-{widen_factor}"
    assert (depth - 4) % 6 == 0
    layer_depth = (depth - 4) // 6
    return ResNet(block, num_classes, layer_depth, in_channels=in_channels,
                  widen_factor=widen_factor, norm=norm, act=act, preact=True, **kwargs)


def resnext(num_classes: int,
            depth: int,
            width_per_group: int,
            groups: int,
            in_channels: int,
            norm: Type[nn.BatchNorm2d] = nn.BatchNorm2d,
            act: Callable[[torch.Tensor], torch.Tensor] = nn.ReLU(),
            block: Type[Bottleneck] = Bottleneck,
            **kwargs
            ) -> ResNet:
    f"resnext-{depth}_{groups}x{width_per_group}d"
    assert (depth - 2) % 9 == 0
    layer_depth = (depth - 2) // 9
    return ResNet(block, num_classes, layer_depth, width=64, in_channels=in_channels, groups=groups,
                  width_per_group=width_per_group, norm=norm, act=act, **kwargs)



def resnet20(num_classes: int = 10,
             in_channels: int = 3
             ) -> ResNet:
    """ ResNet by He+16
    """
    return resnet(num_classes, 20, in_channels)



def resnet32(num_classes: int = 10,
             in_channels: int = 3
             ) -> ResNet:
    """ ResNet by He+16
    """
    return resnet(num_classes, 32, in_channels)



def resnet56(num_classes: int = 10,
             in_channels: int = 3
             ) -> ResNet:
    """ ResNet by He+16
    """
    return resnet(num_classes, 56, in_channels)



def resnet110(num_classes: int = 10,
              in_channels: int = 3
              ) -> ResNet:
    """ ResNet by He+16
    """
    return resnet(num_classes, 110, in_channels)



def se_resnet20(num_classes: int = 10,
                in_channels: int = 3
                ) -> ResNet:
    """ SEResNet by Hu+18
    """
    return resnet(num_classes, 20, in_channels, block=partial(SEBasicBlock, reduction=16))



def se_resnet56(num_classes: int = 10,
                in_channels: int = 3
                ) -> ResNet:
    """ SEResNet by Hu+18
    """
    return resnet(num_classes, 56, in_channels, block=partial(SEBasicBlock, reduction=16))



def wrn16_8(num_classes: int = 10,
            in_channels: int = 3
            ) -> ResNet:
    """ WideResNet by Zagoruyko&Komodakis 17
    """
    return wide_resnet(num_classes, 16, 8, in_channels)



def wrn28_2(num_classes: int = 10,
            in_channels: int = 3
            ) -> ResNet:
    """ WideResNet by Zagoruyko&Komodakis 17
    """
    return wide_resnet(num_classes, 28, 2, in_channels)



def wrn28_10(num_classes: int = 10,
             in_channels: int = 3
             ) -> ResNet:
    """ WideResNet by Zagoruyko&Komodakis 17
    """
    return wide_resnet(num_classes, 28, 10, in_channels)



def wrn40_2(num_classes: int = 10,
            in_channels: int = 3
            ) -> ResNet:
    """ WideResNet by Zagoruyko&Komodakis 17
    """
    return wide_resnet(num_classes, 40, 2, in_channels)



def resnext29_32x4d(num_classes: int = 10,
                    in_channels: int = 3
                    ) -> ResNet:
    """ ResNeXT by Xie+17
    """
    return resnext(num_classes, 29, 4, 32, in_channels)



def resnext29_8x64d(num_classes: int = 10,
                    in_channels: int = 3
                    ) -> ResNet:
    """ ResNeXT by Xie+17
    """
    return resnext(num_classes, 29, 64, 8, in_channels)



def wrn28_2_attention_pool(num_classes: int = 10,
                           in_channels: int = 3,
                           num_heads: int = 2
                           ) -> ResNet:
    return wide_resnet(num_classes, 28, 2, in_channels, final_pool=AttentionPool2d(2 * 64, num_heads))



def wrn28_10_attention_pool(num_classes: int = 10,
                            in_channels: int = 3,
                            num_heads: int = 10
                            ) -> ResNet:
    return wide_resnet(num_classes, 28, 10, in_channels, final_pool=AttentionPool2d(10 * 64, num_heads))


class TVResNet(models.ResNet):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.maxpool = nn.Identity()



def cifar_resnet18(num_classes: int = 10,
                   ) -> TVResNet:
    return TVResNet(models.resnet.BasicBlock, [2, 2, 2, 2], num_classes=num_classes)



def cifar_resnet50(num_classes: int = 10,
                   ) -> TVResNet:
    return TVResNet(models.resnet.Bottleneck, [3, 4, 6, 3], num_classes=num_classes)



import torch
from torch import nn
from torch.nn import functional as F



class KeyValAttention(nn.Module):
    """ Key-value attention.

    :param scaling:
    :param dropout_prob:
    """

    def __init__(self,
                 scaling: bool = False,
                 dropout_prob: float = 0):
        super(KeyValAttention, self).__init__()
        self._scaling = scaling
        self._dropout = dropout_prob

    def forward(self,
                query: torch.Tensor,
                key: torch.Tensor,
                value: torch.Tensor,
                mask: torch.Tensor = None,
                additive_mask: torch.Tensor = None,
                ) -> tuple[torch.Tensor, torch.Tensor]:
        """ See `functional.attention.kv_attention` for details

        :param query:
        :param key:
        :param value:
        :param mask:
        :param additive_mask:
        :return:
        """

        return kv_attention(query, key, value, mask, additive_mask,
                            self.training, self._dropout, self._scaling)


class AttentionPool2d(nn.Module):
    # from openAI clip
    def __init__(self,
                 embed_dim: int,
                 num_heads: int):
        super().__init__()
        self.k_proj = nn.Parameter(torch.randn(embed_dim, embed_dim))
        self.q_proj = nn.Parameter(torch.randn(embed_dim, embed_dim))
        self.v_proj = nn.Parameter(torch.randn(embed_dim, embed_dim))
        self.bias = nn.Parameter(torch.randn(3 * embed_dim))
        self.c_proj = nn.Linear(embed_dim, embed_dim)
        self.num_heads = num_heads
        self.initialize_weights()

    def forward(self,
                x: torch.Tensor
                ) -> torch.Tensor:
        x = x.flatten(-2).permute(2, 0, 1)  # NCHW -> (HW)NC
        x = torch.cat([x.mean(dim=0, keepdim=True), x], dim=0)  # (1+HW)NC
        x, _ = F.multi_head_attention_forward(
            query=x, key=x, value=x,
            embed_dim_to_check=x.shape[-1],
            num_heads=self.num_heads,
            q_proj_weight=self.q_proj,
            k_proj_weight=self.k_proj,
            v_proj_weight=self.v_proj,
            in_proj_weight=None,
            in_proj_bias=self.bias,
            bias_k=None,
            bias_v=None,
            add_zero_attn=False,
            dropout_p=0,
            out_proj_weight=self.c_proj.weight,
            out_proj_bias=self.c_proj.bias,
            use_separate_proj_weight=True,
            training=self.training,
            need_weights=False
        )
        return x[0]

    def initialize_weights(self):
        std = self.c_proj.in_features ** -0.5
        nn.init.normal_(self.k_proj, std=std)
        nn.init.normal_(self.q_proj, std=std)
        nn.init.normal_(self.v_proj, std=std)
        nn.init.normal_(self.c_proj.weight, std=std)
        nn.init.zeros_(self.bias)

import torch
from torch import nn


def init_parameters(module: nn.Module):
    """initialize parameters using kaiming normal"""
    for m in module.modules():
        if isinstance(m, nn.Conv2d):
            # todo: check if fan_out is valid
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)


def conv3x3(in_planes: int,
            out_planes: int,
            stride: int = 1,
            bias: bool = False,
            groups: int = 1
            ) -> nn.Conv2d:
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=bias, groups=groups)


def conv1x1(in_planes: int,
            out_planes: int,
            stride=1,
            bias: bool = False
            ) -> nn.Conv2d:
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, padding=0, bias=bias)


class SELayer(nn.Module):
    def __init__(self,
                 planes: int,
                 reduction: int):
        super().__init__()
        self.module = nn.Sequential(nn.AdaptiveAvgPool2d(1),
                                    conv1x1(planes, planes // reduction, bias=False),
                                    nn.ReLU(inplace=True),
                                    conv1x1(planes // reduction, planes, bias=False),
                                    nn.Sigmoid())

    def forward(self,
                x: torch.Tensor
                ) -> torch.Tensor:
        return x * self.module(x)


import torch
from torch.nn import functional as F


def kv_attention(query: torch.Tensor,
                 key: torch.Tensor,
                 value: torch.Tensor,
                 mask: torch.Tensor = None,
                 additive_mask: torch.Tensor = None,
                 training: bool = True,
                 dropout_prob: float = 0,
                 scaling: bool = True
                 ) -> tuple[torch.Tensor, torch.Tensor]:
    """Attention using queries, keys and value

    :param query: `...JxM`
    :param key: `...KxM`
    :param value: `...KxM`
    :param mask: `...JxK`
    :param additive_mask:
    :param training:
    :param dropout_prob:
    :param scaling:
    :return: torch.Tensor whose shape of `...JxM`
    """

    if scaling:
        query /= (query.size(-1) ** 0.5)
    attn = torch.einsum('...jm,...km->...jk', query, key).softmax(dim=-1)
    if mask is not None:
        if mask.dim() < attn.dim():
            mask.unsqueeze_(0)
        attn = attn.masked_fill(mask == 0, 1e-9)
    if additive_mask is not None:
        attn += additive_mask
    if training and dropout_prob > 0:
        attn = F.dropout(attn, p=dropout_prob)

    return torch.einsum('...jk,...km->...jm', attn, value), attn