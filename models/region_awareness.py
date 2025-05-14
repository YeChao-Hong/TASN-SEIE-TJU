import torch
from torch import Tensor
import torch.nn as nn
from typing import Type, Any, Callable, Union, List, Optional
from torch.nn.functional import softmax
import math
from torch.nn import TransformerEncoder, TransformerEncoderLayer

try:
    from torch.hub import load_state_dict_from_url
except ImportError:
    from torch.utils.model_zoo import load_url as load_state_dict_from_url

model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-f37072fd.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-b627a593.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-0676ba61.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-63fe2227.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-394f9c45.pth',
    'resnext50_32x4d': 'https://download.pytorch.org/models/resnext50_32x4d-7cdf4587.pth',
    'resnext101_32x8d': 'https://download.pytorch.org/models/resnext101_32x8d-8ba56ff5.pth',
    'wideget_backbone50_2': 'https://download.pytorch.org/models/wideget_backbone50_2-95faca4d.pth',
    'wideget_backbone101_2': 'https://download.pytorch.org/models/wideget_backbone101_2-32ee1156.pth',
}




def conv3x3(in_planes: int, out_planes: int, stride: int = 1, groups: int = 1, dilation: int = 1) -> nn.Conv2d:
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

# 先把维度调成(N,5,2816)才可以输入这个位置编码，然后再变回(5,N,2816)进入encoder
# class PositionalEncoding(nn.Module):
#     def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
#         super().__init__()
#         self.dropout = nn.Dropout(p=dropout)
#         position = torch.arange(max_len).unsqueeze(1)
#         div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
#         pe = torch.zeros(max_len, d_model)
#         pe[:, 0::2] = torch.sin(position * div_term)
#         pe[:, 1::2] = torch.cos(position * div_term)
#         self.register_buffer('pe', pe)
#
#     def forward(self, x: Tensor) -> Tensor:
#         x = x + self.pe[:x.size(1)]
#         return self.dropout(x)

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)

class BasicBlock(nn.Module):
    expansion: int = 1

    def __init__(
            self,
            inplanes: int,
            planes: int,
            stride: int = 1,
            downsample: Optional[nn.Module] = None,
            groups: int = 1,
            base_width: int = 64,
            dilation: int = 1,
            norm_layer: Optional[Callable[..., nn.Module]] = None
                 ) -> None:
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:     #膨胀卷积参数？
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion: int = 4

    def __init__(
            self,
            inplanes: int,
            planes: int,
            stride: int = 1,
            downsample: Optional[nn.Module] = None,
            groups: int = 1,
            base_width: int = 64,
            dilation: int = 1,
            norm_layer: Optional[Callable[..., nn.Module]] = None
    ) -> None:
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(
            # self,
            # block: Type[Union[BasicBlock, Bottleneck]],
            # layers: List[int],
            # num_classes: int = 1000,
            # zero_init_residual: bool = False,
            # groups: int = 1,
            # width_per_group: int = 64,
            # replace_stride_with_dilation: Optional[List[bool]] = None,
            # norm_layer: Optional[Callable[..., nn.Module]] = None
            self,
            block: Type[Union[BasicBlock, Bottleneck]],
            layers: List[int],
            num_classes: int = 1000,
            zero_init_residual: bool = False,
            groups: int = 1,
            width_per_group: int = 64,
            replace_stride_with_dilation: Optional[List[bool]] = None,
            norm_layer: Optional[Callable[..., nn.Module]] = None
    ) -> None:
        super(ResNet, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2,
                                       dilate=replace_stride_with_dilation[2])
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.lip_conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.lip_bn1 = norm_layer(self.inplanes)
        self.lip_relu = nn.ReLU(inplace=True)
        self.lip_maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.lip_layer1 = self._make_layer(block, 64, layers[0])
        self.lip_layer2 = self._make_layer(block, 128, layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0])
        self.lip_layer3 = self._make_layer(block, 256, layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1])
        self.lip_layer4 = self._make_layer(block, 512, layers[3], stride=2,
                                       dilate=replace_stride_with_dilation[2])
        self.lip_avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.get_weight = nn.Sequential(
            nn.Linear(512 * block.expansion + 768, 1),  # TODO: 768 is the length of global feature
            nn.Sigmoid()
        )
        self.transformer = nn.ModuleDict({
            'encoder': TransformerEncoder(
                encoder_layer=TransformerEncoderLayer(
                    d_model=512 * block.expansion,  # 输入特征维度
                    nhead=8,
                    dim_feedforward=2048,
                    dropout=0.1
                ),
                num_layers=2
            ),
            'pos_encoder': PositionalEncoding(
                d_model=512 * block.expansion,
                dropout=0.1,
                max_len=4
            )
        })
        # 修改RNN输入维度为ResNet特征维度
        self.rnn = nn.LSTM(
            input_size=512 * block.expansion,  # 仅使用ResNet特征
            hidden_size=512,
            num_layers=2,
            bidirectional=True,
            batch_first=True
        )
        self.norm1 = nn.LayerNorm(normalized_shape=1024)  # 对 lip_transformer_output 的最后 2048 维做归一化
        self.norm2 = nn.LayerNorm(normalized_shape=2816)  # 对 原输出 的最后 816 维做归一化
        # 修改最后的全连接层
        # self.fc = nn.Sequential(
        #     nn.Linear(2 * (512 * block.expansion) + 768, 1024),
        #     nn.ReLU(),
        #     nn.Linear(1024, 1)
        # )
        # self.fc = nn.Sequential(
        #     # 第一降维层 (4864 -> 2048)
        #     nn.Linear(2 * (512 * block.expansion) + 768, 2048),
        #     nn.ReLU(),
        #     nn.Dropout(p=0.3),  # 防止过拟合
        #
        #     # 第二降维层 (2048 -> 512)
        #     nn.Linear(2048, 512),
        #     nn.ReLU(),
        #     nn.Dropout(p=0.2),  # 递减的dropout
        #
        #     # 最终输出层 (512 -> 1)
        #     nn.Linear(512, 1)
        # )

        # self.fc = nn.Linear(512 * block.expansion + 768, 1)
        self.fc = nn.Linear(512 * block.expansion + 768 + 1024, 1)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)  # type: ignore[arg-type]
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)  # type: ignore[arg-type]

    def _make_layer(self, block: Type[Union[BasicBlock, Bottleneck]], planes: int, blocks: int,
                    stride: int = 1, dilate: bool = False) -> nn.Sequential:
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))

        return nn.Sequential(*layers)


    def _forward_impl(self, x, feature):
        def standardize(tensor):
            mean = tensor.mean(dim=0, keepdim=True)
            std = tensor.std(dim=0, keepdim=True)
            return (tensor - mean) / std
        # The comment resolution is based on input size is 224*224 imagenet
        # f.shape: (batch_size, 3, 224, 224), feature.shape: (batch_size, 768)
        features, weights, parts, weights_org, weights_max = [list() for i in range(5)]
        lip_features = []
        for i in range(0, len(x[2])):
            f = x[2][i]-x[2][i-1]
            f = self.lip_conv1(f)
            f = self.lip_bn1(f)
            f = self.lip_relu(f)
            f = self.lip_maxpool(f)
            f = self.lip_layer1(f)
            f = self.lip_layer2(f)
            f = self.lip_layer3(f)
            f = self.lip_layer4(f)
            f = self.lip_avgpool(f)
            f = torch.flatten(f, 1)
            lip_features.append(f)
        # lip_transformer_input = torch.stack(lip_features, dim=0)
        # lip_transformer_input = self.transformer['pos_encoder'](lip_transformer_input)
        # lip_transformer_output = self.transformer['encoder'](lip_transformer_input)  # (4, N, 2048)
        # # 使用CLS token（取第一个位置）或平均池化
        # # out = transformer_output.mean(dim=0)  # (N, 2048)
        # lip_transformer_output = lip_transformer_output[0]  # 取第一个时间步
        rnn_input = torch.stack(lip_features, dim=1)
        rnn_out, _ = self.rnn(rnn_input)
        lip_transformer_output = rnn_out[:, -1, :]  # 取最后时间步 (N,1024)

        for i in range(len(x[0])):
            features.clear()
            weights.clear()
            for j in range(len(x)):
                #头、面、嘴
                f = x[j][i]
                f = self.conv1(f)
                f = self.bn1(f)
                f = self.relu(f)
                f = self.maxpool(f)
                f = self.layer1(f)
                f = self.layer2(f)
                f = self.layer3(f)
                f = self.layer4(f)
                f = self.avgpool(f)
                f = torch.flatten(f, 1)

                # features.append(f)

                features.append(torch.cat([f, feature], dim=1))  # concat regional feature with global feature
                weights.append(self.get_weight(features[-1]))

            features_stack = torch.stack(features, dim=2)
            weights_stack = torch.stack(weights, dim=2)
            weights_stack = softmax(weights_stack, dim=2)

            weights_max.append(weights_stack[:, :, :len(x)].max(dim=2)[0])
            weights_org.append(weights_stack[:, :, 0])
            parts.append(features_stack.mul(weights_stack).sum(2).div(weights_stack.sum(2)))
        parts_stack = torch.stack(parts, dim=0)  # (5, N, 2816)
        out = self.norm2(parts_stack.sum(0).div(parts_stack.shape[0]))
        lip_transformer_output = self.norm1(lip_transformer_output)
        # print("lip_transformer_output shape:", lip_transformer_output.shape)
        # print("out shape:", out.shape)
        out = torch.cat([lip_transformer_output, out], dim=1)
        # transformer_output_0_std = standardize(lip_transformer_output)
        # parts_stack_avg_std = standardize(parts_stack.sum(0).div(parts_stack.shape[0]))
        # out = torch.cat([transformer_output_0_std, parts_stack_avg_std], dim=1)
        pred_score = self.fc(out)

        return pred_score, weights_max, weights_org

    def forward(self, x, feature):
        return self._forward_impl(x, feature)




def _get_backbone(
        arch: str,
        block: Type[Union[BasicBlock, Bottleneck]],
        layers: List[int],
        pretrained: bool,
        progress: bool,
        **kwargs: Any
) -> ResNet:
    model = ResNet(block, layers, num_classes=1, **kwargs)
    if pretrained:
        state_dict = load_state_dict_from_url(model_urls[arch], progress=progress)
        model.load_state_dict(state_dict)
    return model


def get_backbone(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> ResNet:
    r"""ResNet-50 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _get_backbone('resnet50', Bottleneck, [3, 4, 6, 3], pretrained, progress, **kwargs)


if __name__ == '__main__':
    model = get_backbone()
    data = [[] for i in range(3)]
    for i in range(3):
        for j in range(5):
            data[i].append(torch.rand((10, 3, 224, 224)))
    feature = torch.rand((10, 768))
    pred_score, weights_max, weights_org = model(data, feature)
    pass

# import math
# import copy
# from functools import partial
# from collections import OrderedDict
# from typing import Optional, Callable
# from torch.nn.functional import softmax
# import torch
# import torch.nn as nn
# from torch import Tensor
# from torch.nn import functional as F
# from typing import Any, Callable, Optional
#
# """
# 该方法就是将传入的channel的个数调整到8的整数倍，使得对硬件友好
# """
#
#
# def _make_divisible(ch, divisor=8, min_ch=None):
#     """
#     This function is taken from the original tf repo.
#     It ensures that all layers have a channel number that is divisible by 8
#     It can be seen here:
#     https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
#     """
#     if min_ch is None:
#         min_ch = divisor
#     new_ch = max(min_ch, int(ch + divisor / 2) // divisor * divisor)
#     # Make sure that round down does not go down by more than 10%.
#     if new_ch < 0.9 * ch:
#         new_ch += divisor
#     return new_ch
#
#
# """
#
# """
#
#
# def drop_path(x, drop_prob: float = 0., training: bool = False):
#     """
#     Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).
#     "Deep Networks with Stochastic Depth", https://arxiv.org/pdf/1603.09382.pdf
#
#     This function is taken from the rwightman.
#     It can be seen here:
#     https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/layers/drop.py#L140
#     """
#     if drop_prob == 0. or not training:
#         return x
#     keep_prob = 1 - drop_prob
#     shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
#     random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
#     random_tensor.floor_()  # binarize
#     output = x.div(keep_prob) * random_tensor
#     return output
#
#
# """
#
# """
#
#
# class DropPath(nn.Module):
#     """
#     Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
#     "Deep Networks with Stochastic Depth", https://arxiv.org/pdf/1603.09382.pdf
#     """
#
#     def __init__(self, drop_prob=None):
#         super(DropPath, self).__init__()
#         self.drop_prob = drop_prob
#
#     def forward(self, x):
#         return drop_path(x, self.drop_prob, self.training)
#
#
# """
# 定义BN和激活函数
# """
#
#
# class ConvBNActivation(nn.Sequential):
#     def __init__(self,
#                  in_planes: int,
#                  out_planes: int,
#                  kernel_size: int = 3,
#                  stride: int = 1,
#                  groups: int = 1,
#                  norm_layer: Optional[Callable[..., nn.Module]] = None,  # BN结构，默认为None
#                  activation_layer: Optional[Callable[..., nn.Module]] = None):  # ac结构，默认为None
#         padding = (kernel_size - 1) // 2
#         if norm_layer is None:
#             """
#             不指定norm的话，默认BN结构
#             """
#             norm_layer = nn.BatchNorm2d
#         if activation_layer is None:
#             """
#             如果不指定激活函数，则默认SiLU激活函数
#             """
#             activation_layer = nn.SiLU  # alias Swish  (torch>=1.7)激活函数使用的是SiLU，在版本1.70以上才有，建议1.7.1版本
#
#         # 定义层结构[in_channels,out_channels,kernel_size,stride,padding,groups,bias=False],-->BN-->ac
#         super(ConvBNActivation, self).__init__(nn.Conv2d(in_channels=in_planes,
#                                                          out_channels=out_planes,
#                                                          kernel_size=kernel_size,
#                                                          stride=stride,
#                                                          padding=padding,
#                                                          groups=groups,
#                                                          bias=False),
#                                                norm_layer(out_planes),
#                                                activation_layer())
#
#
# """
# 定义SE模块
# """
#
#
# class SqueezeExcitation(nn.Module):
#     def __init__(self,
#                  input_c: int,  # block input channel
#                  expand_c: int,  # block expand channel
#                  squeeze_factor: int = 4):
#         super(SqueezeExcitation, self).__init__()
#         squeeze_c = input_c // squeeze_factor
#         self.fc1 = nn.Conv2d(expand_c, squeeze_c, 1)
#         self.ac1 = nn.SiLU()  # alias Swish
#         self.fc2 = nn.Conv2d(squeeze_c, expand_c, 1)
#         self.ac2 = nn.Sigmoid()
#
#     def forward(self, x: Tensor) -> Tensor:
#         scale = F.adaptive_avg_pool2d(x, output_size=(1, 1))
#         scale = self.fc1(scale)
#         scale = self.ac1(scale)
#         scale = self.fc2(scale)
#         scale = self.ac2(scale)
#         return scale * x
#
#
# """
#
# """
#
#
# class InvertedResidualConfig:
#     # kernel_size, in_channel, out_channel, exp_ratio, strides, use_SE, drop_connect_rate
#     def __init__(self,
#                  kernel: int,  # 3 or 5
#                  input_c: int,
#                  out_c: int,
#                  expanded_ratio: int,  # 1 or 6
#                  stride: int,  # 1 or 2
#                  use_se: bool,  # True
#                  drop_rate: float,
#                  index: str,  # 1a, 2a, 2b, ...
#                  width_coefficient: float):
#         self.input_c = self.adjust_channels(input_c, width_coefficient)
#         self.kernel = kernel
#         self.expanded_c = self.input_c * expanded_ratio
#         self.out_c = self.adjust_channels(out_c, width_coefficient)
#         self.use_se = use_se
#         self.stride = stride
#         self.drop_rate = drop_rate
#         self.index = index
#
#     @staticmethod
#     def adjust_channels(channels: int, width_coefficient: float):
#         return _make_divisible(channels * width_coefficient, 8)
#
#
# """
#
# """
#
#
# class InvertedResidual(nn.Module):
#     def __init__(self,
#                  cnf: InvertedResidualConfig,
#                  norm_layer: Callable[..., nn.Module]):
#         super(InvertedResidual, self).__init__()
#
#         if cnf.stride not in [1, 2]:
#             raise ValueError("illegal stride value.")
#
#         self.use_res_connect = (cnf.stride == 1 and cnf.input_c == cnf.out_c)
#
#         layers = OrderedDict()
#         activation_layer = nn.SiLU  # alias Swish
#
#         # expand
#         if cnf.expanded_c != cnf.input_c:
#             layers.update({"expand_conv": ConvBNActivation(cnf.input_c,
#                                                            cnf.expanded_c,
#                                                            kernel_size=1,
#                                                            norm_layer=norm_layer,
#                                                            activation_layer=activation_layer)})
#
#         # depthwise
#         layers.update({"dwconv": ConvBNActivation(cnf.expanded_c,
#                                                   cnf.expanded_c,
#                                                   kernel_size=cnf.kernel,
#                                                   stride=cnf.stride,
#                                                   groups=cnf.expanded_c,
#                                                   norm_layer=norm_layer,
#                                                   activation_layer=activation_layer)})
#
#         if cnf.use_se:
#             layers.update({"se": SqueezeExcitation(cnf.input_c,
#                                                    cnf.expanded_c)})
#
#         # project
#         layers.update({"project_conv": ConvBNActivation(cnf.expanded_c,
#                                                         cnf.out_c,
#                                                         kernel_size=1,
#                                                         norm_layer=norm_layer,
#                                                         activation_layer=nn.Identity)})
#
#         self.block = nn.Sequential(layers)
#         self.out_channels = cnf.out_c
#         self.is_strided = cnf.stride > 1
#
#         # 只有在使用shortcut连接时才使用dropout层
#         if self.use_res_connect and cnf.drop_rate > 0:
#             self.dropout = DropPath(cnf.drop_rate)
#         else:
#             self.dropout = nn.Identity()
#
#     def forward(self, x: Tensor) -> Tensor:
#         result = self.block(x)
#         result = self.dropout(result)
#         if self.use_res_connect:
#             result += x
#
#         return result
#
#
# """
#
# """
#
#
# class EfficientNet(nn.Module):
#     def __init__(self,
#                  width_coefficient: float,
#                  depth_coefficient: float,
#                  num_classes: int = 1000,
#                  dropout_rate: float = 0.2,
#                  drop_connect_rate: float = 0.2,
#                  block: Optional[Callable[..., nn.Module]] = None,
#                  norm_layer: Optional[Callable[..., nn.Module]] = None
#                  ):
#         super(EfficientNet, self).__init__()
#
#         # kernel_size, in_channel, out_channel, exp_ratio, strides, use_SE, drop_connect_rate, repeats
#         default_cnf = [[3, 32, 16, 1, 1, True, drop_connect_rate, 1],
#                        [3, 16, 24, 6, 2, True, drop_connect_rate, 2],
#                        [5, 24, 40, 6, 2, True, drop_connect_rate, 2],
#                        [3, 40, 80, 6, 2, True, drop_connect_rate, 3],
#                        [5, 80, 112, 6, 1, True, drop_connect_rate, 3],
#                        [5, 112, 192, 6, 2, True, drop_connect_rate, 4],
#                        [3, 192, 320, 6, 1, True, drop_connect_rate, 1]]
#
#         def round_repeats(repeats):
#             """Round number of repeats based on depth multiplier."""
#             return int(math.ceil(depth_coefficient * repeats))
#
#         if block is None:
#             block = InvertedResidual
#
#         if norm_layer is None:
#             norm_layer = partial(nn.BatchNorm2d, eps=1e-3, momentum=0.1)
#
#         adjust_channels = partial(InvertedResidualConfig.adjust_channels,
#                                   width_coefficient=width_coefficient)
#
#         # build inverted_residual_setting
#         bneck_conf = partial(InvertedResidualConfig,
#                              width_coefficient=width_coefficient)
#
#         b = 0
#         num_blocks = float(sum(round_repeats(i[-1]) for i in default_cnf))
#         inverted_residual_setting = []
#         for stage, args in enumerate(default_cnf):
#             cnf = copy.copy(args)
#             for i in range(round_repeats(cnf.pop(-1))):
#                 if i > 0:
#                     # strides equal 1 except first cnf
#                     cnf[-3] = 1  # strides
#                     cnf[1] = cnf[2]  # input_channel equal output_channel
#
#                 cnf[-1] = args[-2] * b / num_blocks  # update dropout ratio
#                 index = str(stage + 1) + chr(i + 97)  # 1a, 2a, 2b, ...
#                 inverted_residual_setting.append(bneck_conf(*cnf, index))
#                 b += 1
#
#         # create layers
#         layers = OrderedDict()
#
#         # first conv
#         layers.update({"stem_conv": ConvBNActivation(in_planes=3,
#                                                      out_planes=adjust_channels(32),
#                                                      kernel_size=3,
#                                                      stride=2,
#                                                      norm_layer=norm_layer)})
#
#         # building inverted residual blocks
#         for cnf in inverted_residual_setting:
#             layers.update({cnf.index: block(cnf, norm_layer)})
#
#         # build top
#         last_conv_input_c = inverted_residual_setting[-1].out_c
#         last_conv_output_c = adjust_channels(1280)
#         layers.update({"top": ConvBNActivation(in_planes=last_conv_input_c,
#                                                out_planes=last_conv_output_c,
#                                                kernel_size=1,
#                                                norm_layer=norm_layer)})
#
#         self.features = nn.Sequential(layers)
#         self.avgpool = nn.AdaptiveAvgPool2d(1)
#
#         classifier = []
#         if dropout_rate > 0:
#             classifier.append(nn.Dropout(p=dropout_rate, inplace=True))
#         classifier.append(nn.Linear(last_conv_output_c + 768, num_classes))
#
#         self.get_weight = nn.Sequential(
#             nn.Linear(last_conv_output_c + 768, 1),  # TODO: 768 is the length of global feature
#             nn.Sigmoid()
#         )
#         self.classifier = nn.Sequential(*classifier)
#
#         # initial weights
#         for m in self.modules():
#             if isinstance(m, nn.Conv2d):
#                 nn.init.kaiming_normal_(m.weight, mode="fan_out")
#                 if m.bias is not None:
#                     nn.init.zeros_(m.bias)
#             elif isinstance(m, nn.BatchNorm2d):
#                 nn.init.ones_(m.weight)
#                 nn.init.zeros_(m.bias)
#             elif isinstance(m, nn.Linear):
#                 nn.init.normal_(m.weight, 0, 0.01)
#                 nn.init.zeros_(m.bias)
#
#     def _forward_impl(self, x: Tensor, feature) -> Tensor:
#         features, weights, parts, weights_org, weights_max = [list() for i in range(5)]
#         for i in range(len(x[0])):
#             features.clear()
#             weights.clear()
#             for j in range(len(x)):
#                 y = x[j][i]
#                 y = self.features(y)
#                 y = self.avgpool(y)
#                 y = torch.flatten(y, 1)
#                 # self.get_weight = nn.Sequential(
#                 # nn.Linear(y.shape[1] + 768, 1),
#                 # nn.Sigmoid()
#                 # )
#                 features.append(torch.cat([y, feature], dim=1))  # concat regional feature with global feature
#                 weights.append(self.get_weight(features[-1]))
#
#             features_stack = torch.stack(features, dim=2)
#             weights_stack = torch.stack(weights, dim=2)
#             weights_stack = softmax(weights_stack, dim=2)
#
#             weights_max.append(weights_stack[:, :, :len(x)].max(dim=2)[0])
#             weights_org.append(weights_stack[:, :, 0])
#             parts.append(features_stack.mul(weights_stack).sum(2).div(weights_stack.sum(2)))
#         parts_stack = torch.stack(parts, dim=0)
#         out = parts_stack.sum(0).div(parts_stack.shape[0])
#
#         pred_score = self.classifier(out)
#
#         return pred_score, weights_max, weights_org
#
#     def forward(self, x: Tensor, feature) -> Tensor:
#         return self._forward_impl(x, feature)
#
#
# def get_backbone() -> EfficientNet:
#     return EfficientNet(width_coefficient=1.2,
#                         depth_coefficient=1.4,
#                         dropout_rate=0.3,
#                         num_classes=1)
#
#
# """
#
# """
#
#
# def efficientnet_b0(num_classes=1000):
#     # input image size 224x224
#     return EfficientNet(width_coefficient=1.0,
#                         depth_coefficient=1.0,
#                         dropout_rate=0.2,
#                         num_classes=num_classes)
#
#
# """
#
# """
#
#
# def efficientnet_b1(num_classes=1000):
#     # input image size 240x240
#     return EfficientNet(width_coefficient=1.0,
#                         depth_coefficient=1.1,
#                         dropout_rate=0.2,
#                         num_classes=num_classes)
#
#
# """
#
# """
#
#
# def efficientnet_b2(num_classes=1000):
#     # input image size 260x260
#     return EfficientNet(width_coefficient=1.1,
#                         depth_coefficient=1.2,
#                         dropout_rate=0.3,
#                         num_classes=num_classes)
#
#
# """
#
# """
#
#
# def efficientnet_b3(num_classes=1000):
#     # input image size 300x300
#     return EfficientNet(width_coefficient=1.2,
#                         depth_coefficient=1.4,
#                         dropout_rate=0.3,
#                         num_classes=num_classes)
#
#
# """
#
# """
#
#
# def efficientnet_b4(num_classes=1000):
#     # input image size 380x380
#     return EfficientNet(width_coefficient=1.4,
#                         depth_coefficient=1.8,
#                         dropout_rate=0.4,
#                         num_classes=num_classes)
#
#
# """
#
# """
#
#
# def efficientnet_b5(num_classes=1000):
#     # input image size 456x456
#     return EfficientNet(width_coefficient=1.6,
#                         depth_coefficient=2.2,
#                         dropout_rate=0.4,
#                         num_classes=num_classes)
#
#
# """
#
# """
#
#
# def efficientnet_b6(num_classes=1000):
#     # input image size 528x528
#     return EfficientNet(width_coefficient=1.8,
#                         depth_coefficient=2.6,
#                         dropout_rate=0.5,
#                         num_classes=num_classes)
#
#
# """
#
# """
#
#
# def efficientnet_b7(num_classes=1000):
#     # input image size 600x600
#     return EfficientNet(width_coefficient=2.0,
#                         depth_coefficient=3.1,
#                         dropout_rate=0.5,
#                         num_classes=num_classes)
#
#
# if __name__ == '__main__':
#     model = get_backbone()
#     data = [[] for i in range(3)]
#     for i in range(3):
#         for j in range(5):
#             data[i].append(torch.rand((10, 3, 224, 224)))
#     feature = torch.rand((10, 768))
#     pred_score, weights_max, weights_org = model(data, feature)
#     pass
















# import torch
# from torch import Tensor
# import torch.nn as nn
# from typing import Type, Any, Callable, Union, List, Optional
# from torch.nn.functional import softmax
#
# try:
#     from torch.hub import load_state_dict_from_url
# except ImportError:
#     from torch.utils.model_zoo import load_url as load_state_dict_from_url
#
# model_urls = {
#     'resnet18': 'https://download.pytorch.org/models/resnet18-f37072fd.pth',
#     'resnet34': 'https://download.pytorch.org/models/resnet34-b627a593.pth',
#     'resnet50': 'https://download.pytorch.org/models/resnet50-0676ba61.pth',
#     'resnet101': 'https://download.pytorch.org/models/resnet101-63fe2227.pth',
#     'resnet152': 'https://download.pytorch.org/models/resnet152-394f9c45.pth',
#     'resnext50_32x4d': 'https://download.pytorch.org/models/resnext50_32x4d-7cdf4587.pth',
#     'resnext101_32x8d': 'https://download.pytorch.org/models/resnext101_32x8d-8ba56ff5.pth',
#     'wideget_backbone50_2': 'https://download.pytorch.org/models/wideget_backbone50_2-95faca4d.pth',
#     'wideget_backbone101_2': 'https://download.pytorch.org/models/wideget_backbone101_2-32ee1156.pth',
# }
#
#
#
#
# def conv3x3(in_planes: int, out_planes: int, stride: int = 1, groups: int = 1, dilation: int = 1) -> nn.Conv2d:
#     """3x3 convolution with padding"""
#     return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
#                      padding=dilation, groups=groups, bias=False, dilation=dilation)
#
#
# def conv1x1(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
#     """1x1 convolution"""
#     return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)
#
#
# class BasicBlock(nn.Module):
#     expansion: int = 1
#
#     def __init__(
#             self,
#             inplanes: int,
#             planes: int,
#             stride: int = 1,
#             downsample: Optional[nn.Module] = None,
#             groups: int = 1,
#             base_width: int = 64,
#             dilation: int = 1,
#             norm_layer: Optional[Callable[..., nn.Module]] = None
#                  ) -> None:
#         super(BasicBlock, self).__init__()
#         if norm_layer is None:
#             norm_layer = nn.BatchNorm2d
#         if groups != 1 or base_width != 64:
#             raise ValueError('BasicBlock only supports groups=1 and base_width=64')
#         if dilation > 1:     #膨胀卷积参数？
#             raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
#         # Both self.conv1 and self.downsample layers downsample the input when stride != 1
#         self.conv1 = conv3x3(inplanes, planes, stride)
#         self.bn1 = norm_layer(planes)
#         self.relu = nn.ReLU(inplace=True)
#         self.conv2 = conv3x3(planes, planes)
#         self.bn2 = norm_layer(planes)
#         self.downsample = downsample
#         self.stride = stride
#
#     def forward(self, x: Tensor) -> Tensor:
#         identity = x
#
#         out = self.conv1(x)
#         out = self.bn1(out)
#         out = self.relu(out)
#
#         out = self.conv2(out)
#         out = self.bn2(out)
#
#         if self.downsample is not None:
#             identity = self.downsample(x)
#
#         out += identity
#         out = self.relu(out)
#
#         return out
#
#
# class Bottleneck(nn.Module):
#     expansion: int = 4
#
#     def __init__(
#             self,
#             inplanes: int,
#             planes: int,
#             stride: int = 1,
#             downsample: Optional[nn.Module] = None,
#             groups: int = 1,
#             base_width: int = 64,
#             dilation: int = 1,
#             norm_layer: Optional[Callable[..., nn.Module]] = None
#     ) -> None:
#         super(Bottleneck, self).__init__()
#         if norm_layer is None:
#             norm_layer = nn.BatchNorm2d
#         width = int(planes * (base_width / 64.)) * groups
#         # Both self.conv2 and self.downsample layers downsample the input when stride != 1
#         self.conv1 = conv1x1(inplanes, width)
#         self.bn1 = norm_layer(width)
#         self.conv2 = conv3x3(width, width, stride, groups, dilation)
#         self.bn2 = norm_layer(width)
#         self.conv3 = conv1x1(width, planes * self.expansion)
#         self.bn3 = norm_layer(planes * self.expansion)
#         self.relu = nn.ReLU(inplace=True)
#         self.downsample = downsample
#         self.stride = stride
#
#     def forward(self, x: Tensor) -> Tensor:
#         identity = x
#
#         out = self.conv1(x)
#         out = self.bn1(out)
#         out = self.relu(out)
#
#         out = self.conv2(out)
#         out = self.bn2(out)
#         out = self.relu(out)
#
#         out = self.conv3(out)
#         out = self.bn3(out)
#
#         if self.downsample is not None:
#             identity = self.downsample(x)
#
#         out += identity
#         out = self.relu(out)
#
#         return out
#
#
# class ResNet(nn.Module):
#
#     def __init__(
#             self,
#             block: Type[Union[BasicBlock, Bottleneck]],
#             layers: List[int],
#             num_classes: int = 1000,
#             zero_init_residual: bool = False,
#             groups: int = 1,
#             width_per_group: int = 64,
#             replace_stride_with_dilation: Optional[List[bool]] = None,
#             norm_layer: Optional[Callable[..., nn.Module]] = None
#     ) -> None:
#         super(ResNet, self).__init__()
#         if norm_layer is None:
#             norm_layer = nn.BatchNorm2d
#         self._norm_layer = norm_layer
#
#         self.inplanes = 64
#         self.dilation = 1
#         if replace_stride_with_dilation is None:
#             # each element in the tuple indicates if we should replace
#             # the 2x2 stride with a dilated convolution instead
#             replace_stride_with_dilation = [False, False, False]
#         if len(replace_stride_with_dilation) != 3:
#             raise ValueError("replace_stride_with_dilation should be None "
#                              "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
#         self.groups = groups
#         self.base_width = width_per_group
#         self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3,
#                                bias=False)
#         self.bn1 = norm_layer(self.inplanes)
#         self.relu = nn.ReLU(inplace=True)
#         self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
#         self.layer1 = self._make_layer(block, 64, layers[0])
#         self.layer2 = self._make_layer(block, 128, layers[1], stride=2,
#                                        dilate=replace_stride_with_dilation[0])
#         self.layer3 = self._make_layer(block, 256, layers[2], stride=2,
#                                        dilate=replace_stride_with_dilation[1])
#         self.layer4 = self._make_layer(block, 512, layers[3], stride=2,
#                                        dilate=replace_stride_with_dilation[2])
#         self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
#         self.get_weight = nn.Sequential(
#             nn.Linear(512 * block.expansion + 768, 1),  # TODO: 768 is the length of global feature
#             nn.Sigmoid()
#         )
#         self.fc = nn.Linear(512 * block.expansion + 768, 1)
#
#         for m in self.modules():
#             if isinstance(m, nn.Conv2d):
#                 nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
#             elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
#                 nn.init.constant_(m.weight, 1)
#                 nn.init.constant_(m.bias, 0)
#
#         # Zero-initialize the last BN in each residual branch,
#         # so that the residual branch starts with zeros, and each residual block behaves like an identity.
#         # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
#         if zero_init_residual:
#             for m in self.modules():
#                 if isinstance(m, Bottleneck):
#                     nn.init.constant_(m.bn3.weight, 0)  # type: ignore[arg-type]
#                 elif isinstance(m, BasicBlock):
#                     nn.init.constant_(m.bn2.weight, 0)  # type: ignore[arg-type]
#
#     def _make_layer(self, block: Type[Union[BasicBlock, Bottleneck]], planes: int, blocks: int,
#                     stride: int = 1, dilate: bool = False) -> nn.Sequential:
#         norm_layer = self._norm_layer
#         downsample = None
#         previous_dilation = self.dilation
#         if dilate:
#             self.dilation *= stride
#             stride = 1
#         if stride != 1 or self.inplanes != planes * block.expansion:
#             downsample = nn.Sequential(
#                 conv1x1(self.inplanes, planes * block.expansion, stride),
#                 norm_layer(planes * block.expansion),
#             )
#
#         layers = []
#         layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
#                             self.base_width, previous_dilation, norm_layer))
#         self.inplanes = planes * block.expansion
#         for _ in range(1, blocks):
#             layers.append(block(self.inplanes, planes, groups=self.groups,
#                                 base_width=self.base_width, dilation=self.dilation,
#                                 norm_layer=norm_layer))
#
#         return nn.Sequential(*layers)
#
#     def _forward_impl(self, x, feature):
#         # The comment resolution is based on input size is 224*224 imagenet
#         # f.shape: (batch_size, 3, 224, 224), feature.shape: (batch_size, 768)
#         features, weights, parts, weights_org, weights_max = [list() for i in range(5)]
#         for i in range(len(x[0])):
#             features.clear()
#             weights.clear()
#             for j in range(len(x)):
#                 f = x[j][i]
#                 f = self.conv1(f)
#                 f = self.bn1(f)
#                 f = self.relu(f)
#                 f = self.maxpool(f)
#                 f = self.layer1(f)
#                 f = self.layer2(f)
#                 f = self.layer3(f)
#                 f = self.layer4(f)
#                 f = self.avgpool(f)
#                 f = torch.flatten(f, 1)
#
#                 # features.append(f)
#                 features.append(torch.cat([f, feature], dim=1))  # concat regional feature with global feature
#                 weights.append(self.get_weight(features[-1]))
#
#             features_stack = torch.stack(features, dim=2)
#             weights_stack = torch.stack(weights, dim=2)
#             weights_stack = softmax(weights_stack, dim=2)
#
#             weights_max.append(weights_stack[:, :, :len(x)].max(dim=2)[0])
#             weights_org.append(weights_stack[:, :, 0])
#             parts.append(features_stack.mul(weights_stack).sum(2).div(weights_stack.sum(2)))
#         parts_stack = torch.stack(parts, dim=0)
#         out = parts_stack.sum(0).div(parts_stack.shape[0])
#
#         pred_score = self.fc(out)
#
#         return pred_score, weights_max, weights_org
#
#     def forward(self, x, feature):
#         return self._forward_impl(x, feature)
#
#
# def _get_backbone(
#         arch: str,
#         block: Type[Union[BasicBlock, Bottleneck]],
#         layers: List[int],
#         pretrained: bool,
#         progress: bool,
#         **kwargs: Any
# ) -> ResNet:
#     model = ResNet(block, layers, num_classes=1, **kwargs)
#     if pretrained:
#         state_dict = load_state_dict_from_url(model_urls[arch], progress=progress)
#         model.load_state_dict(state_dict)
#     return model
#
#
# def get_backbone(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> ResNet:
#     r"""ResNet-50 model from
#     `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_.
#
#     Args:
#         pretrained (bool): If True, returns a model pre-trained on ImageNet
#         progress (bool): If True, displays a progress bar of the download to stderr
#     """
#     return _get_backbone('resnet50', Bottleneck, [3, 4, 6, 3], pretrained, progress, **kwargs)
#
#
# if __name__ == '__main__':
#     model = get_backbone()
#     data = [[] for i in range(3)]
#     for i in range(3):
#         for j in range(5):
#             data[i].append(torch.rand((10, 3, 224, 224)))
#     feature = torch.rand((10, 768))
#     pred_score, weights_max, weights_org = model(data, feature)
#     pass