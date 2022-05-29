import os
from turtle import forward
from venv import create
from typing import Type, Any, Callable, Union, List, Optional
from cv2 import norm


import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from modelsummary import summary
from torchvision.models.resnet import conv1x1, conv3x3, BasicBlock, Bottleneck



def create_resnet():
    model = torchvision.models.resnet18(pretrained=False, num_classes=10)
    #model.conv1 = nn.Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
    return model


class SimpleResNet(nn.Module):
    def __init__(self, layers: List[int], num_classes: int = 10, inplanes = 16, zero_init_residual: bool = False,  
                replace_stride_with_dilation: Optional[List[bool]] = None, aggressive_downsampling = False) -> None:
        super().__init__()

        self._norm_layer = norm_layer = nn.BatchNorm2d

         
        self.inplanes = inplanes 
        self.dilation = 1 
        # for compatibility 
        self.groups = 1 
        self.base_width = 64 

        # get number of channels / planes as multiples of input channels / planes
        self.inplanes_ar = [inplanes*(2**k) for k in range(len(layers))]
        
        if aggressive_downsampling:
            # aggressively downsample in case original image size is large, e.g. 224x224 
            self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False)
            self.bn1 = norm_layer(self.inplanes)
            self.relu = nn.ReLU(inplace=True)
            self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        else: 
            self.conv1 = conv3x3(in_planes=3, out_planes=self.inplanes, stride=1, )
            self.bn1 = norm_layer(self.inplanes)
            self.relu = nn.ReLU(inplace=True)
            self.maxpool = nn.Identity() # no maxpooling 
        
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False, False]
        else:
            # to have consistent external interface, the replace_stride_with_dilation is assumed to be a three element list 
            replace_stride_with_dilation = [False, *replace_stride_with_dilation]

        if len(replace_stride_with_dilation)-1 != 3:
            raise ValueError(
                "replace_stride_with_dilation should be None "
                f"or a 3-element tuple, got {replace_stride_with_dilation[1:]}"
            )

        resnet_layers = [] 
        layer = self._make_layer(BasicBlock, self.inplanes_ar[0], layers[0])
        resnet_layers.append(layer)

        self.resnet_layers = nn.ModuleList(
            self._make_layer(BasicBlock, self.inplanes_ar[i], layers[i], stride=2, dilate=replace_stride_with_dilation[i]) for i in range(1,len(layers))
        )

    
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(self.inplanes_ar[-1] * BasicBlock.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
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

    def _make_layer(
        self,
        block: Type[Union[BasicBlock, Bottleneck]],
        planes: int,
        blocks: int,
        stride: int = 1,
        dilate: bool = False,
    ) -> nn.Sequential:
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
        layers.append(
            block(
                self.inplanes, planes, stride, downsample, self.groups, self.base_width, previous_dilation, norm_layer
            )
        )
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(
                block(
                    self.inplanes,
                    planes,
                    groups=self.groups,
                    base_width=self.base_width,
                    dilation=self.dilation,
                    norm_layer=norm_layer,
                )
            )

        return nn.Sequential(*layers)
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        for layer in self.resnet_layers: 
            x = layer(x)

        
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        
        return x 



if __name__ == '__main__':
    resnet = SimpleResNet(layers=[1,1,1,1], num_classes=10, inplanes=12, zero_init_residual=True, aggressive_downsampling=False)

    x = torch.rand(10,3, 32,32)
    y = resnet(x)
    print('Input shape = x', x.shape)
    print('output shape = y',y.shape)

    summary(resnet, x)