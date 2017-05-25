import sys
import torch
import torch.nn as nn
import torchvision.models as models

from .noatt import MLBNoAtt, MutanNoAtt
from .att import MLBAtt, MutanAtt

class ResNet(nn.Module):

    def __init__(self, resnet, pooling, fix_until=None):
        # pooling: boolean
        # fix_until: None or layer name (included)
        super(ResNet, self).__init__()
        self.resnet = resnet
        self.pooling = pooling
        if fix_until is not None:
            self.fixable_layers = [
                'conv1', 'bn1', 'relu', 'maxpool',
                'layer1', 'layer2', 'layer3', 'layer4']
            if fix_until in self.fixable_layers:
                self.fix_until = fix_until
                self._fix_layers(fix_until)
            else:
                raise ValueError

    def _fix_layers(self, fix_until):
        for layer in self.fixable_layers:
            print('Warning models/utils.py: Fix cnn layer '+layer)
            for p in getattr(self.resnet, layer).parameters():
                p.requires_grad = False
            if layer == self.fix_until:
                break

    def forward(self, x):
        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        x = self.resnet.maxpool(x)
        x = self.resnet.layer1(x)
        x = self.resnet.layer2(x)
        x = self.resnet.layer3(x)
        x = self.resnet.layer4(x)
        if self.pooling:
            x = self.resnet.avgpool(x)
            x = x.view(x.size(0), -1)
        # x = self.fc(x)
        return x
