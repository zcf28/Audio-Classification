from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.utils import weights_init
from models.additional_layers import Flatten, Transpose
from models.densenet2 import DenseNet
from models.senet import SEBottleneck, SELayer

class EnvReLu(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, bias=False):
        super(EnvReLu, self).__init__()
        self.layer = nn.Sequential(
            nn.Conv2d(in_channels=in_channels,
                      out_channels=out_channels,
                      kernel_size=kernel_size,
                      stride=stride,
                      padding=padding,
                      bias=bias),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )

    def forward(self, input):
        return self.layer(input)

class EnvNet4(nn.Module):
    def __init__(self, n_classes):
        super(EnvNet4, self).__init__()
        self.model = nn.Sequential(OrderedDict([
            ('conv1', EnvReLu(in_channels=1,
                              out_channels=32,
                              kernel_size=(1, 64),
                              stride=(1, 2),
                              padding=0)), #[b,32, 1, 33294]
            ('conv2', EnvReLu(in_channels=32,
                              out_channels=64,
                              kernel_size=(1, 16),
                              stride=(1, 2),
                              padding=0)), #[b, 64, 1, 16640]
            ('max_pool2', nn.MaxPool2d(kernel_size=(1, 64),
                                       stride=(1, 64),
                                       ceil_mode=True)), #[b, 64, 1, 260]
            ('transpose', Transpose()), #[b, 1, 64, 260]

            ('densenet', DenseNet(growth_rate=16, block_config=(6, 12, 24, 16))), #[b, 1024,8,32]
            # growth_rate= 32 --> 16 解决当前服务器内存不够

            ('senet1', SEBottleneck(inplanes=136, planes=136//4)),
            ('senet2', SEBottleneck(inplanes=260, planes=260//4)),

            # ('senet1', SELayer(channel=256)),
            # ('senet2', SELayer(channel=512)),

            ('global avgpool', nn.AdaptiveAvgPool2d(1)),

            ('flatten', Flatten()),

            ('fc13', nn.Linear(in_features=(136+260+516), out_features=n_classes, bias=True))#[b, n]
            ]))

        # # params initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        for i in range(len(self.model)):

            if i < 4:
                x = self.model[i](x)

            elif i == 4:
                out1, out2, out3 = self.model[i](x)
                break

        out1 = self.model[7](out1)

        out2 = self.model[5](out2)
        out2 = self.model[7](out2)

        out3 = self.model[6](out3)
        out3= self.model[7](out3)

        # print(f'out1.shape = {out1.shape}')
        # print(f'out3.shape = {out3.shape}')
        # print(f'out5.shape = {out5.shape}')

        out = torch.cat((out1, out2, out3), dim=1)
        # print(f'out.shape = {out.shape}')

        out = self.model[8](out)
        out = self.model[9](out)

        return out

def print_network(model, name):
    """Print out the network information."""
    num_params = 0
    for p in model.parameters():
        num_params += p.numel()
    print(model)
    print(f'The model name is: {name}')
    print("The number of parameters: {}".format(num_params))

if __name__ == "__main__":

    model = EnvNet4(50)
    # print_network(model, 'envnet4')
    for m in model.parameters():
        print(m)