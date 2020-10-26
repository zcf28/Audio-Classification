from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.utils import weights_init
from models.additional_layers import Flatten, Transpose
from models.densenet1 import DenseNet

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

class EnvNet3(nn.Module):
    def __init__(self, n_classes):
        super(EnvNet3, self).__init__()
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

            ('densenet', DenseNet(growth_rate=16, block_config=(6, 12, 24, 16))),
            # growth_rate= 32 --> 16 解决当前服务器内存不够

            ('flatten', Flatten()),
            ('fc11', nn.Linear(in_features=516 * 8 * 32, out_features=1024, bias=True)),
            ('relu11', nn.ReLU()),
            ('dropout11', nn.Dropout()),
            ('fc13', nn.Linear(in_features=1024, out_features=n_classes, bias=True))#[b, n]
            ]))

        # params initialization
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
                nn.init.constant_(m.bias, 0)



    def forward(self, x):

        return self.model(x)


if __name__ == '__main__':
    device = torch.device("cuda: 0" if torch.cuda.is_available else "cpu")
    from torchsummary import summary

    model = EnvNet3(50)
    model.to(device)
    # model.apply(weights_init)

    summary(model, (1, 1, 66650))
    # input = torch.randn(1,1,1,66650).to(device)
    # out = model(input)