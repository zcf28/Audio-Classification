from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.utils import weights_init
from models.additional_layers import Flatten, Transpose
from models.densenet3 import DenseNet
from models.senet import SEBottleneck, SELayer


class EnvNet6(nn.Module):
    def __init__(self, n_classes):
        super(EnvNet6, self).__init__()
        self.model = nn.Sequential(OrderedDict([

            ('densenet', DenseNet(growth_rate=32, block_config=(6, 12, 24, 16))), #[b, 1024,9,8]
            # growth_rate= 32 --> 16 解决当前服务器内存不够

            ('senet1', SEBottleneck(inplanes=256, planes=256//4)),
            ('senet2', SEBottleneck(inplanes=512, planes=512//4)),
            # ('senet1', SELayer(channel=136)),
            # ('senet2', SELayer(channel=260)),

            ('global_avg_pool2', nn.AdaptiveAvgPool2d(1)),

            ('flatten', Flatten()),
  
            ('fc1', nn.Linear(in_features=(256+512+1024), out_features=n_classes, bias=True)),#[b, n]

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
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)



    def forward(self, x):
        # return self.model(x)

        out1, out2, out3 = self.model[0](x)


        out1 = self.model[3](out1)

        out2 = self.model[1](out2)
        out2 = self.model[3](out2)

        out3 = self.model[2](out3)
        out3 = self.model[3](out3)

        out = torch.cat((out1, out2, out3), dim=1)

        out = self.model[4](out)
        out = self.model[5](out)

        return out

def print_network(model, name):
    """Print out the network information."""
    num_params = 0
    for p in model.parameters():
        num_params += p.numel()
    print(model)
    print(f'The model name is: {name}')
    print("The number of parameters: {}".format(num_params))

if __name__ == '__main__':
    device = torch.device("cuda: 0" if torch.cuda.is_available else "cpu")
    from torchsummary import summary


    model = EnvNet6(50)
    model.to(device)

    # print(model)
    print_network(model, "1")

    # summary(model, (1, 150, 128))
    # print(model.named_children())




"""

"""