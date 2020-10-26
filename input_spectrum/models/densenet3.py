import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict


class _DenseLayer(nn.Sequential):
    """Basic unit of DenseBlock (using bottleneck layer) """
    def __init__(self, num_input_features, growth_rate, bn_size, drop_rate, kernel_size, padding):
        super(_DenseLayer, self).__init__()
        self.add_module("norm1", nn.BatchNorm2d(num_input_features))
        self.add_module("relu1", nn.ReLU(inplace=True))
        self.add_module("conv1", nn.Conv2d(num_input_features, bn_size*growth_rate,
                                           kernel_size=1, stride=1, bias=False))
        self.add_module("norm2", nn.BatchNorm2d(bn_size*growth_rate))
        self.add_module("relu2", nn.ReLU(inplace=True))
        self.add_module("conv2", nn.Conv2d(bn_size*growth_rate, growth_rate,
                                           kernel_size=kernel_size, stride=1, padding=padding, bias=False))
        self.drop_rate = drop_rate

    def forward(self, x):
        new_features = super(_DenseLayer, self).forward(x)
        # print(f'new_feature = {new_features.shape}')
        if self.drop_rate > 0:
            new_features = F.dropout(new_features, p=self.drop_rate, training=self.training)
        return torch.cat([x, new_features], 1)

class _DenseBlock(nn.Sequential):
    """DenseBlock"""
    def __init__(self, num_layers, num_input_features, bn_size, growth_rate, drop_rate, flag):
        super(_DenseBlock, self).__init__()
        if flag in [2, 3]:
            for i in range(num_layers):
                layer = _DenseLayer(num_input_features + i * growth_rate, growth_rate, bn_size,
                                    drop_rate, kernel_size=(3, 3), padding=(1, 1))
                self.add_module("denselayer%d" % (i + 1,), layer)
        else:
            for i in range(num_layers):
                if i % 2 == 0:
                    layer = _DenseLayer(num_input_features + i * growth_rate, growth_rate, bn_size,
                                        drop_rate, kernel_size=(1, 3), padding=(0, 1))
                else:
                    layer = _DenseLayer(num_input_features + i * growth_rate, growth_rate, bn_size,
                                        drop_rate, kernel_size=(3, 1), padding=(1, 0))

                self.add_module("denselayer%d" % (i + 1,), layer)

class _Transition(nn.Sequential):
    """Transition layer between two adjacent DenseBlock"""
    def __init__(self, num_input_feature, num_output_features):
        super(_Transition, self).__init__()
        self.add_module("norm", nn.BatchNorm2d(num_input_feature))
        self.add_module("relu", nn.ReLU(inplace=True))
        self.add_module("conv", nn.Conv2d(num_input_feature, num_output_features,
                                          kernel_size=1, stride=1, bias=False))
        self.add_module("pool", nn.AvgPool2d(2, stride=2))


class DenseNet(nn.Module):
    "DenseNet-BC model"
    def __init__(self, growth_rate=32, block_config=(6, 12, 24, 16), num_init_features=64,
                 bn_size=4, compression_rate=0.5, drop_rate=0, num_classes=1000):
        """
        :param growth_rate: (int) number of filters used in DenseLayer, `k` in the paper
        :param block_config: (list of 4 ints) number of layers in each DenseBlock
        :param num_init_features: (int) number of filters in the first Conv2d
        :param bn_size: (int) the factor using in the bottleneck layer
        :param compression_rate: (float) the compression rate used in Transition Layer
        :param drop_rate: (float) the drop rate after each DenseLayer
        :param num_classes: (int) number of classes for classification
        """
        super(DenseNet, self).__init__()

        self.block_config = block_config

        # first Conv2d
        self.features = nn.Sequential(OrderedDict([
            ("conv0", nn.Conv2d(1, num_init_features, kernel_size=3, stride=2, padding=1, bias=False)),
            ("norm0", nn.BatchNorm2d(num_init_features)),
            ("relu0", nn.ReLU(inplace=True)),
        ]))


        # DenseBlock
        num_features = num_init_features
        for i, num_layers in enumerate(block_config):
            block = _DenseBlock(num_layers, num_features, bn_size, growth_rate, drop_rate, flag=i)
            self.features.add_module("denseblock%d" % (i + 1), block)
            num_features += num_layers*growth_rate
            if i != len(block_config) - 1:
                transition = _Transition(num_features, int(num_features*compression_rate))
                self.features.add_module("transition%d" % (i + 1), transition)
                num_features = int(num_features * compression_rate)

        # final bn+ReLU
        self.features.add_module("norm5", nn.BatchNorm2d(num_features))
        self.features.add_module("relu5", nn.ReLU(inplace=True))

        # classification layer
        # self.classifier = nn.Linear(in_features= num_features,out_features=num_classes)



    def forward(self, x):
        # features = self.features(x)
        if len(self.block_config) == 3:
            for i in range(len(self.features)):
                if i == 4:
                    out1 = self.features[i](x)

                if i == 6:
                    out2 = self.features[i](x)

                x = self.features[i](x)

            return x, out1, out2

        if len(self.block_config) == 4:
            for i in range(len(self.features)):
                if i == 6:
                    out1 = self.features[i](x)

                if i == 8:
                    out2 = self.features[i](x)

                x = self.features[i](x)

            return x, out1, out2



if __name__ == '__main__':
    device = torch.device("cuda: 0" if torch.cuda.is_available else "cpu")
    from torchsummary import summary
    densenet = DenseNet(growth_rate=16, block_config=(6, 12, 24, 16))
    densenet.to(device)
    # print(densenet)

    # summary(densenet, (64,64,260))

    x = torch.randn(1,1,64,64).to(device)
    out, out1,  out3 = densenet(x)
    print(f'out.shape = {out.shape}')
    print(f'out1.shape = {out1.shape}')
    print(f'out3.shape = {out3.shape}')




