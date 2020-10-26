import torch
import torch.nn as nn
import torch.nn.functional as F

class Mish(nn.Module):
    def __init__(self):
        super(Mish, self).__init__()

    def forward(self, x):
        return x*(torch.tanh(F.softplus(x)))


class Conv_Block(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size, stride, padding):
        super(Conv_Block, self).__init__()
        self.conv1 = nn.Conv2d(
            in_channels=in_ch,
            out_channels=in_ch,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=1,
            groups=1,
            bias=False
        )
        self.conv2 = nn.Conv2d(
            in_channels=in_ch,
            out_channels=out_ch,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=1,
            groups=1,
            bias=False
        )
        self.conv3 = nn.Conv2d(
            in_channels=in_ch,
            out_channels=out_ch,
            kernel_size=1,
            stride=1,
            padding=0,
            dilation=1,
            groups=1,
            bias=False
        )

        self.bn = nn.BatchNorm2d(out_ch)
        self.mish = Mish()


    def forward(self, x):
        residual = self.conv3(x)

        x1 = self.conv1(x)
        x2 = self.mish(x1)
        x3 = self.conv2(x2)
        x4 = self.mish(x3)
        x5 = self.bn(x4)

        out = residual + x5
        return out

class MaxPool(nn.Module):
    def __init__(self, kernel_size, stride, padding):
        super(MaxPool, self).__init__()
        self.max_pool = nn.MaxPool2d(
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
        )

    def forward(self, x):
        return self.max_pool(x)

class SpecNet2(nn.Module):
    def __init__(self, n_classes=50, out_ch=32):
        super(SpecNet2, self).__init__()

        self.first_conv = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=out_ch, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            Mish(),
        )
        self.conv2 = nn.Sequential(
            Conv_Block(out_ch, out_ch, (1,3), (1,1), (0,1)),
            MaxPool((1,3), (1,2),(0,1))
        )
        self.conv3 = nn.Sequential(
            Conv_Block(out_ch, out_ch*2, (5,1), (1,1), (2,0)),
            MaxPool((3,1), (2,1), (1,0)),
        )
        self.conv4 = nn.Sequential(
            Conv_Block(out_ch*2, out_ch*2, (1, 3), (1, 1), (0, 1)),
            MaxPool((1, 3), (1, 2), (0, 1))
        )
        self.conv5 = nn.Sequential(
            Conv_Block(out_ch*2, out_ch*4, (5, 1), (1, 1), (2, 0)),
            MaxPool((3, 1), (2, 1), (1, 0)),
        )
        self.conv6 = nn.Sequential(
            Conv_Block(out_ch*4, out_ch*4, (1, 3), (1, 1), (0, 1)),
            MaxPool((1, 3), (1, 2), (0, 1))
        )
        self.conv7 = nn.Sequential(
            Conv_Block(out_ch*4, out_ch*8, (5, 1), (1, 1), (2, 0)),
            MaxPool((3, 1), (2, 1), (1, 0)),
        )
        self.conv8 = nn.Sequential(
            Conv_Block(out_ch*8, out_ch*16, (5, 3), (1, 1), (2, 1)),
            MaxPool((3, 3), (2, 2), (1, 1))
        )
        self.conv9 = nn.Sequential(
            Conv_Block(out_ch*16, out_ch*32, (5, 3), (1, 1), (2, 1)),
            MaxPool((3, 3), (2, 2), (1, 1)),
        )
        self.linear = nn.Sequential(
            nn.Linear(256 + 512 + 1024, 512),
            Mish(),
            nn.Dropout(p=0.5),
            nn.Linear(512, n_classes),
        )

        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)

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
        x1 = self.first_conv(x)
        x2 = self.conv2(x1)
        x3 = self.conv3(x2)
        x4 = self.conv4(x3)
        x5 = self.conv5(x4)
        x6 = self.conv6(x5)
        x7 = self.conv7(x6)
        x8 = self.conv8(x7)
        x9 = self.conv9(x8)

        out1 = self.global_avg_pool(x7)
        out2 = self.global_avg_pool(x8)
        out3 = self.global_avg_pool(x9)

        out = torch.cat((out1, out2, out3), dim=1)
        out = out.view(out.size(0), -1)
        out = self.linear(out)

        return out


if __name__ == '__main__':
    device = torch.device("cuda: 0" if torch.cuda.is_available else "cpu")

    from torchsummary import summary

    model = SpecNet2()

    model.to(device)

    summary(model, (1, 150,128))