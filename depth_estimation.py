import torch.nn as nn

class ResidualBlock(nn.Module):
    '''
        Used as a basic residual module for a ResidualBlock.

        Parameters:
            in_channels - {int} - number of input channels
            out_channels - {int} - number of output channels
            stride - {int} - stride to use in Convolution layers
    '''

    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.IN1 = nn.InstanceNorm2d(in_channels)

        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.IN2 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.IN1(out)
        out = self.conv2(out)
        out = self.bn2(out)
        return out + residual


class ResidualLayer(nn.Module):
    '''
        Used as a Conv-IN-Relu Layer.

        Parameters:
            in_channels - {int} - number of input channels
            out_channels - {int} - number of output channels
            stride - {int} - stride to use in Convolution layers
    '''

    def __init__(self, in_channels, out_channels, stride=1, kernel_size=3, type='conv'):
        super(ResidualLayer, self).__init__()
        self.relu = nn.ReLU(inplace=True)
        if type == 'conv':
            self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=1)
        else:
            self.conv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=1)
        self.IN = nn.InstanceNorm2d(in_channels)

    def forward(self, x):
        out = self.conv(x)
        out = self.IN(out)
        out = self.relu(out)
        return out



class Stream1(nn.Module):
    def __init__(self):
        super(Stream1, self).__init__()
        self.layer1 = ResidualLayer(3, 128)
        self.layer2 = ResidualBlock(128, 128)
        self.layer3 = ResidualLayer(128, 128, 2)
        self.layer4 = ResidualBlock(128, 128)
        self.layer5 = ResidualLayer(128, 128, 2)
        self.layer6 = ResidualBlock(128, 128)
        self.layer7 = ResidualLayer(128, 128, 2)
        self.layer8 = ResidualBlock(128, 128)
        self.layer9 = ResidualLayer(128, 128, 2)
        self.layer10 = ResidualLayer(128, 128, 2, type='deconv')
        self.layer11 = ResidualBlock(128, 128)
        self.layer12 = ResidualLayer(128, 128, 2, type='deconv')
        self.layer13 = ResidualBlock(128, 128)
        self.layer14 = ResidualLayer(128, 128, 2, type='deconv')
        self.layer15 = ResidualBlock(128, 128)
        self.layer16 = ResidualLayer(128, 128, 2, type='deconv')
        self.layer17 = ResidualBlock(128, 128)

        self.feature1 = ResidualLayer(128, 128, 8, type='deconv')
        self.feature2 = ResidualLayer(128, 128, 4, type='deconv')
        self.feature3 = ResidualLayer(128, 128, 2, type='deconv')

    def forward(self, x):
        out1 = self.layer1(x)
        out2 = self.layer2(out1)
        out3 = self.layer3(out2)
        out4 = self.layer4(out3)
        out5 = self.layer5(out4)
        out6 = self.layer6(out5)
        out7 = self.layer7(out6)
        out8 = self.layer8(out7)
        out9 = self.layer9(out8)
        out10 = self.layer10(out9)
        out11 = self.layer11(out10) + out7
        out12 = self.layer12(out11)
        out13 = self.layer13(out12) + out5
        out14 = self.layer14(out13)
        out15 = self.layer15(out14) + out3
        out16 = self.layer16(out15)
        out17 = self.layer17(out16)

        return self.feature1(out11), self.feature2(out13), self.feature3(out15), out17


class Stream2(nn.Module):
    def __init__(self):
        super(Stream2, self).__init__()
        self.layer1 = ResidualLayer(3, 64)
        self.layer2 = ResidualLayer(64, 128)
        self.layer3 = ResidualLayer(128, 128)
        self.layer4 = ResidualBlock(128, 128)
        self.layer5 = ResidualBlock(128, 128)
        self.layer6 = ResidualBlock(128, 128)
        self.layer7 = ResidualBlock(128, 128)
        self.layer8 = ResidualBlock(128, 128)
        self.layer9 = ResidualLayer(128, 128, type='deconv')
        self.layer10 = ResidualLayer(128, 128, type='deconv')

    def forward(self, x):
        im, feature1, feature2, feature3 = x
        out1 = self.layer1(im)
        out2 = self.layer2(out1)
        out3 = self.layer3(out2)
        out4 = self.layer4(out3) + feature1
        out5 = self.layer5(out4) + feature2
        out6 = self.layer6(out5) + feature3
        out7 = self.layer7(out6)
        out8 = self.layer8(out7)
        out9 = self.layer9(out8)
        out10 = self.layer10(out9)

        return out10

class DepthNet(nn.Module):
    def __init__(self):
        super(DepthNet, self).__init__()
        self.stream1 = Stream1()
        self.stream2 = Stream2()
        self.features = ResidualLayer(128, 128, 1)

    def forward(self, x):
        f1, f2, f3, stream1_out = self.stream1(x)
        stream2_out = self.stream2([x,f1, f2, f3])
        return self.features(stream2_out + stream1_out)


