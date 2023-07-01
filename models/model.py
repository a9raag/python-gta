import torch.nn.init
from torch import nn


class ResidualBlock(nn.Module):
    # O = ((I - K + 2P) / S) + 1
    def __init__(self, in_channels, out_channels, stride=1, downsample=None) -> None:
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU()
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels)
        )
        self.conv1.apply(init_kaiming)
        self.conv2.apply(init_kaiming)
        self.downsample = downsample
        self.relu = nn.LeakyReLU()
        self.out_channels = out_channels

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.conv2(out)
        if self.downsample:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out


def init_kaiming(m):
    if type(m) == nn.Linear:
        nn.init.kaiming_uniform_(m.weight, nonlinearity='leaky_relu')


class ResNet(nn.Module):
    def __init__(self, block, layers, num_classes=2):
        super(ResNet, self).__init__()
        self.inplanes = 64
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.LeakyReLU()
        )
        self.conv1.apply(init_kaiming)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.dropout = nn.Dropout()
        self.layer0 = self._make_layer(block, 64, layers[0], stride=1)
        self.layer1 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer2 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer3 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AvgPool2d(7, stride=1)
        self.relu = nn.LeakyReLU()
        self.fc = nn.Linear(512, num_classes)
        torch.nn.init.normal_(self.fc.weight, mean=0.0, std=0.01)
        # torch.nn.init.kaiming_uniform_(self.fc.weight, nonlinearity='leaky_relu')

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes, kernel_size=1, stride=stride),
                nn.BatchNorm2d(planes)
            )
            downsample.apply(init_kaiming)
        layers = [block(self.inplanes, planes, stride, downsample)]
        self.inplanes = planes
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.dropout(x)
        x = self.maxpool(x)
        x = self.layer0(x)
        x = self.dropout(x)
        x = self.layer1(x)
        x = self.dropout(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.dropout(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc(x)
        return x

    def view_grads(self):
        pass


if __name__ == '__main__':
    from torchsummary import summary

    model = ResNet(ResidualBlock, [2, 2, 2, 2], num_classes=2)
    model.to("cuda")
    summary(model, (3, 224, 224))