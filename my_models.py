import torch.nn as nn

class BasicBlock(nn.Module):
    expansion = 1
    def __init__(self, in_channels, out_channels, stride = 1):
        super(BasicBlock, self).__init__()
        self.straight = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size = 3, stride = stride, padding = 1, bias = False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace = True),
            nn.Conv2d(out_channels, out_channels, kernel_size = 3, stride = 1, padding = 1, bias = False),
            nn.BatchNorm2d(out_channels)
        )

        self.shortcut = nn.Sequential()

        if (stride != 1) or (in_channels != out_channels):
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size = 1, stride = stride, bias = False),
                nn.BatchNorm2d(out_channels)
            )
        self.ReLU = nn.ReLU(inplace = True)

    def forward(self, x):
        output = self.straight(x) + self.shortcut(x)
        output = self.ReLU(output)

        return output

class BottleNeck(nn.Module):
    expansion = 4
    def __init__(self, in_channels, out_channels, stride = 1):
        super(BottleNeck, self).__init__()
        self.straight = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size = 1, bias = False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace = True),
            nn.Conv2d(out_channels, out_channels, stride = stride, kernel_size = 3, padding = 1, bias = False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace = True),
            nn.Conv2d(out_channels, out_channels * BottleNeck.expansion, kernel_size = 1, bias = False),
            nn.BatchNorm2d(out_channels * BottleNeck.expansion)
        )

        self.shortcut = nn.Sequential()

        if (stride != 1) or (in_channels != out_channels * BottleNeck.expansion):
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels * BottleNeck.expansion, stride = stride, kernel_size = 1, bias = False),
                nn.BatchNorm2d(out_channels * BottleNeck.expansion) 
            )

        self.ReLU = nn.ReLU(inplace = True)
    def forward(self, x):
        output = self.straight(x) + self.shortcut(x)
        output = self.ReLU(output)

        return output

class ResNet(nn.Module):
    def __init__(self, block, num_blocks, in_channels, num_classes):
        super(ResNet, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size = 3,stride = 1, padding = 1, bias = False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace = True)
        )
        self.conv2_x = self.make_layers(block, 64, num_blocks[0], 1)
        self.conv3_x = self.make_layers(block, 128, num_blocks[1], 2)
        self.conv4_x = self.make_layers(block, 256, num_blocks[2], 2)
        self.conv5_x = self.make_layers(block, 512, num_blocks[3], 2)
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

    def make_layers(self, block, out_channels, num_block, stride):
        strides = [stride] + [1] * (num_block - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, out_channels, stride))
            self.in_planes  = out_channels * block.expansion

        return nn.Sequential(*layers)

    def forward(self, x):
        output = self.conv1(x)
        output = self.conv2_x(output)
        output = self.conv3_x(output)
        output = self.conv4_x(output)
        output = self.conv5_x(output)
        output = self.avg_pool(output)
        output = output.view(output.size(0), -1)
        output = self.fc(output)

        return output

def ResNet18(in_channels, num_classes):
    return ResNet(BasicBlock, [2, 2, 2, 2], in_channels = in_channels, num_classes = num_classes)

def ResNet50(in_channels, num_classes):
    return ResNet(BottleNeck, [3, 4, 6, 3], in_channels = in_channels, num_classes = num_classes)
    
def ResNet101(in_channels, num_classes):
    return ResNet(BottleNeck, [3, 4, 23, 3], in_channels = in_channels, num_classes = 
    num_classes)