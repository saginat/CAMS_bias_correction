import torch
from torch import nn

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=2, kernel_size=3, dilation=1, padding=1, reduce=False):
        super(ResidualBlock, self).__init__()

        '''
        Residual block to be used in NN architcture 
        
        '''
        
        if reduce:
            self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, bias=False)
            self.bn1 = nn.BatchNorm2d(out_channels)
            self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=kernel_size, stride=stride-1, padding=padding, dilation=dilation, bias=False)
            self.bn2 = nn.BatchNorm2d(out_channels)
            self.shortcut = nn.MaxPool2d((2, 2))
        else:
            self.conv1 = nn.Conv2d(out_channels, out_channels, kernel_size=kernel_size, stride=stride-1, padding=padding, dilation=dilation, bias=False)
            self.bn1 = nn.BatchNorm2d(out_channels)
            self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=kernel_size, stride=stride-1, padding=padding, dilation=dilation, bias=False)
            self.bn2 = nn.BatchNorm2d(out_channels)
            self.shortcut = None

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        identity = x.clone() if self.shortcut is None else self.shortcut(x)
        
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += identity
        out = self.relu(out)
        
        return out

class ClassificationResNet(nn.Module):
    def __init__(self, input_shape, num_classes, num_stations, channel_zero=8):
        super(ClassificationResNet, self).__init__()

        channel, lat_index, lon_index = input_shape

        self.conv1 = nn.Conv2d(channel, channel_zero, kernel_size=(8, 10), stride=1, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(channel_zero)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.stage1 = self.make_stage(channel_zero, channel_zero, num_blocks=1, stride=2, dilation=1, reduce=False)
        self.conv2 = nn.Conv2d(channel_zero, channel_zero * 2, kernel_size=3, stride=2, padding=1, dilation=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channel_zero * 2)
        
        self.stage2 = self.make_stage(channel_zero * 2, channel_zero * 2, num_blocks=2, stride=2, dilation=1, reduce=False)
        self.conv3 = nn.Conv2d(channel_zero * 2, channel_zero * 4, kernel_size=3, stride=2, padding=1, dilation=1, bias=False)
        self.bn3 = nn.BatchNorm2d(channel_zero * 4)
        
        self.stage3 = self.make_stage(channel_zero * 4, channel_zero * 4, num_blocks=2, stride=2, dilation=1, reduce=False)
        self.conv4 = nn.Conv2d(channel_zero * 4, channel_zero * 8, kernel_size=3, stride=2, padding=1, dilation=1, bias=False)
        self.bn4 = nn.BatchNorm2d(channel_zero * 8)
        
        self.stage4 = self.make_stage(channel_zero * 8, channel_zero * 8, num_blocks=2, stride=2, dilation=1, reduce=False)
        self.conv5 = nn.Conv2d(channel_zero * 8, channel_zero * 16, kernel_size=3, stride=2, padding=2, dilation=2, bias=False)
        self.bn5 = nn.BatchNorm2d(channel_zero * 16)
        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc1 = nn.Linear(channel_zero * 16, channel_zero * 8)
        self.fc2 = nn.Linear(channel_zero * 8, num_stations * num_classes)
        self.dropout = nn.Dropout(0.5)

        self.num_stations = num_stations
        self.num_classes = num_classes

    def make_stage(self, in_channels, out_channels, num_blocks, stride, dilation, reduce=False):
        layers = [ResidualBlock(in_channels, out_channels, stride, dilation, reduce)]
        for _ in range(1, num_blocks):
            layers.append(ResidualBlock(out_channels, out_channels, stride, dilation, reduce))
        return nn.Sequential(*layers)

    def forward(self, x):

        '''
        Input : torch.tensor [batch, channels(h), H, W]
        Output: torch.tensor [batch, num_classes, num_stations]
        
        '''
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.maxpool(out)

        out = self.stage1(out)
        out = self.relu(self.bn2(self.conv2(out)))
        
        out = self.stage2(out)
        out = self.relu(self.bn3(self.conv3(out)))
        
        out = self.stage3(out)
        out = self.relu(self.bn4(self.conv4(out)))

        out = self.stage4(out)
        out = self.relu(self.bn5(self.conv5(out)))
        
        out = self.avgpool(out)
        out = torch.flatten(out, 1)
        out = self.relu(self.fc1(out))
        out = self.dropout(out)
        out = self.fc2(out)

        out = out.view(-1, self.num_classes, self.num_stations)
        out = out.permute(0, 2, 1)
        
        return out

# Example usage
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#Example input size: (c=1,h=65, W=67), num_stations=38, num_classes=11(0:NaN,1-11:quantiles)
Net = ClassificationResNet((1, 65, 67), 38, 11).to(device)
