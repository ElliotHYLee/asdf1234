import torch
import torch.nn as nn
import torch.nn.functional as F

class Encoder(nn.Module):
    def __init__(self, input_channels=1, conv_channels=[16, 32], fc_sizes=[128, 64, 32]):
        super(Encoder, self).__init__()
        # 2D convolutional layers
        self.conv1 = nn.Conv2d(input_channels, conv_channels[0], kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(conv_channels[0])
        self.conv2 = nn.Conv2d(conv_channels[0], conv_channels[1], kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(conv_channels[1])
        
        # Fully connected layers
        self.fc1 = nn.Linear(conv_channels[1] * 8 * 8, fc_sizes[0])
        self.fc2 = nn.Linear(fc_sizes[0], fc_sizes[1])
        self.fc3 = nn.Linear(fc_sizes[1], fc_sizes[2])
        
    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class Decoder(nn.Module):
    def __init__(self, conv_channels=[32, 16], fc_sizes=[64, 128], output_channels=1):
        super(Decoder, self).__init__()
        # Fully connected layers
        self.fc1 = nn.Linear(32, fc_sizes[0])
        self.fc2 = nn.Linear(fc_sizes[0], fc_sizes[1])
        self.fc3 = nn.Linear(fc_sizes[1], conv_channels[0] * 8 * 8)
        
        # 2D transposed convolutional layers
        self.deconv1 = nn.ConvTranspose2d(conv_channels[0], conv_channels[1], kernel_size=3, stride=2, padding=1, output_padding=1)
        self.bn1 = nn.BatchNorm2d(conv_channels[1])
        self.deconv2 = nn.ConvTranspose2d(conv_channels[1], output_channels, kernel_size=3, stride=2, padding=1, output_padding=1)
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = x.view(x.size(0), 32, 8, 8)
        x = F.relu(self.bn1(self.deconv1(x)))
        x = self.deconv2(x)
        return x

# Instantiate the models
encoder = Encoder()
decoder = Decoder()

# Print the models
print(encoder)
print(decoder)
