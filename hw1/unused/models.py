import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

class Encoder(nn.Module):
    def __init__(self,):
        super(Encoder, self).__init__()
        # 2D convolutional layers
        self.conv1 = nn.Conv2d(36, 5, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(5)
        self.conv2 = nn.Conv2d(5, 1, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(1)
        
        # Fully connected layers
        self.fc1 = nn.Linear(480, 96)
        self.fc2 = nn.Linear(96, 32)
        self.fc3 = nn.Linear(32, 10)
        
    def forward(self, x):
        x = self.bn1(F.relu(self.conv1(x)))
        x = self.bn2(F.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    
class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        # Fully connected layers
        self.fc1 = nn.Linear(10, 64)
        self.fc2 = nn.Linear(64, 512)
        
        # 2D transposed convolutional layers, expecting: (N, 256, 1, 2) 
        self.deconv1 = nn.ConvTranspose2d(256, 82, kernel_size=5, stride=1, padding=0, output_padding=0)
        self.bn1 = nn.BatchNorm2d(82)
        self.deconv2 = nn.ConvTranspose2d(82, 60, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.bn2 = nn.BatchNorm2d(60)
        self.deconv3 = nn.ConvTranspose2d(60, 36, kernel_size=3, stride=2, padding=1, output_padding=1)
        # self.bn3 = nn.BatchNorm2d(36)
        # # Additional ConvTranspose2d layers
        # self.deconv4 = nn.ConvTranspose2d(36, 36, kernel_size=3, stride=1, padding=1, output_padding=0)
        # self.bn4 = nn.BatchNorm2d(36)
        # self.deconv5 = nn.ConvTranspose2d(36, 36, kernel_size=3, stride=1, padding=1, output_padding=0)
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = x.view(x.size(0), 256, 1, 2)
        x = self.bn1(F.relu(self.deconv1(x)))
        x = self.bn2(F.relu(self.deconv2(x)))
        x = self.deconv3(x)
        
        # x = self.bn3(F.relu(self.deconv3(x)))
        # x = self.bn4(F.relu(self.deconv4(x)))
        # x = self.deconv5(x)

        return x

class Autoencoder(nn.Module):
    def __init__(self, encoder, decoder):
        super(Autoencoder, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        
    def forward(self, x):
        z = self.encoder(x)
        x_reconstructed = self.decoder(z)
        return x_reconstructed
    

def trace_encoder_shapes(encoder, input_data):
    print("Tracing Encoder Shapes")
    x = input_data
    print(f'{"Input:":<25} {x.shape}')
    x = F.relu(encoder.bn1(encoder.conv1(x)))
    print(f'{"After conv1:":<25} {x.shape}')
    x = F.relu(encoder.bn2(encoder.conv2(x)))
    print(f'{"After conv2:":<25} {x.shape}')
    x = x.view(x.size(0), -1)
    print(f'{"After Flattend:":<25} {x.shape}')
    x = F.relu(encoder.fc1(x))
    print(f'{"After fc1:":<25} {x.shape}')
    x = F.relu(encoder.fc2(x))
    print(f'{"After fc2:":<25} {x.shape}')
    x = encoder.fc3(x)
    print(f'{"After fc3:":<25} {x.shape}')

def trace_decoder_shapes(decoder, encoded_data):
    print("Tracing Decoder Shapes")
    x = encoded_data
    print(f'{"Input:":<25} {x.shape}')
    x = F.relu(decoder.fc1(x))
    print(f'{"After fc1:":<25} {x.shape}')
    x = F.relu(decoder.fc2(x))
    print(f'{"After fc2:":<25} {x.shape}')
    x = x.view(x.size(0), 256, 1, 2)
    print(f'{"After Unflattend:":<25} {x.shape}')
    x = F.relu(decoder.bn1(decoder.deconv1(x)))
    print(f'{"After deconv1:":<25} {x.shape}')
    x = F.relu(decoder.bn2(decoder.deconv2(x)))
    print(f'{"After deconv2:":<25} {x.shape}')
    x = decoder.deconv3(x)
    print(f'{"After deconv3:":<25} {x.shape}')
    # x = F.relu(decoder.bn3(decoder.deconv3(x)))
    # print(f'{"After deconv3:":<25} {x.shape}')
    # x = F.relu(decoder.bn4(decoder.deconv4(x)))
    # print(f'{"After deconv4:":<25} {x.shape}')
    # x = decoder.deconv5(x)
    # print(f'{"After deconv5:":<25} {x.shape}')

if __name__ == "__main__":
    # Create instances of the encoder and decoder
    encoder = Encoder()
    decoder = Decoder()
    autoencoder = Autoencoder(encoder, decoder)

    # Create a random tensor with the shape (N, 26, 24, 20)
    N = 1234  # Batch size
    input_data = torch.randn(N, 36, 24, 20)
    encoded_data = encoder(input_data)
    # Trace the shapes through the encoder and decoder
    print(encoder)
    trace_encoder_shapes(encoder, input_data)
    print(decoder)
    trace_decoder_shapes(decoder, encoded_data)    