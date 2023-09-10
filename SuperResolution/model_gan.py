import torch
import torch.nn as nn

class Generator(nn.Module):
    def __init__(self, UPSCALE_FACTOR):
        super(Generator, self).__init__()
        
        self.upscale_factor = UPSCALE_FACTOR
        self.conv1 = nn.Conv2d(3, 64, (3, 3), (1, 1), (1, 1))
        self.conv2 = nn.Conv2d(64, 32, (3, 3), (1, 1), (1, 1))
        self.conv3 = nn.Conv2d(32, 3 * (self.upscale_factor ** 2), (3, 3), (1, 1), (1, 1))
        self.pixel_shuffle = nn.PixelShuffle(self.upscale_factor)

    def forward(self, x):
        x = torch.tanh(self.conv1(x))
        x = torch.tanh(self.conv2(x))
        x = torch.sigmoid(self.pixel_shuffle(self.conv3(x)))
        return x

class DownSalmpe(nn.Module):
    def __init__(self, input_channel, output_channel,  stride, kernel_size=3, padding=1):
        super().__init__()
        self.layer = nn.Sequential(
            nn.Conv2d(input_channel, output_channel, kernel_size, stride, padding),
            nn.BatchNorm2d(output_channel),
            nn.LeakyReLU(inplace=True)
        )

    def forward(self, x):
        x = self.layer(x)
        return x

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        
        # Convolutional and pooling layers
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, 3, stride=1, padding=1),
            nn.LeakyReLU(inplace=True),

            DownSalmpe(64, 64, stride=2, padding=1),
            DownSalmpe(64, 128, stride=1, padding=1),
            DownSalmpe(128, 128, stride=2, padding=1),
            DownSalmpe(128, 256, stride=1, padding=1),
            DownSalmpe(256, 256, stride=2, padding=1),
            DownSalmpe(256, 512, stride=1, padding=1),
            DownSalmpe(512, 512, stride=2, padding=1),
        )
        
        # Fully connected layers
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(512, 1024, 1),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(1024, 1, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

if __name__ == '__main__':
    g = Generator(4)
    a = torch.rand([1, 3, 64, 64])
    print('# parameters:', sum(param.numel() for param in g.parameters()))
    print(g(a).shape)
    d = Discriminator()
    b = torch.rand([8, 3, 512, 512])
    print(d(b).shape)
    print('# parameters:', sum(param.numel() for param in d.parameters()))
