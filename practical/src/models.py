"""
Deep Learning Architectures for Virtual Staining:
1. Standard 2D U-Net (Regression Baseline)
2. Autoencoder & Convolutional Variational Autoencoder (VAE with Generative Sampling)
3. Pix2Pix Generator & 70x70 PatchGAN Discriminator (Conditional GAN)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    """Two consecutive Conv2D -> BatchNorm -> ReLU blocks."""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)


class Autoencoder(nn.Module):
    """
    Deterministic Convolutional Autoencoder (AE) for Representation Learning.
    """
    def __init__(self, in_channels=1, latent_dim=32):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 16, 3, stride=2, padding=1),  # 128x128
            nn.ReLU(),
            nn.Conv2d(16, 32, 3, stride=2, padding=1),           # 64x64
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),           # 32x32
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(64 * 32 * 32, latent_dim)
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 64 * 32 * 32),
            nn.Unflatten(1, (64, 32, 32)),
            nn.ConvTranspose2d(64, 32, 3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 16, 3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(16, in_channels, 3, stride=2, padding=1, output_padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z)


class ConvVAE(nn.Module):
    """
    Convolutional Variational Autoencoder (VAE) supporting Generative Sampling.
    """
    def __init__(self, in_channels=1, latent_dim=32):
        super().__init__()
        self.latent_dim = latent_dim

        # Shared Feature Encoder
        self.encoder_conv = nn.Sequential(
            nn.Conv2d(in_channels, 16, 3, stride=2, padding=1), # 128x128
            nn.ReLU(),
            nn.Conv2d(16, 32, 3, stride=2, padding=1),          # 64x64
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),          # 32x32
            nn.ReLU(),
            nn.Flatten()
        )

        # Latent parameters: Mean (mu) & Log-Variance (log_var)
        self.fc_mu = nn.Linear(64 * 32 * 32, latent_dim)
        self.fc_log_var = nn.Linear(64 * 32 * 32, latent_dim)

        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 64 * 32 * 32),
            nn.Unflatten(1, (64, 32, 32)),
            nn.ConvTranspose2d(64, 32, 3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 16, 3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(16, in_channels, 3, stride=2, padding=1, output_padding=1),
            nn.Sigmoid()
        )

    def encode(self, x):
        h = self.encoder_conv(x)
        mu = self.fc_mu(h)
        log_var = self.fc_log_var(h)
        return mu, log_var

    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + std * eps

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        mu, log_var = self.encode(x)
        z = self.reparameterize(mu, log_var)
        return self.decode(z), mu, log_var

    def sample(self, num_samples=16, device="cpu"):
        """Generative sampling: Draw z ~ N(0, I) and decode into synthetic cell images."""
        z = torch.randn(num_samples, self.latent_dim, device=device)
        return self.decode(z)


class UNet(nn.Module):
    """
    Standard 2D U-Net Encoder-Decoder Architecture for Virtual Staining Regression.
    """
    def __init__(self, in_channels=1, out_channels=1, features=[64, 128, 256, 512]):
        super().__init__()
        self.downs = nn.ModuleList()
        self.ups = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Encoder (Downsampling)
        curr_channels = in_channels
        for feature in features:
            self.downs.append(DoubleConv(curr_channels, feature))
            curr_channels = feature

        # Bottleneck
        self.bottleneck = DoubleConv(features[-1], features[-1] * 2)

        # Decoder (Upsampling)
        for feature in reversed(features):
            self.ups.append(
                nn.ConvTranspose2d(feature * 2, feature, kernel_size=2, stride=2)
            )
            self.ups.append(DoubleConv(feature * 2, feature))

        # Final output conv
        self.final_conv = nn.Sequential(
            nn.Conv2d(features[0], out_channels, kernel_size=1),
            nn.Sigmoid()  # Output range [0, 1]
        )

    def forward(self, x):
        skip_connections = []

        for down in self.downs:
            x = down(x)
            skip_connections.append(x)
            x = self.pool(x)

        x = self.bottleneck(x)
        skip_connections = skip_connections[::-1]

        for idx in range(0, len(self.ups), 2):
            x = self.ups[idx](x)
            skip_connection = skip_connections[idx // 2]
            concat_x = torch.cat((skip_connection, x), dim=1)
            x = self.ups[idx + 1](concat_x)

        return self.final_conv(x)


class Pix2PixGenerator(nn.Module):
    """
    Pix2Pix U-Net Generator with Tanh activation (output range [-1, 1]).
    """
    def __init__(self, in_channels=1, out_channels=1, features=[64, 128, 256, 512]):
        super().__init__()
        self.unet = UNet(in_channels=in_channels, out_channels=out_channels, features=features)
        # Modify final layer for Tanh range [-1, 1]
        self.unet.final_conv = nn.Sequential(
            nn.Conv2d(features[0], out_channels, kernel_size=1),
            nn.Tanh()
        )

    def forward(self, x):
        return self.unet(x)


class PatchGANDiscriminator(nn.Module):
    """
    70x70 PatchGAN Discriminator for Conditional GAN (evaluates input y and output x pair).
    """
    def __init__(self, in_channels=2, features=[64, 128, 256, 512]):
        super().__init__()
        layers = []
        curr_channels = in_channels
        
        for i, feature in enumerate(features):
            stride = 1 if i == len(features) - 1 else 2
            layers.append(
                nn.Sequential(
                    nn.Conv2d(curr_channels, feature, kernel_size=4, stride=stride, padding=1, bias=False),
                    nn.BatchNorm2d(feature) if i > 0 else nn.Identity(),
                    nn.LeakyReLU(0.2, inplace=True)
                )
            )
            curr_channels = feature
            
        layers.append(nn.Conv2d(curr_channels, 1, kernel_size=4, stride=1, padding=1))
        self.model = nn.Sequential(*layers)

    def forward(self, y, x):
        # Concatenate condition y and target/generated x along channel dim
        pair = torch.cat([y, x], dim=1)
        return self.model(pair)
