import torch
from torch import nn
from torch.nn import functional as F
from server.pix2pix.modules.attention.attention import SelfAttention


class VAE_AttentionBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.groupnorm = nn.GroupNorm(32, channels)
        self.attention = SelfAttention(1, channels)

    def forward(self, x):
        residue = x

        # (c,h,w)
        x = self.groupnorm(x)

        n, c, h, w = x.shape

        # (c,h,w) -> (c,h*w)
        x = x.view(n, c, h * w)
        # (c,h*w) -> (h*w,c)
        # transpose because attention input is (seq,dim)
        x = x.transpose(-1, -2)
        # (h*w,c) -> (h*w,c)
        x = self.attention(x)
        # (h*w,c) -> (c,h*w)
        # undo previous transpose
        x = x.transpose(-1, -2)
        # (c,h*w) -> (c,h,w)
        x = x.view(n, c, h, w)

        return x + residue


class VAE_ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.groupnorm_1 = nn.GroupNorm(32, in_channels)
        self.conv_1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)

        self.groupnorm_2 = nn.GroupNorm(32, out_channels)
        self.conv_2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)

        if in_channels != out_channels:
            self.residual_layer = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        else:
            self.residual_layer = nn.Identity()

    def forward(self, x):
        # x: (h,w)
        residue = x
        x = self.groupnorm_1(x)
        x = F.silu(x)
        x = self.conv_1(x)
        x = self.groupnorm_2(x)
        x = F.silu(x)
        x = self.conv_2(x)
        return x + self.residual_layer(residue)


class VAE_Decoder(nn.Sequential):
    def __init__(self):
        super().__init__(
            # (4,h/8,w/8)
            nn.Conv2d(4, 4, kernel_size=1, padding=0),
            # (4,h/8,w/8) -> (512, h/8, w/8)
            nn.Conv2d(4, 512, kernel_size=3, padding=1),
            # (512, h/8, w/8)
            VAE_ResidualBlock(512, 512),
            # (512, h/8, w/8)
            VAE_AttentionBlock(512),
            # (512, h/8, w/8)
            VAE_ResidualBlock(512, 512),
            # (512, h/8, w/8)
            VAE_ResidualBlock(512, 512),
            # (512, h/8, w/8)
            VAE_ResidualBlock(512, 512),
            # (512, h/8, w/8)
            VAE_ResidualBlock(512, 512),
            # (512, h/8, w/8) -> (512, h/4, w/4)
            nn.Upsample(scale_factor=2),
            # (512, h/4, w/4)
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            # (512, h/4, w/4)
            VAE_ResidualBlock(512, 512),
            # (512, h/4, w/4)
            VAE_ResidualBlock(512, 512),
            # (512, h/4, w/4)
            VAE_ResidualBlock(512, 512),
            # (512, h/4, w/4) -> (512, h/2, w/2)
            nn.Upsample(scale_factor=2),
            # (512, h/2, w/2)
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            # (512, h/2, w/2) -> (256, h/2, w/2)
            VAE_ResidualBlock(512, 256),
            # (256, h/2, w/2)
            VAE_ResidualBlock(256, 256),
            # (256, h/2, w/2)
            VAE_ResidualBlock(256, 256),
            # (256, h/2, w/2) -> (256, h, w)
            nn.Upsample(scale_factor=2),
            # (256, h, w)
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            # (256, h, w) -> (128, h, w)
            VAE_ResidualBlock(256, 128),
            # (128, h, w)
            VAE_ResidualBlock(128, 128),
            # (128, h, w)
            VAE_ResidualBlock(128, 128),
            # (128, h, w)
            nn.GroupNorm(32, 128),
            # (128, h, w)
            nn.SiLU(),
            # (128, h, w) -> (3, h, w)
            # back to original image shape!
            nn.Conv2d(128, 3, kernel_size=3, padding=1),
        )

    def forward(self, x):
        # x: (4,h/8,w/8), latent size

        # https://github.com/huggingface/diffusers/issues/437
        x /= 0.18215

        for module in self:
            x = module(x)

        return x
