import torch
from torch import nn
from torch.nn import functional as F
from decoder import VAE_AttentionBlock, VAE_ResidualBlock


class VAE_Encoder(nn.Sequential):
    def __init__(self):
        super().__init__(
            # (h,w)->(h,w)
            nn.Conv2d(3, 128, kernel_size=3, padding=1),
            # (h,w)->(h,w)
            VAE_ResidualBlock(128, 128),
            # (h,w)->(h,w)
            VAE_ResidualBlock(128, 128),
            # (h,w)->(h/2,w/2)
            nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=0),
            # (h/2,w/2)->(h/2,w/2)
            VAE_ResidualBlock(128, 256),
            # (h/2,w/2)->(h/2,w/2)
            VAE_ResidualBlock(256, 256),
            # (h/2,w/2)->(h/4,w/4)
            nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=0),
            # (h/2,w/2)->(h/4,w/4)
            VAE_ResidualBlock(256, 512),
            # (h/2,w/2)->(h/4,w/4)
            VAE_ResidualBlock(512, 512),
            # (h/4,w/4)->(h/8,w/8)
            nn.Conv2d(512, 512, kernel_size=3, stride=2, padding=0),
            # (h/8,w/8)->(h/8,w/8)
            VAE_ResidualBlock(512, 512),
            # (h/8,w/8)->(h/8,w/8)
            VAE_ResidualBlock(512, 512),
            # (h/8,w/8)->(h/8,w/8)
            VAE_ResidualBlock(512, 512),
            # (h/8,w/8)->(h/8,w/8)
            VAE_AttentionBlock(512),
            # (h/8,w/8)->(h/8,w/8)
            VAE_ResidualBlock(512, 512),
            # (h/8,w/8)->(h/8,w/8)
            nn.GroupNorm(32, 512),
            # (h/8,w/8)->(h/8,w/8)
            nn.SiLU(),
            # (h/8,w/8)->(h/8,w/8)
            nn.Conv2d(512, 8, kernel_size=3, padding=1),
            # (h/8,w/8)->(h/8,w/8)
            nn.Conv2d(8, 8, kernel_size=1, padding=0),
        )

    def forward(self, x, noise):
        # x: (h,w)
        # noise: (h/8, w/8), same dims as encoder output
        for module in self:
            if getattr(module, "stride", None) == (2, 2):
                x = F.pad(x, (0, 1, 0, 1))
            x = module(x)

        # (8,h/8,w/8)->[(4,h/8,w/8),(4,h/8,w/8)]
        mean, log_var = torch.chunk(x, 2, dim=1)
        log_var = torch.clamp(log_var, -30, 20)
        var = log_var.exp()
        stdev = var.sqrt()

        x = mean + stdev * noise

        # https://github.com/huggingface/diffusers/issues/437
        x *= 0.18215

        return x
