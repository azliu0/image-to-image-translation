import torch
from torch import nn
from torch.nn import functional as F
from server.pix2pix.modules.attention.attention import SelfAttention, CrossAttention


class TimeEmbedding(nn.Module):
    def __init__(self, n_embd):
        super().__init__()
        self.linear_1 = nn.Linear(n_embd, 4 * n_embd)
        self.linear_2 = nn.Linear(4 * n_embd, 4 * n_embd)

    def forward(self, x):
        x = self.linear_1(x)
        x = F.silu(x)
        x = self.linear_2(x)
        return x


class UNET_ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, n_time=1280):
        super().__init__()

        # group norm
        self.groupnorm_feature = nn.GroupNorm(32, in_channels)

        # conv layer
        self.conv_feature = nn.Conv2d(
            in_channels, out_channels, kernel_size=3, padding=1
        )

        # time layer
        self.linear_time = nn.Linear(n_time, out_channels)

        # group norm for merged time+feature vector
        self.groupnorm_merged = nn.GroupNorm(32, out_channels)

        # conv layer for merged time+feature vector
        self.conv_merged = nn.Conv2d(
            out_channels, out_channels, kernel_size=3, padding=1
        )

        # final residual layer to ensure that residue has the same number of channels as output
        if in_channels == out_channels:
            self.residual_layer = nn.Identity()
        else:
            self.residual_layer = nn.Conv2d(
                in_channels, out_channels, kernel_size=1, padding=0
            )

    def forward(self, feature, time):
        # feature: (in_channels,h,w)
        # time: (1, 1280)

        residue = feature

        # (in_channels,h,w)
        feature = self.groupnorm_feature(feature)

        # (in_channels,h,w)
        feature = F.silu(feature)

        # (out_channels,h,w)
        feature = self.conv_feature(feature)

        # (1280)
        time = F.silu(time)

        # (out_channels)
        time = self.linear_time(time)

        # (out_channels,h,w)
        merged = feature + time.unsqueeze(-1).unsqueeze(-1)

        # (out_channels,h,w)
        merged = self.groupnorm_merged(merged)

        # (out_channels,h,w)
        merged = F.silu(merged)

        # (out_channels,h,w)
        merged = self.conv_merged(merged)

        # join residue
        return merged + self.residual_layer(residue)


class UNET_AttentionBlock(nn.Module):
    def __init__(self, n_head, n_embd, d_context=768):
        super().__init__()
        channels = n_head * n_embd

        # normalization
        self.groupnorm = nn.GroupNorm(32, channels, eps=1e-6)
        self.conv_input = nn.Conv2d(channels, channels, kernel_size=1, padding=0)

        # self attention
        self.layernorm_1 = nn.LayerNorm(channels)
        self.attention_1 = SelfAttention(n_head, channels, in_proj_bias=False)

        # cross attention
        self.layernorm_2 = nn.LayerNorm(channels)
        self.attention_2 = CrossAttention(
            n_head, channels, d_context, in_proj_bias=False
        )

        # geglu activation
        self.layernorm_3 = nn.LayerNorm(channels)
        self.linear_geglu_1 = nn.Linear(channels, 4 * channels * 2)
        self.linear_geglu_2 = nn.Linear(4 * channels, channels)

        # output
        self.conv_output = nn.Conv2d(channels, channels, kernel_size=1, padding=0)

    def forward(self, x, context):
        # x : (c,h,w)
        # context: (seq, dim)

        # (c,h,w)
        residue_long = x

        # (c,h,w)
        x = self.groupnorm(x)
        # (c,h,w)
        x = self.conv_input(x)

        n, c, h, w = x.shape

        # (c,h,w) -> (c,h*w)
        x = x.view((n, c, h * w))
        # (c,h*w) -> (h*w,c)
        # transpose because attention input is (seq,dim)
        x = x.transpose(-1, -2)

        #  self attention with skip connection
        residue_short = x
        x = self.layernorm_1(x)
        x = self.attention_1(x)
        x += residue_short

        # cross attention with skip connection
        residue_short = x
        x = self.layernorm_2(x)
        x = self.attention_2(x, context)
        x += residue_short

        # FF with geglu and skip connection
        residue_short = x
        # (h*w,c)
        x = self.layernorm_3(x)
        # (h*w,c) -> 2*(h*w,c)
        x, gate = self.linear_geglu_1(x).chunk(2, dim=-1)
        # (h*w,c)
        x = x * F.gelu(gate)

        # (h*w,c)
        x = self.linear_geglu_2(x)
        # (h*w,c)
        x += residue_short

        # (c,h*w), undoing original transpose
        x = x.transpose(-1, -2)

        # (c,h,w)
        x = x.view((n, c, h, w))

        # add back long residual
        return self.conv_output(x) + residue_long


class Upsample(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, kernel_size=3, padding=1)

    def forward(self, x):
        # (h,w) -> (h*2,w*2)
        x = F.interpolate(x, scale_factor=2, mode="nearest")
        return self.conv(x)


class SwitchSequential(nn.Sequential):
    def forward(self, x, context, time):
        for layer in self:
            if isinstance(layer, UNET_AttentionBlock):
                x = layer(x, context)
            elif isinstance(layer, UNET_ResidualBlock):
                x = layer(x, time)
            else:
                x = layer(x)
        return x


class UNET(nn.Module):
    def __init__(self):
        super().__init__()

        # the #s in front of each comment represent corresponding skip connections

        self.encoders = nn.ModuleList(
            [
                # 12. (4, h/8, w/8) -> (320, h/8, w/8)
                SwitchSequential(nn.Conv2d(4, 320, kernel_size=3, padding=1)),
                # 11. (320, h/8, w/8)
                SwitchSequential(
                    UNET_ResidualBlock(320, 320), UNET_AttentionBlock(8, 40)
                ),
                # 10. (320, h/8, w/8)
                SwitchSequential(
                    UNET_ResidualBlock(320, 320), UNET_AttentionBlock(8, 40)
                ),
                # 9. (320, h/8, w/8) -> (320, h/16, w/16)
                SwitchSequential(
                    nn.Conv2d(320, 320, kernel_size=3, stride=2, padding=1)
                ),
                # 8. (320, h/16, w/16)
                SwitchSequential(
                    UNET_ResidualBlock(320, 640), UNET_AttentionBlock(8, 80)
                ),
                # 7. (320, h/16, w/16)
                SwitchSequential(
                    UNET_ResidualBlock(640, 640), UNET_AttentionBlock(8, 80)
                ),
                # 6. (320, h/16, w/16) -> (640, h/32, w/32)
                SwitchSequential(
                    nn.Conv2d(640, 640, kernel_size=3, stride=2, padding=1)
                ),
                # 5. (640, h/32, w/32)
                SwitchSequential(
                    UNET_ResidualBlock(640, 1280), UNET_AttentionBlock(8, 160)
                ),
                # 4. (640, h/32, w/32)
                SwitchSequential(
                    UNET_ResidualBlock(1280, 1280), UNET_AttentionBlock(8, 160)
                ),
                # 3. (640, h/32, w/32) -> (1280, h/64, w/64)
                SwitchSequential(
                    nn.Conv2d(1280, 1280, kernel_size=3, stride=2, padding=1)
                ),
                # 2. (1280, h/64, w/64)
                SwitchSequential(UNET_ResidualBlock(1280, 1280)),
                # 1. (1280, h/64, w/64)
                SwitchSequential(UNET_ResidualBlock(1280, 1280)),
            ]
        )

        self.bottleneck = SwitchSequential(
            UNET_ResidualBlock(1280, 1280),
            UNET_AttentionBlock(8, 160),
            UNET_ResidualBlock(1280, 1280),
        )

        self.decoders = nn.ModuleList(
            [
                # 1. (1280*2, h/64, w/64), due to skip connections
                SwitchSequential(UNET_ResidualBlock(2560, 1280)),
                # 2. (1280*2, h/64, w/64)
                SwitchSequential(UNET_ResidualBlock(2560, 1280)),
                # 3. (1280*2, h/32, w/32)
                SwitchSequential(UNET_ResidualBlock(2560, 1280), Upsample(1280)),
                # 4.
                SwitchSequential(
                    UNET_ResidualBlock(2560, 1280), UNET_AttentionBlock(8, 160)
                ),
                # 5.
                SwitchSequential(
                    UNET_ResidualBlock(2560, 1280), UNET_AttentionBlock(8, 160)
                ),
                # 6.
                # skip connections: (640, h/32, w/32)
                # inputs: (1280, h/32, w/32)
                # final output: (1280, h/16, w/16)
                SwitchSequential(
                    UNET_ResidualBlock(1920, 1280),
                    UNET_AttentionBlock(8, 160),
                    Upsample(1280),
                ),
                # 7.
                # skip connections: (640, h/16, w/16)
                # inputs: (1280, h/16, w/16)
                # final output: (640, h/16, w/16)
                SwitchSequential(
                    UNET_ResidualBlock(1920, 640), UNET_AttentionBlock(8, 80)
                ),
                # 8.
                # skip connections: (640, h/16, w/16)
                # inputs: (640, h/16, w/16)
                # final output: (640, h/16, w/16)
                SwitchSequential(
                    UNET_ResidualBlock(1280, 640), UNET_AttentionBlock(8, 80)
                ),
                # 9.
                # skip connections: (320, h/16, w/16)
                # inputs: (640, h/16, w/16)
                # final output: (640, h/8, w/8)
                SwitchSequential(
                    UNET_ResidualBlock(960, 640),
                    UNET_AttentionBlock(8, 80),
                    Upsample(640),
                ),
                # 10.
                # skip connections: (320, h/8, w/8)
                # inputs: (640, h/8, w/8)
                # final output: (320, h/8, w/8)
                SwitchSequential(
                    UNET_ResidualBlock(960, 320), UNET_AttentionBlock(8, 40)
                ),
                # 11.
                # skip connections: (320, h/8, w/8)
                # inputs: (320, h/8, w/8)
                # final output: (320, h/8, w/8)
                SwitchSequential(
                    UNET_ResidualBlock(640, 320), UNET_AttentionBlock(8, 40)
                ),
                # 12.
                # skip connections: (320, h/8, w/8)
                # inputs: (320, h/8, w/8)
                # final output: (320, h/8, w/8)
                SwitchSequential(
                    UNET_ResidualBlock(640, 320), UNET_AttentionBlock(8, 40)
                ),
            ]
        )

    def forward(self, x, context, time):
        # x: (4, h/8, w/8), latent image size
        # context: (seq, dim)
        # time: (1, 1280)

        skip_connections = []
        for layers in self.encoders:
            x = layers(x, context, time)
            skip_connections.append(x)

        x = self.bottleneck(x, context, time)

        for layers in self.decoders:
            # concatenate along the channels dimension
            x = torch.cat((x, skip_connections.pop()), dim=1)
            x = layers(x, context, time)

        return x


class UNET_OutputLayer(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.groupnorm = nn.GroupNorm(32, in_channels)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)

    def forward(self, x):
        # x: (320, h/8, w/8)

        x = self.groupnorm(x)
        x = F.silu(x)

        # (4, h/8, w/8)
        x = self.conv(x)
        return x


class Diffusion(nn.Module):
    def __init__(self):
        super().__init__()
        self.time_embedding = TimeEmbedding(320)
        self.unet = UNET()
        self.final = UNET_OutputLayer(320, 4)

    def forward(self, latent, context, time):
        # latent: (4, h/8, w/8)
        # context: (seq, dim)
        # time: (1, 320)

        # (1, 320) -> (1, 1280)
        time = self.time_embedding(time)

        # (4, h/8, h/8) -> (320, h/8, h/8)
        output = self.unet(latent, context, time)

        # (320, h/8, h/8) -> (4, h/8, h/8)
        output = self.final(output)

        return output
