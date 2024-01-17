---
title: Model Architectures
author: Jack Chen and Andrew Liu
date: January 15, 2024
time: 1 min
---

## Models details

### CLIP Transformer Encoder

<details>

<summary>Module</summary>

```
CLIP(
  (embedding): CLIPEmbedding(
    (token_embedding): Embedding(49408, 768)
  )
  (layers): ModuleList(
    (0-11): 12 x CLIPLayer(
      (layernorm_1): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
      (attention): SelfAttention(
        (in_proj): Linear(in_features=768, out_features=2304, bias=True)
        (out_proj): Linear(in_features=768, out_features=768, bias=True)
      )
      (layernorm_2): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
      (linear_1): Linear(in_features=768, out_features=3072, bias=True)
      (linear_2): Linear(in_features=3072, out_features=768, bias=True)
    )
  )
  (layernorm): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
)
```

</details>

### Variational Autoencoder Encoder

<details>

<summary>Module</summary>

```
VAE_Encoder(
  (0): Conv2d(3, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
  (1): VAE_ResidualBlock(
    (groupnorm_1): GroupNorm(32, 128, eps=1e-05, affine=True)
    (conv_1): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (groupnorm_2): GroupNorm(32, 128, eps=1e-05, affine=True)
    (conv_2): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (residual_layer): Identity()
  )
  (2): VAE_ResidualBlock(
    (groupnorm_1): GroupNorm(32, 128, eps=1e-05, affine=True)
    (conv_1): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (groupnorm_2): GroupNorm(32, 128, eps=1e-05, affine=True)
    (conv_2): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (residual_layer): Identity()
  )
  (3): Conv2d(128, 128, kernel_size=(3, 3), stride=(2, 2))
  (4): VAE_ResidualBlock(
    (groupnorm_1): GroupNorm(32, 128, eps=1e-05, affine=True)
    (conv_1): Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (groupnorm_2): GroupNorm(32, 256, eps=1e-05, affine=True)
    (conv_2): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (residual_layer): Conv2d(128, 256, kernel_size=(1, 1), stride=(1, 1))
  )
  (5): VAE_ResidualBlock(
    (groupnorm_1): GroupNorm(32, 256, eps=1e-05, affine=True)
    (conv_1): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (groupnorm_2): GroupNorm(32, 256, eps=1e-05, affine=True)
    (conv_2): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (residual_layer): Identity()
  )
  (6): Conv2d(256, 256, kernel_size=(3, 3), stride=(2, 2))
  (7): VAE_ResidualBlock(
    (groupnorm_1): GroupNorm(32, 256, eps=1e-05, affine=True)
    (conv_1): Conv2d(256, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (groupnorm_2): GroupNorm(32, 512, eps=1e-05, affine=True)
    (conv_2): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (residual_layer): Conv2d(256, 512, kernel_size=(1, 1), stride=(1, 1))
  )
  (8): VAE_ResidualBlock(
    (groupnorm_1): GroupNorm(32, 512, eps=1e-05, affine=True)
    (conv_1): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (groupnorm_2): GroupNorm(32, 512, eps=1e-05, affine=True)
    (conv_2): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (residual_layer): Identity()
  )
  (9): Conv2d(512, 512, kernel_size=(3, 3), stride=(2, 2))
  (10): VAE_ResidualBlock(
    (groupnorm_1): GroupNorm(32, 512, eps=1e-05, affine=True)
    (conv_1): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (groupnorm_2): GroupNorm(32, 512, eps=1e-05, affine=True)
    (conv_2): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (residual_layer): Identity()
  )
  (11): VAE_ResidualBlock(
    (groupnorm_1): GroupNorm(32, 512, eps=1e-05, affine=True)
    (conv_1): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (groupnorm_2): GroupNorm(32, 512, eps=1e-05, affine=True)
    (conv_2): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (residual_layer): Identity()
  )
  (12): VAE_ResidualBlock(
    (groupnorm_1): GroupNorm(32, 512, eps=1e-05, affine=True)
    (conv_1): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (groupnorm_2): GroupNorm(32, 512, eps=1e-05, affine=True)
    (conv_2): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (residual_layer): Identity()
  )
  (13): VAE_AttentionBlock(
    (groupnorm): GroupNorm(32, 512, eps=1e-05, affine=True)
    (attention): SelfAttention(
      (in_proj): Linear(in_features=512, out_features=1536, bias=True)
      (out_proj): Linear(in_features=512, out_features=512, bias=True)
    )
  )
  (14): VAE_ResidualBlock(
    (groupnorm_1): GroupNorm(32, 512, eps=1e-05, affine=True)
    (conv_1): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (groupnorm_2): GroupNorm(32, 512, eps=1e-05, affine=True)
    (conv_2): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (residual_layer): Identity()
  )
  (15): GroupNorm(32, 512, eps=1e-05, affine=True)
  (16): SiLU()
  (17): Conv2d(512, 8, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
  (18): Conv2d(8, 8, kernel_size=(1, 1), stride=(1, 1))
)
```

</details>

### Variational Autoencoder Decoder

<details>

<summary>Module</summary>

```
VAE_Decoder(
  (0): Conv2d(4, 4, kernel_size=(1, 1), stride=(1, 1))
  (1): Conv2d(4, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
  (2): VAE_ResidualBlock(
    (groupnorm_1): GroupNorm(32, 512, eps=1e-05, affine=True)
    (conv_1): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (groupnorm_2): GroupNorm(32, 512, eps=1e-05, affine=True)
    (conv_2): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (residual_layer): Identity()
  )
  (3): VAE_AttentionBlock(
    (groupnorm): GroupNorm(32, 512, eps=1e-05, affine=True)
    (attention): SelfAttention(
      (in_proj): Linear(in_features=512, out_features=1536, bias=True)
      (out_proj): Linear(in_features=512, out_features=512, bias=True)
    )
  )
  (4): VAE_ResidualBlock(
    (groupnorm_1): GroupNorm(32, 512, eps=1e-05, affine=True)
    (conv_1): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (groupnorm_2): GroupNorm(32, 512, eps=1e-05, affine=True)
    (conv_2): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (residual_layer): Identity()
  )
  (5): VAE_ResidualBlock(
    (groupnorm_1): GroupNorm(32, 512, eps=1e-05, affine=True)
    (conv_1): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (groupnorm_2): GroupNorm(32, 512, eps=1e-05, affine=True)
    (conv_2): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (residual_layer): Identity()
  )
  (6): VAE_ResidualBlock(
    (groupnorm_1): GroupNorm(32, 512, eps=1e-05, affine=True)
    (conv_1): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (groupnorm_2): GroupNorm(32, 512, eps=1e-05, affine=True)
    (conv_2): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (residual_layer): Identity()
  )
  (7): VAE_ResidualBlock(
    (groupnorm_1): GroupNorm(32, 512, eps=1e-05, affine=True)
    (conv_1): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (groupnorm_2): GroupNorm(32, 512, eps=1e-05, affine=True)
    (conv_2): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (residual_layer): Identity()
  )
  (8): Upsample(scale_factor=2.0, mode='nearest')
  (9): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
  (10): VAE_ResidualBlock(
    (groupnorm_1): GroupNorm(32, 512, eps=1e-05, affine=True)
    (conv_1): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (groupnorm_2): GroupNorm(32, 512, eps=1e-05, affine=True)
    (conv_2): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (residual_layer): Identity()
  )
  (11): VAE_ResidualBlock(
    (groupnorm_1): GroupNorm(32, 512, eps=1e-05, affine=True)
    (conv_1): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (groupnorm_2): GroupNorm(32, 512, eps=1e-05, affine=True)
    (conv_2): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (residual_layer): Identity()
  )
  (12): VAE_ResidualBlock(
    (groupnorm_1): GroupNorm(32, 512, eps=1e-05, affine=True)
    (conv_1): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (groupnorm_2): GroupNorm(32, 512, eps=1e-05, affine=True)
    (conv_2): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (residual_layer): Identity()
  )
  (13): Upsample(scale_factor=2.0, mode='nearest')
  (14): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
  (15): VAE_ResidualBlock(
    (groupnorm_1): GroupNorm(32, 512, eps=1e-05, affine=True)
    (conv_1): Conv2d(512, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (groupnorm_2): GroupNorm(32, 256, eps=1e-05, affine=True)
    (conv_2): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (residual_layer): Conv2d(512, 256, kernel_size=(1, 1), stride=(1, 1))
  )
  (16): VAE_ResidualBlock(
    (groupnorm_1): GroupNorm(32, 256, eps=1e-05, affine=True)
    (conv_1): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (groupnorm_2): GroupNorm(32, 256, eps=1e-05, affine=True)
    (conv_2): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (residual_layer): Identity()
  )
  (17): VAE_ResidualBlock(
    (groupnorm_1): GroupNorm(32, 256, eps=1e-05, affine=True)
    (conv_1): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (groupnorm_2): GroupNorm(32, 256, eps=1e-05, affine=True)
    (conv_2): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (residual_layer): Identity()
  )
  (18): Upsample(scale_factor=2.0, mode='nearest')
  (19): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
  (20): VAE_ResidualBlock(
    (groupnorm_1): GroupNorm(32, 256, eps=1e-05, affine=True)
    (conv_1): Conv2d(256, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (groupnorm_2): GroupNorm(32, 128, eps=1e-05, affine=True)
    (conv_2): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (residual_layer): Conv2d(256, 128, kernel_size=(1, 1), stride=(1, 1))
  )
  (21): VAE_ResidualBlock(
    (groupnorm_1): GroupNorm(32, 128, eps=1e-05, affine=True)
    (conv_1): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (groupnorm_2): GroupNorm(32, 128, eps=1e-05, affine=True)
    (conv_2): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (residual_layer): Identity()
  )
  (22): VAE_ResidualBlock(
    (groupnorm_1): GroupNorm(32, 128, eps=1e-05, affine=True)
    (conv_1): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (groupnorm_2): GroupNorm(32, 128, eps=1e-05, affine=True)
    (conv_2): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (residual_layer): Identity()
  )
  (23): GroupNorm(32, 128, eps=1e-05, affine=True)
  (24): SiLU()
  (25): Conv2d(128, 3, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
)
```

</details>

### Latent Diffusion Model UNet

<details>

<summary>Module</summary>

```
UNET(
  (encoders): ModuleList(
    (0): SwitchSequential(
      (0): Conv2d(4, 320, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    )
    (1-2): 2 x SwitchSequential(
      (0): UNET_ResidualBlock(
        (groupnorm_feature): GroupNorm(32, 320, eps=1e-05, affine=True)
        (conv_feature): Conv2d(320, 320, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (linear_time): Linear(in_features=1280, out_features=320, bias=True)
        (groupnorm_merged): GroupNorm(32, 320, eps=1e-05, affine=True)
        (conv_merged): Conv2d(320, 320, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (residual_layer): Identity()
      )
      (1): UNET_AttentionBlock(
        (groupnorm): GroupNorm(32, 320, eps=1e-06, affine=True)
        (conv_input): Conv2d(320, 320, kernel_size=(1, 1), stride=(1, 1))
        (layernorm_1): LayerNorm((320,), eps=1e-05, elementwise_affine=True)
        (attention_1): SelfAttention(
          (in_proj): Linear(in_features=320, out_features=960, bias=False)
          (out_proj): Linear(in_features=320, out_features=320, bias=True)
        )
        (layernorm_2): LayerNorm((320,), eps=1e-05, elementwise_affine=True)
        (attention_2): CrossAttention(
          (q_proj): Linear(in_features=320, out_features=320, bias=False)
          (k_proj): Linear(in_features=768, out_features=320, bias=False)
          (v_proj): Linear(in_features=768, out_features=320, bias=False)
          (out_proj): Linear(in_features=320, out_features=320, bias=True)
        )
        (layernorm_3): LayerNorm((320,), eps=1e-05, elementwise_affine=True)
        (linear_geglu_1): Linear(in_features=320, out_features=2560, bias=True)
        (linear_geglu_2): Linear(in_features=1280, out_features=320, bias=True)
        (conv_output): Conv2d(320, 320, kernel_size=(1, 1), stride=(1, 1))
      )
    )
    (3): SwitchSequential(
      (0): Conv2d(320, 320, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
    )
    (4): SwitchSequential(
      (0): UNET_ResidualBlock(
        (groupnorm_feature): GroupNorm(32, 320, eps=1e-05, affine=True)
        (conv_feature): Conv2d(320, 640, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (linear_time): Linear(in_features=1280, out_features=640, bias=True)
        (groupnorm_merged): GroupNorm(32, 640, eps=1e-05, affine=True)
        (conv_merged): Conv2d(640, 640, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (residual_layer): Conv2d(320, 640, kernel_size=(1, 1), stride=(1, 1))
      )
      (1): UNET_AttentionBlock(
        (groupnorm): GroupNorm(32, 640, eps=1e-06, affine=True)
        (conv_input): Conv2d(640, 640, kernel_size=(1, 1), stride=(1, 1))
        (layernorm_1): LayerNorm((640,), eps=1e-05, elementwise_affine=True)
        (attention_1): SelfAttention(
          (in_proj): Linear(in_features=640, out_features=1920, bias=False)
          (out_proj): Linear(in_features=640, out_features=640, bias=True)
        )
        (layernorm_2): LayerNorm((640,), eps=1e-05, elementwise_affine=True)
        (attention_2): CrossAttention(
          (q_proj): Linear(in_features=640, out_features=640, bias=False)
          (k_proj): Linear(in_features=768, out_features=640, bias=False)
          (v_proj): Linear(in_features=768, out_features=640, bias=False)
          (out_proj): Linear(in_features=640, out_features=640, bias=True)
        )
        (layernorm_3): LayerNorm((640,), eps=1e-05, elementwise_affine=True)
        (linear_geglu_1): Linear(in_features=640, out_features=5120, bias=True)
        (linear_geglu_2): Linear(in_features=2560, out_features=640, bias=True)
        (conv_output): Conv2d(640, 640, kernel_size=(1, 1), stride=(1, 1))
      )
    )
    (5): SwitchSequential(
      (0): UNET_ResidualBlock(
        (groupnorm_feature): GroupNorm(32, 640, eps=1e-05, affine=True)
        (conv_feature): Conv2d(640, 640, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (linear_time): Linear(in_features=1280, out_features=640, bias=True)
        (groupnorm_merged): GroupNorm(32, 640, eps=1e-05, affine=True)
        (conv_merged): Conv2d(640, 640, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (residual_layer): Identity()
      )
      (1): UNET_AttentionBlock(
        (groupnorm): GroupNorm(32, 640, eps=1e-06, affine=True)
        (conv_input): Conv2d(640, 640, kernel_size=(1, 1), stride=(1, 1))
        (layernorm_1): LayerNorm((640,), eps=1e-05, elementwise_affine=True)
        (attention_1): SelfAttention(
          (in_proj): Linear(in_features=640, out_features=1920, bias=False)
          (out_proj): Linear(in_features=640, out_features=640, bias=True)
        )
        (layernorm_2): LayerNorm((640,), eps=1e-05, elementwise_affine=True)
        (attention_2): CrossAttention(
          (q_proj): Linear(in_features=640, out_features=640, bias=False)
          (k_proj): Linear(in_features=768, out_features=640, bias=False)
          (v_proj): Linear(in_features=768, out_features=640, bias=False)
          (out_proj): Linear(in_features=640, out_features=640, bias=True)
        )
        (layernorm_3): LayerNorm((640,), eps=1e-05, elementwise_affine=True)
        (linear_geglu_1): Linear(in_features=640, out_features=5120, bias=True)
        (linear_geglu_2): Linear(in_features=2560, out_features=640, bias=True)
        (conv_output): Conv2d(640, 640, kernel_size=(1, 1), stride=(1, 1))
      )
    )
    (6): SwitchSequential(
      (0): Conv2d(640, 640, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
    )
    (7): SwitchSequential(
      (0): UNET_ResidualBlock(
        (groupnorm_feature): GroupNorm(32, 640, eps=1e-05, affine=True)
        (conv_feature): Conv2d(640, 1280, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (linear_time): Linear(in_features=1280, out_features=1280, bias=True)
        (groupnorm_merged): GroupNorm(32, 1280, eps=1e-05, affine=True)
        (conv_merged): Conv2d(1280, 1280, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (residual_layer): Conv2d(640, 1280, kernel_size=(1, 1), stride=(1, 1))
      )
      (1): UNET_AttentionBlock(
        (groupnorm): GroupNorm(32, 1280, eps=1e-06, affine=True)
        (conv_input): Conv2d(1280, 1280, kernel_size=(1, 1), stride=(1, 1))
        (layernorm_1): LayerNorm((1280,), eps=1e-05, elementwise_affine=True)
        (attention_1): SelfAttention(
          (in_proj): Linear(in_features=1280, out_features=3840, bias=False)
          (out_proj): Linear(in_features=1280, out_features=1280, bias=True)
        )
        (layernorm_2): LayerNorm((1280,), eps=1e-05, elementwise_affine=True)
        (attention_2): CrossAttention(
          (q_proj): Linear(in_features=1280, out_features=1280, bias=False)
          (k_proj): Linear(in_features=768, out_features=1280, bias=False)
          (v_proj): Linear(in_features=768, out_features=1280, bias=False)
          (out_proj): Linear(in_features=1280, out_features=1280, bias=True)
        )
        (layernorm_3): LayerNorm((1280,), eps=1e-05, elementwise_affine=True)
        (linear_geglu_1): Linear(in_features=1280, out_features=10240, bias=True)
        (linear_geglu_2): Linear(in_features=5120, out_features=1280, bias=True)
        (conv_output): Conv2d(1280, 1280, kernel_size=(1, 1), stride=(1, 1))
      )
    )
    (8): SwitchSequential(
      (0): UNET_ResidualBlock(
        (groupnorm_feature): GroupNorm(32, 1280, eps=1e-05, affine=True)
        (conv_feature): Conv2d(1280, 1280, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (linear_time): Linear(in_features=1280, out_features=1280, bias=True)
        (groupnorm_merged): GroupNorm(32, 1280, eps=1e-05, affine=True)
        (conv_merged): Conv2d(1280, 1280, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (residual_layer): Identity()
      )
      (1): UNET_AttentionBlock(
        (groupnorm): GroupNorm(32, 1280, eps=1e-06, affine=True)
        (conv_input): Conv2d(1280, 1280, kernel_size=(1, 1), stride=(1, 1))
        (layernorm_1): LayerNorm((1280,), eps=1e-05, elementwise_affine=True)
        (attention_1): SelfAttention(
          (in_proj): Linear(in_features=1280, out_features=3840, bias=False)
          (out_proj): Linear(in_features=1280, out_features=1280, bias=True)
        )
        (layernorm_2): LayerNorm((1280,), eps=1e-05, elementwise_affine=True)
        (attention_2): CrossAttention(
          (q_proj): Linear(in_features=1280, out_features=1280, bias=False)
          (k_proj): Linear(in_features=768, out_features=1280, bias=False)
          (v_proj): Linear(in_features=768, out_features=1280, bias=False)
          (out_proj): Linear(in_features=1280, out_features=1280, bias=True)
        )
        (layernorm_3): LayerNorm((1280,), eps=1e-05, elementwise_affine=True)
        (linear_geglu_1): Linear(in_features=1280, out_features=10240, bias=True)
        (linear_geglu_2): Linear(in_features=5120, out_features=1280, bias=True)
        (conv_output): Conv2d(1280, 1280, kernel_size=(1, 1), stride=(1, 1))
      )
    )
    (9): SwitchSequential(
      (0): Conv2d(1280, 1280, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
    )
    (10-11): 2 x SwitchSequential(
      (0): UNET_ResidualBlock(
        (groupnorm_feature): GroupNorm(32, 1280, eps=1e-05, affine=True)
        (conv_feature): Conv2d(1280, 1280, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (linear_time): Linear(in_features=1280, out_features=1280, bias=True)
        (groupnorm_merged): GroupNorm(32, 1280, eps=1e-05, affine=True)
        (conv_merged): Conv2d(1280, 1280, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (residual_layer): Identity()
      )
    )
  )
  (bottleneck): SwitchSequential(
    (0): UNET_ResidualBlock(
      (groupnorm_feature): GroupNorm(32, 1280, eps=1e-05, affine=True)
      (conv_feature): Conv2d(1280, 1280, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      (linear_time): Linear(in_features=1280, out_features=1280, bias=True)
      (groupnorm_merged): GroupNorm(32, 1280, eps=1e-05, affine=True)
      (conv_merged): Conv2d(1280, 1280, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      (residual_layer): Identity()
    )
    (1): UNET_AttentionBlock(
      (groupnorm): GroupNorm(32, 1280, eps=1e-06, affine=True)
      (conv_input): Conv2d(1280, 1280, kernel_size=(1, 1), stride=(1, 1))
      (layernorm_1): LayerNorm((1280,), eps=1e-05, elementwise_affine=True)
      (attention_1): SelfAttention(
        (in_proj): Linear(in_features=1280, out_features=3840, bias=False)
        (out_proj): Linear(in_features=1280, out_features=1280, bias=True)
      )
      (layernorm_2): LayerNorm((1280,), eps=1e-05, elementwise_affine=True)
      (attention_2): CrossAttention(
        (q_proj): Linear(in_features=1280, out_features=1280, bias=False)
        (k_proj): Linear(in_features=768, out_features=1280, bias=False)
        (v_proj): Linear(in_features=768, out_features=1280, bias=False)
        (out_proj): Linear(in_features=1280, out_features=1280, bias=True)
      )
      (layernorm_3): LayerNorm((1280,), eps=1e-05, elementwise_affine=True)
      (linear_geglu_1): Linear(in_features=1280, out_features=10240, bias=True)
      (linear_geglu_2): Linear(in_features=5120, out_features=1280, bias=True)
      (conv_output): Conv2d(1280, 1280, kernel_size=(1, 1), stride=(1, 1))
    )
    (2): UNET_ResidualBlock(
      (groupnorm_feature): GroupNorm(32, 1280, eps=1e-05, affine=True)
      (conv_feature): Conv2d(1280, 1280, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      (linear_time): Linear(in_features=1280, out_features=1280, bias=True)
      (groupnorm_merged): GroupNorm(32, 1280, eps=1e-05, affine=True)
      (conv_merged): Conv2d(1280, 1280, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      (residual_layer): Identity()
    )
  )
  (decoders): ModuleList(
    (0-1): 2 x SwitchSequential(
      (0): UNET_ResidualBlock(
        (groupnorm_feature): GroupNorm(32, 2560, eps=1e-05, affine=True)
        (conv_feature): Conv2d(2560, 1280, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (linear_time): Linear(in_features=1280, out_features=1280, bias=True)
        (groupnorm_merged): GroupNorm(32, 1280, eps=1e-05, affine=True)
        (conv_merged): Conv2d(1280, 1280, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (residual_layer): Conv2d(2560, 1280, kernel_size=(1, 1), stride=(1, 1))
      )
    )
    (2): SwitchSequential(
      (0): UNET_ResidualBlock(
        (groupnorm_feature): GroupNorm(32, 2560, eps=1e-05, affine=True)
        (conv_feature): Conv2d(2560, 1280, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (linear_time): Linear(in_features=1280, out_features=1280, bias=True)
        (groupnorm_merged): GroupNorm(32, 1280, eps=1e-05, affine=True)
        (conv_merged): Conv2d(1280, 1280, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (residual_layer): Conv2d(2560, 1280, kernel_size=(1, 1), stride=(1, 1))
      )
      (1): Upsample(
        (conv): Conv2d(1280, 1280, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      )
    )
    (3-4): 2 x SwitchSequential(
      (0): UNET_ResidualBlock(
        (groupnorm_feature): GroupNorm(32, 2560, eps=1e-05, affine=True)
        (conv_feature): Conv2d(2560, 1280, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (linear_time): Linear(in_features=1280, out_features=1280, bias=True)
        (groupnorm_merged): GroupNorm(32, 1280, eps=1e-05, affine=True)
        (conv_merged): Conv2d(1280, 1280, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (residual_layer): Conv2d(2560, 1280, kernel_size=(1, 1), stride=(1, 1))
      )
      (1): UNET_AttentionBlock(
        (groupnorm): GroupNorm(32, 1280, eps=1e-06, affine=True)
        (conv_input): Conv2d(1280, 1280, kernel_size=(1, 1), stride=(1, 1))
        (layernorm_1): LayerNorm((1280,), eps=1e-05, elementwise_affine=True)
        (attention_1): SelfAttention(
          (in_proj): Linear(in_features=1280, out_features=3840, bias=False)
          (out_proj): Linear(in_features=1280, out_features=1280, bias=True)
        )
        (layernorm_2): LayerNorm((1280,), eps=1e-05, elementwise_affine=True)
        (attention_2): CrossAttention(
          (q_proj): Linear(in_features=1280, out_features=1280, bias=False)
          (k_proj): Linear(in_features=768, out_features=1280, bias=False)
          (v_proj): Linear(in_features=768, out_features=1280, bias=False)
          (out_proj): Linear(in_features=1280, out_features=1280, bias=True)
        )
        (layernorm_3): LayerNorm((1280,), eps=1e-05, elementwise_affine=True)
        (linear_geglu_1): Linear(in_features=1280, out_features=10240, bias=True)
        (linear_geglu_2): Linear(in_features=5120, out_features=1280, bias=True)
        (conv_output): Conv2d(1280, 1280, kernel_size=(1, 1), stride=(1, 1))
      )
    )
    (5): SwitchSequential(
      (0): UNET_ResidualBlock(
        (groupnorm_feature): GroupNorm(32, 1920, eps=1e-05, affine=True)
        (conv_feature): Conv2d(1920, 1280, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (linear_time): Linear(in_features=1280, out_features=1280, bias=True)
        (groupnorm_merged): GroupNorm(32, 1280, eps=1e-05, affine=True)
        (conv_merged): Conv2d(1280, 1280, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (residual_layer): Conv2d(1920, 1280, kernel_size=(1, 1), stride=(1, 1))
      )
      (1): UNET_AttentionBlock(
        (groupnorm): GroupNorm(32, 1280, eps=1e-06, affine=True)
        (conv_input): Conv2d(1280, 1280, kernel_size=(1, 1), stride=(1, 1))
        (layernorm_1): LayerNorm((1280,), eps=1e-05, elementwise_affine=True)
        (attention_1): SelfAttention(
          (in_proj): Linear(in_features=1280, out_features=3840, bias=False)
          (out_proj): Linear(in_features=1280, out_features=1280, bias=True)
        )
        (layernorm_2): LayerNorm((1280,), eps=1e-05, elementwise_affine=True)
        (attention_2): CrossAttention(
          (q_proj): Linear(in_features=1280, out_features=1280, bias=False)
          (k_proj): Linear(in_features=768, out_features=1280, bias=False)
          (v_proj): Linear(in_features=768, out_features=1280, bias=False)
          (out_proj): Linear(in_features=1280, out_features=1280, bias=True)
        )
        (layernorm_3): LayerNorm((1280,), eps=1e-05, elementwise_affine=True)
        (linear_geglu_1): Linear(in_features=1280, out_features=10240, bias=True)
        (linear_geglu_2): Linear(in_features=5120, out_features=1280, bias=True)
        (conv_output): Conv2d(1280, 1280, kernel_size=(1, 1), stride=(1, 1))
      )
      (2): Upsample(
        (conv): Conv2d(1280, 1280, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      )
    )
    (6): SwitchSequential(
      (0): UNET_ResidualBlock(
        (groupnorm_feature): GroupNorm(32, 1920, eps=1e-05, affine=True)
        (conv_feature): Conv2d(1920, 640, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (linear_time): Linear(in_features=1280, out_features=640, bias=True)
        (groupnorm_merged): GroupNorm(32, 640, eps=1e-05, affine=True)
        (conv_merged): Conv2d(640, 640, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (residual_layer): Conv2d(1920, 640, kernel_size=(1, 1), stride=(1, 1))
      )
      (1): UNET_AttentionBlock(
        (groupnorm): GroupNorm(32, 640, eps=1e-06, affine=True)
        (conv_input): Conv2d(640, 640, kernel_size=(1, 1), stride=(1, 1))
        (layernorm_1): LayerNorm((640,), eps=1e-05, elementwise_affine=True)
        (attention_1): SelfAttention(
          (in_proj): Linear(in_features=640, out_features=1920, bias=False)
          (out_proj): Linear(in_features=640, out_features=640, bias=True)
        )
        (layernorm_2): LayerNorm((640,), eps=1e-05, elementwise_affine=True)
        (attention_2): CrossAttention(
          (q_proj): Linear(in_features=640, out_features=640, bias=False)
          (k_proj): Linear(in_features=768, out_features=640, bias=False)
          (v_proj): Linear(in_features=768, out_features=640, bias=False)
          (out_proj): Linear(in_features=640, out_features=640, bias=True)
        )
        (layernorm_3): LayerNorm((640,), eps=1e-05, elementwise_affine=True)
        (linear_geglu_1): Linear(in_features=640, out_features=5120, bias=True)
        (linear_geglu_2): Linear(in_features=2560, out_features=640, bias=True)
        (conv_output): Conv2d(640, 640, kernel_size=(1, 1), stride=(1, 1))
      )
    )
    (7): SwitchSequential(
      (0): UNET_ResidualBlock(
        (groupnorm_feature): GroupNorm(32, 1280, eps=1e-05, affine=True)
        (conv_feature): Conv2d(1280, 640, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (linear_time): Linear(in_features=1280, out_features=640, bias=True)
        (groupnorm_merged): GroupNorm(32, 640, eps=1e-05, affine=True)
        (conv_merged): Conv2d(640, 640, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (residual_layer): Conv2d(1280, 640, kernel_size=(1, 1), stride=(1, 1))
      )
      (1): UNET_AttentionBlock(
        (groupnorm): GroupNorm(32, 640, eps=1e-06, affine=True)
        (conv_input): Conv2d(640, 640, kernel_size=(1, 1), stride=(1, 1))
        (layernorm_1): LayerNorm((640,), eps=1e-05, elementwise_affine=True)
        (attention_1): SelfAttention(
          (in_proj): Linear(in_features=640, out_features=1920, bias=False)
          (out_proj): Linear(in_features=640, out_features=640, bias=True)
        )
        (layernorm_2): LayerNorm((640,), eps=1e-05, elementwise_affine=True)
        (attention_2): CrossAttention(
          (q_proj): Linear(in_features=640, out_features=640, bias=False)
          (k_proj): Linear(in_features=768, out_features=640, bias=False)
          (v_proj): Linear(in_features=768, out_features=640, bias=False)
          (out_proj): Linear(in_features=640, out_features=640, bias=True)
        )
        (layernorm_3): LayerNorm((640,), eps=1e-05, elementwise_affine=True)
        (linear_geglu_1): Linear(in_features=640, out_features=5120, bias=True)
        (linear_geglu_2): Linear(in_features=2560, out_features=640, bias=True)
        (conv_output): Conv2d(640, 640, kernel_size=(1, 1), stride=(1, 1))
      )
    )
    (8): SwitchSequential(
      (0): UNET_ResidualBlock(
        (groupnorm_feature): GroupNorm(32, 960, eps=1e-05, affine=True)
        (conv_feature): Conv2d(960, 640, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (linear_time): Linear(in_features=1280, out_features=640, bias=True)
        (groupnorm_merged): GroupNorm(32, 640, eps=1e-05, affine=True)
        (conv_merged): Conv2d(640, 640, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (residual_layer): Conv2d(960, 640, kernel_size=(1, 1), stride=(1, 1))
      )
      (1): UNET_AttentionBlock(
        (groupnorm): GroupNorm(32, 640, eps=1e-06, affine=True)
        (conv_input): Conv2d(640, 640, kernel_size=(1, 1), stride=(1, 1))
        (layernorm_1): LayerNorm((640,), eps=1e-05, elementwise_affine=True)
        (attention_1): SelfAttention(
          (in_proj): Linear(in_features=640, out_features=1920, bias=False)
          (out_proj): Linear(in_features=640, out_features=640, bias=True)
        )
        (layernorm_2): LayerNorm((640,), eps=1e-05, elementwise_affine=True)
        (attention_2): CrossAttention(
          (q_proj): Linear(in_features=640, out_features=640, bias=False)
          (k_proj): Linear(in_features=768, out_features=640, bias=False)
          (v_proj): Linear(in_features=768, out_features=640, bias=False)
          (out_proj): Linear(in_features=640, out_features=640, bias=True)
        )
        (layernorm_3): LayerNorm((640,), eps=1e-05, elementwise_affine=True)
        (linear_geglu_1): Linear(in_features=640, out_features=5120, bias=True)
        (linear_geglu_2): Linear(in_features=2560, out_features=640, bias=True)
        (conv_output): Conv2d(640, 640, kernel_size=(1, 1), stride=(1, 1))
      )
      (2): Upsample(
        (conv): Conv2d(640, 640, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      )
    )
    (9): SwitchSequential(
      (0): UNET_ResidualBlock(
        (groupnorm_feature): GroupNorm(32, 960, eps=1e-05, affine=True)
        (conv_feature): Conv2d(960, 320, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (linear_time): Linear(in_features=1280, out_features=320, bias=True)
        (groupnorm_merged): GroupNorm(32, 320, eps=1e-05, affine=True)
        (conv_merged): Conv2d(320, 320, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (residual_layer): Conv2d(960, 320, kernel_size=(1, 1), stride=(1, 1))
      )
      (1): UNET_AttentionBlock(
        (groupnorm): GroupNorm(32, 320, eps=1e-06, affine=True)
        (conv_input): Conv2d(320, 320, kernel_size=(1, 1), stride=(1, 1))
        (layernorm_1): LayerNorm((320,), eps=1e-05, elementwise_affine=True)
        (attention_1): SelfAttention(
          (in_proj): Linear(in_features=320, out_features=960, bias=False)
          (out_proj): Linear(in_features=320, out_features=320, bias=True)
        )
        (layernorm_2): LayerNorm((320,), eps=1e-05, elementwise_affine=True)
        (attention_2): CrossAttention(
          (q_proj): Linear(in_features=320, out_features=320, bias=False)
          (k_proj): Linear(in_features=768, out_features=320, bias=False)
          (v_proj): Linear(in_features=768, out_features=320, bias=False)
          (out_proj): Linear(in_features=320, out_features=320, bias=True)
        )
        (layernorm_3): LayerNorm((320,), eps=1e-05, elementwise_affine=True)
        (linear_geglu_1): Linear(in_features=320, out_features=2560, bias=True)
        (linear_geglu_2): Linear(in_features=1280, out_features=320, bias=True)
        (conv_output): Conv2d(320, 320, kernel_size=(1, 1), stride=(1, 1))
      )
    )
    (10-11): 2 x SwitchSequential(
      (0): UNET_ResidualBlock(
        (groupnorm_feature): GroupNorm(32, 640, eps=1e-05, affine=True)
        (conv_feature): Conv2d(640, 320, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (linear_time): Linear(in_features=1280, out_features=320, bias=True)
        (groupnorm_merged): GroupNorm(32, 320, eps=1e-05, affine=True)
        (conv_merged): Conv2d(320, 320, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (residual_layer): Conv2d(640, 320, kernel_size=(1, 1), stride=(1, 1))
      )
      (1): UNET_AttentionBlock(
        (groupnorm): GroupNorm(32, 320, eps=1e-06, affine=True)
        (conv_input): Conv2d(320, 320, kernel_size=(1, 1), stride=(1, 1))
        (layernorm_1): LayerNorm((320,), eps=1e-05, elementwise_affine=True)
        (attention_1): SelfAttention(
          (in_proj): Linear(in_features=320, out_features=960, bias=False)
          (out_proj): Linear(in_features=320, out_features=320, bias=True)
        )
        (layernorm_2): LayerNorm((320,), eps=1e-05, elementwise_affine=True)
        (attention_2): CrossAttention(
          (q_proj): Linear(in_features=320, out_features=320, bias=False)
          (k_proj): Linear(in_features=768, out_features=320, bias=False)
          (v_proj): Linear(in_features=768, out_features=320, bias=False)
          (out_proj): Linear(in_features=320, out_features=320, bias=True)
        )
        (layernorm_3): LayerNorm((320,), eps=1e-05, elementwise_affine=True)
        (linear_geglu_1): Linear(in_features=320, out_features=2560, bias=True)
        (linear_geglu_2): Linear(in_features=1280, out_features=320, bias=True)
        (conv_output): Conv2d(320, 320, kernel_size=(1, 1), stride=(1, 1))
      )
    )
  )
)
```

</details>

<br/>