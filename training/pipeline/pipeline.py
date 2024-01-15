import sys

sys.path.append(".")
sys.path.append("..")

import torch
import numpy as np
import tqdm as tqdm
from diffusion.ddpm import DDPMSampler
from config import MAX_SEQ_LENGTH, TIME_EMBEDDING_SIZE

WIDTH = 512
HEIGHT = 512
LATENT_WIDTH = WIDTH // 8
LATENT_HEIGHT = HEIGHT // 8


def generate(
    prompt,
    uncond_prompt,
    input_image,
    strength=0.8,
    do_cfg=True,
    cfg_scale=7.5,
    sampler_name="ddpm",
    n_inference_steps=50,
    models={},
    seed=None,
    device=None,
    tokenizer=None,
):
    with torch.no_grad():
        if not (0 < strength <= 1):
            raise ValueError("strength must be between 0 and 1")

        generator = torch.Generator(device=device)
        if seed is None:
            generate.seed()
        else:
            generator.manual_seed(seed)

        clip = models["clip"]
        clip = clip.to(device)

        if do_cfg:
            cond_tokens = tokenizer.batch_encode_plus(
                [prompt], padding="max_length", max_length=MAX_SEQ_LENGTH
            ).input_ids
            cond_tokens = torch.tensor(cond_tokens, dtype=torch.long, device=device)
            cond_context = clip(cond_tokens)

            uncond_tokens = tokenizer.batch_encode_plus(
                [uncond_prompt], padding="max_length", max_length=MAX_SEQ_LENGTH
            ).input_ids
            uncond_tokens = torch.tensor(uncond_tokens, dtype=torch.long, device=device)
            uncond_context = clip(uncond_tokens)

            # (2, MAX_SEQ_LENGTH, TOKEN_EMBEDDING_SIZE)
            context = torch.cat([cond_context, uncond_context])
        else:
            tokens = tokenizer.batch_encode_plus(
                [prompt], padding="max_length", max_length=MAX_SEQ_LENGTH
            ).input_ids
            tokens = torch.tensor(tokens, dtype=torch.long, device=device)

            # (1, MAX_SEQ_LENGTH, TOKEN_EMBEDDING_SIZE)
            context = clip(tokens)

        if sampler_name == "ddpm":
            sampler = DDPMSampler(generator)
            sampler.set_inference_timesteps(n_inference_steps)
        else:
            raise ValueError("Unknown Sampler")

        latents_shape = (1, 4, LATENT_HEIGHT, LATENT_WIDTH)

        if input_image:
            # start with a reference image (image-to-image)
            encoder = models["encoder"]
            encoder = encoder.to(device)

            input_image_tensor = input_image.resize([WIDTH, HEIGHT])
            input_image_tensor = np.array(input_image_tensor)
            # (h,w,c)
            input_image_tensor = torch.tensor(input_image_tensor, dtype=torch.float32)
            # rescale image pixel values from [0,255] to [-1,1]
            input_image_tensor = rescale(input_image_tensor, (0, 255), (-1, 1))
            # add batch dimension: (b,h,w,c)
            input_image_tensor = input_image_tensor.unsqueeze(0)
            # (b,c,h,w)
            input_image_tensor = input_image_tensor.permute(0, 3, 1, 2)
            # gaussian noise for reparamaterization input to encoder
            encoder_noise = torch.randn(
                latents_shape, generator=generator, device=device
            )
            latents = encoder(input_image_tensor, encoder_noise)

            # high strength = high noise = more variance
            sampler.set_strength(strength=strength)
            latents = sampler.add_noise(latents, sampler.timesteps[0])
        else:
            # start with random noise (text-to-image)
            latents = torch.randn(latents_shape, generator=generator, device=device)

        diffusion = models["diffusion"]
        diffusion.to(device)

        timesteps = tqdm.tqdm(sampler.timesteps)
        for i, timestep in enumerate(timesteps):
            time_embedding = get_time_embedding(timestep).to(device)
            # (4,h/8,w/8)
            model_input = latents

            if do_cfg:
                model_input = model_input.repeat(2, 1, 1, 1)

            model_output = diffusion(model_input, context, time_embedding)

            if do_cfg:
                output_cond, output_uncond = model_output.chunk(2)
                model_output = cfg_scale * (output_cond - output_uncond) + output_uncond

            latents = sampler.step(timestep, latents, model_output)

        decoder = models["decoder"]

        images = decoder(latents)

        images = rescale(images, (-1, 1), (0, 255), clamp=True)
        images = images.permute(0, 2, 3, 1)
        images = images.numpy()
        return images[0]


def rescale(x, old_range, new_range, clamp=False):
    old_min, old_max = old_range
    new_min, new_max = new_range
    x -= old_min
    x *= (new_max - new_min) / (old_max - old_min)
    x += new_min

    if clamp:
        x = x.clamp(new_min, new_max)

    return x


def get_time_embedding(timestep):
    freqs = torch.pow(
        10000,
        -torch.arange(start=0, end=(TIME_EMBEDDING_SIZE // 2), dtype=torch.float32)
        / (TIME_EMBEDDING_SIZE // 2),
    )
    x = torch.tensor([timestep], dtype=torch.float32)[:, None] * freqs[None]
    return torch.cat([torch.cos(x), torch.sin(x)], dim=-1)
