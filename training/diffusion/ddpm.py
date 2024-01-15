import torch
import numpy as np


class DDPMSampler:
    def __init__(
        self, generator, num_training_steps=1000, beta_start=0.00085, beta_end=0.0120
    ):
        # scaled beta schedule
        self.betas = (
            torch.linspace(
                beta_start**0.5,
                beta_end**0.5,
                num_training_steps,
                dtype=torch.float32,
            )
            ** 2
        )
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, 0)
        self.one = torch.tensor(1.0)

        self.generator = generator
        self.num_training_steps = num_training_steps
        self.timesteps = torch.from_numpy(np.arange(0, num_training_steps)[::-1].copy())

    def set_inference_timesteps(self, num_inference_steps=50):
        self.num_inference_steps = num_inference_steps
        step_ratio = self.num_training_steps // self.num_inference_steps
        timesteps = (
            (np.arange(0, num_inference_steps) * step_ratio)
            .round()[::-1]
            .copy()
            .astype(np.int64)
        )
        self.timesteps = torch.from_numpy(timesteps)

    def _get_previous_timestep(self, timestep):
        prev_t = timestep - (self.num_training_steps // self.num_inference_steps)
        return prev_t

    def _get_variance(self, timestep):
        prev_t = self._get_previous_timestep(timestep)

        alpha_prod_t = self.alpha_cumprod[timestep]
        alpha_prod_t_prev = self.alpha_cumprod[prev_t] if prev_t >= 0 else self.one
        current_alpha_t = alpha_prod_t / alpha_prod_t_prev
        current_beta_t = 1 - current_alpha_t

        variance = (
            (1 - alpha_prod_t_prev) / (1 - alpha_prod_t) * current_beta_t
        ).clamp(min=1e-20)

        return variance

    def set_strength(self, strength=1):
        start_step = self.num_inference_steps - int(self.num_inference_steps * strength)
        self.timesteps = self.timesteps[start_step:]
        self.start_step = start_step

    def step(self, timestep, latents, model_output):
        # spam formulas from ddpm
        prev_t = self._get_previous_timestep(timestep)

        alpha_prod_t = self.alpha_cumprod[timestep]
        alpha_prod_t_prev = self.alpha_cumprod[prev_t] if prev_t >= 0 else self.one
        beta_prod_t = 1 - alpha_prod_t
        beta_prod_t_prev = 1 - alpha_prod_t_prev
        current_alpha_t = alpha_prod_t / alpha_prod_t_prev
        current_beta_t = 1 - current_alpha_t

        pred_x0 = (latents - beta_prod_t**0.5 * model_output) / alpha_prod_t**0.5
        pred_x0_coeff = (alpha_prod_t_prev**0.5 * current_beta_t) / beta_prod_t
        xt_coeff = current_alpha_t**0.5 * beta_prod_t_prev / beta_prod_t

        pred_prev_sample = pred_x0_coeff * pred_x0 + xt_coeff * latents

        variance = 0
        if timestep > 0:
            device = model_output.device
            noise = torch.randn(
                model_output.shape,
                generator=self.generator,
                device=device,
                dtype=model_output.dtype,
            )
            variance = self._get_variance(timestep) ** 0.5 * noise

        pred_prev_sample = pred_prev_sample + variance

    def add_noise(self, original_samples, timestep):
        alpha_cumprod = self.alphas_cumprod.to(
            device=original_samples.device, dtype=original_samples.dtype
        )
        timestep = timestep.to(original_samples.device)

        sqrt_alpha_prod = alpha_cumprod[timestep] ** 0.5
        sqrt_alpha_prod = sqrt_alpha_prod.flatten()
        while len(sqrt_alpha_prod.shape) < len(original_samples.shape):
            sqrt_alpha_prod = sqrt_alpha_prod.unsqueeze(-1)

        sqrt_one_minus_alpha_prod = (1 - alpha_cumprod[timestep]) ** 0.5
        sqrt_one_minus_alpha_prod = sqrt_one_minus_alpha_prod.flatten()
        while len(sqrt_one_minus_alpha_prod.shape) < len(original_samples.shape):
            sqrt_one_minus_alpha_prod = sqrt_one_minus_alpha_prod.unsqueeze(-1)

        noise = torch.randn(
            original_samples.shape,
            generator=self.generator,
            device=original_samples.device,
            dtype=original_samples.dtype,
        )
        noisy_samples = (sqrt_alpha_prod * original_samples) + (
            sqrt_one_minus_alpha_prod
        ) * noise
        return noisy_samples
