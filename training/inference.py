import numpy as np
import model_loader
import pipeline.pipeline
from PIL import Image
from transformers import CLIPTokenizer
import torch

DEVICE = "cpu"

tokenizer = CLIPTokenizer(
    "../data/tokenizer_vocab.json", merges_file="../data/tokenizer_merges.txt"
)

model_file = "../data/v1-5-pruned-emaonly.ckpt"

models = model_loader.preload_models_from_standard_weights(model_file, DEVICE)


prompt = "Dog with hat"
uncond_prompt = ""  # Also known as negative prompt

image_path = "Dog_Breeds.jpg"
input_image = Image.open(image_path)
# Higher values means more noise will be added to the input image, so the result will further from the input image.
# Lower values means less noise is added to the input image, so output will be closer to the input image.
strength = 0.5

## SAMPLER

sampler = "ddpm"
num_inference_steps = 50

do_cfg = True
cfg_scale = 8  # min: 1, max: 14

output_image = pipeline.pipeline.generate(
    prompt=prompt,
    uncond_prompt=uncond_prompt,
    input_image=input_image,
    strength=strength,
    do_cfg=do_cfg,
    cfg_scale=cfg_scale,
    sampler_name=sampler,
    n_inference_steps=num_inference_steps,
    models=models,
    device=DEVICE,
    tokenizer=tokenizer,
)

# Combine the input image and the output image into a single image.
output_image = output_image.astype(np.uint8)
output_image = Image.fromarray(output_image)
output_image.save("output_image.jpg")
