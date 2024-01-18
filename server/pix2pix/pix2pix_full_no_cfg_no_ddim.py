import numpy as np
import server.pix2pix.modules.model_loader as model_loader
from server.pix2pix.modules.pipeline.pipeline import generate
from PIL import Image
from transformers import CLIPTokenizer
import torch

DEVICE = "cpu"

tokenizer = CLIPTokenizer(
    "data/tokenizer_vocab.json", merges_file="data/tokenizer_merges.txt"
)

model_file = "data/v1-5-pruned-emaonly.ckpt"

models = model_loader.preload_models_from_standard_weights(model_file, DEVICE)

strength = 0.5

sampler = "ddpm"


def inference_pix2pix_full_no_cfg_no_ddim(opts, image):
    output_image = generate(
        prompt=opts["prompt"],
        uncond_prompt=opts["negativePrompt"],
        input_image=image,
        strength=opts["temperature"],
        do_cfg=True,
        cfg_scale=opts["CFG"],
        sampler_name=sampler,
        n_inference_steps=opts["inferenceSteps"],
        models=models,
        device=DEVICE,
        tokenizer=tokenizer,
    )

    output_image = output_image.astype(np.uint8)
    output_image = Image.fromarray(output_image)
    output_image.save("output.png")
