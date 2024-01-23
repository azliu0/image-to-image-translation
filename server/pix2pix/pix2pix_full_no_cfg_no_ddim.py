import numpy as np
import server.pix2pix.modules.model_loader as model_loader
from server.pix2pix.modules.pipeline.pipeline import generate
from PIL import Image
from transformers import CLIPTokenizer
import boto3
from server.utils.s3 import s3_to_pil, pil_to_s3
from server.config import IMAGE_HEIGHT, IMAGE_WIDTH

DEVICE = "cpu"

tokenizer = CLIPTokenizer(
    "data/tokenizer_vocab.json", merges_file="data/tokenizer_merges.txt"
)

MODEL_PATH = "data/model.ckpt"

models = model_loader.preload_models_from_standard_weights(MODEL_PATH, DEVICE)

# TODO: make this configurable when add ddim
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


def inference_pix2pix_full_no_cfg_no_ddim_modelbit(opts):
    try:
        image = s3_to_pil()
    except Exception as e:
        raise Exception(f"{e}")

    image = image.resize([IMAGE_WIDTH, IMAGE_HEIGHT])
    print(image)
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

    try:
        pil_to_s3(output_image)
    except Exception as e:
        raise Exception(f"{e}")
