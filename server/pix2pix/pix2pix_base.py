import PIL
import requests
import torch
from diffusers import (
    StableDiffusionInstructPix2PixPipeline,
    EulerAncestralDiscreteScheduler,
)
from server.config import IMAGE_HEIGHT, IMAGE_WIDTH
from server.utils.s3 import s3_to_pil, pil_to_s3

model_id = "timbrooks/instruct-pix2pix"
pipe = StableDiffusionInstructPix2PixPipeline.from_pretrained(
    model_id, torch_dtype=torch.float32, safety_checker=None
)
pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config)


def inference_pix2pix_base(opts, image):
    image = image.resize([IMAGE_WIDTH, IMAGE_HEIGHT])
    image = pipe(
        opts["prompt"],
        image,
        num_inference_steps=opts["inferenceSteps"],
        image_guidance_scale=1.0 / opts["temperature"],
    ).images[0]
    image.save("output.png")


def inference_pix2pix_base_modelbit(opts):
    try:
        image = s3_to_pil()
    except Exception as e:
        raise Exception(f"{e}")

    image = image.resize([IMAGE_WIDTH, IMAGE_HEIGHT])
    print(image)
    output_image = pipe(
        opts["prompt"],
        image,
        num_inference_steps=opts["inferenceSteps"],
        image_guidance_scale=1.0 / opts["temperature"],
    ).images[0]

    try:
        pil_to_s3(output_image)
    except Exception as e:
        raise Exception(f"{e}")
