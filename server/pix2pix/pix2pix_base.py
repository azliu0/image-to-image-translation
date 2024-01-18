import PIL
import requests
import torch
from diffusers import (
    StableDiffusionInstructPix2PixPipeline,
    EulerAncestralDiscreteScheduler,
)
from server.config import IMAGE_HEIGHT, IMAGE_WIDTH

model_id = "timbrooks/instruct-pix2pix"
pipe = StableDiffusionInstructPix2PixPipeline.from_pretrained(
    model_id, torch_dtype=torch.float32, safety_checker=None
)
pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config)


def inference_pix2pix_base(opts, image):
    image.resize([IMAGE_HEIGHT, IMAGE_WIDTH])
    images = pipe(
        opts["prompt"],
        image,
        num_inference_steps=opts["num_inference_steps"],
        image_guidance_scale=1.0 / opts["temperature"],
    )
    images[0].save("output.png")
