import PIL
import requests
import torch
from diffusers import (
    StableDiffusionInstructPix2PixPipeline,
    EulerAncestralDiscreteScheduler,
)

model_id = "timbrooks/instruct-pix2pix"
pipe = StableDiffusionInstructPix2PixPipeline.from_pretrained(
    model_id, torch_dtype=torch.float32, safety_checker=None
)
# pipe.to("cuda")
pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config)

# url = "https://raw.githubusercontent.com/timothybrooks/instruct-pix2pix/main/imgs/example.jpg"
# def download_image(url):
#     image = PIL.Image.open(requests.get(url, stream=True).raw)
#     image = PIL.ImageOps.exif_transpose(image)
#     image = image.convert("RGB")
#     return image

image_path = "../../training/Dog_Breeds.jpg"

# Open the image file using PIL
image = PIL.Image.open(image_path)
image = image.resize([512, 512])
print(image)

prompt = "turn him into cyborg"
images = pipe(
    prompt,
    image=image,
    num_inference_steps=50,
    image_guidance_scale=1.5,
).images

images[0].save("output.jpg")
