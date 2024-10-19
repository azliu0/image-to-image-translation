import torch
import modal
from PIL import Image
from typing import cast
from diffusers.pipelines.stable_diffusion.pipeline_output import (
    StableDiffusionPipelineOutput,
)
from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion_instruct_pix2pix import (
    StableDiffusionInstructPix2PixPipeline,
)
from diffusers.schedulers.scheduling_euler_ancestral_discrete import (
    EulerAncestralDiscreteScheduler,
)
from server.config import IMAGE_HEIGHT, IMAGE_WIDTH
from server.utils.s3 import s3_to_pil, generate_image_key, pil_to_s3
from server.pix2pix.remote import modal_app, modal_image, modal_volume


def inference_pix2pix_base(opts: dict, image: Image.Image):
    model_id = "timbrooks/instruct-pix2pix"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    pipe = StableDiffusionInstructPix2PixPipeline.from_pretrained(
        model_id, torch_dtype=torch.float32, safety_checker=None
    ).to(device)
    pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config)

    image = image.resize((IMAGE_WIDTH, IMAGE_HEIGHT))
    image = cast(
        StableDiffusionPipelineOutput,
        pipe.__call__(
            prompt=opts["prompt"],
            image=image,
            num_inference_steps=opts["inferenceSteps"],
            guidance_scale=1.0 / opts["temperature"],
        ),
    ).images[0]
    image.save("output.png")
    return image


@modal_app.function(
        image=modal_image, 
        volumes={"/root/data": modal_volume}, 
        gpu="A100", 
        secrets=[modal.Secret.from_dotenv()],
        container_idle_timeout=60 * 10 # 10 minutes
)
@modal.web_endpoint(method="POST")
def inference_pix2pix_base_modal(opts: dict, image_s3_key: str):
    image = s3_to_pil(image_s3_key)
    image = inference_pix2pix_base(opts, image)
    out_key = generate_image_key("out")
    return {"image_s3_key": pil_to_s3(image, out_key)}
