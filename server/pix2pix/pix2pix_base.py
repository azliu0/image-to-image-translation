import torch
from typing import cast
from diffusers.pipelines.stable_diffusion.pipeline_output import StableDiffusionPipelineOutput
from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion_instruct_pix2pix import StableDiffusionInstructPix2PixPipeline
from diffusers.schedulers.scheduling_euler_ancestral_discrete import EulerAncestralDiscreteScheduler
from server.config import IMAGE_HEIGHT, IMAGE_WIDTH

model_id = "timbrooks/instruct-pix2pix"
pipe = StableDiffusionInstructPix2PixPipeline.from_pretrained(
    model_id, torch_dtype=torch.float32, safety_checker=None
)
pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config)


def inference_pix2pix_base(opts, image):
    image = image.resize([IMAGE_WIDTH, IMAGE_HEIGHT])
    out = cast(StableDiffusionPipelineOutput, pipe.__call__(
        prompt=opts["prompt"],
        image=image,
        num_inference_steps=opts["inferenceSteps"],
        guidance_scale=1.0 / opts["temperature"],
    ))
    image = out.images[0]
    image.save("output.png")
