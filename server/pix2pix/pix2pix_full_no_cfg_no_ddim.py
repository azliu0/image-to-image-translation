import numpy as np
import server.pix2pix.modules.model_loader as model_loader
from server.pix2pix.modules.pipeline.pipeline import generate
from server.utils.s3 import s3_to_pil
from PIL import Image
from transformers import CLIPTokenizer

import modal
from server.utils.s3 import s3_to_pil, generate_image_key, pil_to_s3
from server.pix2pix.remote import modal_app, modal_image, modal_volume

import torch
from pathlib import Path

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print("running on device:", DEVICE)

data_path = Path(__file__).parent.parent.parent / "data"
print("data_path", data_path)
tokenizer = CLIPTokenizer(
    str(data_path / "tokenizer_vocab.json"), str(data_path / "tokenizer_merges.txt")
)

MODEL_PATH = str(data_path / "model.ckpt")

class ModelLoader:
    def load(self):
        self.models = model_loader.preload_models_from_standard_weights(MODEL_PATH, DEVICE)

loader = ModelLoader()

# TODO: make this configurable when add ddim
sampler = "ddpm"


def inference_pix2pix_full_no_cfg_no_ddim(opts, image):
    if not loader.models:
        loader.load()

    output_image = generate(
        prompt=opts["prompt"],
        uncond_prompt=opts["negativePrompt"],
        input_image=image,
        strength=opts["temperature"],
        do_cfg=True,
        cfg_scale=opts["CFG"],
        sampler_name=sampler,
        n_inference_steps=opts["inferenceSteps"],
        models=loader.models,
        device=DEVICE,
        tokenizer=tokenizer,
    )

    output_image = output_image.astype(np.uint8)
    output_image = Image.fromarray(output_image)
    output_image.save("output.png")
    return output_image


@modal_app.function(
        image=modal_image, 
        volumes={"/root/data": modal_volume}, 
        gpu="A100", 
        secrets=[modal.Secret.from_dotenv()],
        container_idle_timeout=60 * 10 # 10 minutes
)
@modal.web_endpoint(method="POST")
def inference_pix2pix_full_no_cfg_no_ddim_modal(opts: dict, image_s3_key: str):
    image = s3_to_pil(image_s3_key)
    image = inference_pix2pix_full_no_cfg_no_ddim(opts, image)
    out_key = generate_image_key("out")
    return {"image_s3_key": pil_to_s3(image, out_key)}
