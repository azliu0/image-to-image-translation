import numpy as np
import server.pix2pix.modules.model_loader as model_loader
from server.pix2pix.modules.pipeline.pipeline import generate
from PIL import Image
from transformers import CLIPTokenizer
import boto3
from server.config import AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, AWS_BUCKET_NAME

DEVICE = "cpu"

tokenizer = CLIPTokenizer(
    "data/tokenizer_vocab.json", merges_file="data/tokenizer_merges.txt"
)

MODEL_PATH = "data/model.ckpt"
MODEL_REMOTE_PATH = "v1-5-pruned-emaonly.ckpt"

# Download the model weights from S3
s3 = boto3.client(
    "s3",
    aws_access_key_id=AWS_ACCESS_KEY_ID,
    aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
)
s3.download_file(AWS_BUCKET_NAME, MODEL_REMOTE_PATH, MODEL_PATH)

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
