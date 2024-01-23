import boto3
from PIL import Image
from server.config import (
    AWS_ACCESS_KEY_ID,
    AWS_SECRET_ACCESS_KEY,
    AWS_BUCKET_NAME,
    TARGET_IMAGE_PATH,
)

MODEL_REMOTE_PATH = "v1-5-pruned-emaonly.ckpt"

s3 = boto3.client(
    "s3",
    aws_access_key_id=AWS_ACCESS_KEY_ID,
    aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
)


def pil_to_s3(image):
    # image: pil image
    image.save(TARGET_IMAGE_PATH)
    s3.upload_file(TARGET_IMAGE_PATH, AWS_BUCKET_NAME, TARGET_IMAGE_PATH)


def s3_to_pil():
    s3.download_file(AWS_BUCKET_NAME, TARGET_IMAGE_PATH, TARGET_IMAGE_PATH)
    return Image.open(TARGET_IMAGE_PATH)
