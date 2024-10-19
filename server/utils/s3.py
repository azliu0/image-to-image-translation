import boto3
from PIL import Image
from io import BytesIO
from server.config import (
    AWS_ACCESS_KEY_ID,
    AWS_SECRET_ACCESS_KEY,
    AWS_BUCKET_NAME,
)
import uuid

MODEL_REMOTE_PATH = "v1-5-pruned-emaonly.ckpt"

s3 = boto3.client(
    "s3",
    aws_access_key_id=AWS_ACCESS_KEY_ID,
    aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
)


def generate_image_key(pfx: str) -> str:
    return f"{pfx}-{uuid.uuid4()}.png"


def pil_to_s3(image: Image.Image, key: str) -> str:
    buffer = BytesIO()
    image.save(buffer, format="png")
    buffer.seek(0)
    print("uploading image to s3...")
    s3.upload_fileobj(buffer, AWS_BUCKET_NAME, key)
    print("image uploaded to s3 successfully.")
    buffer.close()
    return key


def s3_to_pil(key: str) -> Image.Image:
    print("downloading image from s3...")
    response = s3.get_object(Bucket=AWS_BUCKET_NAME, Key=key)
    print("image downloaded from s3 successfully.")
    image_data = response["Body"].read()
    image_bytes = BytesIO(image_data)
    image = Image.open(image_bytes)
    if image.mode == "RGBA":
        image = image.convert("RGB")
    return image
