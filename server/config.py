import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv(os.path.dirname(__file__) / Path("../.env"))

# server config
DEBUG = os.environ.get("DEBUG", True)
FLASK_RUN_PORT = 3000
FRONTEND_URL = os.environ.get("FRONTEND_URL", "http://localhost:5173")
BACKEND_URL = os.environ.get("BACKEND_URL", "http://localhost:3000")
ALLOWED_DOMAINS = [FRONTEND_URL, "https://www.image.azliu.cc"]
AWS_ACCESS_KEY_ID = os.environ.get("AWS_ACCESS_KEY_ID")
AWS_SECRET_ACCESS_KEY = os.environ.get("AWS_SECRET_ACCESS_KEY")
AWS_BUCKET_NAME = os.environ.get("AWS_BUCKET_NAME")
USE_REMOTE = os.environ.get("USE_REMOTE", True)

# prod config
MODAL_TOKEN_ID = os.environ.get("MODAL_TOKEN_ID", "")
MODAL_TOKEN_SECRET = os.environ.get("MODAL_TOKEN_SECRET", "")
ENV = os.environ.get("ENV", "development")

# model config
VOCAB_SIZE = 49408
MAX_SEQ_LENGTH = 77
TOKEN_EMBEDDING_SIZE = 768
TIME_EMBEDDING_SIZE = 320
IMAGE_HEIGHT = 512
IMAGE_WIDTH = 512
OPTS = [
    {"key": "model", "int": False, "float": False},
    {"key": "prompt", "int": False, "float": False},
    {"key": "inferenceSteps", "int": True, "float": False},
    {"key": "temperature", "int": False, "float": True},
    {"key": "CFG", "int": True, "float": False},
    {"key": "negativePrompt", "int": False, "float": False},
]
INFERENCE_URLS = {
    "pix2pix-base": "https://azliu0--image-to-image-translation-inference-pix2pix-base-modal.modal.run",
    "pix2pix-full-no-cfg-no-ddim": "https://azliu0--image-to-image-translation-inference-pix2pix-ful-cf33b1.modal.run",
}
MODELS = list(INFERENCE_URLS.keys())
TARGET_IMAGE_PATH = "output.png"
