# server modules
from flask import request, jsonify, send_file
from apiflask import APIBlueprint
from PIL import Image
from io import BytesIO
from server.utils.ModelNotFoundException import ModelNotFoundException
from server.utils.s3 import pil_to_s3, generate_image_key, s3_to_pil
import requests

# inference modules
from server.config import (
    OPTS,
    MODELS,
    INFERENCE_URLS,
    IMAGE_HEIGHT,
    IMAGE_WIDTH,
    USE_REMOTE,
)
from server.pix2pix import (
    inference_pix2pix_base,
    inference_pix2pix_full_no_cfg_no_ddim,
)


api = APIBlueprint("api", __name__, url_prefix="/api")


def get_pil_image(file_storage_obj):
    bytes = file_storage_obj.read()
    stream = BytesIO(bytes)
    return Image.open(stream)


def save_pil(pil):
    pil.save("output.png")


def validate_imgtype(imgtype):
    if imgtype not in ["image/png", "image/jpeg"]:
        raise Exception("Image must be of type png or jpeg")


def validate_model(model):
    if model not in MODELS:
        raise ModelNotFoundException()


def make_request(url: str, opts: dict, image_s3_key: str):
    params = {"image_s3_key": image_s3_key}
    response = requests.post(url, params=params, json=opts)
    body = response.json()
    if "image_s3_key" not in body:
        raise Exception(f"Unexpected response: {body}")
    return body["image_s3_key"]


def do_inference_remote(opts: dict, image: Image.Image):
    in_key = generate_image_key("in")
    image_s3_key = pil_to_s3(image, in_key)
    match opts["model"]:
        case "pix2pix-base":
            out_key = make_request(INFERENCE_URLS["pix2pix-base"], opts, image_s3_key)
        case "pix2pix-full-no-cfg-no-ddim":
            out_key = make_request(
                INFERENCE_URLS["pix2pix-full-no-cfg-no-ddim"], opts, image_s3_key
            )
    image = s3_to_pil(out_key)
    image.save("output.png")


def do_inference_local(opts: dict, image: Image.Image):
    match opts["model"]:
        case "pix2pix-base":
            inference_pix2pix_base(opts, image)
        case "pix2pix-full-no-cfg-no-ddim":
            inference_pix2pix_full_no_cfg_no_ddim(opts, image)


def do_inference(opts: dict, image: Image.Image):
    if USE_REMOTE:
        return do_inference_remote(opts, image)
    else:
        return do_inference_local(opts, image)


@api.route("/")
def hello_world():
    return "hello world"


@api.route("/inference", methods=["POST"])
def inference():
    # ensure that file exists
    if "files[]" not in request.files:
        resp = jsonify({"message": "No file in the request!"})
        resp.status_code = 400
        return resp

    # file exists, so we can extract input image
    image = request.files.getlist("files[]")[0]

    # validate image type
    try:
        validate_imgtype(image.mimetype)
    except Exception as e:
        resp = jsonify({"message": f"Error: {e}"})
        resp.status_code = 400
        return resp

    # transform to pil image
    image = get_pil_image(image)
    image = image.resize((IMAGE_WIDTH, IMAGE_HEIGHT))
    print(image, flush=True)

    # convert form to dictionary
    form = request.form.to_dict()

    # ensure all option keys are in form
    for opt in OPTS:
        if opt["key"] not in form:
            resp = jsonify({"message": "Model options not specified correctly"})
            resp.status_code = 400
            return resp

    # fill out model options
    opts = {}
    for opt in OPTS:
        opts[opt["key"]] = form[opt["key"]]
        if opt["int"]:
            opts[opt["key"]] = int(opts[opt["key"]])
        elif opt["float"]:
            opts[opt["key"]] = float(opts[opt["key"]])

    print(opts, flush=True)
    # validate model type
    try:
        validate_model(opts["model"])
    except ModelNotFoundException:
        resp = jsonify({"message": "Error: model type does not exist!"})
        resp.status_code = 400
        return resp

    # do inference!
    try:
        do_inference(opts, image)
    except ModelNotFoundException:
        resp = jsonify({"message": "Error: model type does not exist!"})
        resp.status_code = 400
        return resp
    except Exception as e:
        resp = jsonify({"message": f"Error: {e}"})
        resp.status_code = 400
        return resp

    return send_file(
        "../output.png",
        as_attachment=True,
        mimetype="image/png",
    )
