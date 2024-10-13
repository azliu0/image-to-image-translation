# server modules
from flask import request, jsonify, send_file
from apiflask import APIBlueprint
from PIL import Image
from io import BytesIO
from server.utils.ModelNotFoundException import ModelNotFoundException

# inference modules
from server.config import OPTS, MODELS, IMAGE_HEIGHT, IMAGE_WIDTH
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

def do_inference(opts, image):
    match opts["model"]:
        case "pix2pix-base":
            inference_pix2pix_base(opts, image)
        case "pix2pix-full-no-cfg-no-ddim":
            inference_pix2pix_full_no_cfg_no_ddim(opts, image)


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
        resp = jsonify({"message": f"Error: model type does not exist!"})
        resp.status_code = 400
        return resp

    # do inference!
    try:
        do_inference(opts, image)
    except ModelNotFoundException:
        resp = jsonify({"message": f"Error: model type does not exist!"})
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
