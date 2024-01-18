# server modules
from flask import request, Response, jsonify, send_file
from apiflask import APIBlueprint
from PIL import Image
from io import BytesIO
from werkzeug.datastructures import FileStorage
from server.utils.ModelNotFoundException import ModelNotFoundException
import asyncio

# inference modules
from server.config import OPTS, MODELS
from server.pix2pix.pix2pix_base import inference_pix2pix_base
from server.pix2pix.pix2pix_full_no_cfg_no_ddim import (
    inference_pix2pix_full_no_cfg_no_ddim,
)

api = APIBlueprint("api", __name__, url_prefix="/api")


def get_pil_image(file_storage_obj):
    bytes = file_storage_obj.read()
    stream = BytesIO(bytes)
    return Image.open(stream)


def save_pil(pil):
    pil.save("output.png")


def validate_model(model):
    if model not in MODELS:
        raise ModelNotFoundException()


def do_inference(opts, image):
    print(opts)
    match opts["model"]:
        case "pix2pix-base":
            inference_pix2pix_base(opts, image)
        case "pix2pix-full-no-cfg-no-ddim":
            inference_pix2pix_full_no_cfg_no_ddim(opts, image)
        case _:
            raise ModelNotFoundException()


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
    image = get_pil_image(image)

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
