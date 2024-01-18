from flask import request, Response, jsonify, send_file
from apiflask import APIBlueprint
from PIL import Image
from io import BytesIO
from werkzeug.datastructures import FileStorage
from server.config import OPTS
from server.utils.ModelNotFoundException import ModelNotFoundException

api = APIBlueprint("api", __name__, url_prefix="/api")


def get_pil_image(file_storage_obj):
    bytes = file_storage_obj.read()
    stream = BytesIO(bytes)
    return Image.open(stream)


def save_pil(pil):
    pil.save("output.png")


def do_inference(opts):
    print(opts)
    raise ModelNotFoundException()


@api.route("/")
def hello_world():
    return "hello world"


@api.route("/inference", methods=["POST"])
def inference():
    if "files[]" not in request.files:
        resp = jsonify({"message": "No file in the request!"})
        resp.status_code = 400
        return resp
    form = request.form.to_dict()

    for opt in OPTS:
        if opt["key"] not in form:
            resp = jsonify({"message": "Model options not specified correctly"})
            resp.status_code = 400
            return resp
    opts = {}
    for opt in OPTS:
        opts[opt["key"]] = form[opt["key"]]
        if opt["int"]:
            opts[opt["key"]] = int(opts[opt["key"]])
        elif opt["float"]:
            opts[opt["key"]] = float(opts[opt["key"]])

    try:
        do_inference(opts)
    except ModelNotFoundException:
        resp = jsonify({"message": f"Error: model type does not exist!"})
        resp.status_code = 400
        return resp
    except Exception as e:
        resp = jsonify({"message": f"Error: {e}"})
        resp.status_code = 400
        return resp

    image = request.files.getlist("files[]")[0]
    image = get_pil_image(image)
    save_pil(image)
    return send_file(
        "../output.png",
        as_attachment=True,
        mimetype="image/png",
    )
