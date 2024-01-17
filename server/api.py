from flask import request, Response, jsonify, send_file
from apiflask import APIBlueprint
from PIL import Image
from io import BytesIO
from werkzeug.datastructures import FileStorage

api = APIBlueprint("api", __name__, url_prefix="/api")


def get_pil_image(file_storage_obj):
    bytes = file_storage_obj.read()
    stream = BytesIO(bytes)
    return Image.open(stream)


def save_pil(pil):
    pil.save("output.png")


@api.route("/")
def hello_world():
    return "hello world"


@api.route("/inference", methods=["POST"])
def inference():
    print(request.form)
    if "files[]" not in request.files:
        resp = jsonify({"message": "No file in the request!"})
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
