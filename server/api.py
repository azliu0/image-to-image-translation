from flask import request, Response
from apiflask import APIBlueprint

api = APIBlueprint("api", __name__, url_prefix="/api")


@api.route("/")
def hello_world():
    return "hello world"


@api.route("/inference", methods=["POST"])
def inference():
    data = request.get_json()
    print(data)
    return data
