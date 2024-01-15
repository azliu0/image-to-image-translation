from flask import request
from apiflask import APIBlueprint

api = APIBlueprint("api", __name__, url_prefix="/api", tag="api")


@api.route("/")
def hello_world():
    return "hello world"
