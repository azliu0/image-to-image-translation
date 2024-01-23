import requests
import json
from server.config import MODELBIT_URLS


def modelbit_inference(opts, model):
    modelbit_url = MODELBIT_URLS[model]
    body = {"data": opts}
    body = json.dumps(body)
    headers = {"Content-Type": "application/json"}
    response = requests.post(modelbit_url, data=body, headers=headers)
    if response.status_code != 200:
        raise Exception("modelbit error")
