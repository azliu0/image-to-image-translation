import requests
import json
from server.config import MODELBIT_URL


def modelbit_pix2pix_full_no_cfg_no_ddim(opts):
    modelbit_url = MODELBIT_URL
    body = {"data": opts}
    body = json.dumps(body)
    headers = {"Content-Type": "application/json"}
    response = requests.post(modelbit_url, data=body, headers=headers)
    if response.status_code != 200:
        raise Exception("modelbit error")
