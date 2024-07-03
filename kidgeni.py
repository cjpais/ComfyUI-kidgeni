import websocket #NOTE: websocket-client (https://github.com/websocket-client/websocket-client)
import uuid
import json
import urllib.request
import urllib.parse

from io import BytesIO

from fastapi import FastAPI, Response
from pydantic import BaseModel

from PIL import Image

app = FastAPI()
server_address = "127.0.0.1:8188"

BASE_PROMPT = """
{
    "3": {
        "inputs": {
            "seed": 1052946731162720,
            "steps": 4,
            "cfg": 1,
            "sampler_name": "euler",
            "scheduler": "sgm_uniform",
            "denoise": 1,
            "model": [
                "4",
                0
            ],
            "positive": [
                "6",
                0
            ],
            "negative": [
                "7",
                0
            ],
            "latent_image": [
                "5",
                0
            ]
        },
        "class_type": "KSampler"
    },
    "4": {
        "inputs": {
            "ckpt_name": "sdxl_lightning_4step.safetensors"
        },
        "class_type": "CheckpointLoaderSimple"
    },
    "5": {
        "inputs": {
            "width": 1024,
            "height": 1024,
            "batch_size": 1
        },
        "class_type": "EmptyLatentImage"
    },
    "6": {
        "inputs": {
            "text": "",
            "clip": [
                "4",
                1
            ]
        },
        "class_type": "CLIPTextEncode"
    },
    "7": {
        "inputs": {
            "text": "",
            "clip": [
                "4",
                1
            ]
        },
        "class_type": "CLIPTextEncode"
    },
    "8": {
        "inputs": {
            "samples": [
                "3",
                0
            ],
            "vae": [
                "4",
                2
            ]
        },
        "class_type": "VAEDecode"
    },
    "9": {
        "inputs": {
            "filename_prefix": "ComfyUI",
            "images": [
                "14",
                0
            ]
        },
        "class_type": "SaveImage"
    },
    "14": {
        "inputs": {
            "sensitivity": 0.5,
            "images": [
                "8",
                0
            ]
        },
        "class_type": "Safety Checker"
    }
}
"""

def queue_prompt(prompt, client_id):
    p = {"prompt": prompt, "client_id": client_id}
    data = json.dumps(p).encode('utf-8')
    req =  urllib.request.Request("http://{}/prompt".format(server_address), data=data)
    return json.loads(urllib.request.urlopen(req).read())

def get_image(filename, subfolder, folder_type):
    data = {"filename": filename, "subfolder": subfolder, "type": folder_type}
    url_values = urllib.parse.urlencode(data)
    with urllib.request.urlopen("http://{}/view?{}".format(server_address, url_values)) as response:
        return response.read()

def get_history(prompt_id):
    with urllib.request.urlopen("http://{}/history/{}".format(server_address, prompt_id)) as response:
        return json.loads(response.read())

def get_images(ws, prompt, client_id):
    prompt_id = queue_prompt(prompt, client_id)['prompt_id']
    output_images = {}
    while True:
        out = ws.recv()
        if isinstance(out, str):
            message = json.loads(out)
            if message['type'] == 'executing':
                data = message['data']
                if data['node'] is None and data['prompt_id'] == prompt_id:
                    break #Execution is done
        else:
            continue #previews are binary data

    history = get_history(prompt_id)[prompt_id]
    for o in history['outputs']:
        for node_id in history['outputs']:
            node_output = history['outputs'][node_id]
        if 'images' in node_output:
            images_output = []
            for image in node_output['images']:
                image_data = get_image(image['filename'], image['subfolder'], image['type'])
                images_output.append(image_data)
                output_images[node_id] = images_output

    return output_images

class CreateRequest(BaseModel):
    prompt: str
    negative: str = ""
    nsfw_sensitivity: float = 0.5
    seed: int = 1052946731162720

@app.get("/")
def read_root():
    return {"Hello": "World"}

@app.post("/create")
def create(req: CreateRequest):
    prompt = json.loads(BASE_PROMPT)
    prompt['14']['inputs']['sensitivity'] = req.nsfw_sensitivity
    prompt['6']['inputs']['text'] = req.prompt
    prompt['7']['inputs']['text'] = req.negative
    prompt['3']['inputs']['seed'] = req.seed

    client_id = str(uuid.uuid4())
    ws = websocket.WebSocket()
    ws.connect("ws://{}/ws?clientId={}".format(server_address, client_id))
    images = get_images(ws, prompt, client_id)

    for node_id in images:
        for image_data in images[node_id]:
            image = Image.open(BytesIO(image_data))
            buff = BytesIO()
            image.save(buff, format="JPEG")

            return Response(content=buff.getvalue(), media_type="image/jpeg")
