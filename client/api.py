#This is an example that uses the websockets api to know when a prompt execution is done
#Once the prompt execution is done it downloads the images using the /history endpoint

import websocket #NOTE: websocket-client (https://github.com/websocket-client/websocket-client)
import os
import uuid
import json
import urllib.request
import urllib.parse
import requests

server_address = "127.0.0.1:8080"
client_id = str(uuid.uuid4())

def queue_prompt(p):
    p['client_id'] = client_id
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

def get_images(ws, data, save_dir):
    prompt_id = queue_prompt(data)['prompt_id']
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
            # If you want to be able to decode the binary stream for latent previews, here is how you can do it:
            # bytesIO = BytesIO(out[8:])
            # preview_image = Image.open(bytesIO) # This is your preview in PIL image format, store it in a global
            continue #previews are binary data

    history = get_history(prompt_id)[prompt_id]
    for node_id in history['outputs']:
        node_output = history['outputs'][node_id]
        images_output = []
        if 'images' in node_output:
            for idx, image in enumerate(node_output['images']):
                image_data = get_image(image['filename'], image['subfolder'], image['type'])
                save_name = "{}_{}_{}".format(node_id, idx, image['filename'])
                save_path = os.path.join(save_dir, save_name)
                with open(save_path, 'wb') as fout:
                    fout.write(image_data)
                print("saved to {}".format(save_path))


def fmt_wan_t2v_data(prompt):
    template_json_path = "prompts/wan_t2v.json"
    data = json.loads(open(template_json_path).read())
    data['prompt']['12']['inputs']['prompt'] = prompt
    return data


def fmt_hunyuan_t2v_data(prompt):
    template_json_path = "prompts/hunyuan_t2v.json"
    data = json.loads(open(template_json_path).read())
    data['prompt']['3']['inputs']['prompt'] = prompt
    return data


def process_t2v(model_cls, prompt, save_dir="outputs"):
    data = None
    if model_cls == 'hunyuan':
        data = fmt_hunyuan_t2v_data(prompt)
    elif model_cls == 'wan2.1':
        data = fmt_wan_t2v_data(prompt)
    else:
        raise Exception("Unknown model_cls: {}".format(model_cls))
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    ws = websocket.WebSocket()
    ws.connect("ws://{}/ws?clientId={}".format(server_address, client_id))
    images = get_images(ws, data, save_dir)
    ws.close()

if __name__ == "__main__":
    process_t2v("wan2.1", "a fish is swimming in the sea.")
    process_t2v("wan2.1", "a bird is flying in the sky.")
    process_t2v("wan2.1", "a girl is flying in the sky.")
    process_t2v("hunyuan", "a bird is flying in the blue sky.")
