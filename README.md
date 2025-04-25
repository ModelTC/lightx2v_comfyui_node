# lightx2v-comfyui-node

#### 启动镜像
```
docker pull lightx2v/lightx2v:latest
docker run -it --rm --gpus all --ipc=host --network=host -v /path/of/x2v_models:/x2v_models --entrypoint bash lightx2v/lightx2v:latest
```

#### 安装 ComfyUI
```
git clone https://github.com/comfyanonymous/ComfyUI.git
cd ComfyUI/
pip install -r requirements.txt
```

#### 安装 lightx2v node 和 lightx2v
```
cd ComfyUI/custom_nodes/
git clone https://github.com/ModelTC/lightx2v_comfyui_node.git
cd lightx2v_comfyui_node/
git checkout develop
git submodule update --init --recursive
pip install -r lightx2v/requirements.txt
pip install transformers==4.45.2
pip install --upgrade sgl_kernel
```

#### 启动 ComfyUI
```
export PYTHONPATH=custom_nodes/lightx2v_comfyui_node/lightx2v:$PYTHONPATH
cd ComfyUI/
python main.py --input-directory custom_nodes/lightx2v_comfyui_node/input --listen 0.0.0.0 --enable-cors-header --port 8080
```

#### 访问 ComfyUI
浏览器访问 `http://${server-ip}:8080`，工作流->浏览模板->lightx2v_comfyui_node，可找到 wan t2v，wan i2v, hunyuan t2v, hunyuan i2v 四个workflows 模板

