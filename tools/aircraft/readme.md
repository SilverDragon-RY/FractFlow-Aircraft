### 配置.env
将需要的api key放在
/path/to/FractFlow-Aircraft/.env

### 进入aircraft目录
```
cd /path/to/FractFlow-Aircraft/tools/aircraft
```

### 创建环境
```
uv venv
```

### 激活环境
```
source .venv/bin/activate
```

## 配置基础环境
(optional) pip清华源`-i https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple`

```
uv pip install openai pyyaml loguru dotenv mcp pillow replicate websocket json-repair tokencost gradio -i https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple
```

### 配置SAM
```
cd ./sam

git clone https://github.com/facebookresearch/segment-anything.git`

cd ./segment-anything`

uv pip install -e .`

uv pip install opencv-python pycocotools matplotlib onnxruntime onnx flask -i https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple

uv pip install torch torchvision torchaudio -i https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple
```

### 下载权重：
下载sam_vit_h_4b8939.pth到/path/to/segment-anything/sam_vit_h_4b8939.pth

下载源：https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth

### 环境路径设置(optional)
```
export PYTHONPATH=$PYTHONPATH:/path/to/segment-anything (要自己设置segment-anything的路径)

export PYTHONPATH=$PYTHONPATH:/path/to/FractFlow-Aircraft
```

### 设置可见gpu
```
export CUDA_VISIBLE_DEVICES="1"
```

## 运行：
请在三个不同的终端分别运行下面的代码：

（记得激活venv环境）

1. 运行sam_server.py
（本条不需要执行，yuxin在校园网的http://10.30.58.120:5000部署了sam server，并且已经hard code到sam_gradio.py了）
进入/FractFlow-Aircraft/tools/aircraft/sam
python sam_server.py
会运行一个sam的server

2. 运行sam_gradio.py (确保当前网络在HKUSTGZ)
进入/FractFlow-Aircraft/tools/aircraft/sam
在sam下创建./tmp/camera/，并且放置测试图片，在sam_gradio.py中修改IMAGE_PATH来索引到文件
```
python sam_gradio.py
```
得到分割结果后，带boundary的图片会被保存到/FractFlow-Aircraft/tools/aircraft/sam/tmp/test_boundary.png

3. 运行safty_agent.py


进入/FractFlow-Aircraft/tools/aircraft/safety_check
```
原图：
python safety_agent.py --query "Image:path/to/FractFlow-Aircraft/tools/aircraft/sam/tmp/test_boundary.png"
cropped图：
python safety_agent.py --query "Image:path/to/FractFlow-Aircraft/tools/aircraft/sam/tmp/test_boundary_cropped.png"
```
(需要修改为自己的路径)


