import requests
import json
import time
import asyncio
from typing import Dict, Any, Optional
from datetime import datetime
from mcp.server.fastmcp import FastMCP
from openai import OpenAI
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Initialize FastMCP server
mcp = FastMCP("flightbrain")

# API端点
API_URL = "http://10.4.147.50:5000/set"


def set_flight_parameter(param_name, param_value):
    """
    设置飞行参数
    
    Args:
        param_name (str): 参数名称
        param_value: 参数值
        
    Returns:
        str: 返回消息
    """
    # 准备请求数据
    payload = {
        "name": param_name,
        "val": param_value
    }

    try:
        # 发送PUT请求到API
        response = requests.put(API_URL, json=payload)
        
        # 检查响应状态码
        if response.status_code == 200:
            # 解析JSON响应
            data = response.json()
            return f"设置成功: {data['message']}"
        else:
            if response.text:
                try:
                    error_data = response.json()
                    return f"请求失败，状态码: {response.status_code}, 错误信息: {error_data['message']}"
                except:
                    return f"请求失败，状态码: {response.status_code}, 响应内容: {response.text}"
            else:
                return f"请求失败，状态码: {response.status_code}"
            
    except requests.exceptions.RequestException as e:
        return f"请求错误: {e}"
    except json.JSONDecodeError:
        return "无法解析JSON响应"
    except Exception as e:
        return f"发生错误: {e}"

@mcp.tool()
def move_forward(time_s):
    """
    Make the aircraft move forward
    
    Args:
        duration (float): Duration to move forward in seconds
    
    Returns:
        None
    """
    print("Moving forward", time_s)
    hover()
    set_flight_parameter("GENERAL_ENG_THROTTLE_LEVER_POSITION_1", 99)
    time.sleep(float(time_s))
    hover()
    print("Moving forward done")
    pass

@mcp.tool()
def move_backward(time_s):
    """
    Make the aircraft move backward
    
    Args:
        duration (float): Duration to move backward in seconds
    
    Returns:
        None
    """
    print("Moving backward", time_s)
    hover()
    set_flight_parameter("GENERAL_ENG_THROTTLE_LEVER_POSITION_1", 0)
    time.sleep(float(time_s))
    hover()
    print("Moving backward done")
    pass

@mcp.tool()
def move_left(time_s):
    """
    Make the aircraft move left
    
    Args:
        duration (float): Duration to move left in seconds
    
    Returns:
        None
    """
    print("Moving left", time_s)
    hover()
    set_flight_parameter("AILERON_POSITION", -1.0)
    time.sleep(float(time_s))
    hover()
    print("Moving left done")
    pass

@mcp.tool()
def move_right(time_s):
    """
    Make the aircraft move right
    
    Args:
        duration (float): Duration to move right in seconds
    
    Returns:
        None
    """
    print("Moving right", time_s)
    hover()
    set_flight_parameter("AILERON_POSITION", 1.0)
    time.sleep(float(time_s))
    hover()
    print("Moving right done")
    pass

@mcp.tool()
def move_ascend(time_s):
    """
    Make the aircraft move ascend
    
    Args:
        duration (float): Duration to move up in seconds
    
    """
    print("Moving up", time_s)
    hover()
    set_flight_parameter("ELEVATOR_POSITION", 1.0)
    time.sleep(float(time_s))
    hover()
    print("Moving up done")
    pass

@mcp.tool()
def move_descend(time_s):
    """
    Make the aircraft move descend
    
    Args:
        duration (float): Duration to move down in seconds
    
    """
    print("Moving down", time_s)
    hover()
    set_flight_parameter("ELEVATOR_POSITION", -1.0)
    time.sleep(float(time_s))
    hover()
    print("Moving down done")
    pass

@mcp.tool()
def hover():
    """
    Make the aircraft hover
    
    Args:
        None
    
    """
    print("Hovering")
    set_flight_parameter("GENERAL_ENG_THROTTLE_LEVER_POSITION_1", 50)
    set_flight_parameter("RUDDER_POSITION", 0.0)
    set_flight_parameter("AILERON_POSITION", 0.0)
    set_flight_parameter("ELEVATOR_POSITION", 0.0)
    time.sleep(0.1)
    pass

@mcp.tool()
def hover_turn_left(time_s):
    """
    Make the aircraft hover and turn left
    
    Args:
        duration (float): Duration to turn left in seconds
    
    """
    print("Hover turning left", time_s)
    hover()
    set_flight_parameter("RUDDER_POSITION", -0.25)
    time.sleep(float(time_s))
    hover()
    print("Hover turning left done")
    pass

@mcp.tool() 
def hover_turn_right(time_s):
    """
    Make the aircraft hover and turn right
    
    Args:
        duration (float): Duration to turn right in seconds
    
    """
    print("Hover turning right", time_s)
    hover()
    set_flight_parameter("RUDDER_POSITION", 0.25)
    time.sleep(float(time_s))
    hover()
    print("Hover turning right done")
    pass

import os
import base64
import io
from PIL import Image

os.sys.path.append("/home/bld/dyx/FractFlow-Aircraft/tools/aircraft")
from qwen.qwen_utils import qwen_tool, QwenClient

client = QwenClient(server_url="http://10.30.58.120:5001")
qwen_client = QwenClient(server_url="http://10.30.58.120:5001")

def normalize_path(path: str) -> str:
    # Expand ~ to user's home directory
    expanded_path = os.path.expanduser(path)
    
    # Convert to absolute path if relative
    if not os.path.isabs(expanded_path):
        expanded_path = os.path.abspath(expanded_path)
        
    return expanded_path

def encode_image(image: Image.Image, size: tuple[int, int] = (512, 512)) -> str:
    image.thumbnail(size)
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    base64_image = base64.b64encode(buffer.getvalue()).decode('utf-8')
    return base64_image

def load_image(image_path: str, size_limit: tuple[int, int] = (512, 512)) -> tuple[str, dict]:
    meta_info = {}
    image = Image.open(image_path)
    meta_info['width'], meta_info['height'] = image.size
    base64_image = encode_image(image, size_limit)
    return base64_image, meta_info

@mcp.tool()
async def flight_decision():
    '''
    This tool uses Qwen2.5-VL-7B-Instruct model to make flight decision.
    
    Args:
        None

    Returns:
        str: A flight decision.
    '''

#     SYSTEM_PROMPT_LOCAL = """
# 你是MSFS的飞行决策AI，负责评估Joby S4 eVTOL的降落过程中图像中的视觉信息。

# # 工作流程
# 1. 指出图像中的红色boundary停机坪相对图像中央的方位（例如：左侧，右侧，上方，下方）
# 2. 分析停机坪周围的环境，给出停机坪周围的环境状况（障碍物分布、空间开阔程度）


# # 输出格式
# - 停机坪位置：[停机坪在图像的哪个位置，并指出停机坪相对图像中央的方位]
# - 停机坪周围环境：[停机坪周围的环境状况（障碍物分布、空间开阔程度）]

# """
    SYSTEM_PROMPT_LOCAL = """
    *1. 角色与核心任务**

​    你是一个先进的飞行决策AI，名为“天穹之眼 (Aether-Eye)”。你的核心任务是自主操控一台JOBY S4 eVTOL飞行器，并将其精准、安全地降落在指定平台上。

​    你输入信息是来自飞行器前置摄像头拍摄的实时图像和专家模型对停机坪可见状态的描述。在图像中，目标停机坪会以一个醒目的**红色边框**标出。

​    你的职责是分析图像，先输出对当前飞行器与红色边框指示的停机坪相对位置的综合描述，然后输出下一步要执行的单一飞行操作指令。

​    **2. 可用操作指令 (Action Space)**

​    你只能从以下指令中选择一个进行输出。对于所有带有 `(time_s)` 参数的指令，根据当前情况**自主决定一个最合理的持续时间（秒）** 是你的关键职责之一。

    * `hover`: 在当前位置和高度悬停，用于观察和重新评估。
        * `move_forward(time_s)`: 向前水平移动。
        * `move_backward(time_s)`:向后水平移动。
        * `move_left(time_s)`: 向左水平移动。
        * `move_right(time_s)`: 向右水平移动。
        * `move_ascend(time_s)`: 垂直上升。
        * `move_descend(time_s)`: 垂直下降。
        * `hover_turn_left(time_s)`: 在原地向左水平转动（掉头）。
        * `hover_turn_right(time_s)`: 在原地向右水平转动（掉头）。

​    **3. 决策逻辑与思考框架**

​    你的决策过程应模拟一个经验丰富的飞行员，并严格遵循以下顺序清晰的阶段。

    请你首先描述图像中的红色边框指示的停机坪与飞行器当前位置的相对空间关系，结合专家模型对停机坪可见状态的描述，解释当前看到的红色边框指示的停机坪的可见状态（是否被部分遮挡，导致看不到完整的停机坪中心（H字母的中心或圆形的中心））。然后根据相对空间关系判断你正处于哪个决策阶段。

* **阶段一：建立下滑道并最终逼近 (Establish Glide Slope & Final Approach)**
        
        * **核心任务**：在此阶段，你的目标是**靠近停机坪并到达它的正上方**
        *一般情况*：你需要综合使用 `move_forward(t)` 和 `move_descend(t)`，其目标是让红色边框指示的停机坪在**视野的中心**稳定地放大。
        *重要情况*：当停机坪的可见状态为[被部分遮挡]，你**必须**优先执行`move_descend(t)`下降，直到停机坪完全可见，再进行其他操作。
        *紧急情况*：当停机坪高于飞行器时，你需要`move_ascend(t)`上升，直到停机坪低于飞行器。
        *结束标志*：当红色边框指示的停机坪占据大部分视野时，你已经到达停机坪的正上方。

    * 如果目标在视野左侧，使用 `hover_turn_left(t)` 向左旋转；如果目标在视野右侧，使用 `hover_turn_right(t)` 向右旋转。
    
* **阶段二：执行降落 (Execute Landing)**

    * **触发条件**：飞行器处于停机坪正上方时。
    * **执行**：开始执行**持续的垂直下降** `move_descend(t)`。持续下降直到任务完成。

    * **特殊情况** 当停机坪在视野正前方非常接近时，你需要`move_forward(t)`前进，直到飞行器处于停机坪正上方。

​    **4. 输出格式**

​    你的输出必须严格遵循以下两行格式。

    1.  **第一行 `解释:`**：请你详细解释你所理解的当前的飞行情况，停机坪的可见状态，和你正处于哪个决策阶段。
    2.  **第二行 `操作:`**：根据你的解释判断出的、即将执行的单一指令。（time_s至少为1秒）

​    **输出示例:**
​    描述: [一段解释文本]
​    操作: hover_turn_left(1.2)

​    """

    image_path = "./sam/tmp/boundary_img.png"
    image_path = normalize_path(image_path)
    base64_image, meta_info = load_image(image_path, (512, 512))

    vertical_position_check_result = await vertical_position_check()
    text_prompt = f"请分析图片进行飞行决策：\n下面是一个专家模型对停机坪可见状态的描述，请你参考：{vertical_position_check_result}"

    client = OpenAI(
        # 若没有配置环境变量，请用百炼API Key将下行替换为：api_key="sk-xxx",
        api_key=os.getenv('QWEN_API_KEY'),
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    )
    completion = client.chat.completions.create(
        model="qwen-vl-max",  # 此处以qwen-vl-plus为例，可按需更换模型名称。模型列表：https://help.aliyun.com/zh/model-studio/getting-started/models
        messages=[
            {"role": "system","content": SYSTEM_PROMPT_LOCAL},
            {"role": "user","content": [
                {"type": "text","text": text_prompt},
                {"type": "image_url",
                "image_url": {"url": f'data:image/png;base64,{base64_image}'}}
                ]}]
    )
    return json.dumps({
        'image_info': completion.choices[0].message.content,
    }, indent=2, ensure_ascii=False)


# @mcp.tool()
async def vertical_position_check():
    """
    Check the vertical position of the aircraft
    
    Args:
        None
    """
    SYSTEM_PROMPT_LOCAL = """
    你是一个停机坪检测专家，负责描述红色边框指示的停机坪的可见状态。
    一个未被遮挡的直升机停机坪（Helipad / Vertiport），通常带有'H'标志或特定几何图案，且是正方形或圆形。否则，认为停机坪被遮挡。
    图片中是飞机驾驶舱向外拍摄的画面，底部的白色轮廓是飞行器的结构。

    输出内容：
    - 解释你看到停机坪的情况，主要解释飞行器轮廓与停机坪的互相遮挡的情况
    - 停机坪是否被飞行器轮廓部分遮挡[是/否]
    """
    image_path = "./sam/tmp/boundary_img.png"

    text_prompt = "请分析图片中红色边框指示的停机坪的情况："
    image_path = normalize_path(image_path)
    base64_image, meta_info = load_image(image_path, (512, 512))
    result = await qwen_tool(client, base64_image, text_prompt, system_prompt=SYSTEM_PROMPT_LOCAL, max_new_tokens=1024)
    print(result)
    return result

if __name__ == "__main__":
    mcp.run(transport='stdio') 
    # asyncio.run(vertical_position_check())