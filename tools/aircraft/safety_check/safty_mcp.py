from typing import List, Dict, Optional, Any
import os
from mcp.server.fastmcp import FastMCP
from openai import OpenAI

os.sys.path.append("/home/bld/dyx/FractFlow-Aircraft/tools/aircraft")
from qwen.qwen_utils import qwen_tool, QwenClient

from dotenv import load_dotenv


from PIL import Image
import base64
import io
import json


load_dotenv()
# Initialize FastMCP server
mcp = FastMCP("Safety_VLM")

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

def parse_result(text: str):
    content = text.find("方框内容识别")
    level = text.find("评估结果")
    reasoning = text.find("核心理由")
    result = {
        "complete": 0,
        "landing_spot_type": "",
        "safety_level": "",
        "safety_reasoning": ""
    }
    if content == -1 or level == -1 or reasoning == -1:
        return result
    else:
        result["complete"] = 1
    for i in range(content+6, len(text)):
        if text[i] == "-" or text[i] == "#":
            break
        if text[i] != "：" and text[i] != ":":
            result["landing_spot_type"] += text[i]
    result["landing_spot_type"] = result["landing_spot_type"][:-2]
    for i in range(level+4, len(text)):
        if text[i] == "-" or text[i] == "#":
            break
        if text[i] != "：" and text[i] != ":":
            result["safety_level"] += text[i]
    result["safety_level"] = result["safety_level"][:-2]
    for i in range(reasoning+4, len(text)):
        if text[i] == "-" or text[i] == "#":
            break
        if text[i] != "：" and text[i] != ":":
            result["safety_reasoning"] += text[i]
    result["safety_reasoning"] = result["safety_reasoning"][:-2]
    return result


@mcp.tool()
async def Safety_VLM_Local() -> str:
    '''
    This tool uses Qwen2.5-VL-7B-Instruct model to analyse the landing safety level of the marked landing spot in current sight.
    
    Args:
        None

    Returns:
        str: A safety level ("适合降落"或"谨慎降落"或"不适合降落") and its reasoning.
    '''

    SYSTEM_PROMPT_LOCAL = """
你是MSFS的视觉分析AI，负责评估Joby S4 eVTOL的降落安全性。

# 工作流程
1. **识别红色boundary内部的内容**：描述方框内的地物类型，忽略方框外的内容，特别要忽略边框外的停机坪
2. **仅根据方框内容的文本描述判断是否为直升机停机坪**：
   - **是停机坪**：继续分析周边环境，给出"适合降落"或"谨慎降落"
   - **不是停机坪**：直接评估为"不适合降落"

# 硬性规则
- 降落点必须是明确的直升机停机坪（有H标志或特定几何图案）
- 非停机坪（屋顶、公路、草地、停车场等）一律"不适合降落"

# 停机坪环境评估要素
- 周边建筑物高度
- 树木、塔吊、电线等障碍物
- 进近/撤离空间是否清晰

# 输出格式
- #方框内容识别：[简短描述]
- #评估结果：[适合降落/谨慎降落/不适合降落]
- #核心理由：[以序号列出关键依据]
- #潜在风险与建议：[风险点和建议]
- #限制声明：“最终的飞行安全始终由操作飞行模拟器的用户（飞行员）负责”
"""

    image_path = "./sam/tmp/cropped_image.png"
    image_path = normalize_path(image_path)
    base64_image, meta_info = load_image(image_path, (512, 512))

    text_prompt = "请分析图片："
    result = await qwen_tool(qwen_client, base64_image, text_prompt, SYSTEM_PROMPT_LOCAL, max_new_tokens=1024)
    print(result)
    result = parse_result(result)
    while True:
        if result["complete"] == 0:
            result = await qwen_tool(qwen_client, base64_image, text_prompt, SYSTEM_PROMPT_LOCAL, max_new_tokens=1024)
        else:
            print(result)
            return json.dumps({
            "landing_spot_type": result["landing_spot_type"],
            "safety_level": result["safety_level"],
            "safety_reasoning": result["safety_reasoning"]
            }, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    # Initialize and run the server
    mcp.run(transport='stdio') 