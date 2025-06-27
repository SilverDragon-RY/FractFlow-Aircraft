from typing import List, Dict, Optional, Any
import os
from mcp.server.fastmcp import FastMCP
from openai import OpenAI

from qwen.qwen_utils import qwen_tool, QwenClient

from dotenv import load_dotenv
load_dotenv()
# Initialize FastMCP server
mcp = FastMCP("Safety_VLM")

client = QwenClient(server_url="http://10.30.58.120:5001")

from PIL import Image
import base64
import io

def normalize_path(path: str) -> str:
    """
    Normalize a file path by expanding ~ to user's home directory
    and resolving relative paths.
    
    Args:
        path: The input path to normalize
        
    Returns:
        The normalized absolute path
    """
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
    """Load an image from the given path.
    
    Args:
        image_path: Path to the image file to load
        
    Returns:
        A tuple containing:
            - Base64 encoded string of the resized image (max 512x512) (this can be put in the image_url field of the user message)
            - Dictionary with metadata including original width and height
    """
    meta_info = {}
    image = Image.open(image_path)
    meta_info['width'], meta_info['height'] = image.size
    base64_image = encode_image(image, size_limit)
    return base64_image, meta_info

# @mcp.tool()
async def Safety_VLM(image_path: str) -> str:
    '''
    This tool uses Qwen-VL-Plus model to analyse the safety level of a landing spot from a given masked image input.
    
    Args:
        image_path (str): Full path to the image file to process. The path should be accessible
                          by the system and point to a valid image file (e.g., JPG, PNG).

    Returns:
        str: A safety level (Green, Yellow, Red) and its reasoning.
    '''
    image_path = normalize_path(image_path)
    base64_image, meta_info = load_image(image_path, (512, 512))

    # --- TBC ---
    client = OpenAI(
        # 若没有配置环境变量，请用百炼API Key将下行替换为：api_key="sk-xxx",
        api_key=os.getenv('QWEN_API_KEY'),
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    )
    # --- --- ---
    prompt = "请分析图中红色boundary区域的是否属于标准停机坪。"
    completion = client.chat.completions.create(
        model="qwen-vl-max",  # 此处以qwen-vl-plus为例，可按需更换模型名称。模型列表：https://help.aliyun.com/zh/model-studio/getting-started/models
        messages=[{"role": "user","content": [
                {"type": "text","text": prompt},
                {"type": "image_url",
                "image_url": {"url": f'data:image/png;base64,{base64_image}'}}
                ]}]
    )
    return completion.choices[0].message.content

@mcp.tool()
async def Safety_VLM_Local(image_path: str) -> str:
    '''
    This tool uses Qwen2.5-VL-7B-Instruct model to analyse the safety level of a landing spot from a given masked image input.
    
    Args:
        image_path (str): Full path to the image file to process. The path should be accessible
                          by the system and point to a valid image file (e.g., JPG, PNG).

    Returns:
        str: A safety level (Green, Yellow, Red) and its reasoning.
    '''
    image_path = normalize_path(image_path)
    # base64_image, meta_info = load_image(image_path, (512, 512))

    image = Image.open(image_path)
    text_prompt = "请分析图中红色boundary区域的是否属于标准停机坪。"
    result = await qwen_tool(client, image, text_prompt, max_new_tokens=1024)
    return result
    # return completion.choices[0].message.content

if __name__ == "__main__":
    # Initialize and run the server
    mcp.run(transport='stdio') 

