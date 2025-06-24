from typing import List, Dict, Optional, Any
import os
from mcp.server.fastmcp import FastMCP
from openai import OpenAI
from dotenv import load_dotenv
load_dotenv()
import replicate
import requests

from sam_gradio import trigger_external_reload, IMAGE_PATH

from PIL import Image
import base64
import io
import cv2
import numpy as np
import json

# Initialize FastMCP server
mcp = FastMCP("SAM_tool")



class SAMClient:
    def __init__(self, server_url="http://localhost:5000"):
        self.server_url = server_url
        
    def initialize_server(self):
        """初始化服务器上的SAM模型"""
        try:
            response = requests.post(f"{self.server_url}/initialize")
            result = response.json()
            print(f"初始化结果: {result['message']}")
            return result['status'] == 'success'
        except Exception as e:
            print(f"初始化失败: {e}")
            return False
    
    def check_health(self):
        """检查服务器健康状态"""
        try:
            response = requests.get(f"{self.server_url}/health")
            result = response.json()
            print(f"服务器状态: {result}")
            return result['status'] == 'healthy'
        except Exception as e:
            print(f"健康检查失败: {e}")
            return False
    
    def encode_image(self, image_path):
        """将图像文件编码为base64"""
        with open(image_path, 'rb') as image_file:
            encoded_string = base64.b64encode(image_file.read()).decode('utf-8')
        return encoded_string
    
    def segment_image(self, image_path, prompt_points, prompt_labels=None):
        """
        发送图像分割请求
        
        Args:
            image_path: 图像文件路径
            prompt_points: 提示点列表 [[x1, y1], [x2, y2], ...]
            prompt_labels: 提示点标签列表 [1, 0, 1, ...] (1为前景，0为背景)
        """
        try:
            # 编码图像
            image_base64 = self.encode_image(image_path)
            
            # 准备请求数据
            data = {
                "image": image_base64,
                "prompt_points": prompt_points,
                "prompt_labels": prompt_labels or [1] * len(prompt_points)
            }
            
            # 发送请求
            response = requests.post(
                f"{self.server_url}/segment",
                json=data,
                headers={'Content-Type': 'application/json'}
            )
            
            result = response.json()
            
            if result['status'] == 'success':
                print(f"分割成功，得分: {result['score']}")
                return self.decode_mask(result['mask'])
            else:
                print(f"分割失败: {result['message']}")
                return None
                
        except Exception as e:
            print(f"请求失败: {e}")
            return None
    
    def decode_mask(self, mask_base64):
        """将base64编码的mask解码为numpy数组"""
        mask_bytes = base64.b64decode(mask_base64)
        mask = cv2.imdecode(np.frombuffer(mask_bytes, np.uint8), cv2.IMREAD_GRAYSCALE)
        return mask
    
    def save_mask(self, mask, output_path):
        """保存mask到文件"""
        cv2.imwrite(output_path, mask)
        print(f"分割结果已保存到: {output_path}")
    
    def visualize_result(self, image_path, mask, prompt_points, output_path=None):
        """可视化分割结果"""
        # 读取原图
        image = cv2.imread(image_path)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # 创建彩色mask
        colored_mask = np.zeros_like(image_rgb)
        colored_mask[mask > 127] = [255, 0, 0]  # 红色表示分割区域
        
        # 混合原图和mask
        alpha = 0.5
        result = cv2.addWeighted(image_rgb, 1-alpha, colored_mask, alpha, 0)
        
        # 标记提示点
        for point in prompt_points:
            cv2.circle(result, tuple(point), 5, (0, 255, 0), -1)
        
        if output_path:
            result_bgr = cv2.cvtColor(result, cv2.COLOR_RGB2BGR)
            cv2.imwrite(output_path, result_bgr)
            print(f"可视化结果已保存到: {output_path}")
        
        return result
    
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

def clear_cache():
    os.system("rm -rf ./tmp/individual_masks/*")
    os.system("rm -rf ./tmp/combined_mask/*")

def request_result_from_replicate(response: dict) -> Image.Image:
    clear_cache()
    def request_url(url: str) -> Any:
        response = requests.get(url)
        content = io.BytesIO(response.content) 
        print(content)
        return content

    # 下载文件内容
    combined_mask_content = request_url(response['combined_mask'].url)
    combined_mask_image = Image.open(combined_mask_content)
    combined_mask_image.save("./tmp/combined_mask/combined_mask.png")

    for i, mask_replicate in enumerate(response['individual_masks']):
        individual_masks_content = request_url(mask_replicate.url)
        individual_masks_image = Image.open(individual_masks_content)
        individual_masks_image.save(f"./tmp/individual_masks/individual_masks_{i}.png")

    trigger_external_reload()


# @mcp.tool()
# async 


def SAM_tool(image_path: str) -> str:
    '''
    This tool uses Qwen-VL-Plus model to analyse the safety level of a landing spot from a given masked image input.
    
    Args:
        image_path (str): Full path to the image file to process. The path should be accessible
                          by the system and point to a valid image file (e.g., JPG, PNG).

    Returns:
        str: A safety level (Green, Yellow, Red) and its reasoning.
    '''
    print('>>> SAM will process image: ', image_path)
    image_path = normalize_path(image_path)

    # load image
    image = open(image_path, "rb")

    input={
        "image": image,
        "use_m2m": True,
        "points_per_side": 32,
        "pred_iou_thresh": 0.88,
        "stability_score_thresh": 0.5
    }

    output = replicate.run(
        "meta/sam-2:fe97b453a6455861e3bac769b441ca1f1086110da7466dbb65cf1eecfd60dc83",
        input=input
    )
    print(output)

    # {'combined_mask': <replicate.helpers.FileOutput object at 0x7f6998dc0970>, 
    # 'individual_masks': [<replicate.helpers.FileOutput object at 0x7f6998c18880>, <replicate.helpers.FileOutput object at 0x7f6998c18820>, <replicate.helpers.FileOutput object at 0x7f6998c189d0>, <replicate.helpers.FileOutput object at 0x7f6998c18970>, <replicate.helpers.FileOutput object at 0x7f6998c18910>, <replicate.helpers.FileOutput object at 0x7f6998c188b0>, <replicate.helpers.FileOutput object at 0x7f6998c18bb0>, <replicate.helpers.FileOutput object at 0x7f6998c18b50>, <replicate.helpers.FileOutput object at 0x7f6998c18af0>, <replicate.helpers.FileOutput object at 0x7f6998c18a90>, <replicate.helpers.FileOutput object at 0x7f6998c18be0>]}
    request_result_from_replicate(output)


if __name__ == "__main__":
    # Initialize and run the server
    # mcp.run(transport='stdio') 
    SAM_tool(IMAGE_PATH)

