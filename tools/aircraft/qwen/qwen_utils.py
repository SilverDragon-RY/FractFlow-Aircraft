from typing import List, Dict, Optional, Any
import os
import requests
from PIL import Image
import base64
import io
import json

class QwenClient:
    def __init__(self, server_url="http://localhost:5001"):
        self.server_url = server_url
        self.initialize_server()
        
    def initialize_server(self):
        """初始化服务器上的Qwen模型"""
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

    def encode_image(self, image: Image.Image) -> str:
        """将PIL图像编码为base64字符串"""
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        base64_image = base64.b64encode(buffer.getvalue()).decode('utf-8')
        return base64_image
    
    def encode_image_from_path(self, image_path: str) -> str:
        """从文件路径读取图像并编码为base64字符串"""
        image = Image.open(image_path)
        return self.encode_image(image)
    
    async def generate_text(self, image64, text_prompt="Describe this image.", system_prompt="You are a helpful assistant.", max_new_tokens=128):
        """
        发送图像和文本到服务器，获取生成的文本
        
        Args:
            image: PIL.Image对象或图像文件路径
            text_prompt: 用户文本提示
            system_prompt: 系统提示
            max_new_tokens: 最大生成token数
            
        Returns:
            生成的文本字符串，失败时返回None
        """
        try:
            # 准备请求数据
            data = {
                "image": image64,
                "text": text_prompt,
                "system_prompt": system_prompt,
                "max_new_tokens": max_new_tokens
            }
            
            # 发送请求
            response = requests.post(
                f"{self.server_url}/generate",
                json=data,
                headers={'Content-Type': 'application/json'}
            )
            
            result = response.json()
            
            if result['status'] == 'success':
                print(f"生成成功")
                return result['output_text']
            else:
                print(f"生成失败: {result['message']}")
                return None
                
        except Exception as e:
            print(f"请求失败: {e}")
            return None
    
    def save_result(self, output_text: str, output_path: str):
        """保存生成结果到文件"""
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(output_text)
        print(f"生成结果已保存到: {output_path}")

def normalize_path(path: str) -> str:
    """
    标准化文件路径
    
    Args:
        path: 输入路径
        
    Returns:
        标准化的绝对路径
    """
    # 展开~到用户主目录
    expanded_path = os.path.expanduser(path)
    
    # 转换为绝对路径
    if not os.path.isabs(expanded_path):
        expanded_path = os.path.abspath(expanded_path)
        
    return expanded_path

async def qwen_tool(client: QwenClient, image64, text_prompt="Describe this image.", system_prompt="You are a helpful assistant.", max_new_tokens=128):
    """
    Qwen工具函数
    
    Args:
        client: QwenClient实例
        image64: 图像base64编码
        text_prompt: 用户文本提示
        system_prompt: 系统提示
        max_new_tokens: 最大生成token数
        
    Returns:
        生成的文本
    """
    # 检查服务器健康状态
    if not client.check_health():
        print("服务器不可用，请先启动服务器")
        return None
    
    # 初始化Qwen模型
    if not client.initialize_server():
        print("模型初始化失败")
        return None

    # 执行文本生成
    output_text = await client.generate_text(image64, text_prompt, system_prompt, max_new_tokens)
    return output_text

if __name__ == "__main__":
    # 示例用法
    client = QwenClient()
    
    # 使用示例图像
    image_path = "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg"
    system_prompt = "你是一个专业的图像分析师，请详细描述图片内容。"
    result = qwen_tool(client, image_path, "请描述这张图片的内容", system_prompt)
    
    if result:
        print(f"生成结果: {result}")
        client.save_result(result, "./output.txt") 