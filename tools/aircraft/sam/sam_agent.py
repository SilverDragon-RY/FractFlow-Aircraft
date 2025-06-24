import os
import sys

# Add the project root directory to the Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '..'))
sys.path.append(project_root)

# Import the FractFlow ToolTemplate
from FractFlow.tool_template import ToolTemplate

class VQATool(ToolTemplate):
    """Visual Question Answering tool using ToolTemplate"""
    
    SYSTEM_PROMPT = """
你是一个图像分割助手，通过SAM模型分割图像。
你会将用户提供的图像路径作为输入，使用SAM模型分割图像，并返回分割结果。
"""
    
    TOOLS = [
        ("tools/aircraft/sam/sam_mcp.py", "segment_image_with_same")
    ]
    
    MCP_SERVER_NAME = "segment_image_with_same"
    
    TOOL_DESCRIPTION = """Segment an image with SAM model
    
    Parameters:
        image_path: str - Include image path (e.g., "Image: /path/photo.jpg")
        
    Returns:
        None
    """
    
    @classmethod
    def create_config(cls):
        """Custom configuration for VQA tool"""
        from FractFlow.infra.config import ConfigManager
        from dotenv import load_dotenv
        
        load_dotenv()
        return ConfigManager(
            provider='qwen',
            qwen_model='qwen-plus',
            max_iterations=5,  # Visual analysis usually completes in one iteration
            custom_system_prompt=cls.SYSTEM_PROMPT,
            tool_calling_version='turbo'
        )

if __name__ == "__main__":
    VQATool.main() 