# python flightbrain_agent.py --query "Image:/home/bld/dyx/FractFlow-Aircraft/tools/aircraft/sam/tmp/test_boundary.png"

import os
import sys
import re

# Add the project root directory to the Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '../..'))
sys.path.append(project_root)

# Import the FractFlow ToolTemplate
from FractFlow.tool_template import ToolTemplate
import json

class FlightBrain_Agent(ToolTemplate):
    """Intelligent Flight Brain Agent that integrates flight control, visual analysis, and safety assessment"""
    
    SYSTEM_PROMPT = """
你是一个智能飞行大脑代理，专门负责为 Joby S4 eVTOL 提供智能飞行决策。

# 工作流程
1. 首先，如果用户提供了图像路径，使用analyze_flight_situation工具分析图像
2. 然后，基于分析结果，选择并执行合适的飞行操作工具

# 可用的飞行操作工具
- hover: 悬停（打印"The aircraft is hovering"）
- hover_turn: 悬停转弯（打印"The aircraft is performing a hover turn X degrees clockwise/counter-clockwise"）
- move_forward: 前进（打印"The aircraft is moving forward X meters"）
- move_backward: 后退（打印"The aircraft is moving backward X meters"）
- move_left: 左移（打印"The aircraft is moving left X meters"）
- move_right: 右移（打印"The aircraft is moving right X meters"）
- ascend: 上升（打印"The aircraft is ascending X meters"）
- descend: 下降（打印"The aircraft is descending X meters"）
- rotate: 旋转（打印"The aircraft is rotating X degrees"）
- land: 降落（打印"The aircraft has landed"）

# 决策逻辑
根据图像分析结果：
- 如果分析显示可以安全降落 → 调用land工具
- 如果有障碍物需要避开 → 调用ascend或move_left/right工具
- 如果需要调整位置 → 调用相应的移动工具
- 如果情况不明确 → 调用hover工具保持安全

# 重要
每次必须选择并调用一个飞行操作工具，这样才能输出相应的print内容。
"""
    
    TOOLS = [
        ("/home/bld/cl/FractFlow-Aircraft/tools/aircraft/flightbrain/flightbrain_mcp.py", "flightbrain_operations")
    ]
    
    MCP_SERVER_NAME = "flightbrain_agent"
    
    TOOL_DESCRIPTION = """Intelligent Flight Brain Agent for eVTOL aircraft management.
    
    This agent reads safety assessment results, analyzes images, and executes appropriate flight operations.
    
    Parameters:
        query: str - Flight management query, can include:
            - Image analysis: "Image:/path/to/image.png"
            - Flight commands based on safety assessment
            
    Returns:
        str - Flight operation execution result with print content
        
    Example queries:
        - "Image:/home/bld/dyx/FractFlow-Aircraft/tools/aircraft/sam/tmp/test_boundary.png"
    """
    
    @classmethod
    def create_config(cls):
        """Custom configuration for FlightBrain Agent"""
        from FractFlow.infra.config import ConfigManager
        from dotenv import load_dotenv
        
        load_dotenv()
        return ConfigManager(
            provider='qwen',
            qwen_model='qwen-max',
            max_iterations=15,
            custom_system_prompt=cls.SYSTEM_PROMPT,
            tool_calling_version='turbo'
        )

    def _read_safety_result(self):
        """读取安全评估结果文件"""
        try:
            safety_file = "/home/bld/dyx/FractFlow-Aircraft/tools/aircraft/sam/tmp/safety_result.txt"
            if os.path.exists(safety_file):
                with open(safety_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                return content
            else:
                return "安全评估文件不存在"
        except Exception as e:
            return f"读取安全评估文件时出错: {str(e)}"

    def _extract_image_path(self, query: str):
        """从查询中提取图像路径"""
        image_pattern = r"[Ii]mage:\s*([^\s]+)"
        match = re.search(image_pattern, query)
        return match.group(1) if match else None

    def _analyze_and_decide(self, safety_result: str, image_path: str = None):
        """基于安全结果和图像分析做出飞行决策"""
        # 简单的决策逻辑，可以根据需要扩展
        safety_lower = safety_result.lower()
        
        if "适合降落" in safety_result or "safe to land" in safety_lower:
            return "land", {}
        elif "不适合降落" in safety_result or "not safe" in safety_lower:
            return "hover", {"duration": 5.0}
        elif "谨慎降落" in safety_result or "caution" in safety_lower:
            return "descend", {"distance": 2.0}
        elif "障碍物" in safety_result or "obstacle" in safety_lower:
            return "ascend", {"distance": 5.0}
        elif "向前" in safety_result or "forward" in safety_lower:
            return "move_forward", {"distance": 3.0}
        elif "向后" in safety_result or "backward" in safety_lower:
            return "move_backward", {"distance": 3.0}
        elif "向左" in safety_result or "left" in safety_lower:
            return "move_left", {"distance": 3.0}
        elif "向右" in safety_result or "right" in safety_lower:
            return "move_right", {"distance": 3.0}
        else:
            # 默认悬停
            return "hover", {"duration": 3.0}

if __name__ == "__main__":
    FlightBrain_Agent.main() 