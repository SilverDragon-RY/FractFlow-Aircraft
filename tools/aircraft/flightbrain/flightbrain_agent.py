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
你是一个智能飞行大脑，负责为 Joby S4 eVTOL 执行飞行决策。你的任务是调用flight_decision工具获得飞行决策，并根据飞行决策调用相应的飞行操作工具，把飞机降落到直升机停机坪上。

# 工作流程
1. 首先，调用flight_decision工具，获得飞行决策
2. 然后，根据flight_decision工具的决策，调用相应的飞行操作工具，把飞机降落到直升机停机坪上。

# 可用的飞行决策工具
- flight_decision: 飞行决策

# 可用的飞行操作工具
- hover: 悬停
- hover_turn(time_s): 悬停转弯
- move_forward(time_s): 前进
- move_backward(time_s): 后退
- move_left(time_s): 左移
- move_right(time_s): 右移
- move_ascend(time_s): 上升
- move_descend(time_s): 下降
- hover_turn_left(time_s): 悬停向左掉头
- hover_turn_right(time_s): 悬停向右掉头


# 重要
- 必须：每次调用工具时，必须先调用flight_decision工具，获得飞行决策，再根据飞行决策调用相应工具。
- time_s: 工具的执行时间，单位为秒。
"""


    TOOLS = [
        ("tools/aircraft/flightbrain/flightbrain_mcp.py", "flightbrain_operations"),
        # ("tools/aircraft/safety_check/safty_mcp.py", "safety_vlm_operations")
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
            max_iterations=50,
            custom_system_prompt=cls.SYSTEM_PROMPT,
            tool_calling_version='turbo'
        )

    # def _read_safety_result(self):
    #     """读取安全评估结果文件"""
    #     try:
    #         safety_file = "/home/bld/dyx/FractFlow-Aircraft/tools/aircraft/sam/tmp/safety_result.txt"
    #         if os.path.exists(safety_file):
    #             with open(safety_file, 'r', encoding='utf-8') as f:
    #                 content = f.read()
    #             return content
    #         else:
    #             return "安全评估文件不存在"
    #     except Exception as e:
    #         return f"读取安全评估文件时出错: {str(e)}"

    # def _extract_image_path(self, query: str):
    #     """从查询中提取图像路径"""
    #     image_pattern = r"[Ii]mage:\s*([^\s]+)"
    #     match = re.search(image_pattern, query)
    #     return match.group(1) if match else None

    # def _analyze_and_decide(self, safety_result: str, image_path: str = None):
    #     """基于安全结果和图像分析做出飞行决策"""
    #     # 简单的决策逻辑，可以根据需要扩展
    #     safety_lower = safety_result.lower()
        
    #     if "适合降落" in safety_result or "safe to land" in safety_lower:
    #         return "land", {}
    #     elif "不适合降落" in safety_result or "not safe" in safety_lower:
    #         return "hover", {"duration": 5.0}
    #     elif "谨慎降落" in safety_result or "caution" in safety_lower:
    #         return "descend", {"distance": 2.0}
    #     elif "障碍物" in safety_result or "obstacle" in safety_lower:
    #         return "ascend", {"distance": 5.0}
    #     elif "向前" in safety_result or "forward" in safety_lower:
    #         return "move_forward", {"distance": 3.0}
    #     elif "向后" in safety_result or "backward" in safety_lower:
    #         return "move_backward", {"distance": 3.0}
    #     elif "向左" in safety_result or "left" in safety_lower:
    #         return "move_left", {"distance": 3.0}
    #     elif "向右" in safety_result or "right" in safety_lower:
    #         return "move_right", {"distance": 3.0}
    #     else:
    #         # 默认悬停
    #         return "hover", {"duration": 3.0}

if __name__ == "__main__":
    FlightBrain_Agent.main() 