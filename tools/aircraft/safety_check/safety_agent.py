# python safety_agent.py --query "Image:path/to/FractFlow-Aircraft/tools/aircraft/sam/tmp/test_boundary_cropped.png"
# python safety_agent.py --query "Image:path/to/FractFlow-Aircraft/tools/aircraft/sam/tmp/test_boundary.png"

import os
import sys
from PIL import Image

# Add the project root directory to the Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '../..'))
sys.path.append(project_root)

# Import the FractFlow ToolTemplate
# from FractFlow.tool_template import ToolTemplate
from aircraft_tool_template import AircraftToolTemplate

from qwen.qwen_utils import qwen_tool, QwenClient

class Safety_Agent(AircraftToolTemplate):
    """Landing Safety Checking Tool using ToolTemplate"""
    
    SYSTEM_PROMPT = """
你是一个为微软模拟飞行（MSFS）设计的、专业的视觉分析与降落安全评估AI。你的核心任务是分析用户提供的、带有一个红色boundary的图像，判断boundary内的区域是否为 Joby S4 eVTOL 的合规且安全的降落点。

# 核心能力
- **识别方框内容:** 准确描述用户输入图像中红色boundary内的地物类型。
- **判断直升机停机坪:** 核心能力是判断方框内的地物是否为一个明确的直升机停机坪（Helipad / Vertiport），通常带有'H'标志或特定几何图案。
- **评估周边环境:** 在确认目标为直升机停机坪后，分析该直升机停机坪紧邻周边的障碍物，如建筑物、树木、塔吊、电线等。
- **提供综合决策:** 基于以上视觉分析，给出明确、有理有据的降落决策。

# 重要限制
- **硬性降落规则：降落点必须是明确可识别的直升机停机坪。** 非直升机停机坪例子：屋顶、公路、草地、停车场、游泳池等，即使区域平坦开阔，必须评估为"不适合降落"。
- **仅在降落点是标准直升机停机坪时，才进行周边环境评估，评估为"适合降落"或"谨慎降落"**
- 评估结果**仅适用**于微软模拟飞行中的 **Joby S4 eVTOL**。
- 所有判断**严格基于**图像中的视觉信息，无法感知图像外的危险或实时的天气变化。
- 最终的飞行安全**始终由操作飞行模拟器的用户（飞行员）负责**。

# 工作流程
1.  接收一张带有红色boundary的图片作为唯一输入。
2.  **第一步：识别与描述。** 首先，仔细观察红色boundary内的内容，并用一句话进行清晰描述。
3.  **第二步：判断与决策分支。**
    * 判断方框内的内容**是否为一个直升机停机坪**。
    * **如果不是直升机停机坪**，立即中止后续分析，直接生成"不适合降落"的评估报告。
    * **如果是直升机停机坪**，则继续进行第三步的深入分析。
4.  **第三步：周边风险评估。** 重点评估该直升机停机坪的**紧邻四周**：
    * 是否存在高出停机坪平面的建筑物结构？
    * 是否有高大的树木侵入进近/撤离空间？
    * 是否有其他潜在障碍物（如天线、栏杆、空调外机等）？
5.  **第四步：生成报告。** 综合所有分析结果，按照下述【输出格式要求】生成结构化的评估报告。

# 输出格式要求
你的回复必须严格包含以下几个部分，且顺序不能改变：
- **方框内容识别:** [对红色boundary内内容的简短、精确的文字描述。例如："一个位于楼顶的圆形直升机停机坪，中心有H标志。"或"一片城市公园的草地。"]
- **评估结果:** [必须是 "**适合降落**"、"**谨慎降落**" 或 "**不适合降落**" 三者之一。]
- **核心理由:** [使用项目符号（bullet points）列出决策的关键依据。第一条必须说明目标是否为直升机停机坪。]
- **潜在风险与建议:** [如果适合或谨慎降落，说明周围环境的主要风险点并提供建议。如果不适合，则说明核心原因。]
- **限制声明:** [每一次回复的结尾都必须附带标准的安全免责声明。]

始终将飞行安全置于首位，提供专业、果断的视觉评估，以辅助飞行员做出最安全的决策。
"""

    SYSTEM_PROMPT_LOCAL = """
你是MSFS的视觉分析AI，负责评估Joby S4 eVTOL的降落安全性。

# 工作流程
1. **识别红色boundary内部的内容**：描述方框内的地物类型，忽略方框外的内容，特别是边框外的停机坪
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
- **方框内容识别：** [简短描述]
- **评估结果：** [适合降落/谨慎降落/不适合降落]
- **核心理由：** [项目符号列出关键依据]
- **潜在风险与建议：** [风险点和建议]
- **限制声明：** “最终的飞行安全始终由操作飞行模拟器的用户（飞行员）负责”
"""

    TOOLS = [
        ("tools/aircraft/safety_check/safety_vlm.py", "landing_safety_check_operations")
    ]
    
    MCP_SERVER_NAME = "landing_safety_checker"
    
    TOOL_DESCRIPTION = """ Check if a marked landing spot in a image is save.
    
    Parameters:
        query: image - The path of the marked image to be checked, (e.g., "Image: /path/photo.jpg")
        
    Returns:
        str - The safety level (Green, Yellow, Red) of the designated landing spot and its reasoning.
        
    Note: Requires accessible image files, automatically resized to 512x512.
    """
    
    client = QwenClient(server_url="http://10.30.58.120:5001")

    @classmethod
    def create_config(cls):
        """Custom configuration for tool"""
        from FractFlow.infra.config import ConfigManager
        from dotenv import load_dotenv
        
        load_dotenv()
        return ConfigManager(
            provider='qwen',
            qwen_model='qwen-vl-max',
            max_iterations=5,  # Visual analysis usually completes in one iteration
            custom_system_prompt=cls.SYSTEM_PROMPT,
            tool_calling_version='turbo'
        )
    
    async def run_online(self):
        query = "Image:./sam/tmp/cropped_image.png"
        result = await super()._run_online(query)
        
        # 保存结果到文件
        os.makedirs("./sam/tmp", exist_ok=True)
        with open("./sam/tmp/safety_result.txt", "w", encoding="utf-8") as f:
            f.write(result)
        
        return result
    
    async def run_local(self):
        image_url = "./sam/tmp/cropped_image.png"
        image = Image.open(image_url)
        text_prompt = "请分析图中红色boundary区域的是否属于标准停机坪。"
        result = await qwen_tool(self.client, image, text_prompt, self.SYSTEM_PROMPT_LOCAL, max_new_tokens=1024)
        
        # 保存结果到文件
        os.makedirs("./sam/tmp", exist_ok=True)
        with open("./sam/tmp/safety_result.txt", "w", encoding="utf-8") as f:
            f.write(result)
        
        return result
    
if __name__ == "__main__":
    Safety_Agent.main() 