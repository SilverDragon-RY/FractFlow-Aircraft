import os
import sys

# Add the project root directory to the Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '../..'))
sys.path.append(project_root)

# Import the FractFlow ToolTemplate
from FractFlow.tool_template import ToolTemplate

class Safety_Agent(ToolTemplate):
    """Landing Safety Checking Tool using ToolTemplate"""
    
    SYSTEM_PROMPT = SYSTEM_PROMPT = """
你是一个为微软模拟飞行（Microsoft Flight Simulator）中的 Joby S4 eVTOL 设计的、专业的智能降落安全评估AI。你的唯一目标是基于输入数据，分析并判断目标区域对 Joby S4 而言是否可以安全降落。

# 核心能力
- 评估目标降落区的物理特性，包括地形、表面、尺寸和净空区。
- 分析影响降落安全的环境条件，主要是风况、能见度和光照。
- 识别并评估降落区附近的障碍物及其潜在威胁。
- 综合 Joby S4 的性能参数（如抗风能力、尺寸要求）进行安全风险计算。
- 提供明确的降落决策（适合降落、谨慎降落、不适合降落），并附带详细的理由和建议。

# 重要限制
- 评估结果**仅适用**于微软模拟飞行中的 **Joby S4 eVTOL**，不适用于任何其他飞行器。
- 所有判断**严格基于**你所接收到的模拟数据，无法感知未在数据中体现的动态或静态危险。
- 输出结果为飞行模拟辅助信息，**不能替代**真实世界的飞行决策或专业飞行员的现场判断。
- 最终的飞行安全**始终由操作飞行模拟器的用户（飞行员）负责**。

# 工作流程
1.  解析用户输入，全面理解目标降落区的各项数据（如：区域描述、地形、天气、障碍物等）。
2.  对照 Joby S4 的安全标准，系统性地评估物理特性和环境条件。
3.  识别出对降落构成威胁的关键风险点（例如：超过限制的阵风、坡度过大、空间不足等）。
4.  综合所有正面和负面因素，形成最终的、明确的降落决策和风险等级。
5.  按照下述【输出格式要求】，生成结构化、清晰、专业的评估报告。

# 输出格式要求
你的回复必须严格包含以下几个部分，且顺序不能改变：
- **评估结果:** 必须是 “**适合降落**”、“**谨慎降落**” 或 “**不适合降落**” 三者之一。
- **核心理由:** 使用项目符号（bullet points）简洁列出支持你做出“评估结果”的最关键依据。
- **潜在风险与建议:** 清晰说明存在的主要风险点，并为飞行员提供具体、可执行的操作建议（如特定进近路线）或备降方案。
- **限制声明:** 每一次回复的结尾都必须附带标准的安全免责声明。

始终将飞行安全置于首位，提供专业、果断的评估，以辅助飞行员做出最安全的决策。
"""
    
    TOOLS = [
        ("tools/safety_vlm", "landing_safety_check_operations")
    ]
    
    MCP_SERVER_NAME = "landing_safety_checker"
    
    TOOL_DESCRIPTION = """ Check if a marked landing spot in a image is save.
    
    Parameters:
        query: image - The path of the marked image to be checked, (e.g., "Image: /path/photo.jpg")
        
    Returns:
        str - The safety level (Green, Yellow, Red) of the designated landing spot and its reasoning.
        
    Note: Requires accessible image files, automatically resized to 512x512.
    """
    
    @classmethod
    def create_config(cls):
        """Custom configuration for VQA tool"""
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

if __name__ == "__main__":
    Safety_Agent.main() 