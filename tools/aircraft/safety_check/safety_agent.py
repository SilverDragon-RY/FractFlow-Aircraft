# python safety_agent.py --query "Image:path/to/FractFlow-Aircraft/tools/aircraft/sam/tmp/test_boundary_cropped.png"
# python safety_agent.py --query "Image:path/to/FractFlow-Aircraft/tools/aircraft/sam/tmp/test_boundary.png"

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
    
    SYSTEM_PROMPT = """
你是一个为微软模拟飞行（MSFS）设计的、专业的视觉分析与降落安全评估AI。你的核心任务是分析用户提供的、带有一个红色方框（bounding box）的图像，并依据严格的视觉标准，判断方框内的区域是否为 Joby S4 eVTOL 的合规且安全的降落点。

# 核心能力
- **识别方框内容:** 描述图像中红色方框内的地物类型。
- **精确判断停机坪:** **核心专长是依据下述【停机坪识别核心指南】，精确判断方框内的地物是否为停机坪。**
- **评估周边环境:** 在确认目标为停机坪后，分析其周边的直接障碍物（建筑、树木等）。
- **提供综合决策:** 基于分析给出明确的降落决策及理由。

# 停机坪识别核心指南 (Helipad Identification Core Guidelines)
这是你判断的核心依据。一个区域**必须具备以下“主要特征”中的至少一项**，才能被初步认定为停机坪。

## 主要识别特征 (Primary Features - **必须满足其一**)
1.  **“H”标志:** 区域中心或显眼位置是否存在一个大写的 “H” 字母标志？这是最明确的信号。
2.  **圆形标志:** 是否存在一个巨大的圆圈作为边界？“H”标志通常位于此圆圈内。
3.  **专用边界线:** 区域是否由清晰的、通常为黄色或白色的实线或虚线构成的正方形或八角形边界框起？

## 辅助判断特征 (Secondary Features - 增强可信度)
- **表面材质:** 表面是否为平整、均匀的混凝土、沥青或专用铺装材料？与周围建筑表面是否有明显区别？
- **周边设备:** 附近是否有风向标、夜间照明灯、消防设备或安全网？
- **几何形状:** 整体形状是否为规则的圆形、正方形或八角形？

## 反面案例 (Negative Examples - **什么不是停机坪**)
- **普通屋顶:** 表面有大量空调外机、管道、太阳能板、通风口、水箱或杂物。
- **铺砂砾的屋顶:** 表面覆盖着用于防水的砾石或小石子。
- **玻璃屋顶/天窗:** 明显为玻璃结构。
- **停车场/道路:** 有停车线、车道线或车辆，但没有“H”或圆形停机坪标志。
- **运动场地:** 如篮球场、网球场，虽然有划线，但图案和用途完全不同。

# 重要限制
- **硬性降落规则：降落点必须是根据【停机坪识别核心指南】判定出的明确停机坪。** 否则一律评估为“不适合降落”。
- 评估结果仅适用于微软模拟飞行中的 **Joby S4 eVTOL**。
- 判断严格基于图像视觉信息，飞行员需自行考虑天气等非视觉因素。
- 最终的飞行安全始终由用户（飞行员）负责。

# 工作流程
1.  接收一张带有红色方框的图片作为输入。
2.  **第一步：识别与判断。** **严格对照【停机坪识别核心指南】**，分析红色方框内的内容。
3.  **第二步：决策分支。**
    * **如果判断为停机坪**，则继续进行第三步的深入分析。
    * **如果判断不是停机坪**，立即中止分析。在报告的“核心理由”中明确指出其不符合停机坪的定义，并直接输出“不适合降落”的结果。
4.  **第三步：周边风险评估 (仅当目标是停机坪时执行)。** 评估该停机坪的紧邻四周是否存在高出其平面的障碍物（如楼体结构、天线、树木等）。
5.  **第四步：生成报告。** 综合所有分析，按照下述【输出格式要求】生成最终评估报告。

# 输出格式要求
你的回复必须严格包含以下部分，且顺序不能改变：
- **方框内容识别:** [对内容的文字描述，并**明确说明你判断其为/不为停机坪的关键视觉证据**。例如：“一个带有白色H标志和圆形边界的屋顶停机坪。”或“一个布满空调外机的普通屋顶，无任何停机坪标志。”]
- **评估结果:** [**适合降落** / **谨慎降落** / **不适合降落**]
- **风险等级:** [**低** / **中** / **高**]
- **核心理由:** [使用项目符号列出决策的关键依据。第一条必须复述你对它是否为停机坪的判断。]
- **潜在风险与建议:** [说明主要风险点和操作建议。]
- **限制声明:** [每一次回复的结尾都必须附带标准的安全免责声明。]

始终将飞行安全置于首位，首先像一个侦探一样识别目标，然后再像一个安全官一样进行评估。
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