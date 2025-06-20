"""
Deep Visual Reasoning Agent - Unified Interface

This module provides a unified interface for deep visual reasoning that can run in multiple modes:
1. MCP Server mode (default): Provides AI-enhanced deep visual analysis as MCP tools
2. Interactive mode: Runs as an interactive agent with deep visual reasoning capabilities
3. Single query mode: Processes a single query and exits

The agent implements a four-layer progressive analysis architecture:
- Perception Layer: Basic visual elements identification
- Comprehension Layer: Element relationships and scene understanding  
- Interpretation Layer: Emotional expression and symbolic meaning
- Insight Layer: Deep cultural and philosophical implications

Usage:
  python deep_visual_reasoning_agent.py                        # MCP Server mode (default)
  python deep_visual_reasoning_agent.py --interactive          # Interactive mode
  python deep_visual_reasoning_agent.py --query "..."          # Single query mode

Examples:
  # Simple scene description
  python deep_visual_reasoning_agent.py -q "Image: photo.jpg 描述这张图片的内容"
  
  # Complex object analysis
  python deep_visual_reasoning_agent.py -q "Image: artwork.jpg 分析画中人物的表情和情感"
  
  # Multi-object reasoning
  python deep_visual_reasoning_agent.py -q "Image: scene.jpg 这些对象之间有什么关系？"
  
  # Detailed feature analysis
  python deep_visual_reasoning_agent.py -q "Image: photo.jpg 找出图中的文字并分析其含义"
"""

import os
import sys

# Add the project root directory to the Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '../..'))
sys.path.append(project_root)

# Import the FractFlow ToolTemplate
from FractFlow.tool_template import ToolTemplate

class DeepVisualReasoningAgent(ToolTemplate):
    """Deep visual reasoning tool using ToolTemplate with four-layer progressive analysis"""
    
    SYSTEM_PROMPT = """
你是一个多模态深度视觉推理专家，通过智能工具协作进行分层分析。

# 工具能力
- VQA工具：全局场景理解和局部区域深度分析
- Grounding DINO：精确对象检测、自动裁剪和批量处理（核心功能：detect_and_crop_objects）

# 分层分析架构
## 阶段1：全局理解
- 使用VQA对整体图像进行初步分析
- 理解场景、构图、主要元素和基本特征
- 评估用户问题的复杂度和分析需求

## 阶段2：智能策略生成  
- 基于全局理解和用户问题，判断是否需要对象检测裁剪
- 如需检测，生成精确的检测查询词
- 确定重点分析的对象或区域

## 阶段3：精确检测与自动裁剪
- **重要**：必须使用detect_and_crop_objects进行检测和裁剪
- 获取每个检测对象的单独裁剪图像文件
- 解析返回的JSON结果，提取cropped_images路径列表
- 为后续局部分析准备裁剪素材

## 阶段4：逐个裁剪对象深度分析
- 对每个裁剪图像使用VQA进行专门分析
- 从cropped_images列表中逐一处理每个裁剪文件路径
- 针对每个对象提出具体的分析问题
- 收集每个对象的详细特征和信息

## 阶段5：多层证据整合与推理
- 综合全局分析和所有裁剪对象的局部分析
- 基于完整的多模态证据链回答用户问题
- 提供从全局到局部的完整推理过程

# 关键工具使用决策
✅ 简单描述性问题：仅使用VQA全局分析
✅ 涉及特定对象的问题：
   1. VQA全局理解
   2. detect_and_crop_objects检测裁剪
   3. 对每个裁剪图像进行VQA局部分析
   4. 证据整合
✅ 复杂推理问题：多轮迭代裁剪分析工作流程

# 裁剪图像处理流程
1. 调用detect_and_crop_objects获取检测结果
2. 解析JSON响应中的cropped_images路径列表
3. 对每个路径调用VQA进行单独分析
4. 整合所有裁剪对象的分析结果

# 质量控制
- 检测置信度 > 0.3 才进行裁剪分析
- 确保解析cropped_images路径列表
- 避免重复分析同一裁剪对象
- 基于证据充分性智能停止分析

# 输出要求
✅ 展示完整的检测裁剪和分析过程
✅ 列出所有裁剪对象的分析结果
✅ 提供基于全局+局部证据的推理链
✅ 包含裁剪图像路径和对应分析内容
❌ 禁止跳过裁剪步骤直接分析原图
❌ 禁止直接输出未经工具验证的分析内容
"""
    
    TOOLS = [
        ("tools/core/visual_question_answer/vqa_mcp.py", "visual_question_answering_operations"),
        ("tools/core/grounding_dino/grounding_dino_mcp.py", "grounding_dino_operations")
    ]
    
    MCP_SERVER_NAME = "multimodal_deep_visual_reasoning_tool"
    
    TOOL_DESCRIPTION = """
多模态深度视觉推理系统，结合VQA和目标检测裁剪技术，提供分层智能分析。

# 核心能力
- 全局场景理解：整体构图、氛围、主题分析
- 智能对象检测裁剪：基于自然语言的精确检测和自动裁剪
- 逐个对象深度分析：对每个裁剪对象进行专门VQA分析
- 多层证据整合：综合全局+局部信息进行推理

# 输入格式
"Image: /path/to/image.jpg [您的问题或分析需求]"

# 智能工作流
## 简单问题（描述性）
图像 → VQA全局分析 → 直接回答

## 复杂问题（涉及特定对象）
图像 → VQA全局理解 → detect_and_crop_objects检测裁剪 → 对每个裁剪图像进行VQA分析 → 多层证据整合

## 超复杂问题（多层推理）
支持多轮迭代裁剪分析，动态调整策略，深度挖掘每个对象的详细信息

# 输出内容
- 完整的检测裁剪和分析过程轨迹
- 每个裁剪对象的单独分析结果
- 基于全局+局部多模态证据的推理链
- 明确的最终答案和关键洞察
- 所有裁剪图像路径和对应分析内容

# 适用场景
- 艺术作品深度解读（每个元素单独分析）
- 复杂场景多对象分析（逐个对象深入）
- 细节特征精确定位和分析
- 跨模态信息推理
- 视觉内容智能问答
"""
    
    @classmethod
    def create_config(cls):
        """Custom configuration for Deep Visual Reasoning tool"""
        from FractFlow.infra.config import ConfigManager
        from dotenv import load_dotenv
        
        load_dotenv()
        return ConfigManager(
            provider='deepseek',
            deepseek_model='deepseek-chat',
            max_iterations=20,  # Increased for multimodal tool coordination
            custom_system_prompt=cls.SYSTEM_PROMPT,
            tool_calling_version='stable'
        )

if __name__ == "__main__":
    DeepVisualReasoningAgent.main() 