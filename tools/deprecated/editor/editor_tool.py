"""
Editor Tool - Unified Interface

This module provides a unified interface for file editing operations that can run in multiple modes:
1. MCP Server mode (default): Provides AI-enhanced file editing as MCP tools
2. Interactive mode: Runs as an interactive agent with file editing capabilities
3. Single query mode: Processes a single query and exits

Usage:
  python editor_tool.py                        # MCP Server mode (default)
  python editor_tool.py --interactive          # Interactive mode
  python editor_tool.py --query "..."          # Single query mode
"""

import os
import sys

# Add the project root directory to the Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '../..'))
sys.path.append(project_root)

# Import the FractFlow ToolTemplate
from FractFlow.tool_template import ToolTemplate

class EditorTool(ToolTemplate):
    """File editor tool using ToolTemplate"""
    
    SYSTEM_PROMPT = """
你是一个专业的文件编辑助手，专门负责帮助用户进行各种文本文件的编辑和处理操作。

# 核心功能
- 文件读取：读取文件内容，支持整个文件、指定行范围、带行号显示
- 文件编写：创建新文件、覆盖现有文件、追加内容、在指定位置插入内容
- 文件管理：检查文件是否存在、统计行数、删除指定行
- 内容编辑：搜索和替换文本、格式化代码、文本处理

# 工作流程

## 文件编辑工作流
1. 在编辑文件之前，总是先检查文件是否存在
2. 对于修改操作，先读取当前内容以了解文件结构
3. 根据需求选择最合适的操作：
   - 创建/覆盖：使用create_file
   - 追加内容：使用append_content
   - 指定位置插入：使用insert_content_at_line
   - 删除内容：使用delete_line_at

## 大文件处理
1. 首先通过统计行数检查文件大小
2. 如果文件很大（>1000行），采用分块策略：
   - 将文件分成可管理的块（500-1000行）
   - 分块处理每个部分，同时保持上下文
   - 考虑在块之间使用重叠（50-100行）以获得更好的连续性

# 输出格式要求
你的回复应该包含以下结构化信息：
- operation_result: 执行的编辑操作的详细描述
- file_content: 相关的文件内容（当执行读取操作时）
- success: 操作是否成功完成
- message: 关于编辑过程的补充信息和说明

# 注意事项
- 行号从1开始计数（第一行是第1行）
- 在进行任何文件修改之前，务必确认文件路径正确
- 对于代码文件，保持适当的缩进和格式
- 提供清晰的操作反馈和错误处理

始终为每个任务选择最高效和合适的工具。
"""
    
    TOOLS = [
        ("server.py", "editor")
    ]
    
    MCP_SERVER_NAME = "editor_tool"
    
    TOOL_DESCRIPTION = """
    Performs intelligent file editing operations based on natural language requests.

This tool can read, write, modify, and manage text files through natural language commands. It supports complex editing workflows including content insertion, replacement, and file management.

Input format:
- Natural language descriptions of file editing operations
- Can include specific file paths and content specifications
- May request complex multi-step editing workflows
- Supports both simple text files and code files

Returns:
- 'operation_result': Description of the editing operation performed
- 'file_content': Relevant file content (when reading)
- 'success': Boolean indicating operation completion
- 'message': Additional information about the editing process

Examples:
- "Read the first 10 lines of config.txt"
- "Create a new Python file with a basic class structure"
- "Insert a comment at line 25 in main.py"
- "Replace all occurrences of 'old_function' with 'new_function'"
- "Delete lines 15-20 from data.txt"
- "Append logging configuration to the end of settings.py"
    """
    
    @classmethod
    def create_config(cls):
        """Custom configuration for Editor tool"""
        from FractFlow.infra.config import ConfigManager
        from dotenv import load_dotenv
        
        load_dotenv()
        return ConfigManager(
            provider='deepseek',
            deepseek_model='deepseek-chat',
            max_iterations=15,  # File editing may require multiple steps
            custom_system_prompt=cls.SYSTEM_PROMPT,
            tool_calling_version='turbo'
        )

if __name__ == "__main__":
    EditorTool.main() 