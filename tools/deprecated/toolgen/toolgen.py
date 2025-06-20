#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Tool Generator (toolgen)

Generates FractFlow tools based on user-provided source code.
Uses FractFlow Agent to analyze source code and generate appropriate MCP tools.
"""

import os
import sys
import re
import argparse
import shutil
import asyncio
import logging
from typing import Dict, Any, List, Optional, Tuple, Set
import jinja2
import json

from dotenv import load_dotenv
from FractFlow.agent import Agent

# Constants
TEMPLATE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "templates")
PROMPT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "prompts")

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class ToolgenAgent:
    """
    Handles agent-related functionality for the toolgen module.
    Creates and manages FractFlow agents for code analysis and generation.
    """
    
    def __init__(self):
        """Initialize the ToolgenAgent."""
        self.agent = None
        self.prompt_types = {
            # Original prompts
            "server_generation": self._load_prompt("server_generation.txt"),
            "ai_server_generation": self._load_prompt("ai_server_generation.txt"),
            
            # New specialized prompts (will need to be created)
            "function_extraction": self._load_prompt_or_fallback("function_extraction.txt", "server_generation.txt"),
            "tool_info_inference": self._load_prompt_or_fallback("tool_info_inference.txt", "server_generation.txt"),
            "system_prompt_generation": self._load_prompt_or_fallback("system_prompt_generation.txt", "ai_server_generation.txt"),
            "tool_docstring_generation": self._load_prompt_or_fallback("tool_docstring_generation.txt", "ai_server_generation.txt"),
            "run_server_prompt_generation": self._load_prompt_or_fallback("run_server_prompt_generation.txt", "ai_server_generation.txt")
        }
    
    def _load_prompt(self, prompt_name: str) -> str:
        """Load a prompt template from the prompts directory."""
        prompt_path = os.path.join(PROMPT_DIR, prompt_name)
        with open(prompt_path, 'r', encoding='utf-8') as f:
            return f.read()
    
    def _load_prompt_or_fallback(self, primary_prompt_name: str, fallback_prompt_name: str) -> str:
        """
        Load a prompt template, falling back to another if the primary doesn't exist.
        This allows gradual addition of specialized prompts while maintaining compatibility.
        """
        primary_path = os.path.join(PROMPT_DIR, primary_prompt_name)
        
        if os.path.exists(primary_path):
            return self._load_prompt(primary_prompt_name)
        else:
            logger.info(f"Specialized prompt {primary_prompt_name} not found, falling back to {fallback_prompt_name}")
            return self._load_prompt(fallback_prompt_name)
    
    async def create_agent(self, prompt_type: str) -> None:
        """
        Create and initialize a FractFlow Agent with the specified prompt type.
        
        Args:
            prompt_type: The type of prompt to use ('server_generation' or 'ai_server_generation')
        """
        # Create a new agent
        self.agent = Agent('code_analysis_agent')
        
        # Configure the agent
        config = self.agent.get_config()
        config['agent']['provider'] = 'deepseek'  # Default to deepseek
        config['deepseek']['api_key'] = os.getenv('DEEPSEEK_API_KEY')
        config['qwen']['api_key'] = os.getenv('QWEN_API_KEY')  # As fallback
        
        # Use qwen if deepseek key is not available
        if not config['deepseek']['api_key'] and config['qwen']['api_key']:
            config['agent']['provider'] = 'qwen'
            logger.info("Using Qwen as the model provider")
        elif not config['deepseek']['api_key']:
            raise ValueError("No API key found for DeepSeek or Qwen. Please set environment variables.")
        
        # Further configuration
        config['deepseek']['model'] = 'deepseek-chat'
        config['agent']['max_iterations'] = 5
        config['agent']['custom_system_prompt'] = self.prompt_types[prompt_type]
        config['tool_calling']['version'] = 'turbo'
        
        # Apply configuration
        self.agent.set_config(config)
        
        # Initialize the agent
        logger.info(f"Initializing code analysis agent with {prompt_type} prompt...")
        await self.agent.initialize()
    
    async def shutdown(self):
        """Shutdown the agent properly."""
        if self.agent:
            await self.agent.shutdown()
            self.agent = None
    
    def _extract_code_from_response(self, response: str) -> str:
        """Extract code blocks from an Agent response."""
        code_pattern = r'```(?:python)?\s*([\s\S]*?)\s*```'
        matches = re.findall(code_pattern, response)
        
        if matches:
            # Return the largest code block
            return max(matches, key=len).strip()
        
        # If no code blocks found, return the original response
        return response.strip()
    
    async def _query_agent(self, query: str) -> str:
        """
        Send a query to the agent and return the response.
        
        Args:
            query: The query to send to the agent
            
        Returns:
            The agent's response
        """
        if not self.agent:
            raise ValueError("Agent not initialized. Call create_agent() first.")
        
        response = await self.agent.process_query(query)
        return response.strip()
    
    async def extract_function_names(self, source_code: str) -> List[str]:
        """
        Analyze source code and identify functions that should be exposed as tools.
        
        Args:
            source_code: Source code to analyze
            
        Returns:
            List of function names
        """
        # Create agent with specialized prompt if not already created
        if not self.agent or self.agent.get_config()['agent']['custom_system_prompt'] != self.prompt_types["function_extraction"]:
            if self.agent:
                await self.shutdown()
            await self.create_agent("function_extraction")
            
        query = f"""
        深入分析以下源代码，确定应该作为工具暴露的函数:
        
        ```python
        {source_code}
        ```
        
        请考虑以下几点:
        1. 工具功能应该低耦合、高内聚
        2. 工具名称和功能应该容易被大语言模型理解
        3. 工具接口应该清晰，避免歧义
        4. 可能需要将现有函数分解、组合或重新命名以创建更合理的工具
        5. 请从用户角度考虑，不一定所有函数都要暴露为工具，请根据实际情况决定
        
        返回一个逗号分隔的函数名列表，这些函数应该在 server.py 中实现。
        如果建议创建新的工具函数，请提供建议的新函数名和说明。
        仅返回函数名列表，不要包含其他解释。
        """
        
        response = await self._query_agent(query)
        
        # Extract and clean function names
        functions = [f.strip() for f in response.split(',')]
        # Remove any empty strings and duplicates while preserving order
        seen = set()
        unique_functions = []
        for f in functions:
            if f and f not in seen:
                seen.add(f)
                unique_functions.append(f)
        
        logger.info(f"Identified functions to implement as tools: {unique_functions}")
        return unique_functions
    
    async def generate_server_components(self, source_code: str, source_module: str) -> Dict[str, str]:
        """
        Generate components for server.py based on source code analysis.
        
        Args:
            source_code: Source code to analyze
            source_module: Name of the source module to import from
            
        Returns:
            Dictionary with components for server.py template
        """
        # Extract function names to import
        functions = await self.extract_function_names(source_code)
        function_list = ', '.join(functions)
        
        query = f"""
        为 {source_module} 模块设计和实现高质量的 MCP 工具代码:

        源代码中的相关函数: {function_list}
        
        设计指南:
        1. 每个工具函数应添加 "tool_" 前缀以避免命名冲突
        2. 深入思考如何设计这些工具，使它们:
           - 功能明确、低耦合、高内聚
           - 易于被大语言模型理解和使用
           - 接口清晰，参数直观
           - 组合使用时能高效解决复杂问题
        3. 可以基于原始函数创建新的工具函数，如果这样能提高使用体验
        4. 提供详细的文档，帮助大语言模型理解如何使用这些工具
        5. 重要：源文件 {source_module}.py 和生成的server.py位于同一目录下
        6. 关键: 每个工具函数必须使用 @mcp.tool() 装饰器修饰

        请实现两部分代码：

        1. 源码特定导入部分（注意：基础导入如 os, sys, typing, mcp 等已在模板中提供，请勿重复）：
           - 从同目录下的 {source_module}.py 文件导入所需的函数
           - 根据源代码分析，添加源码依赖的第三方库导入
           - 仅包含源码特定的导入，不要包含标准库或 MCP 框架的导入

        2. 工具函数定义部分：
           - 所有工具函数的完整实现
           - 详细的文档字符串
           - 必要的类型提示
           - 每个函数前必须添加 @mcp.tool() 装饰器

        重要提醒：
        - 不要导入 os, sys, typing, mcp.server.fastmcp 等，这些已在模板中提供
        - 只导入源码特定的模块和函数
        - 使用绝对导入，不要使用相对导入（如 from .module import）
        - 由于源文件和server.py在同一目录，直接使用模块名导入（如 from {source_module} import）

        请先思考最佳设计方案，再实现代码。返回源码特定导入部分和工具定义部分，不要包括基础导入、初始化FastMCP或运行服务器的代码。
        """
        
        response = await self._query_agent(query)
        full_code = self._extract_code_from_response(response)
        
        # Split the code into source imports and tool definitions
        source_imports, tool_definitions = self._split_code(full_code)
        
        # Clean up source imports to remove any basic imports that might have slipped through
        source_imports = self._clean_source_imports(source_imports)
        
        # Verify and fix missing @mcp.tool() decorators
        tool_definitions = self._ensure_tool_decorators(tool_definitions)
        
        # Extract tool names from generated code
        tool_names = set(re.findall(r'async\s+def\s+([a-zA-Z0-9_]+)\s*\(', tool_definitions))
        
        # Generate available tools list without duplicates
        available_tools = '\n        '.join([f'print("- {name}")' for name in sorted(tool_names)])
        
        return {
            "source_imports": source_imports,
            "tool_definitions": tool_definitions,
            "available_tools": available_tools
        }
    
    def _split_code(self, code: str) -> Tuple[str, str]:
        """Split code into source imports and function definitions."""
        # Try using regex to split
        import_pattern = r'^(.*?)(?=@mcp\.tool\(\)|async\s+def\s+tool_)'
        tool_pattern = r'(@mcp\.tool\(\).*$)'
        
        import_match = re.search(import_pattern, code, re.DOTALL | re.MULTILINE)
        tool_match = re.search(tool_pattern, code, re.DOTALL | re.MULTILINE)
        
        if import_match and tool_match:
            return import_match.group(1).strip(), tool_match.group(1).strip()
        
        # Fallback to line-by-line splitting
        lines = code.split('\n')
        for i, line in enumerate(lines):
            if line.startswith('@mcp.tool()') or (line.startswith('async def tool_')):
                return '\n'.join(lines[:i]).strip(), '\n'.join(lines[i:]).strip()
        
        # If we can't split, return the whole code as tool definitions
        return "", code
    
    def _clean_source_imports(self, imports: str) -> str:
        """
        Clean source imports to remove basic imports that should be in template.
        
        Args:
            imports: Raw import section from agent response
            
        Returns:
            Cleaned imports with basic imports removed
        """
        if not imports.strip():
            return imports
            
        # List of basic imports that should not be in source_imports
        basic_imports = {
            'import os',
            'import sys', 
            'from typing import',
            'from mcp.server.fastmcp import FastMCP',
            'from mcp import',
            'import typing',
            'import asyncio',  # Usually provided in template context
        }
        
        lines = imports.split('\n')
        cleaned_lines = []
        
        for line in lines:
            line_stripped = line.strip()
            
            # Skip empty lines and comments
            if not line_stripped or line_stripped.startswith('#'):
                cleaned_lines.append(line)
                continue
                
            # Check if this line contains any basic import
            is_basic_import = False
            for basic_import in basic_imports:
                if line_stripped.startswith(basic_import):
                    is_basic_import = True
                    break
                    
            # Only keep non-basic imports
            if not is_basic_import:
                cleaned_lines.append(line)
        
        # Remove leading/trailing empty lines
        while cleaned_lines and not cleaned_lines[0].strip():
            cleaned_lines.pop(0)
        while cleaned_lines and not cleaned_lines[-1].strip():
            cleaned_lines.pop()
            
        return '\n'.join(cleaned_lines)
    
    def _ensure_tool_decorators(self, code: str) -> str:
        """
        Ensures that all async def tool_* functions have @mcp.tool() decorators.
        
        Args:
            code: Code string containing tool definitions
            
        Returns:
            Code with @mcp.tool() decorators added where needed
        """
        lines = code.split('\n')
        result = []
        i = 0
        
        while i < len(lines):
            line = lines[i]
            
            # Check if this line defines a tool function
            if re.match(r'^\s*async\s+def\s+tool_', line):
                # Check if the previous line has a decorator
                if i == 0 or not re.match(r'^\s*@mcp\.tool\(\)\s*$', lines[i-1]):
                    # Add the decorator
                    indentation = re.match(r'^\s*', line).group(0)
                    result.append(f"{indentation}@mcp.tool()")
                
            result.append(line)
            i += 1
            
        return '\n'.join(result)
    
    async def generate_component(self, source_code: str, tool_name: str, component_type: str) -> str:
        """
        Generic function to generate different components for the tool.
        
        Args:
            source_code: Source code to analyze
            tool_name: Name of the tool
            component_type: Type of component to generate 
                            ('system_prompt', 'tool_docstring', 'run_server_prompt')
            
        Returns:
            Generated component text
        """
        # Map component types to prompt types
        prompt_type_mapping = {
            'system_prompt': 'system_prompt_generation',
            'tool_docstring': 'tool_docstring_generation',
            'run_server_prompt': 'run_server_prompt_generation'
        }
        
        # Create agent with specialized prompt if needed
        if component_type in prompt_type_mapping:
            specialized_prompt_type = prompt_type_mapping[component_type]
            if not self.agent or self.agent.get_config()['agent']['custom_system_prompt'] != self.prompt_types[specialized_prompt_type]:
                if self.agent:
                    await self.shutdown()
                await self.create_agent(specialized_prompt_type)
        
        prompts = {
            'system_prompt': f"""
                分析源代码和已优化的工具设计，为 {tool_name} 创建一个高效的系统提示:
                
                ```python
                {source_code}
                ```
                
                你的系统提示应当:
                1. 简洁明了地定义 AI 助手的角色和职责
                2. 清晰介绍可用工具及其用途
                3. 提供选择合适工具的指南
                4. 包含常见工作流程和使用模式
                5. 说明如何处理错误情况
                
                重点放在如何有效使用这些工具，而非它们的实现细节。
                使用简短段落和要点列表，提高可读性。
                
                仅返回系统提示文本，不包含任何代码块或其他格式。
            """,
            
            'tool_docstring': f"""
                为 {tool_name} 的主 AI 工具函数创建一个全面但简洁的文档字符串。
                
                函数签名:
                
                async def {tool_name.lower().replace(' ', '_')}_tool(query: str) -> Dict[str, Any]:
                
                文档字符串应当:
                1. 清晰解释工具的总体功能和用途
                2. 描述预期的自然语言查询格式
                3. 说明返回结构和可能的值
                4. 包含 3-5 个多样化的查询示例
                
                避免不必要的冗长，保持信息丰富但简洁。
                仅返回文档字符串文本，不包含函数定义或其他代码。
            """,
            
            'run_server_prompt': f"""
                Based on this source code, generate a concise system prompt for the run_server.py script
                that runs {tool_name}:
                
                ```python
                {source_code}
                ```
                
                The system prompt should be shorter than the one for AI_server.py and should:
                1. Briefly describe what the tool does
                2. Include key capabilities
                3. Provide basic usage guidelines
                
                Return only the system prompt text without any code blocks or other formatting.
            """
        }
        
        if component_type not in prompts:
            raise ValueError(f"Unknown component type: {component_type}")
        
        query = prompts[component_type]
        response = await self._query_agent(query)
        
        # Special handling for docstrings
        if component_type == 'tool_docstring':
            response = self._extract_code_from_response(response)
            if response.startswith('"""') and response.endswith('"""'):
                response = response[3:-3].strip()
        
        return response
    
    async def generate_ai_server_components(self, source_code: str, tool_name: str) -> Dict[str, str]:
        """
        Generate components for AI_server.py.
        
        Args:
            source_code: Source code to analyze
            tool_name: Name of the tool
            
        Returns:
            Dictionary with components for AI_server.py template
        """
        # Generate system prompt and tool docstring concurrently
        system_prompt, tool_docstring = await asyncio.gather(
            self.generate_component(source_code, tool_name, 'system_prompt'),
            self.generate_component(source_code, tool_name, 'tool_docstring')
        )
        
        return {
            "system_prompt": system_prompt,
            "tool_docstring": tool_docstring
        }
    
    async def generate_run_server_components(self, source_code: str, tool_name: str) -> Dict[str, str]:
        """
        Generate components for run_server.py.
        
        Args:
            source_code: Source code to analyze
            tool_name: Name of the tool
            
        Returns:
            Dictionary with components for run_server.py template
        """
        run_server_prompt = await self.generate_component(source_code, tool_name, 'run_server_prompt')
        
        return {
            "run_server_system_prompt": run_server_prompt
        }
    
    async def infer_tool_info(self, source_code: str) -> Dict[str, str]:
        """
        Analyze source code and infer appropriate tool name and description.
        
        Args:
            source_code: Source code to analyze
            
        Returns:
            Dictionary with inferred tool name and description
        """
        # Create agent with specialized prompt if not already created
        if not self.agent or self.agent.get_config()['agent']['custom_system_prompt'] != self.prompt_types["tool_info_inference"]:
            if self.agent:
                await self.shutdown()
            await self.create_agent("tool_info_inference")
            
        query = f"""
        分析以下源代码，推断最合适的工具名称和简短描述:
        
        ```python
        {source_code}
        ```
        
        请考虑:
        1. 源代码的整体功能和目的
        2. 模块和函数的名称和文档字符串
        3. 该工具提供的主要功能和用例
        
        以JSON格式返回，仅包含两个字段:
        {{
          "tool_name": "推断的工具名称",
          "description": "简短的工具描述（不超过100个字符）"
        }}
        
        不要包含任何其他文本或解释。
        """
        
        response = await self._query_agent(query)
        
        try:
            # Try to parse as JSON
            try:
                result = json.loads(response)
            except json.JSONDecodeError:
                # Try to extract JSON from text
                json_match = re.search(r'\{[\s\S]*\}', response)
                if not json_match:
                    raise ValueError("Could not parse response as JSON")
                result = json.loads(json_match.group(0))
            
            # Validate required fields
            if "tool_name" not in result or "description" not in result:
                raise ValueError("Response missing required fields")
            
            logger.info(f"Inferred tool name: {result['tool_name']}")
            logger.info(f"Inferred description: {result['description']}")
            
            return {
                "tool_name": result["tool_name"], 
                "description": result["description"]
            }
            
        except Exception as e:
            logger.warning(f"Failed to infer tool info: {str(e)}. Using defaults.")
            # Generate default name from source file name
            module_name = os.path.basename(os.path.splitext(source_code)[0]) if isinstance(source_code, str) and os.path.exists(source_code) else "unknown"
            default_name = " ".join(word.capitalize() for word in module_name.split("_"))
            
            return {
                "tool_name": default_name,
                "description": f"Tool based on {module_name} module"
            }


class TemplateManager:
    """
    Manages template rendering and file operations for toolgen.
    """
    
    def __init__(self, template_dir: str):
        """Initialize the TemplateManager with the template directory."""
        self.template_env = jinja2.Environment(
            loader=jinja2.FileSystemLoader(searchpath=template_dir)
        )
    
    def render_template(self, template_name: str, context: Dict[str, Any]) -> str:
        """Render a Jinja2 template with the given context."""
        template = self.template_env.get_template(template_name)
        return template.render(**context)
    
    def write_file(self, path: str, content: str) -> None:
        """Write content to a file, creating directories if needed."""
        directory = os.path.dirname(path)
        if directory and not os.path.exists(directory):
            os.makedirs(directory)
            
        with open(path, 'w', encoding='utf-8') as f:
            f.write(content)
    
    def generate_files(self, file_specs: List[Tuple[str, str, str]], context: Dict[str, Any]) -> None:
        """
        Generate multiple files from templates with the given context.
        
        Args:
            file_specs: List of (template_name, output_path, description) tuples
            context: Dictionary with variables for template rendering
        """
        for template_name, output_path, description in file_specs:
            content = self.render_template(template_name, context)
            self.write_file(output_path, content)
            print(f"Created {description} at {output_path}")


async def generate_tool_components(source_path: str, tool_name: str, description: str) -> Dict[str, Any]:
    """Generate tool components based on source code analysis."""
    # Read source file
    with open(source_path, 'r', encoding='utf-8') as f:
        source_code = f.read()
    
    # Get source module name without extension
    source_module = os.path.basename(source_path).split('.')[0]
    
    # Create basic context
    context = {
        "tool_name": tool_name,
        "description": description,
        "tool_name_snake": tool_name.lower().replace(' ', '_'),
        "tool_name_pascal": ''.join(word.capitalize() for word in tool_name.split(' ')),
        "source_module": source_module
    }
    
    # Create agent
    agent = ToolgenAgent()
    
    try:
        # Generate server components using specialized function extraction prompt
        logger.info("Generating server components with specialized prompt...")
        server_components = await agent.generate_server_components(source_code, source_module)
        context.update(server_components)
        
        # Extract tool names for run_server components
        tool_names = []
        if "available_tools" in server_components:
            tools_code = server_components["available_tools"]
            tool_names = [line.split('"- ')[1].split('"')[0] for line in tools_code.split("\n") if '"- ' in line]
        
        # Generate AI server and run server components concurrently with specialized prompts
        logger.info("Generating AI and run server components with specialized prompts...")
        ai_components, run_components = await asyncio.gather(
            agent.generate_ai_server_components(source_code, tool_name),
            agent.generate_run_server_components(source_code, tool_name)
        )
        
        context.update(ai_components)
        context.update(run_components)
        
        # Add available tools to run_server_components if we have tool names
        if tool_names:
            context["available_tools"] = "\n        ".join([f'print("- {name}")' for name in sorted(tool_names)])
        
        return context
    finally:
        # Ensure agent is shut down properly
        await agent.shutdown()


def generate_tool(target_path: str, tool_name: str, description: str, source_path: str) -> None:
    """Generate tool files in the target directory."""
    print(f"Generating tool '{tool_name}' in {target_path}")
    
    # Create template manager
    template_manager = TemplateManager(TEMPLATE_DIR)
    
    try:
        # Run the async function to generate components
        loop = asyncio.get_event_loop()
        context = loop.run_until_complete(
            generate_tool_components(source_path, tool_name, description)
        )
        
        # Define files to generate
        files_to_generate = [
            ("server.py.j2", os.path.join(target_path, "server.py"), "server.py with MCP tools"),
            ("AI_server.py.j2", os.path.join(target_path, "AI_server.py"), "AI_server.py with FractFlow Agent"),
            ("run_server.py.j2", os.path.join(target_path, "run_server.py"), "run_server.py script"),
        ]
        
        # Generate files
        template_manager.generate_files(files_to_generate, context)
        print(f"\nTool generation complete! Your tool is ready to use in {target_path}")
        
    except Exception as e:
        print(f"Error generating tool: {str(e)}")
        raise


def main():
    """Main entry point for the toolgen command line interface."""
    # Load environment variables for API keys
    load_dotenv()
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Generate FractFlow tools based on source code")
    parser.add_argument("source", help="Path to source file containing core implementations")
    args = parser.parse_args()
    
    # Validate source path
    source_path = os.path.abspath(os.path.expanduser(args.source))
    if not os.path.isfile(source_path):
        print(f"Error: Source file does not exist: {source_path}")
        return 1
    
    # Set target path to same folder as source
    target_path = os.path.dirname(source_path)
    print(f"Target path: {target_path}")
    
    try:
        # Read source code
        with open(source_path, 'r', encoding='utf-8') as f:
            source_code = f.read()
        
        # Create agent and infer tool info
        agent = ToolgenAgent()
        loop = asyncio.get_event_loop()
        
        try:
            # Initialize agent with specialized prompt for tool info inference
            print("Inferring tool name and description from source code...")
            inferred_info = loop.run_until_complete(agent.infer_tool_info(source_code))
            tool_name = inferred_info["tool_name"]
            description = inferred_info["description"]
        finally:
            # Ensure agent is shut down
            if agent.agent:
                loop.run_until_complete(agent.shutdown())
        
        print(f"Tool name: {tool_name}")
        print(f"Description: {description}")
        
        # Check for existing files
        existing_files = [
            os.path.join(target_path, file)
            for file in ["server.py", "AI_server.py", "run_server.py"]
            if os.path.exists(os.path.join(target_path, file))
        ]
        
        if existing_files:
            print("The following files already exist:")
            for file_path in existing_files:
                print(f"  - {file_path}")
            
            response = input("Overwrite these files? [y/N] ")
            if response.lower() not in ["y", "yes"]:
                print("Aborting.")
                return 1
        
        # Generate the tool
        generate_tool(target_path, tool_name, description, source_path)
        
        # Print usage instructions
        print(f"\nTo use your tool, run:")
        print(f"cd {target_path}")
        print(f"python run_server.py         # For interactive mode with direct tools")
        print(f"python run_server.py --ai    # For interactive mode with AI-enhanced tools")
        print(f"python run_server.py -q \"Your query here\"  # For single query mode")
        
        return 0
        
    except Exception as e:
        print(f"Error: {str(e)}")
        return 1


if __name__ == "__main__":
    sys.exit(main()) 