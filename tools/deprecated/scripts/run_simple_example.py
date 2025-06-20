#!/usr/bin/env python3
"""
run_simple_example.py
Author: Ying-Cong Chen (yingcong.ian.chen@gmail.com)
Date: 2025-04-08
Description: Example script demonstrating how to use the FractalFlow Agent interface with basic setup and usage.
License: MIT License
"""

"""
Example script demonstrating how to use the FractalFlow Agent interface.

This example shows the basic setup and usage of the FractalFlow Agent
as described in the interface requirements.
"""

import asyncio
import os
import sys
import logging

import os.path as osp
# Add the project root directory to the Python path
project_root = osp.abspath(osp.join(osp.dirname(__file__), '..'))
sys.path.append(project_root)

# Import the FractalFlow Agent
from FractFlow.agent import Agent
from FractFlow.infra.config import ConfigManager
from FractFlow.infra.logging_utils import setup_logging, get_logger

# 设置日志
setup_logging(
    level=logging.DEBUG,  # 根logger设置为INFO
    namespace_levels={
        # "simple agent": logging.DEBUG,  # FractFlow项目输出DEBUG日志
        "httpx": logging.WARNING,    # 降低httpx的日志级别
    }
)


async def main():
    
    # 3. Create configuration with new parameter-based approach
    config = ConfigManager(
        provider='deepseek',
        deepseek_model='deepseek-chat',
        max_iterations=5,
        # custom_system_prompt='你会用萌萌哒的语气回复',
        qwen_base_url='https://dashscope.aliyuncs.com/compatible-mode/v1'
    )
    
    # 4. Create agent with configuration
    agent = Agent(config=config, name='simple agent')
    
    # Add tools to the agent
    # agent.add_tool("./tools/ComfyUITool.py")
    # agent.add_tool("./tools/VisualQestionAnswer.py")
    agent.add_tool("./tools/weather/weather_mcp.py", "forecast_tool")
    # Initialize the agent (starts up the tool servers)
    print("Initializing agent...")
    await agent.initialize()
    
    try:
        # Interactive chat loop
        print("Agent chat started. Type 'exit', 'quit', or 'bye' to end the conversation.")
        while True:
            user_input = input("\nYou: ")
            if user_input.lower() in ('exit', 'quit', 'bye'):
                break
                
            print("\n thinking... \n", end="")
            result = await agent.process_query(user_input)
            print("Agent: {}".format(result))
    finally:
        # Shut down the agent gracefully
        await agent.shutdown()
        print("\nAgent chat ended.")

if __name__ == "__main__":
    asyncio.run(main()) 