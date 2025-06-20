"""
File I/O Tool Server Runner

This module provides the entry point for starting the File I/O Tool server.
It can be run in two modes:
1. Interactive chat mode - continuous processing of user queries until exit
2. Single query mode - processing a single query and then exiting

The module initializes an EnvisionCore agent with the File I/O tool and
handles user interactions according to the chosen mode.

Author: Ying-Cong Chen (yingcong.ian.chen@gmail.com)
Date: 2025-04-27
License: MIT License
"""

import asyncio
import os
import sys
import logging
import argparse

import os.path as osp
# Add the project root directory to the Python path
project_root = osp.abspath(osp.join(osp.dirname(__file__), '..'))
sys.path.append(project_root)

# Import the FractalFlow Agent
from FractFlow.agent import Agent
from FractFlow.infra.config import ConfigManager
from FractFlow.infra.logging_utils import setup_logging, get_logger

# Setup logging
setup_logging(level=logging.INFO)


async def create_agent():
    """Create and initialize the Agent"""
    # Create a new agent
    agent = Agent('file_io_agent')  # No need to specify provider here if it's in config
    config = agent.get_config()
    config['agent']['provider'] = 'deepseek'
    # config['agent']['custom_system_prompt'] = '你会用萌萌哒的语气回复'
    config['deepseek']['model'] = 'deepseek-chat'
    # You can modify configuration values directly
    config['agent']['max_iterations'] = 5  # Properly set as nested value
    # 4. Set configuration loaded from environment
    agent.set_config(config)
    
    # Get the current directory path
    current_dir = os.path.dirname(os.path.abspath(__file__))
    server_path = os.path.join(current_dir, "server.py")
    
    # Add tools to the agent
    # agent.add_tool("./tools/ComfyUITool.py")
    # agent.add_tool("./tools/VisualQestionAnswer.py")
    agent.add_tool(server_path, 'editor')
    # Initialize the agent (starts up the tool servers)
    print("Initializing agent...")
    await agent.initialize()
    
    return agent


async def interactive_mode(agent):
    """Interactive chat mode"""
    print("Agent chat started. Type 'exit', 'quit', or 'bye' to end the conversation.")
    while True:
        user_input = input("\nYou: ")
        if user_input.lower() in ('exit', 'quit', 'bye'):
            break
            
        print("\n thinking... \n", end="")
        result = await agent.process_query(user_input)
        print("Agent: {}".format(result))


async def single_query_mode(agent, query):
    """One-time execution mode"""
    print(f"Processing query: {query}")
    print("\n thinking... \n", end="")
    result = await agent.process_query(query)
    print("Result: {}".format(result))
    return result


async def main():
    # Command line argument parsing
    parser = argparse.ArgumentParser(description='Run File I/O Tool Server')
    parser.add_argument('--user_query', type=str, help='Single query mode: process this query and exit')
    args = parser.parse_args()
    
    # Create Agent
    agent = await create_agent()
    
    try:
        if args.user_query:
            # Single query mode
            await single_query_mode(agent, args.user_query)
        else:
            # Interactive chat mode
            await interactive_mode(agent)
    finally:
        # Close Agent
        await agent.shutdown()
        print("\nAgent session ended.")


if __name__ == "__main__":
    asyncio.run(main()) 