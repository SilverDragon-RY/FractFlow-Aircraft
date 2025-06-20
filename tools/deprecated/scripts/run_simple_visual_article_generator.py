"""
Web Search Tool Server Runner

This module provides the entry point for starting the Web Search Tool server.
It can be run in two modes:
1. Interactive chat mode - continuous processing of user queries until exit
2. Single query mode - processing a single query and then exiting

The module initializes a FractFlow agent with the Web Search tool and
handles user interactions according to the chosen mode.

Author: Xinli Xu (xxu068@connect.hkust-gz.edu.cn) - Envision Lab
Date: 2025-04-28
License: MIT License
"""

import asyncio
import os
import sys
import logging
import argparse
from dotenv import load_dotenv

import os.path as osp
# Add the project root directory to the Python path
project_root = osp.abspath(osp.join(osp.dirname(__file__), '../..'))
sys.path.append(project_root)

# Import the FractFlow Agent
from FractFlow.agent import Agent
from FractFlow.infra.config import ConfigManager
from FractFlow.infra.logging_utils import setup_logging, get_logger

# Setup logging
setup_logging(level=logging.DEBUG)


async def create_agent():
    """Create and initialize the Agent"""
    load_dotenv()
    # Create a new agent
    agent = Agent('Image_Article_Application')  # No need to specify provider here if it's in config
    config = agent.get_config()
    config['agent']['provider'] = 'deepseek'

    config['agent']['custom_system_prompt'] = """
## 🧠 你是一个图文 Markdown 内容生成 Agent

你的职责是撰写结构化的 Markdown 文章，并在适当位置自动插入相关插图，最终生成一篇完整、图文并茂的 Markdown 文件。

---

## 🔁 工作流（循环执行）

### 1. 规划阶段（仅一次）

* 明确主题、结构、段落划分、图像需求
* 在内部完成规划，**不输出**

---

### 2. 段落生成流程（每段循环）

#### 2.1 撰写段落

* 撰写该段 Markdown 内容，结构清晰、语言自然，故事完整，字数不小于500字。
* 在合适位置插入图像路径引用，如：
  `![说明](images/sectionX-figY.png)`
* 内容必须**直接写入 Markdown 文件**，**不得输出到 response 中**

#### 2.2 生成插图

* 根据该段上下文，为引用的路径生成图像
* 图像应与引用路径匹配，保存至 `images/` 子目录

#### 2.3 路径一致性检查

* 检查当前段落图像路径是否：

  * 属于 `images/` 目录
  * 与实际文件匹配
  * 唯一、不重复

---

### 3. 进入下一段

* 重复段落撰写、插图生成、路径校验，直到整篇文章完成

---

## 📁 文件结构约定

* 文章主文件为 Markdown 格式
* 插图统一存于 `images/` 子目录
* 图像命名应基于段落结构，如 `section2-fig1.png`

---

## 🚫 输出规范（必须遵守）

* 不得输出 Markdown 正文或图像信息到 response 中
* 所有正文和图像操作都应**直接执行、写入对应文件和目录**
* **你不是讲述者，而是操作执行者**。只做事，不解释

     """
    config['deepseek']['model'] = 'deepseek-reasoner'
    # You can modify configuration values directly
    config['agent']['max_iterations'] = 20  # Properly set as nested value
    # 4. Set configuration loaded from environment
    agent.set_config(config)
    
    # Add tools to the agent
    agent.add_tool("./tools/gpt_imagen.py", "image_generation_tool")
    agent.add_tool("./tools/editor/server.py", "editor_tool")
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
    parser = argparse.ArgumentParser(description='Run Web Search Tool Server')
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