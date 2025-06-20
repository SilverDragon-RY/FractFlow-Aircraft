"""
Web Search Tool Server Module

This module provides a FastMCP server that exposes web search and browsing operations as tools.
It wraps the core web search logic in a server interface that can be used by the FractFlow
framework. Each tool is exposed as an endpoint that can be called by the FractFlow agent.

The server provides tools for:
- Searching the web using different search engines and optionally browsing results 
- Crawling web pages to extract content

Author: Xinli Xu (xxu068@connect.hkust-gz.edu.cn) - Envision Lab
Date: 2025-05-03
License: MIT License
"""

from mcp.server.fastmcp import FastMCP
import sys
from pathlib import Path
import os


# Add the parent directory to the python path so we can import the core_logic module
current_dir = Path(__file__).parent
parent_dir = current_dir.parent
sys.path.insert(0, str(parent_dir))

from src.core_logic import web_search_and_browse, crawl, MAX_CONTENT_LENGTH

# Initialize MCP server
mcp = FastMCP("web_search_browse_tool")

@mcp.tool()
async def search_and_browse(query: str, search_engine: str = "duckduckgo", num_results: int = 5, 
                           max_browse: int = 1, max_length: int = 40000) -> str:
    """
    搜索并可选择性浏览搜索结果的网页内容
    
    Args:
        query (str): 搜索关键词或短语
        search_engine (str, optional): 要使用的搜索引擎 ("duckduckgo", "baidu", "google")
                                      默认为 "duckduckgo"
        num_results (int, optional): 返回搜索结果数量，默认为5
        max_browse (int, optional): 自动浏览的搜索结果数量
                                   默认为1（只浏览第一个结果）
                                   设置为0表示浏览所有搜索结果
                                   设置为-1表示只搜索不浏览（只获取搜索结果列表）
        max_length (int, optional): 每个网页内容的最大长度，默认为50000字符
                                   注意：当浏览多个页面时，每个页面的实际长度会自动减小
        
    Returns:
        str: 包含搜索结果和网页具体内容的综合信息
        
    Example:
        # 场景1: 只搜索不浏览 - 适合简单问题，搜索结果就能提供答案
        search_and_browse("今天北京天气如何", search_engine="baidu", num_results=3, max_browse=-1)
        
        # 场景2: 搜索并浏览第一个结果（默认行为）- 适合需要详细了解的主题
        search_and_browse("Python字符串操作教程", search_engine="duckduckgo")
        
        # 场景3: 搜索学术/技术内容并浏览第一个结果 - 适合深入理解技术概念
        search_and_browse("ResNet深度学习网络结构介绍", search_engine="google", max_browse=1)
        
        # 场景4: 搜索并浏览多个结果 - 适合需要比较多个来源的信息
        search_and_browse("比特币原理", search_engine="google", max_browse=3)
        
        # 场景5: 搜索并浏览所有结果 - 适合需要全面了解某个主题（请限制结果数量）
        search_and_browse("碳中和概念", num_results=3, max_browse=0)
    """
    return await web_search_and_browse(query, search_engine, num_results, max_browse, max_length)

@mcp.tool()
async def web_crawl(url: str, max_length: int = 40000) -> str:
    """
    爬取网页内容
    
    Args:
        url (str): 要爬取的网页URL
        max_length (int, optional): 返回内容的最大长度
                                   默认为50000字符（约50KB）
        
    Returns:
        str: 包含网址和内容的信息，或错误信息
        
    Example:
        web_crawl("https://www.example.com")
        web_crawl("https://www.python.org", max_length=30000)
    """
    arguments = {
        "url": url, 
        "max_length": max_length
    }
    
    result = await crawl(arguments)
    
    # 如果有错误，返回错误信息
    if "error" in result:
        return f"爬取错误: {result['error']}"
    
    # 构建输出
    is_pdf = result.get("is_pdf", False)
    if is_pdf:
        return f"网址: {result['url']}\n[PDF文件]\n\n内容:\n{result['content']}"
    else:
        return f"网址: {result['url']}\n\n内容:\n{result['content']}"

# If this module is run directly, start the MCP server
if __name__ == "__main__":
    mcp.run(transport="stdio") 