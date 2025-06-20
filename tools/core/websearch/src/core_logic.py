"""
Web Search Tool - Core Logic Module

This module implements the core web search and browsing operations for the FractFlow web search tool.
It provides functions for searching the web using various search engines (Google, Baidu, DuckDuckGo)
and browsing web pages to extract content.

The search engine implementations in the search directory are adapted from https://github.com/FoundationAgents/OpenManus/tree/main/app/tool/search.
The web page browsing functionality is inspired by https://platform.moonshot.cn/docs/guide/use-kimi-api-to-complete-tool-calls#%E6%89%A7%E8%A1%8C%E5%B7%A5%E5%85%B7.


This implementation uses the search engine classes defined in the search directory.

Author: Xinli Xu (xxu068@connect.hkust-gz.edu.cn) - Envision Lab
Date: 2025-05-03
License: MIT License
"""

import asyncio
import requests
import httpx
from typing import List, Dict, Optional, Any, Union
from urllib.parse import urlparse, parse_qs, urljoin
from bs4 import BeautifulSoup
import logging
import sys
import os
import json
import io
from pathlib import Path
from datetime import datetime

# 尝试导入PDF处理库
try:
    import PyPDF2
    PDF_SUPPORT = True
except ImportError:
    PDF_SUPPORT = False

# Import search engines from search directory
current_dir = Path(__file__).parent
# Add current directory to path so we can import from search directory
sys.path.append(str(current_dir))

# Import search models and engines
from search.base import SearchItem, WebSearchEngine
from search.google_search import GoogleSearchEngine
from search.baidu_search import BaiduSearchEngine
from search.duckduckgo_search import DuckDuckGoSearchEngine

# Constants
MAX_CONTENT_LENGTH = 40000  # Maximum content length in characters


def is_pdf_content(content_type, content):
    """
    检查内容是否为PDF
    
    Args:
        content_type (str): 内容类型头
        content (bytes): 二进制内容
        
    Returns:
        bool: 是否为PDF文件
    """
    # 检查content-type
    if content_type and 'application/pdf' in content_type.lower():
        return True
    
    # 检查内容开头是否为PDF签名
    if content and len(content) > 4 and content[:4] == b'%PDF':
        return True
    
    return False


def extract_text_from_pdf(pdf_content):
    """
    从PDF内容中提取文本
    
    Args:
        pdf_content (bytes): PDF文件的二进制内容
        
    Returns:
        str: 提取的文本内容
    """
    if not PDF_SUPPORT:
        return "[PDF文件: PyPDF2库未安装，无法解析PDF内容。请安装PyPDF2: pip install PyPDF2]"
    
    try:
        # 创建一个BytesIO对象
        pdf_file = io.BytesIO(pdf_content)
        
        # 创建PDF阅读器
        pdf_reader = PyPDF2.PdfReader(pdf_file)
        
        # 提取文本
        text = ""
        for page_num in range(len(pdf_reader.pages)):
            page = pdf_reader.pages[page_num]
            text += page.extract_text() + "\n\n"
        
        if not text.strip():
            return "[PDF文件: 无法提取文本内容，可能是扫描件或受保护的PDF]"
        
        return text
    except Exception as e:
        return f"[PDF文件解析错误: {str(e)}]"


async def crawl_impl(url: str) -> Dict[str, Any]:
    """
    根据URL获取网页内容的简单实现
    
    Args:
        url (str): 要获取内容的网页URL
        
    Returns:
        Dict[str, Any]: 包含内容和元数据的字典
    """
    try:
        # 检查URL格式
        parsed_url = urlparse(url)
        if not parsed_url.scheme or not parsed_url.netloc:
            return {"error": "无效的URL，请提供完整URL，包括http://或https://"}
        
        # 设置请求头
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/98.0.4758.102 Safari/537.36",
        }
        
        # 使用httpx获取网页内容
        async with httpx.AsyncClient(timeout=10.0, follow_redirects=True) as client:
            response = await client.get(url, headers=headers)
            response.raise_for_status()
            
            # 检查是否是PDF
            content_type = response.headers.get('content-type', '')
            if is_pdf_content(content_type, response.content):
                # 如果是PDF，提取文本
                content = extract_text_from_pdf(response.content)
                return {
                    "content": content,
                    "url": url,
                    "is_pdf": True
                }
            
            # 如果不是PDF，获取HTML内容
            html_content = response.text
            
            return {
                "content": html_content,
                "url": url,
                "is_pdf": False
            }
            
    except Exception as e:
        return {"error": f"获取网页内容时出错: {str(e)}"}


async def crawl(arguments: dict) -> dict:
    """
    爬取网页内容的简单实现
    
    Args:
        arguments (dict): 包含以下字段的字典:
            - url (str): 要爬取的网页URL
            - max_length (int, optional): 返回内容的最大长度
    
    Returns:
        dict: 包含网页内容的字典
    """
    # 获取参数
    url = arguments.get("url")
    if not url:
        return {"error": "缺少URL参数"}
    
    max_length = arguments.get("max_length", MAX_CONTENT_LENGTH)
    
    # 获取网页内容
    result = await crawl_impl(url)
    
    # 检查是否有错误
    if "error" in result:
        return {"error": result["error"]}
    
    # 限制内容大小
    content = result["content"]
    if len(content) > max_length:
        content = content[:max_length] + f"\n... [内容被截断，总共{len(result['content'])}字符] ..."
    
    # 返回结果，包含是否为PDF的信息
    is_pdf = result.get("is_pdf", False)
    return {
        "content": content,
        "url": url,
        "is_pdf": is_pdf
    }


# Main search functionality
def format_search_results(search_engine: str, query: str, items: List[SearchItem]) -> str:
    """
    Format search results into a readable string
    
    Args:
        search_engine (str): The name of the search engine used
        query (str): The search query
        items (List[SearchItem]): The search results
        
    Returns:
        str: Formatted search results string
    """
    if not items:
        return f"No results found for '{query}' on {search_engine.capitalize()}"
    
    results = [f"🔍 Search results for '{query}' from {search_engine.capitalize()}:"]
    
    for i, item in enumerate(items, 1):
        title = item.title or f"Result {i}"
        url = item.url or "No URL"
        description = item.description or "No description available"
        
        results.append(f"\n📌 {title}")
        results.append(f"🔗 {url}")
        results.append(f"📄 {description}")
    
    return "\n".join(results)


async def web_search(query: str, search_engine: str = "duckduckgo", num_results: int = 5) -> str:
    """
    Performs a web search and returns relevant results
    
    Args:
        query (str): The keywords or phrase to search for
        search_engine (str, optional): The search engine to use
                                      Options: "duckduckgo", "baidu", "google"
                                      Default is "duckduckgo"
        num_results (int, optional): Number of results to return, default is 5
        
    Returns:
        str: Search results including titles, links and descriptions, or an error message if the search fails
    """
    try:
        # Validate parameters
        if not query:
            return "Search query cannot be empty"
            
        if num_results <= 0:
            return "Number of results must be greater than 0"
        
        if num_results > 20:
            num_results = 20
            
        search_engine = search_engine.lower()
        
        # Initialize search engines
        engines = {
            "google": GoogleSearchEngine(),
            "baidu": BaiduSearchEngine(),
            "duckduckgo": DuckDuckGoSearchEngine()
        }
        
        if search_engine not in engines:
            return f"Unsupported search engine: {search_engine}. Supported options: {', '.join(engines.keys())}"
        
        # Perform search
        engine = engines[search_engine]
        search_results = engine.perform_search(query, num_results=num_results)
        
        # Format and return results
        return format_search_results(search_engine, query, search_results)
            
    except requests.exceptions.RequestException as e:
        return f"Search request failed: {str(e)}"
    except Exception as e:
        return f"An error occurred while performing the search: {str(e)}"


async def web_search_and_browse(query: str, search_engine: str = "duckduckgo", num_results: int = 5, max_browse: int = 1, max_length: int = MAX_CONTENT_LENGTH) -> str:
    """
    搜索并自动浏览搜索结果页面
    
    Args:
        query (str): 搜索关键词
        search_engine (str): 搜索引擎
        num_results (int): 返回的搜索结果数量
        max_browse (int): 最多浏览几个搜索结果
                         设置为0表示浏览所有结果
                         设置为负数表示只搜索不浏览
        max_length (int): 每个网页内容的最大长度
    
    Returns:
        str: 包含搜索结果和网页内容的综合信息
    """
    try:
        # 首先执行搜索
        search_results = await web_search(query, search_engine, num_results)
        
        # 如果max_browse为负数，表示只执行搜索不浏览
        if max_browse < 0:
            return search_results
        
        # 解析搜索结果中的URL
        urls = []
        for line in search_results.split('\n'):
            if line.startswith('🔗 '):
                url = line[2:].strip()
                urls.append(url)
        
        if not urls:
            return f"搜索结果: {search_results}\n\n无法从搜索结果中提取URL。"
        
        # 如果max_browse为0，则浏览所有结果
        if max_browse == 0:
            max_browse = len(urls)
        
        # 限制浏览的URL数量
        urls = urls[:max_browse]
        
        # 准备输出
        output = [f"搜索结果: {search_results}\n\n--- 网页内容（浏览 {len(urls)}/{len(urls)} 结果）---\n"]
        
        # 爬取每个URL的内容
        for i, url in enumerate(urls):
            try:
                # 减小单个页面的最大长度，防止总内容过大
                page_max_length = max(5000, max_length // max(len(urls), 1))
                
                result = await crawl({"url": url, "max_length": page_max_length})
                if "error" in result:
                    page_content = f"⚠️ 无法获取内容: {result['error']}"
                else:
                    is_pdf = result.get("is_pdf", False)
                    page_content = result["content"]
                    
                output.append(f"\n\n[结果 {i+1}] - {'[PDF文件]' if is_pdf else ''} {url}\n{page_content}")
            except Exception as e:
                output.append(f"\n\n[结果 {i+1}] - {url}\n⚠️ 处理时出错: {str(e)}")
        
        return "\n".join(output)
            
    except Exception as e:
        return f"搜索和浏览过程中出错: {str(e)}" 