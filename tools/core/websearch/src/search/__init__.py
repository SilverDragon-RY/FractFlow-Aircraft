"""
Search Engine Package

This package provides implementations of various web search engines.

Author: Xinli Xu
Date: 2025-05-02
"""

from .base import SearchItem, WebSearchEngine
from .google_search import GoogleSearchEngine
from .baidu_search import BaiduSearchEngine
from .duckduckgo_search import DuckDuckGoSearchEngine

__all__ = [
    "WebSearchEngine",
    "SearchItem",
    "GoogleSearchEngine",
    "BaiduSearchEngine",
    "DuckDuckGoSearchEngine",
]
