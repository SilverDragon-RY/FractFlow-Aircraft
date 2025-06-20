"""
MCP integration for the agent system.

Provides classes and functions for integrating with MCP servers
and managing tool providers.
"""

# 导出主要的类和函数
from .client_pool import MCPClientPool, get_client_pool
from .launcher import MCPLauncher
from .tool_loader import MCPToolLoader

__all__ = [
    'MCPClientPool',
    'get_client_pool',
    'MCPLauncher',
    'MCPToolLoader',
] 