"""
tool_executor.py
Author: Ying-Cong Chen (yingcong.ian.chen@gmail.com)
Date: 2025-04-08
Description: Handles the execution of tools based on model requests, providing abstraction between the model and actual tools.
License: MIT License
"""

"""
Tool executor.

Handles the execution of tools based on model requests.
"""

from typing import Dict, Any, Optional
from ..infra.config import ConfigManager
from ..infra.error_handling import ToolExecutionError, handle_error
from ..infra.logging_utils import get_logger

class ToolExecutor:
    """
    Executes tools based on model requests.
    
    Provides a layer of abstraction between the model and the actual tools,
    handling errors and formatting results.
    """
    
    def __init__(self, config: Optional[ConfigManager] = None):
        """
        Initialize the tool executor.
        
        Args:
            config: Configuration manager instance to use
        """
        self.config = config or ConfigManager()
        
        # Push component name to call path
        self.config.push_to_call_path("tool_executor")
        
        # Initialize logger
        self.logger = get_logger(self.config.get_call_path())
        self.logger.debug("Tool executor initialized")
        
    async def execute_tool(self, tool_name: str, arguments: Dict[str, Any]) -> str:
        """
        Execute a tool with the given arguments.
        
        Args:
            tool_name: Name of the tool to execute
            arguments: Dictionary of arguments to pass to the tool
            
        Returns:
            The result of the tool execution as a string
            
        Raises:
            ToolExecutionError: If the tool execution fails
        """
        try:
            self.logger.debug(f"Executing tool", {"tool": tool_name, "args": arguments})
            
            # Get the client pool from the MCP module
            # This is imported here to avoid circular imports
            from ..mcpcore import get_client_pool
            
            # Call the tool using the MCP client pool
            client_pool = get_client_pool()
            result = await client_pool.call(tool_name, arguments)
            
            self.logger.debug(f"Tool execution successful", {"tool": tool_name, "result_length": len(result) if result else 0})
            return result
            
        except Exception as e:
            error = handle_error(e, {"tool_name": tool_name, "arguments": arguments})
            self.logger.error(f"Error executing tool {tool_name}: {error}")
            raise ToolExecutionError(f"Failed to execute tool {tool_name}: {str(error)}", e) 