"""
MCP tool loader implementation.

Provides functionality to load tool schemas from MCP services and
convert them to a standard format for use with the agent system.
"""

from typing import List, Dict, Any, Optional
from mcp.client.session import ClientSession

from ..infra.config import ConfigManager
from ..infra.logging_utils import get_logger

class MCPToolLoader:
    """
    Loads tool schemas from MCP services.
    
    Converts MCP tool schemas to a standardized format for use
    with language models.
    """
    
    def __init__(self, config: Optional[ConfigManager] = None):
        """
        Initialize the tool loader.
        
        Args:
            config: Configuration manager instance to use
        """
        self.config = config or ConfigManager()
        
        # Push component name to call path
        self.config.push_to_call_path("tool_loader")
        
        # Initialize logger
        self.logger = get_logger(self.config.get_call_path())
        
        self.logger.debug("Tool loader initialized")
    
    async def load_tools(self, session: ClientSession) -> List[Dict[str, Any]]:
        """
        Load tool schemas from an MCP client session.
        
        Args:
            session: An initialized MCP ClientSession
            
        Returns:
            List of tool schemas in the standardized format
            
        Raises:
            Exception: If tools can't be loaded
        """
        try:
            self.logger.debug("Loading tools from session")
            # Get available tools from the MCP server
            response = await session.list_tools()
            tools = self.convert_to_standard_format(response.tools)
            self.logger.debug("Tools loaded", {"count": len(tools)})
            return tools
        except Exception as e:
            error_msg = f"Failed to load tools: {str(e)}"
            self.logger.error(error_msg)
            raise ValueError(error_msg)
    
    @staticmethod
    def convert_to_standard_format(tools_data: Any) -> List[Dict[str, Any]]:
        """
        Convert MCP tool schemas to a standardized format.
        
        Args:
            tools_data: MCP tool data
            
        Returns:
            List of tool schemas in the standardized format
        """
        tools = []
        for tool in tools_data:
            tools.append({
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": tool.inputSchema
                }
            })
            
        return tools 