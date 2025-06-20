"""
MCP launcher implementation.

Provides functionality to manage and launch multiple MCP tool servers.
"""

import os
from typing import Dict, List, Optional

from .client_pool import get_client_pool
from ..infra.config import ConfigManager
from ..infra.logging_utils import get_logger

class MCPLauncher:
    """
    Manages and launches multiple MCP tool servers.
    
    Provides a unified interface to access all available tools.
    """
    
    def __init__(self, config: Optional[ConfigManager] = None):
        """
        Initialize the MCP launcher.
        
        Args:
            config: Configuration manager instance to use
        """
        self.config = config or ConfigManager()
        
        # Push component name to call path
        self.config.push_to_call_path("launcher")
        
        # Initialize logger
        self.logger = get_logger(self.config.get_call_path())
        
        self.client_pool = get_client_pool()
        self.server_paths: Dict[str, str] = {}
        
        self.logger.debug("Launcher initialized")
        
    def register_server(self, server_name: str, script_path: str) -> None:
        """
        Register an MCP server to be launched.
        
        Args:
            server_name: A unique name for this server
            script_path: Path to the server script
            
        Raises:
            FileNotFoundError: If the server script doesn't exist
        """
        if not os.path.exists(script_path):
            error_msg = f"Server script not found: {script_path}"
            self.logger.error(error_msg, {"server": server_name, "path": script_path})
            raise FileNotFoundError(error_msg)
            
        self.server_paths[server_name] = script_path
        self.logger.debug(f"Registered server", {"name": server_name, "path": script_path})
        
    async def launch_all(self) -> None:
        """
        Launch all registered MCP servers and connect clients.
        
        Raises:
            Exception: If any server fails to launch
        """
        self.logger.debug(f"Launching servers", {"count": len(self.server_paths)})
        
        try:
            for server_name, script_path in self.server_paths.items():
                self.logger.debug(f"Launching server", {"name": server_name})
                await self.client_pool.add_client(server_name, script_path)
                
            self.logger.info("All servers launched successfully")
        except Exception as e:
            self.logger.error(f"Error launching servers", {"error": str(e)})
            raise
        
    async def shutdown(self) -> None:
        """
        Shutdown all MCP servers and clients.
        
        Raises:
            Exception: If shutdown fails
        """
        try:
            self.logger.debug("Shutting down servers")
            await self.client_pool.cleanup()
            self.logger.info("All servers and clients shut down")
        except Exception as e:
            self.logger.error(f"Error shutting down servers", {"error": str(e)})
            raise 