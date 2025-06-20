"""
agent.py
Author: Ying-Cong Chen (yingcong.ian.chen@gmail.com)
Date: 2025-04-08
Description: Main agent interface for FractalFlow providing user-friendly interaction with the agent system.
License: MIT License
"""

"""
Agent interface for FractalFlow.

Provides a simple user-friendly interface for interacting with the FractalFlow agent system.
"""

import os
import asyncio
from typing import Dict, Any, Optional, List

from .core.orchestrator import Orchestrator
from .core.query_processor import QueryProcessor
from .core.tool_executor import ToolExecutor
from .infra.config import ConfigManager
from .infra.logging_utils import get_logger

class Agent:
    """
    Main interface for using the FractalFlow agent.
    
    This class provides a simplified interface for configuring and using
    the FractalFlow agent system.
    """
    
    def __init__(self, config: ConfigManager, name: str = 'agent'):
        """
        Initialize the FractalFlow agent.
        
        Args:
            config: ConfigManager instance with all configuration settings
            name: Name for the agent
        """
        # Use provided configuration
        self.config = config
        self.name = name
        
        # Push agent name to call path
        self.config.push_to_call_path(self.name)
        
        # Initialize logger with call path
        self.logger = get_logger(self.config.get_call_path())
        
        # Initialize tool configs
        self.tool_configs = {}
        
        # These will be initialized when needed
        self._orchestrator = None
        self._query_processor = None
        self._tool_executor = None
        self._is_initialized = False
        
        self.logger.info(f"Agent '{self.name}' initialized")
        
    def add_tool(self, tool_path: str, tool_name: str) -> None:
        """
        Add a tool to the agent.
        
        Args:
            tool_path: Path to the tool script
            tool_name: Optional name for the tool. If not provided, the basename of the path will be used.
        """
        if not os.path.exists(tool_path):
            raise ValueError(f"Tool script not found: {tool_path}")
        
        self.tool_configs[tool_name] = tool_path
    
    def _ensure_initialized(self) -> None:
        """Initialize components if they haven't been initialized yet."""
        if not self._is_initialized:
            # Get provider from config
            provider = self.config.get('agent.provider')
            
            self.logger.debug("Initializing agent components")
            
            # Create components with the config_manager
            self._orchestrator = Orchestrator(
                tool_configs=self.tool_configs,
                provider=provider,
                config=self.config
            )
            self._tool_executor = ToolExecutor(config=self.config)
            self._query_processor = QueryProcessor(
                self._orchestrator, 
                self._tool_executor,
                config=self.config
            )
            self._is_initialized = True
            
            self.logger.info("Agent components initialized")
    
    async def initialize(self) -> None:
        """Initialize and start the agent system."""
        self._ensure_initialized()
        self.logger.info("Starting orchestrator")
        await self._orchestrator.start()
        self.logger.info("Agent system started")
    
    async def shutdown(self) -> None:
        """Shut down the agent system."""
        if self._orchestrator:
            self.logger.info("Shutting down agent system")
            await self._orchestrator.shutdown()
            self._is_initialized = False
            self.logger.info("Agent system shut down")
    
    async def process_query(self, query: str) -> str:
        """
        Process a user query.
        
        Args:
            query: The user's input query
            
        Returns:
            The agent's response
        """
        # Initialize if not already initialized
        self._ensure_initialized()
        
        # Start the orchestrator if not already started
        if not hasattr(self._orchestrator, "launcher") or self._orchestrator.launcher is None:
            self.logger.info("Starting orchestrator")
            await self._orchestrator.start()
        
        # Log the incoming query
        self.logger.info(f"Processing query", {"query": query})
        
        # Process the query
        result = await self._query_processor.process_query(query)
        
        return result 
        
    def get_history(self) -> List[Dict[str, Any]]:
        """
        Get the conversation history from the current session.
        
        Returns:
            The current conversation history as a list of message dictionaries
        """
        self._ensure_initialized()
        return self._query_processor.get_history() 