"""
aircraft_tool_template.py
Author: FractFlow
Date: 2025-05-31
Description: Aircraft tool template that inherits from FractFlow ToolTemplate with persistent agent support.
License: MIT License
"""

import asyncio
from typing import Optional
from FractFlow.tool_template import ToolTemplate
from FractFlow.agent import Agent


class AircraftToolTemplate(ToolTemplate):
    """
    Aircraft tool template with persistent agent support.
    
    This class extends ToolTemplate to provide an online mode where the agent
    persists throughout the object's lifetime, allowing for more efficient
    multi-query processing without recreating the agent each time.
    """
    
    def __init__(self):
        """Initialize the aircraft tool with no agent initially."""
        self.agent: Optional[Agent] = None
    
    async def _run_online(self, query=None) -> str:
        """
        Process a query using a persistent agent.
        
        On first call, creates and initializes an agent that persists for the
        lifetime of this object. Subsequent calls reuse the same agent.
        
        Args:
            query (str): The query to process
            
        Returns:
            str: The result from the agent
        """
        # Create agent on first call
        if self.agent is None:
            self.agent = await self.create_agent('agent')
        
        # Process the query using the persistent agent
        print(f"Process the query using the persistent agent. Query: {query}")
        result = await self.agent.process_query(query)
        print(f"Result: {result}")
        return result
    
    async def shutdown(self):
        """
        Shutdown the persistent agent if it exists.
        
        Call this method when you're done with the tool to properly
        clean up resources.
        """
        if self.agent is not None:
            await self.agent.shutdown()
            self.agent = None 