"""
Base model interface.

Defines the abstract base class for all model implementations, providing
a standardized interface for model-specific functionality.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional

class BaseModel(ABC):
    """
    Abstract base class for model implementations.
    
    Defines the interface that all model implementations must implement,
    providing a standardized way to execute models and manage conversation history.
    """
    
    @abstractmethod
    async def execute(self, tools: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]:
        """
        Execute the model with the current conversation history.
        
        Args:
            tools: List of tools available to the model, in a standardized format
            
        Returns:
            Response with content and optional tool calls
        """
        pass
    
    @abstractmethod
    def add_user_message(self, message: str) -> None:
        """
        Add a user message to the conversation history.
        
        Args:
            message: The message content from the user
        """
        pass
    
    @abstractmethod
    def add_assistant_message(self, message: str, tool_calls: Optional[List[Dict[str, Any]]] = None) -> None:
        """
        Add an assistant message to the conversation history.
        
        Args:
            message: The message content from the assistant
            tool_calls: Optional list of tool calls made by the assistant
        """
        pass
    
    @abstractmethod
    def add_tool_result(self, tool_name: str, result: str, tool_call_id: Optional[str] = None) -> None:
        """
        Add a tool result to the conversation history.
        
        Args:
            tool_name: Name of the tool that was called
            result: Result returned by the tool
            tool_call_id: Optional ID of the tool call this is responding to
        """
        pass 