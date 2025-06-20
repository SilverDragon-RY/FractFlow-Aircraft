"""
Error handling utilities for the agent system.

Provides custom exceptions and error handling functionality to ensure
consistent error management across the agent system.
"""

import traceback
from typing import Optional, Dict, Any
from .logging_utils import get_logger

# 使用新的日志工具替代基础配置
logger = get_logger(__name__)

class AgentError(Exception):
    """Base exception class for all agent errors."""
    
    def __init__(self, message: str, cause: Optional[Exception] = None):
        """
        Initialize the error.
        
        Args:
            message: Error message
            cause: Original exception that caused this error, if any
        """
        self.message = message
        self.cause = cause
        super().__init__(message)

class ConfigurationError(AgentError):
    """Exception raised for configuration-related errors."""
    pass

class ToolExecutionError(AgentError):
    """Exception raised when a tool execution fails."""
    pass

class ClientError(AgentError):
    """Exception raised for client-related errors."""
    pass

class LLMError(AgentError):
    """Exception raised for language model-related errors."""
    pass

def handle_error(error: Exception, context: Optional[Dict[str, Any]] = None) -> AgentError:
    """
    Handle and transform exceptions into appropriate AgentError types.
    
    Args:
        error: The original exception
        context: Additional context information about the error
        
    Returns:
        An appropriate AgentError instance
    """
    # Error messages should remain highlighted to draw attention
    if context:
        logger.error("Error occurred", {"Error message": str(error), "Context": context})
    else:
        logger.error("Error occurred", {"Error message": str(error)})
    
    # Include stack trace for debugging
    logger.debug(traceback.format_exc())
    
    # Transform common exceptions into agent-specific exceptions
    if isinstance(error, AgentError):
        return error
    
    if "configuration" in str(error).lower() or "config" in str(error).lower():
        return ConfigurationError(str(error), error)
    
    if "tool" in str(error).lower() and ("execution" in str(error).lower() or "call" in str(error).lower()):
        return ToolExecutionError(str(error), error)
    
    if "client" in str(error).lower() or "connection" in str(error).lower() or "mcp" in str(error).lower():
        return ClientError(str(error), error)
    
    if "openai" in str(error).lower() or "model" in str(error).lower() or "completion" in str(error).lower():
        return LLMError(str(error), error)
    
    # Default case
    return AgentError(str(error), error)

def create_error_response(error: Exception) -> Dict[str, Any]:
    """
    Create a standardized error response object.
    
    Args:
        error: The exception to create a response for
        
    Returns:
        A standardized error response dictionary
    """
    agent_error = handle_error(error) if not isinstance(error, AgentError) else error
    
    return {
        "choices": [{
            "message": {
                "content": f"Error: {agent_error.message}",
                "tool_calls": None
            }
        }]
    } 