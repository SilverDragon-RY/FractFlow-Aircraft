"""
Base history adapter.

Defines the interface for provider-specific conversation history adapters.
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional

class HistoryAdapter(ABC):
    """
    Abstract base class for history adapters.
    
    Defines the interface that all history adapters must implement, providing a
    standardized way to format conversation history for different AI providers.
    """
    
    def format_for_model(self, messages: List[Dict[str, Any]], tools: Optional[List[Dict[str, Any]]] = None) -> List[Dict[str, Any]]:
        """
        Format conversation history for a specific model.
        
        Args:
            messages: The raw conversation history
            tools: Optional list of available tools
            
        Returns:
            Formatted conversation history appropriate for the model
        """
        formatted_messages = []
        tools_desc = None
        
        # Format tool descriptions if provided
        if tools:
            tools_desc = self._format_tools_description(tools)
        
        for i, message in enumerate(messages):
            role = message["role"]
            
            if role == "system":
                # System messages are directly supported
                formatted_messages.append({
                    "role": "system",
                    "content": message["content"]
                })
                
            elif role == "user":
                # For user messages, we might need to append tool descriptions
                content = message["content"]
                
                # Only append tools description to the last user message if needed
                if i == len(messages) - 1 and tools_desc and not any(self._contains_tool_desc(msg) for msg in formatted_messages):
                    content = f"{content}\n\nAvailable tools:\n{tools_desc}"
                    
                formatted_messages.append({
                    "role": "user",
                    "content": content
                })
                
            elif role == "assistant":
                # Assistant messages are directly supported
                formatted_messages.append({
                    "role": "assistant", 
                    "content": message["content"]
                })
                
            elif role == "tool":
                # For models, tool results need to be formatted as user messages
                tool_name = message.get("tool_name", "unknown tool")
                formatted_messages.append({
                    "role": "user",
                    "content": f"Tool result from {tool_name}:\n{message['content']}"
                })
        
        # Check if we have successive user messages and fix them
        self._ensure_alternating_messages(formatted_messages)
                
        return formatted_messages
    
    def _format_tools_description(self, tools: List[Dict[str, Any]]) -> str:
        """
        Format tool descriptions for inclusion in prompts.
        
        Args:
            tools: List of available tools
            
        Returns:
            Formatted string describing available tools
        """
        descriptions = []
        
        # Add a concise guide for tool calling at the beginning
        format_explanation = """Note that ONLY the following tools are available:
"""
        descriptions.append(format_explanation)
        
        # Generate description for each tool
        for tool in tools:
            name = tool.get("function", {}).get("name")
            description = tool.get("function", {}).get("description")
            params = tool.get("function", {}).get("parameters", {}).get("properties", {})
            required_params = tool.get("function", {}).get("parameters", {}).get("required", [])
            
            if not name or not description:
                continue
                
            # Format parameter descriptions
            param_desc = []
            for param_name, param_info in params.items():
                param_type = param_info.get("type", "any")
                param_description = param_info.get("description", "")
                required = "required" if param_name in required_params else "optional"
                param_desc.append(f"  - {param_name} ({param_type}, {required}): {param_description}")
                
            param_text = "\n".join(param_desc) if param_desc else "  No parameters"
            
                
            # Combine all into the tool description
            descriptions.append(f"**Available Tool**: {name}\nDescription: {description}\nParameters:\n{param_text}")
            
        return "\n\n".join(descriptions)
    
    def _contains_tool_desc(self, message: Dict[str, Any]) -> bool:
        """
        Check if a message already contains tool descriptions.
        
        Args:
            message: The message to check
            
        Returns:
            True if the message contains tool descriptions, False otherwise
        """
        return (message.get("role") == "user" and 
                "Available tools:" in message.get("content", ""))
    
    def _ensure_alternating_messages(self, messages: List[Dict[str, Any]]) -> None:
        """
        Ensure that user and assistant messages alternate properly.
        
        This is needed for models which require strict alternation.
        The function modifies the messages list in-place.
        
        Args:
            messages: The formatted messages to check and fix
        """
        # Skip if we only have system messages or empty list
        if len(messages) <= 1:
            return
            
        # Find the first non-system message
        start_idx = 0
        for i, msg in enumerate(messages):
            if msg["role"] != "system":
                start_idx = i
                break
                
        # Iterate through messages starting from first non-system
        i = start_idx
        while i < len(messages) - 1:
            current = messages[i]
            next_msg = messages[i+1]
            
            # If we have two consecutive messages of the same role (except system)
            if current["role"] == next_msg["role"] and current["role"] != "system":
                # Special handling for consecutive user messages
                if current["role"] == "user":
                    # Combine the content
                    combined_content = f"{current['content']}\n\n{next_msg['content']}"
                    messages[i]["content"] = combined_content
                    
                    # Remove the duplicate
                    messages.pop(i+1)
                    
                    # Don't increment i since we removed an element
                    continue
                    
                # Special handling for consecutive assistant messages
                elif current["role"] == "assistant":
                    # Combine the content
                    combined_content = f"{current['content']}\n\n{next_msg['content']}"
                    messages[i]["content"] = combined_content
                    
                    # Combine tool_calls if present
                    if "tool_calls" in current and "tool_calls" in next_msg:
                        messages[i]["tool_calls"] = current["tool_calls"] + next_msg["tool_calls"]
                    elif "tool_calls" in next_msg:
                        messages[i]["tool_calls"] = next_msg["tool_calls"]
                        
                    # Remove the duplicate
                    messages.pop(i+1)
                    
                    # Don't increment i since we removed an element
                    continue
            
            # Move to next message
            i += 1
            
    def format_debug_output(self, formatted_messages: List[Dict[str, Any]], tools: Optional[List[Dict[str, Any]]] = None, title: str = "ADAPTER DEBUG OUTPUT") -> str:
        """
        Generate debug output for the formatted messages.
        
        Args:
            formatted_messages: The formatted messages
            tools: Optional list of tools
            title: Title for the debug output
            
        Returns:
            String containing debug information
        """
        lines = [f"===== {title} DEBUG OUTPUT ====="]
        
        # Add info about tools
        if tools:
            lines.append(f"Tools: {len(tools)} available")
        else:
            lines.append("Tools: None")
            
        # Add formatted messages
        lines.append("\nFormatted Messages:")
        for i, msg in enumerate(formatted_messages):
            role = msg.get("role", "unknown")
            content = msg.get("content", "")[:50] + ("..." if len(msg.get("content", "")) > 50 else "")
            
            has_tool_calls = "tool_calls" in msg and msg["tool_calls"]
            tool_calls_info = f", {len(msg['tool_calls'])} tool calls" if has_tool_calls else ""
            
            lines.append(f"{i+1}. Role: {role}{tool_calls_info}")
            lines.append(f"   Content: {content}")
            
        return "\n".join(lines) 