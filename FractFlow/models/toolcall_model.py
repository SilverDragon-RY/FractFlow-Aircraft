import json
import uuid
from typing import List, Dict, Any, Optional, Tuple
from json_repair import repair_json
from tokencost import calculate_prompt_cost

from openai import OpenAI

from ..infra.config import ConfigManager
from ..infra.error_handling import handle_error
from ..infra.logging_utils import get_logger

class ToolCallHelper_v1:
    """
    Tool calling helper for generating tool calls from instructions.
    
    Uses OpenAI-compatible APIs to generate tool calls in a standardized format.
    Can be configured to work with different model providers through configuration.
    """
    
    def __init__(self, config: Optional[ConfigManager] = None):
        """
        Initialize the tool calling helper with configuration.
        
        Args:
            config: Configuration manager instance to use
        """
        self.config = config or ConfigManager()
        
        # Push component name to call path
        self.config.push_to_call_path("tool_call_helper")
        
        # Initialize logger
        self.logger = get_logger(self.config.get_call_path())
        
        self.client = None
        
        # Load configuration with defaults
        self.max_retries = self.config.get('tool_calling.max_retries', 5)
        self.base_url = self.config.get('tool_calling.base_url', 'https://api.deepseek.com')
        self.api_key = self.config.get('tool_calling.api_key', self.config.get('deepseek.api_key'))
        self.model = self.config.get('tool_calling.model', 'deepseek-chat')
        self.temperature = self.config.get('tool_calling.temperature', 0)
        # self.default_max_tokens = self.config.get('tool_calling.default_max_tokens', 8192)
        self.logger.debug("Tool call helper initialized", {
            "model": self.model,
            "max_retries": self.max_retries
        })
        
    async def initialize_client(self) -> OpenAI:
        """
        Initialize the OpenAI-compatible client.
        
        Returns:
            Configured OpenAI client
        """
        if self.client is None:
            self.logger.debug("Initializing OpenAI client", {"base_url": self.base_url})
            self.client = OpenAI(
                base_url=self.base_url,
                api_key=self.api_key
            )
        return self.client
        
    def create_system_prompt(self, tools: List[Dict[str, Any]]) -> str:
        """
        Create a system prompt for JSON tool calling.
        
        Args:
            tools: List of available tools
            
        Returns:
            System prompt for the tool calling model
        """
        # List all available tools with names and descriptions
        tool_details = []
        for tool in tools:
            tool_name = tool['function']['name']
            description = tool['function'].get('description', 'No description available')
            
            # List all parameters for this tool
            params = tool['function'].get('parameters', {}).get('properties', {})
            param_list = ", ".join(params.keys()) if params else "No parameters"
            
            tool_details.append(f"- {tool_name}: {description}\n  Parameters: {param_list}")
        
        tools_text = "\n".join(tool_details)
        
        self.logger.debug("Creating system prompt", {"tools_count": len(tools)})
        
        json_example = """{
    "tool_calls": [
        {
            "function": {
                "name": "tool_name",
                "arguments": {
                    "param1": "value1",
                    "param2": "value2"
                }
            }
        },
        {
            "function": {
                "name": "another_tool_name",
                "arguments": {
                    "param1": "value1",
                    "param2": "value2"
                }
            }
        }
    ]
}"""

        return f"""You are a tool calling expert. Your task is to generate correct JSON format tool calls based ONLY on the tools that are available.

AVAILABLE TOOLS (ONLY USE THESE - DO NOT INVENT NEW ONES):
{tools_text}

IMPORTANT RULES:
1. ONLY use tool names from the list above - never invent new tool names
2. ONLY use parameter names that are listed for each tool - never invent new parameters
3. If a requested tool doesn't exactly match any available tool, use the closest matching one
4. YOU CAN USE MULTIPLE TOOLS OR THE SAME TOOL MULTIPLE TIMES if the request requires it
5. If only one tool is needed, still use the proper array format with a single element

You must output strictly in the following JSON format:
{json_example}

The number of tool calls in the array should match exactly what's needed - don't add unnecessary calls.
For simple requests needing only one tool call, return an array with just one element.
Output JSON only, no other text. The arguments must be a valid JSON object."""
    
    def _estimate_token_count(self, messages: List[Dict[str, str]]) -> int:
        """
        Estimate the token count for a given set of messages.
        
        Args:
            messages: List of message dictionaries with role and content
            
        Returns:
            Estimated token count
        """
        return sum(len(m.get("content", "")) for m in messages) // 2
        # try:
        #     # Use tokencost to calculate the token count
        #     # We use the model name to determine the encoding
        #     token_cost = calculate_prompt_cost(messages, self.model)
        #     # Convert the cost to token count (approximate)
        #     # This is a rough estimate based on pricing
        #     estimated_tokens = int(token_cost * 1000 * 1000)  # Convert from $ to token count approximation
            
        #     self.logger.debug("Estimated token count", {
        #         "messages_count": len(messages),
        #         "estimated_tokens": estimated_tokens
        #     })
            
        #     return estimated_tokens
        # except Exception as e:
        #     self.logger.warning(f"Error estimating token count", {"error": str(e)})
        #     # Return a conservative estimate
        #     return sum(len(m.get("content", "")) for m in messages) // 2
    
    def _calculate_max_tokens(self, messages: List[Dict[str, str]]) -> int:
        """
        Calculate appropriate max_tokens based on input token count.
        
        Args:
            messages: List of message dictionaries
            
        Returns:
            Calculated max_tokens value
        """
        # Estimate input token count
        input_tokens = self._estimate_token_count(messages)
        
        # Set context window size based on model
        context_window = 8192
        
        # Calculate max_tokens to leave room for input and a reasonable output
        # Reserve at least 25% of context window for response
        # max_output_tokens = min(
        #     self.default_max_tokens,
        #     max(512, context_window - input_tokens - 50)  # 50 tokens buffer
        # )
        max_output_tokens = max(512, context_window - input_tokens - 50)  # 50 tokens buffer
        self.logger.debug("Calculated max_tokens", {
            "input_tokens": input_tokens,
            "context_window": context_window,
            "max_output_tokens": max_output_tokens
        })
        
        return max_output_tokens
    
    async def _create_chat_completion(self, **kwargs) -> Tuple[Optional[Any], Optional[Exception]]:
        """
        Handle API call to the model provider.
        
        Args:
            **kwargs: Arguments to pass to the chat completions API
            
        Returns:
            Tuple containing:
            - The API response or None if failed
            - The exception if an error occurred, or None if successful
        """
        try:
            # Make sure client is initialized
            if not self.client:
                await self.initialize_client()
                
            # Add model if not provided
            if 'model' not in kwargs:
                kwargs['model'] = self.model
            kwargs['temperature'] = self.temperature
            
            # Set max_tokens dynamically if not explicitly provided
            if 'max_tokens' not in kwargs and 'messages' in kwargs:
                kwargs['max_tokens'] = self._calculate_max_tokens(kwargs['messages'])
                
            # Call the API
            self.logger.debug("Calling chat completion API", {
                "model": kwargs.get('model'),
                "max_tokens": kwargs.get('max_tokens')
            })
            result = self.client.chat.completions.create(**kwargs)
            self.logger.debug("API call successful")
            return await result if hasattr(result, "__await__") else result, None
        except Exception as e:
            error = handle_error(e, {"kwargs": kwargs})
            self.logger.error(f"API call error", {"error": str(error)})
            return None, error
    
    async def _parse_model_response(self, response: Any) -> Optional[List[Dict[str, Any]]]:
        """
        Parse the response from the model into a list of tool calls.
        
        Args:
            response: Raw response from the model
            
        Returns:
            List of tool calls or None if failed
        """
        if not response or not response.choices:
            self.logger.warning("Empty response from model")
            return None
            
        content = response.choices[0].message.content.strip()
        
        # Apply json_repair directly
        content = repair_json(content)
        if not content:
            self.logger.error("JSON repair returned empty string, JSON was too broken")
            return None
        
        # Parse the JSON response
        try:
            self.logger.debug("Parsing model response")
            model_response = json.loads(content)
            
            # Process multiple tool calls
            if "tool_calls" in model_response and isinstance(model_response["tool_calls"], list):
                # Handle standard multiple tool calls format
                tool_calls = []
                for i, call_data in enumerate(model_response["tool_calls"]):
                    if "function" not in call_data:
                        self.logger.error("Tool call missing function object", {"index": i})
                        continue
                        
                    function_data = call_data.get("function", {})
                    
                    # Ensure arguments is a proper dictionary
                    if "arguments" in function_data and isinstance(function_data["arguments"], str):
                        try:
                            # Convert arguments string to dictionary if needed (for backward compatibility)
                            function_data["arguments"] = json.loads(function_data["arguments"])
                        except json.JSONDecodeError:
                            self.logger.error("Failed to parse arguments string as JSON", {"index": i})
                            continue
                            
                    # Add ID and type fields to each call
                    call_id = self.generate_call_id()
                    tool_call = {
                        "id": call_id,
                        "type": "function",
                        "function": function_data
                    }
                    tool_calls.append(tool_call)
                
                self.logger.debug("Parsed tool calls", {"count": len(tool_calls)})
                return tool_calls
                
            # Handle single tool call format (for backward compatibility)
            elif "function" in model_response:
                # Convert single tool call to list format
                call_id = self.generate_call_id()
                tool_call = {
                    "id": call_id,
                    "type": "function",
                    "function": model_response.get("function", {})
                }
                
                self.logger.debug("Parsed single tool call")
                return [tool_call]
            else:
                self.logger.error("Response does not contain tool_calls array or function object")
                return None
                
        except json.JSONDecodeError as e:
            self.logger.error(f"JSON parsing error", {"error": str(e)})
            return None
    
    async def call_tool(self, instruction: str, tools: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        """
        Execute a tool call using adaptive retry mechanism.
        Can generate multiple tool calls from a single instruction.
        
        Args:
            instruction: The instruction to execute
            tools: List of available tools
            
        Returns:
            Tuple containing:
            - List of valid tool calls in OpenAI format
            - Stats dictionary with success/failure information
        """
        available_tools = [tool['function']['name'] for tool in tools]
        stats = {
            "attempts": 0,
            "success": False,
            "valid_calls": 0,
            "invalid_calls": 0,
            "total_calls": 0,
            "errors": []
        }
        
        self.logger.debug("Starting tool call generation", {"tools_available": len(tools)})
        
        # Track modifications for adaptive retries
        current_instruction = instruction
        current_tools = tools
        
        # Try multiple times to get valid tool calls
        for attempt in range(self.max_retries):
            stats["attempts"] = attempt + 1
            self.logger.debug(f"Tool call attempt", {"attempt": attempt+1, "max": self.max_retries})
            
            # Call the internal method to get tool calls
            tool_calls, error = await self._internal_call_tool(current_instruction, current_tools)
            
            # If there was an error, adapt parameters for next attempt
            if error or not tool_calls:
                error_str = str(error)
                self.logger.warning(f"Error on attempt {attempt+1}: {error_str}")
                stats["errors"].append(error_str)
                
                # Implement adaptive strategy based on previous errors
                current_instruction, current_tools = await self._adapt_parameters(
                    current_instruction, 
                    current_tools, 
                    error, 
                    attempt
                )
                self.logger.debug("Toolcall model: Modified Instruction and Tools", {"instruction": current_instruction, "tools": current_tools})
                continue
            
            
            stats["total_calls"] = len(tool_calls)
            
            # Validate each tool call and keep valid ones
            valid_tool_calls = []
            for i, call in enumerate(tool_calls):
                if self._validate_tool_call(call, available_tools):
                    valid_tool_calls.append(call)
                    stats["valid_calls"] += 1
                else:
                    self.logger.warning(f"Invalid tool call", {
                        "index": i,
                        "attempt": attempt+1,
                        "tool": call.get("function", {}).get("name", "unknown")
                    })
                    stats["invalid_calls"] += 1
            
            # If we have at least one valid call, consider it a success
            if valid_tool_calls:
                self.logger.debug(f"Generated valid tool calls", {
                    "count": len(valid_tool_calls), 
                    "attempt": attempt+1
                })
                stats["success"] = True
                return valid_tool_calls, stats
            else:
                self.logger.warning(f"No valid tool calls on attempt", {"attempt": attempt+1})
                continue
            
        # If we exhausted all retries without success
        self.logger.error(f"Failed to generate valid tool calls after all attempts", {"max": self.max_retries})
        stats["success"] = False
        return [], stats
    
    async def _internal_call_tool(self, instruction: str, tools: List[Dict[str, Any]]) -> Tuple[Optional[List[Dict[str, Any]]], Optional[Exception]]:
        """
        Internal method to call the model and get tool call responses.
        
        Args:
            instruction: The instruction to execute
            tools: List of available tools
            
        Returns:
            Tuple containing:
            - List of tool calls or None if failed
            - Exception if an error occurred, or None if successful
        """
        try:
            # Build messages for the API call
            messages = [
                {"role": "system", "content": self.create_system_prompt(tools)},
                {"role": "user", "content": instruction}
            ]
            
            # Call the model with appropriate formatting
            self.logger.debug("Calling model with instruction")
            response, error = await self._create_chat_completion(
                messages=messages,
                response_format={"type": "json_object"}
            )
            
            if error:
                return None, error
                
            if not response:
                self.logger.warning("No response from model")
                return None, None
            
            # Parse the response
            tool_calls = await self._parse_model_response(response)
            return tool_calls, None
                
        except Exception as e:
            error = handle_error(e, {"instruction": instruction})
            self.logger.error(f"Internal tool calling error", {"error": str(error)})
            return None, error
    
    async def _adapt_parameters(self, instruction: str, tools: List[Dict[str, Any]], 
                              error: Exception, attempt: int) -> Tuple[str, List[Dict[str, Any]]]:
        """
        Adapt parameters based on error by using a LLM to rewrite the instruction.
        
        Args:
            instruction: Current instruction
            tools: Current tools list
            error: Error from previous attempt
            attempt: Attempt number (0-indexed)
            
        Returns:
            Tuple of (adapted_instruction, adapted_tools)
        """
        error_str = str(error)
        tools_count = len(tools)
        
        # For first attempt, we might try a simple approach by reducing tools
        if attempt == 0 and tools_count > 3:
            # Just reduce tools on first error to avoid unnecessary API calls
            keep_count = max(3, tools_count // 2)
            adapted_tools = tools[:keep_count]
            self.logger.info(f"First attempt adapting: reducing tool count", {
                "original_count": tools_count,
                "new_count": keep_count
            })
            return instruction, adapted_tools
        
        # For subsequent attempts, use LLM to rewrite the instruction
        try:
            # Create a prompt for the LLM to rewrite the instruction
            system_prompt = """You are an AI assistant that helps rewrite instructions that failed due to API errors.
Your task is to rewrite the instruction to make it more concise while preserving its core intent.=
DO NOT add explanations or commentary - just provide the rewritten instruction."""

            user_prompt = f"""The following instruction caused an error when processed:

ORIGINAL INSTRUCTION:
{instruction}

ERROR MESSAGE:
{error_str}

Please rewrite this instruction based on the error message while maintaining its core intent.
"""

            # Call the API to rewrite the instruction
            response, rewrite_error = await self._create_chat_completion(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                model=self.model  # Use same model for consistency
            )
            
            if rewrite_error or not response:
                self.logger.warning(f"Failed to rewrite instruction using LLM", {
                    "error": str(rewrite_error) if rewrite_error else "No response"
                })
                # Fall back to simple truncation if LLM rewrite fails
                adaptation_factor = min(0.3 * (attempt + 1), 0.8)
                reduced_length = max(50, int(len(instruction) * (1 - adaptation_factor)))
                adapted_instruction = instruction[:reduced_length]
            else:
                # Extract the rewritten instruction
                adapted_instruction = response.choices[0].message.content.strip()
                
            # Always reduce tools count for each retry
            keep_count = max(1, int(tools_count * (1 - min(0.25 * attempt, 0.75))))
            adapted_tools = tools[:keep_count] if keep_count < tools_count else tools.copy()
            
            self.logger.debug(f"Adapting parameters for attempt {attempt+1}", {
                "original_instruction_length": len(instruction),
                "adapted_instruction_length": len(adapted_instruction),
                "original_tools_count": tools_count,
                "adapted_tools_count": len(adapted_tools),
                "rewritten_by_llm": rewrite_error is None and response is not None
            })
            
            return adapted_instruction, adapted_tools
            
        except Exception as e:
            self.logger.error(f"Error during instruction adaptation", {"error": str(e)})
            # In case of any error in adaptation, fall back to simple truncation
            if len(instruction) > 100:
                adapted_instruction = instruction[:100]
            else:
                adapted_instruction = instruction
                
            # Maybe reduce tools if we have many
            if tools_count > 2:
                adapted_tools = tools[:2]  # Keep just first two tools in case of error
            else:
                adapted_tools = tools.copy()
                
            return adapted_instruction, adapted_tools
    
    def _validate_tool_call(self, tool_call: Dict[str, Any], available_tools: List[str]) -> bool:
        """
        Validate that a tool call has the correct format and references an available tool.
        
        Args:
            tool_call: The tool call to validate
            available_tools: List of available tool names
            
        Returns:
            True if the tool call is valid, False otherwise
        """
        # Check that the required fields are present
        if not isinstance(tool_call, dict):
            self.logger.error("Tool call is not a dictionary")
            return False
            
        if "type" not in tool_call or tool_call["type"] != "function":
            self.logger.error("Tool call is not a function call")
            return False
            
        if "function" not in tool_call or not isinstance(tool_call["function"], dict):
            self.logger.error("Tool call has no function object")
            return False
            
        function = tool_call["function"]
        if "name" not in function or "arguments" not in function:
            self.logger.error("Function object missing name or arguments")
            return False
            
        # Verify that the tool exists
        tool_name = function["name"]
        if tool_name not in available_tools:
            self.logger.error("Tool not in available tools", {"tool": tool_name})
            return False
            
        # Verify that the arguments is a valid JSON object (dictionary)
        arguments = function["arguments"]
        if not isinstance(arguments, dict):
            self.logger.error("Arguments must be a JSON object (dictionary)")
            return False
            
        # Everything looks good
        return True
            
    def generate_call_id(self) -> str:
        """
        Generate a unique ID for a tool call.
        
        Returns:
            Unique ID string
        """
        return f"call_{str(uuid.uuid4())[:8]}"


class ToolCallHelper_v2:
    def __init__(self, config: ConfigManager):
        self.config = config
        
        # Push component name to call path
        self.config.push_to_call_path("tool_call_helper_v2")
        
        # Initialize logger
        self.logger = get_logger(self.config.get_call_path())
        
        # Load configuration with defaults
        self.max_retries = self.config.get('tool_calling.max_retries', 5)
        self.base_url = self.config.get('tool_calling.base_url', 'https://api.deepseek.com')
        self.api_key = self.config.get('tool_calling.api_key', self.config.get('deepseek.api_key'))
        self.model = self.config.get('tool_calling.model', 'deepseek-chat')
        self.temperature = self.config.get('tool_calling.temperature', 0)
        
        self.client = None
        
        self.logger.debug("ToolCallHelper_v2 initialized", {
            "model": self.model,
            "max_retries": self.max_retries
        })
    
    async def initialize_client(self) -> OpenAI:
        """
        Initialize the OpenAI-compatible client.
        
        Returns:
            Configured OpenAI client
        """
        if self.client is None:
            self.logger.debug("Initializing OpenAI client", {"base_url": self.base_url})
            self.client = OpenAI(
                base_url=self.base_url,
                api_key=self.api_key
            )
        return self.client

    async def _create_chat_completion(self, **kwargs) -> Tuple[Optional[Any], Optional[Exception]]:
        """
        Handle API call to the model provider.
        
        Args:
            **kwargs: Arguments to pass to the chat completions API
            
        Returns:
            Tuple containing:
            - The API response or None if failed
            - The exception if an error occurred, or None if successful
        """
        try:
            # Make sure client is initialized
            if not self.client:
                await self.initialize_client()
                
            # Add model if not provided
            if 'model' not in kwargs:
                kwargs['model'] = self.model
            kwargs['temperature'] = self.temperature
            
            # Set max_tokens dynamically if not explicitly provided
            if 'max_tokens' not in kwargs and 'messages' in kwargs:
                input_tokens = sum(len(m.get("content", "")) for m in kwargs['messages']) // 4  # Rough estimate
                max_output_tokens = max(512, 8192 - input_tokens - 50)  # 50 tokens buffer
                kwargs['max_tokens'] = max_output_tokens
                
            # Call the API
            self.logger.debug("Calling chat completion API", {
                "model": kwargs.get('model'),
                "max_tokens": kwargs.get('max_tokens')
            })
            result = self.client.chat.completions.create(**kwargs)
            self.logger.debug("API call successful")
            return await result if hasattr(result, "__await__") else result, None
        except Exception as e:
            error = handle_error(e, {"kwargs": kwargs})
            self.logger.error(f"API call error", {"error": str(error)})
            return None, error

    async def call_tool(self, instruction: str, tools: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        """
        Execute a tool call using adaptive retry mechanism.
        Can generate multiple tool calls from a single instruction.
        
        Args:
            instruction: The instruction to execute
            tools: List of available tools
            
        Returns:
            Tuple containing:
            - List of valid tool calls in OpenAI format
            - Stats dictionary with success/failure information
        """
        stats = {
            "attempts": 1,
            "success": False,
            "valid_calls": 0,
            "invalid_calls": 0,
            "total_calls": 0,
            "errors": []
        }
        
        self.logger.info("Starting tool call processing", {"tools_available": len(tools)})
        
        try:
            # Try to repair and parse the JSON from the instruction
            # fixed_json = repair_json(instruction)
            self.logger.debug("Parsing instruction JSON", {"instruction_length": len(instruction)})
            parsed_json = json.loads(instruction)
            
            # Repair and validate the instruction with available tools
            self.logger.debug("Starting instruction repair and validation")
            repaired_tool_calls, repair_stats = await self.repair_instruction(parsed_json, tools)
            stats.update(repair_stats)
            
            if repaired_tool_calls:
                stats["success"] = True
                stats["total_calls"] = len(repaired_tool_calls)
                stats["valid_calls"] = len(repaired_tool_calls)
                self.logger.info("Successfully repaired tool calls", {
                    "total_calls": len(repaired_tool_calls),
                    "stats": repair_stats
                })
                return repaired_tool_calls, stats
            else:
                stats["errors"].append("Failed to repair tool calls")
                self.logger.warning("Failed to repair any tool calls", {"stats": repair_stats})
                return [], stats
                
        except json.JSONDecodeError as e:
            error_msg = f"JSON parsing error: {str(e)}"
            stats["errors"].append(error_msg)
            self.logger.error(error_msg, {"error_location": str(e.pos)})
            return [], stats
        except Exception as e:
            error_msg = f"Unexpected error in call_tool: {str(e)}"
            stats["errors"].append(error_msg)
            self.logger.error(error_msg, {"error_type": type(e).__name__})
            return [], stats
            
    async def repair_instruction(self, parsed_json: Dict[str, Any], available_tools: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        """
        Validates and repairs tool calls in the instruction.
        
        Args:
            parsed_json: Parsed JSON from the instruction
            available_tools: List of available tools
            
        Returns:
            Tuple containing:
            - List of repaired tool calls
            - Stats dictionary with repair information
        """
        repair_stats = {
            "validated_calls": 0,
            "repaired_calls": 0,
            "failed_repairs": 0,
            "param_optimizations": 0
        }
        
        # Extract available tool names and their parameters
        tool_map = {}
        for tool in available_tools:
            tool_name = tool['function']['name']
            parameters = {}
            if 'parameters' in tool['function'] and 'properties' in tool['function']['parameters']:
                parameters = tool['function']['parameters']['properties']
            tool_map[tool_name] = {
                'parameters': parameters,
                'description': tool['function'].get('description', ''),
                'required': tool['function'].get('parameters', {}).get('required', [])
            }
        
        self.logger.debug("Available tools mapped", {
            "tool_count": len(tool_map), 
            "tools": list(tool_map.keys())
        })
        
        # Parameter value mapping for optimization
        param_value_map = {}
        
        # Process tool calls
        result_tool_calls = []
        
        if "tool_calls" not in parsed_json or not isinstance(parsed_json["tool_calls"], list):
            self.logger.warning("No valid tool_calls array found in JSON", {
                "json_keys": list(parsed_json.keys())
            })
            return [], repair_stats
            
        tool_calls_array = parsed_json["tool_calls"]
        self.logger.debug("Processing tool calls array", {"count": len(tool_calls_array)})
        
        for call_idx, call_data in enumerate(tool_calls_array):
            self.logger.debug(f"Processing tool call", {"index": call_idx})
            
            if "function" not in call_data:
                self.logger.warning("Tool call missing function object", {"index": call_idx})
                repair_stats["failed_repairs"] += 1
                continue
            
            function_data = call_data.get("function", {})
            tool_name = function_data.get("name", "")
            arguments = function_data.get("arguments", {})
            
            self.logger.debug(f"Tool call details", {
                "index": call_idx,
                "tool": tool_name,
                "arg_count": len(arguments) if isinstance(arguments, dict) else 0
            })
            
            # Ensure arguments is a dictionary
            if isinstance(arguments, str):
                try:
                    self.logger.debug("Converting string arguments to JSON", {"index": call_idx})
                    arguments = json.loads(arguments)
                except json.JSONDecodeError as e:
                    self.logger.warning("Failed to parse arguments string as JSON", {
                        "index": call_idx,
                        "error": str(e)
                    })
                    arguments = {}
            
            # Check if tool exists
            valid_tool_name = tool_name
            if tool_name not in tool_map:
                self.logger.warning(f"Invalid tool name", {
                    "tool": tool_name,
                    "available_tools": list(tool_map.keys())
                })
                
                # Find closest match
                self.logger.debug(f"Attempting to find closest tool match", {"invalid_tool": tool_name})
                closest_tool = await self._find_closest_tool(tool_name, tool_map, function_data)
                
                if closest_tool:
                    valid_tool_name = closest_tool
                    repair_stats["repaired_calls"] += 1
                    self.logger.info(f"Repaired invalid tool name", {
                        "original": tool_name,
                        "repaired": closest_tool
                    })
                else:
                    self.logger.error(f"Failed to find closest tool match", {"invalid_tool": tool_name})
                    repair_stats["failed_repairs"] += 1
                    continue
            
            # Validate and optimize parameters
            valid_args = {}
            if valid_tool_name in tool_map:
                valid_params = tool_map[valid_tool_name]['parameters']
                required_params = tool_map[valid_tool_name]['required']
                
                self.logger.debug(f"Validating parameters", {
                    "tool": valid_tool_name,
                    "valid_params": list(valid_params.keys()),
                    "required_params": required_params
                })
                
                # Check each parameter
                for param_name, param_value in arguments.items():
                    # If parameter exists in tool definition
                    if param_name in valid_params:
                        # Optimize large parameter values
                        if isinstance(param_value, str) and len(param_value) > 100:
                            param_id = f"PARAM_{len(param_value_map)}"
                            param_value_map[param_id] = param_value
                            valid_args[param_name] = param_id
                            repair_stats["param_optimizations"] += 1
                            self.logger.debug(f"Optimized large parameter", {
                                "param": param_name,
                                "original_size": len(param_value),
                                "placeholder": param_id
                            })
                        else:
                            valid_args[param_name] = param_value
                    else:
                        self.logger.warning(f"Invalid parameter name", {
                            "tool": valid_tool_name, 
                            "param": param_name,
                            "valid_params": list(valid_params.keys())
                        })
                
                # Add required parameters if missing
                missing_required = []
                for req_param in required_params:
                    if req_param not in valid_args:
                        valid_args[req_param] = "" # Empty placeholder for required params
                        missing_required.append(req_param)
                        
                if missing_required:
                    self.logger.warning(f"Added missing required parameters", {
                        "tool": valid_tool_name,
                        "missing_params": missing_required
                    })
            
            # Create repaired tool call
            call_id = call_data.get("id", f"call_{str(uuid.uuid4())[:8]}")
            repaired_call = {
                "id": call_id,
                "type": "function",
                "function": {
                    "name": valid_tool_name,
                    "arguments": valid_args
                }
            }
            
            result_tool_calls.append(repaired_call)
            repair_stats["validated_calls"] += 1
            self.logger.debug(f"Validated tool call", {
                "index": call_idx,
                "tool": valid_tool_name,
                "param_count": len(valid_args)
            })
        
        # Restore optimized parameter values
        if param_value_map:
            self.logger.debug(f"Restoring optimized parameters", {"count": len(param_value_map)})
            
        for tool_call in result_tool_calls:
            arguments = tool_call["function"]["arguments"]
            for param_name, param_value in list(arguments.items()):
                if isinstance(param_value, str) and param_value.startswith("PARAM_"):
                    if param_value in param_value_map:
                        arguments[param_name] = param_value_map[param_value]
                        self.logger.debug(f"Restored parameter value", {
                            "param": param_name,
                            "tool": tool_call["function"]["name"],
                            "placeholder": param_value,
                            "restored_size": len(param_value_map[param_value])
                        })
        
        self.logger.info(f"Repair instruction completed", {
            "original_count": len(parsed_json.get("tool_calls", [])),
            "repaired_count": len(result_tool_calls),
            "stats": repair_stats
        })
        
        return result_tool_calls, repair_stats
    
    async def _find_closest_tool(self, invalid_tool: str, tool_map: Dict[str, Any], function_data: Dict[str, Any]) -> Optional[str]:
        """
        Find the closest matching tool using LLM assistance.
        
        Args:
            invalid_tool: Invalid tool name
            tool_map: Map of available tools
            function_data: Function data with arguments
            
        Returns:
            Name of closest matching tool or None if not found
        """
        try:
            self.logger.debug(f"Finding closest tool match for '{invalid_tool}'")
            
            # Create a list of available tools with descriptions
            tool_descriptions = []
            for name, info in tool_map.items():
                params = ", ".join(info['parameters'].keys())
                tool_descriptions.append(f"- {name}: {info['description']}\n  Parameters: {params}")
            
            available_tools_str = "\n".join(tool_descriptions)
            
            # Format the invalid tool with its arguments
            args_str = json.dumps(function_data.get("arguments", {}), indent=2)
            
            # Create prompt for LLM
            system_prompt = """You are a tool matching expert. Your task is to find the closest matching 
tool from the available tools list that matches the intent of the invalid tool call.
Only respond with the exact name of the closest matching tool - no explanation or other text."""
            
            user_prompt = f"""Invalid tool: {invalid_tool}
Arguments: {args_str}

Available tools:
{available_tools_str}

What is the closest matching tool from the available tools list? Respond with ONLY the exact tool name."""
            
            # Call LLM for tool recommendation
            self.logger.debug(f"Calling LLM for tool recommendation")
            response, error = await self._create_chat_completion(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.1,
                max_tokens=50
            )
            
            if error:
                self.logger.warning(f"LLM recommendation error", {"error": str(error)})
                return None
                
            if not response:
                self.logger.warning(f"Empty LLM response for tool recommendation")
                return None
                
            # Extract suggested tool name
            suggested_tool = response.choices[0].message.content.strip()
            self.logger.debug(f"LLM suggested tool", {"suggestion": suggested_tool})
            
            # Validate the suggested tool exists
            if suggested_tool in tool_map:
                self.logger.info(f"Using LLM suggested tool", {
                    "invalid": invalid_tool,
                    "suggestion": suggested_tool
                })
                return suggested_tool
            else:
                self.logger.warning(f"LLM suggested invalid tool", {
                    "suggestion": suggested_tool,
                    "valid_tools": list(tool_map.keys())
                })
                
                # Fall back to best effort string matching
                self.logger.debug(f"Falling back to string similarity matching")
                best_match = None
                best_score = 0
                for tool_name in tool_map.keys():
                    # Simple similarity score based on common characters
                    score = sum(1 for a, b in zip(invalid_tool, tool_name) if a == b)
                    if score > best_score:
                        best_score = score
                        best_match = tool_name
                
                # Only return if we have a reasonable match
                similarity_threshold = len(invalid_tool) * 0.3
                if best_score > similarity_threshold:
                    self.logger.info(f"Found similar tool via string matching", {
                        "invalid": invalid_tool,
                        "match": best_match,
                        "score": best_score,
                        "threshold": similarity_threshold
                    })
                    return best_match
                else:
                    self.logger.warning(f"No similar tool found", {
                        "invalid": invalid_tool,
                        "best_score": best_score,
                        "threshold": similarity_threshold
                    })
                    
                return None
                
        except Exception as e:
            self.logger.error(f"Error finding closest tool", {
                "invalid_tool": invalid_tool,
                "error": str(e),
                "error_type": type(e).__name__
            })
            return None

class ToolCallFactory:
    def __init__(self, config: ConfigManager):
        self.config = config

    def create_tool_call_helper(self):
        if self.config.get('tool_calling.version') == 'stable':
            return ToolCallHelper_v1(self.config)
        elif self.config.get('tool_calling.version') == 'turbo':
            return ToolCallHelper_v2(self.config)
        else:
            raise ValueError(f"Unsupported tool calling version: {self.config.get('tool_calling.version')}")

    def create_tool_call_instruction(self):
        # Instructions for the main reasoner model on how to request tool calls with ReAct methodology
        TOOL_REQUEST_INSTRUCTIONS_v1 = """You operate using the ReAct (Reasoning + Acting) methodology, with support for both sequential and parallel tool execution.

        GENERAL PRINCIPLES:
        1. First THINK about what information you need and which tools would provide it
        2. Then REQUEST appropriate tools using the <tool_request> tag
        3. After receiving results, OBSERVE the outcomes
        4. Then DECIDE whether to use more tools or provide a final answer

        SEQUENTIAL VS PARALLEL TOOL CALLS:
        - SEQUENTIAL: Use when the output of one tool is needed as input for another tool
        - PARALLEL: Use when multiple independent pieces of information are needed at once

        For SEQUENTIAL dependencies (when tools depend on each other's results):
        - Request ONE tool at a time using a single <tool_request> tag
        - Wait for each result before requesting the next tool
        - Use previous results to inform subsequent tool requests

        For PARALLEL execution (when tools don't depend on each other):
        - You may include MULTIPLE <tool_request> tags in one response
        - Each tag should contain ONE specific tool instruction
        - All parallel tools will execute before you receive any results

        TO REQUEST TOOLS:
        <tool_request>Clear instruction for exactly ONE tool call</tool_request>

        EXAMPLES:

        Example 1 - Sequential dependency (correct approach):
        User: Find security vulnerabilities in my code and fix them.
        Assistant: I'll first search for potential vulnerabilities.
        <tool_request>Scan the codebase for security vulnerabilities.</tool_request>

        [After receiving scan results]
        Assistant: I found some vulnerabilities. Now I'll fix the most critical one.
        <tool_request>Apply security patch to fix SQL injection in login.php.</tool_request>

        Example 2 - Parallel execution (correct approach):
        User: Summarize today's weather and news headlines.
        Assistant: I'll gather both weather and news information for you.
        <tool_request>Get today's weather forecast for the user's location.</tool_request>
        <tool_request>Retrieve today's top news headlines.</tool_request>

        Example 3 - Mixed approach:
        User: Compare performance metrics across our three products and suggest improvements.
        Assistant: I'll gather performance data for all products simultaneously.
        <tool_request>Get performance metrics for Product A.</tool_request>
        <tool_request>Get performance metrics for Product B.</tool_request>
        <tool_request>Get performance metrics for Product C.</tool_request>

        [After receiving all metrics]
        Assistant: Now I'll analyze which product needs the most improvement.
        <tool_request>Run detailed analysis on Product B's performance bottlenecks.</tool_request>

        If no tool is needed, simply provide a direct answer without any <tool_request> tags."""


        TOOL_REQUEST_INSTRUCTIONS_v2 = """You operate using the ReAct (Reasoning + Acting) methodology, with support for both sequential and parallel tool execution.

        GENERAL PRINCIPLES:
        1. First THINK about what information you need and which tools would provide it
        2. Then REQUEST appropriate tools using JSON format
        3. After receiving results, OBSERVE the outcomes
        4. Then DECIDE whether to use more tools or provide a final answer

        SEQUENTIAL VS PARALLEL TOOL CALLS:
        - SEQUENTIAL: Use when the output of one tool is needed as input for another tool
        - PARALLEL: Use when multiple independent pieces of information are needed at once

        For SEQUENTIAL dependencies (when tools depend on each other's results):
        - Request ONE tool at a time using a single JSON object
        - Wait for each result before requesting the next tool
        - Use previous results to inform subsequent tool requests

        For PARALLEL execution (when tools don't depend on each other):
        - Include MULTIPLE function calls in the tool_calls array
        - Each function should contain ONE specific tool instruction
        - All parallel tools will execute before you receive any results

        TO REQUEST TOOLS, OUTPUT JSON IN THIS FORMAT:
        <tool_request>
        {
            "tool_calls": [
                {
                    "function": {
                        "name": "tool_name",
                        "arguments": {
                            "param1": "value1",
                            "param2": "value2"
                        }
                    }
                }
            ]
        }
        </tool_request>

        -----

        EXAMPLES:

        Example 1 - Sequential dependency (correct approach):
        User: Find security vulnerabilities in my code and fix them.
        Assistant: 
        <tool_request>
        {
            "tool_calls": [
                {
                    "function": {
                        "name": "scan_security_vulnerabilities",
                        "arguments": {
                            "target": "codebase"
                        }
                    }
                }
            ]
        }
        </tool_request>
        [After receiving scan results]
        Assistant:
        <tool_request>
        {
            "tool_calls": [
                {
                    "function": {
                        "name": "apply_security_patch",
                        "arguments": {
                            "file": "login.php",
                            "vulnerability_type": "sql_injection"
                        }
                    }
                }
            ]
        }
        </tool_request>
        Example 2 - Parallel execution (correct approach):
        User: Summarize today's weather and news headlines.
        Assistant:
        <tool_request>
        {
            "tool_calls": [
                {
                    "function": {
                        "name": "get_weather_forecast",
                        "arguments": {
                            "location": "user_location"
                        }
                    }
                },
                {
                    "function": {
                        "name": "get_news_headlines",
                        "arguments": {
                            "category": "top"
                        }
                    }
                }
            ]
        }
        </tool_request>
        Example 3 - Mixed approach:
        User: Compare performance metrics across our three products and suggest improvements.
        Assistant:
        <tool_request>
        {
            "tool_calls": [
                {
                    "function": {
                        "name": "get_performance_metrics",
                        "arguments": {
                            "product": "Product A"
                        }
                    }
                },
                {
                    "function": {
                        "name": "get_performance_metrics",
                        "arguments": {
                            "product": "Product B"
                        }
                    }
                },
                {
                    "function": {
                        "name": "get_performance_metrics",
                        "arguments": {
                            "product": "Product C"
                        }
                    }
                }
            ]
        }
        </tool_request>
        [After receiving all metrics]
        Assistant:
        <tool_request>
        {
            "tool_calls": [
                {
                    "function": {
                        "name": "analyze_performance_bottlenecks",
                        "arguments": {
                            "product": "Product B"
                        }
                    }
                }
            ]
        }
        </tool_request>
        If no tool is needed, simply provide a direct answer without any JSON tool call format.
        ------

        IMPORTANT RULES:
        1. ONLY use tool names from the list in the Available tools section - never invent new tool names
        2. ONLY use parameter names that are listed for each tool - never invent new parameters
        3. YOU CAN USE MULTIPLE TOOLS OR THE SAME TOOL MULTIPLE TIMES if the request requires it
        4. If only one tool is needed, still use the proper array format with a single element
        

        """



        if self.config.get('tool_calling.version') == 'stable':
            return TOOL_REQUEST_INSTRUCTIONS_v1
        elif self.config.get('tool_calling.version') == 'turbo':
            return TOOL_REQUEST_INSTRUCTIONS_v2
        else:
            raise ValueError(f"Unsupported tool calling version: {self.config.get('tool_calling.version')}")