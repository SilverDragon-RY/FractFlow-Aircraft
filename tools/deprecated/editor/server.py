"""
OpenHands ACI Tool Server Module

This module provides a FastMCP server that exposes OpenHands ACI (Auto Code Intelligence) 
operations as tools. It wraps the core OpenHands ACI functionality in a server interface 
that can be used by the EnvisionCore framework.

The server provides tools for:
- Code editing and file operations (view, create, modify, etc.)
- Code linting and static analysis
- Shell command execution
- File caching for performance optimization

Author: Ying-Cong Chen (yingcong.ian.chen@gmail.com)
Date: 2025-04-27
License: MIT License
"""

from mcp.server.fastmcp import FastMCP
import sys
from pathlib import Path
import os
from typing import Optional, List, Dict, Any, Union

# Add the parent directory to the path so we can import the modules
sys.path.insert(0, str(Path(__file__).parent))

# Import the actual modules available in OpenHands ACI
from openhands_aci.editor import file_editor, Command, OHEditor
from openhands_aci.editor.file_cache import FileCache
from openhands_aci.linter import DefaultLinter
from openhands_aci.utils.shell import run_shell_cmd

# Initialize MCP server
mcp = FastMCP("openhands_aci_tool")

def normalize_path(path: str) -> str:
    """
    Normalize a file path by expanding ~ to user's home directory
    and resolving relative paths.
    
    Parameters:
        path (str): The input path to normalize
        
    Returns:
        str: The normalized absolute path
    """
    # Expand ~ to user's home directory
    expanded_path = os.path.expanduser(path)
    
    # Convert to absolute path if relative
    if not os.path.isabs(expanded_path):
        expanded_path = os.path.abspath(expanded_path)
        
    return expanded_path

@mcp.tool()
async def view_file(path: str, view_range: Optional[List[int]] = None):
    """
    View the content of a file.
    
    Retrieves and returns the content of the specified file. Can return a specific range of lines if requested.
    
    Parameters:
        path (str): Absolute path to the file to view
        view_range (List[int], optional): Lines to view: [start, end] (1-based indexing)
    
    Returns:
        str: Output block in format <oh_aci_output_{id}>...<\oh_aci_output_{id}>
            Includes JSON with keys:
            - 'path': The file path
            - 'content': The file content or specified range
            - 'error': Error message if operation failed
    
    Raises:
        ValueError: If the path is invalid or file cannot be read
        
    Notes:
        - For binary or oversized files, the operation will fail
        - If view_range is specified, only the specified lines will be returned
        - Line numbers are 1-based (first line is line 1)
    
    Typical Use Cases:
        - Inspect file content: view the entire file
        - Examine specific text section: view a range of lines in any file
    """
    normalized_path = normalize_path(path)
    result = file_editor(
        command="view",
        path=normalized_path,
        view_range=view_range,
    )
    return result

@mcp.tool()
async def create_file(path: str, file_text: str, enable_linting: bool = False):
    """
    Create a new file with the specified content.
    
    Creates a new file at the specified path with the provided content. The operation will FAIL if the file already exists.
    
    Parameters:
        path (str): Absolute path where the new file should be created
        file_text (str): Content to write to the new file
        enable_linting (bool, optional): Run linting on the content after creation (default: False)
    
    Returns:
        str: Output block in format <oh_aci_output_{id}>...<\oh_aci_output_{id}>
            Includes JSON with keys:
            - 'path': The file path
            - 'new_content': The content written to the file
            - 'error': Error message if operation failed
            - 'prev_exist': Always false for successful creation
    
    Raises:
        ValueError: If the file already exists or path is invalid
        
    Notes:
        - DONOT use this tool to overwrite an existing file.
        - Use a delete operation first if overwriting an existing file is intended
    
    Typical Use Cases:
        - Create new script: create a new Python file with initial code
        - Generate configuration file: create JSON or YAML configuration
    """
    normalized_path = normalize_path(path)
    result = file_editor(
        command="create",
        path=normalized_path,
        file_text=file_text,
        enable_linting=enable_linting,
    )
    return result

@mcp.tool()
async def str_replace_in_file(path: str, old_str: str, new_str: str, enable_linting: bool = False):
    """
    Replace an exact text match in a file.
    
    Searches for the exact text match in the file and replaces it with the new string.
    The operation fails if the match is not found exactly once in the file.
    
    Parameters:
        path (str): Absolute path to the file to modify
        old_str (str): String to search for and replace (must appear exactly once)
        new_str (str): String to insert in place of old_str
        enable_linting (bool, optional): Run linting after edit (default: False)
    
    Returns:
        str: Output block in format <oh_aci_output_{id}>...<\oh_aci_output_{id}>
            Includes JSON with keys:
            - 'path': The file path
            - 'old_content': The original file content
            - 'new_content': The modified file content
            - 'error': Error message if operation failed
    
    Raises:
        ValueError: If old_str is not found exactly once in the file
        
    Notes:
        - The match must be exact and must appear exactly once
        - If multiple matches are found, the operation will fail
        - If no matches are found, the operation will fail
        - For binary or oversized files, the operation will fail
    
    Typical Use Cases:
        - Update function implementation: replace entire function body
        - Fix typo: replace misspelled variable name
        - Change configuration value: replace exact key-value pair
    """
    normalized_path = normalize_path(path)
    result = file_editor(
        command="str_replace",
        path=normalized_path,
        old_str=old_str,
        new_str=new_str,
        enable_linting=enable_linting,
    )
    return result

@mcp.tool()
async def insert_in_file(path: str, insert_line: int, new_str: str, enable_linting: bool = False):
    """
    Insert text at a specific line in a file.
    
    Inserts the provided text at the specified line number in the file.
    Line 1 represents insertion before the first line of the file.
    
    Parameters:
        path (str): Absolute path to the file to modify
        insert_line (int): Line number for insertion (STARTING FROM 1, 1 = before first line)
        new_str (str): Text to insert at the specified line
        enable_linting (bool, optional): Run linting after edit (default: False)
    
    Returns:
        str: Output block in format <oh_aci_output_{id}>...<\oh_aci_output_{id}>
            Includes JSON with keys:
            - 'path': The file path
            - 'old_content': The original file content
            - 'new_content': The modified file content
            - 'error': Error message if operation failed
    
    Raises:
        ValueError: If insert_line is less than 1 or exceeds file length
        
    Notes:
        - Line numbering starts at 1 (first line is line 1)
        - Use insert_line=1 to insert at the beginning of the file
        - Use a line number beyond the end of the file to append to the file
        - For binary or oversized files, the operation will fail
    
    Typical Use Cases:
        - Add import statement: insert at the top of the file (line 1)
        - Add new method: insert at specific line in a class
        - Add comment: insert comment above code at specific line
    """
    # Adjust line number from 1-based (API) to 0-based (internal implementation)
    adjusted_line = insert_line - 1
    
    normalized_path = normalize_path(path)
    result = file_editor(
        command="insert",
        path=normalized_path,
        insert_line=adjusted_line,
        new_str=new_str,
        enable_linting=enable_linting,
    )
    return result

@mcp.tool()
async def remove_lines_in_file(path: str, start_line: int, end_line: int = -1, enable_linting: bool = False):
    """
    Remove specified lines from a file.
    
    Removes the lines between start_line and end_line (inclusive).
    If end_line is -1, removes from start_line to the end of the file.
    
    Parameters:
        path (str): Absolute path to the file to modify
        start_line (int): Starting line number to remove (STARTING FROM 1)
        end_line (int, optional): Ending line number to remove (inclusive). Default is -1 (end of file)
        enable_linting (bool, optional): Run linting after edit (default: False)
    
    Returns:
        str: Output block in format <oh_aci_output_{id}>...<\oh_aci_output_{id}>
            Includes JSON with keys:
            - 'path': The file path
            - 'old_content': The original file content
            - 'new_content': The modified file content
            - 'error': Error message if operation failed
    
    Raises:
        ValueError: If the file doesn't exist or can't be modified
        
    Notes:
        - Line numbering starts at 1 (first line is line 1)
        - If end_line is -1, it means removing from start_line to the end of the file
        - If the line range is outside file boundaries, the function will adjust accordingly
        - For binary or oversized files, the operation will fail
    
    Typical Use Cases:
        - Remove unused code: delete deprecated functions or imports
        - Clean up files: remove debug statements or commented code
        - Refactor: remove code blocks that have been moved elsewhere
    """
    try:
        normalized_path = normalize_path(path)
        # Read the file directly
        with open(normalized_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            original_content = ''.join(lines)
        
        # Calculate actual line numbers
        total_lines = len(lines)
        
        # Convert start_line to 0-indexed
        start_idx = max(0, start_line - 1)
        
        # Handle end_line
        if end_line == -1:
            end_idx = total_lines - 1
        else:
            end_idx = min(end_line - 1, total_lines - 1)
        
        # Remove the specified lines
        new_lines = lines[:start_idx] + lines[end_idx + 1:]
        new_content = ''.join(new_lines)
        
        # Use str_replace to perform the edit
        result = file_editor(
            command="str_replace",
            path=normalized_path,
            old_str=original_content,
            new_str=new_content,
            enable_linting=enable_linting,
        )
        
        return result
    except Exception as e:
        import json
        marker_id = "error"
        error_result = {
            "path": path,
            "error": f"Error removing lines: {str(e)}"
        }
        return f'<oh_aci_output_{marker_id}>\n{json.dumps(error_result)}\n</oh_aci_output_{marker_id}>'

@mcp.tool()
async def append_in_file(path: str, text_to_append: str, ensure_newline: bool = True, enable_linting: bool = False):
    """
    Append text to the end of a file.
    
    Adds the provided text at the end of the specified file. Optionally ensures the file
    ends with a newline before appending the new text.
    
    Parameters:
        path (str): Absolute path to the file to modify
        text_to_append (str): Text to append to the end of the file
        ensure_newline (bool, optional): Ensure file ends with newline before appending (default: True)
        enable_linting (bool, optional): Run linting after edit (default: False)
    
    Returns:
        str: Output block in format <oh_aci_output_{id}>...<\oh_aci_output_{id}>
            Includes JSON with keys:
            - 'path': The file path
            - 'old_content': The original file content
            - 'new_content': The modified file content
            - 'error': Error message if operation failed
    
    Raises:
        ValueError: If the file doesn't exist or can't be modified
        
    Notes:
        - If ensure_newline is True, a newline will be added before the new text if the file doesn't already end with one
        - For binary or oversized files, the operation will fail
        - If the file does not exist, the operation will fail (use create_file instead)
    
    Typical Use Cases:
        - Add new entries: append entries to a log file
        - Extend configurations: add new configuration options
        - Add imports: append import statements to a file
    """
    import json
    import uuid

    try:
        normalized_path = normalize_path(path)
        # Check if file exists
        if not os.path.exists(normalized_path):
            raise ValueError(f"File does not exist: {normalized_path}")
        
        # Read the file directly
        with open(normalized_path, 'r', encoding='utf-8') as f:
            original_content = f.read()
        
        # Prepare new content
        new_content = original_content
        
        # Ensure the file ends with a newline if requested
        if ensure_newline and original_content and not original_content.endswith('\n'):
            new_content += '\n'
        
        # Append the new text
        new_content += text_to_append
        
        # Write the modified content back to the file
        with open(normalized_path, 'w', encoding='utf-8') as f:
            f.write(new_content)
        
        # Run linting if enabled
        lint_results = None
        if enable_linting:
            # Determine language based on file extension
            _, ext = os.path.splitext(normalized_path)
            language_map = {
                '.py': 'python',
                '.js': 'javascript',
                '.ts': 'typescript',
                '.html': 'html',
                '.css': 'css',
                '.json': 'json'
            }
            language = language_map.get(ext.lower(), '')
            
            if language:
                linter = DefaultLinter()
                lint_results = linter.run(normalized_path, language)
        
        # Create result with unique identifier
        marker_id = str(uuid.uuid4())[:8]
        result = {
            "path": normalized_path,
            "old_content": original_content,
            "new_content": new_content
        }
        
        if lint_results:
            result["lint_results"] = lint_results
        
        # Format the result as required
        return f'<oh_aci_output_{marker_id}>\n{json.dumps(result)}\n</oh_aci_output_{marker_id}>'
    
    except Exception as e:
        marker_id = "error"
        error_result = {
            "path": path,
            "error": f"Error appending to file: {str(e)}"
        }
        return f'<oh_aci_output_{marker_id}>\n{json.dumps(error_result)}\n</oh_aci_output_{marker_id}>'

# @mcp.tool()
# async def undo_edit(path: str):
#     """
#     Revert the last edit made to a file.
    
#     Undoes the most recent modification made to the specified file,
#     restoring it to its previous state. Only affects the last operation.
    
#     Parameters:
#         path (str): Absolute path to the file whose last edit should be undone
    
#     Returns:
#         str: Output block in format <oh_aci_output_{id}>...<\oh_aci_output_{id}>
#             Includes JSON with keys:
#             - 'path': The file path
#             - 'old_content': The content after the undo (previously the content before the last edit)
#             - 'new_content': The content before the undo (previously the content after the last edit)
#             - 'error': Error message if operation failed
    
#     Raises:
#         ValueError: If there is no edit history for the specified file
        
#     Notes:
#         - Only the most recent edit can be undone
#         - If no edit history exists for the file, the operation will fail
#         - File creation cannot be undone using this function
#         - Multiple consecutive undos are not supported
    
#     Typical Use Cases:
#         - Revert accidental change: undo after making an incorrect modification
#         - Cancel experiment: undo after testing a code change that didn't work
#     """
#     normalized_path = normalize_path(path)
#     result = file_editor(
#         command="undo_edit",
#         path=normalized_path,
#     )
#     return result

@mcp.tool()
async def run_linter(code: str, language: str):
    """
    Run the linter on the provided code to identify potential issues and errors.
    
    Parameters:
        code (str): Source code to analyze
        language (str): Programming language of the code (e.g., 'python', 'javascript')
        
    Returns:
        Dict[str, Any]: A dictionary containing:
            - issues (List[Dict]): List of linting issues found, each with:
                - line (int): Line number where the issue was found
                - column (int): Column number where the issue was found
                - message (str): Description of the issue
                - severity (str): Severity level (e.g., 'error', 'warning')
                - rule (str): The linting rule that was violated
            - success (bool): Whether the linting operation completed successfully
            - error (str, optional): Error message if linting failed
            
    Raises:
        ValueError: If the language is not supported or the code cannot be parsed
    """
    linter = DefaultLinter()
    try:
        with open("temp_lint_file.py", "w") as f:
            f.write(code)
        
        results = linter.run("temp_lint_file.py", language)
        
        return {
            "issues": results,
            "success": True
        }
    except Exception as e:
        return {
            "issues": [],
            "success": False,
            "error": str(e)
        }
    finally:
        # Clean up temporary file
        if os.path.exists("temp_lint_file.py"):
            os.remove("temp_lint_file.py")

@mcp.tool()
async def execute_shell(cmd: str, timeout: float = 60.0):
    """
    Execute a shell command and return the result, including stdout and stderr.
    
    Parameters:
        cmd (str): Shell command to execute (runs in the server's environment)
        timeout (float, optional): Maximum time in seconds to wait for completion. Defaults to 60.0.
        
    Returns:
        Dict[str, Any]: A dictionary containing:
            - returncode (int): Return code of the command (0 typically indicates success)
            - stdout (str): Standard output captured from the command
            - stderr (str): Standard error captured from the command
            - success (bool): Whether the command completed successfully (returncode == 0)
            
    Raises:
        TimeoutError: If the command execution exceeds the specified timeout
        
    Notes:
        - Commands are executed in the server's environment, not the client's
        - For security reasons, certain commands may be restricted
        - Large output may be truncated
    """
    try:
        returncode, stdout, stderr = run_shell_cmd(cmd, timeout=timeout)
        return {
            "returncode": returncode,
            "stdout": stdout,
            "stderr": stderr,
            "success": returncode == 0
        }
    except TimeoutError as e:
        return {
            "returncode": -1,
            "stdout": "",
            "stderr": str(e),
            "success": False
        }

# @mcp.tool()
# async def cache_file(path: str, content: Optional[str] = None, directory: str = "/tmp/filecache"):
#     """
#     Cache a file for faster access or store content without writing to disk.
    
#     Parameters:
#         path (str): Path to identify the file in the cache (used as the cache key)
#         content (str, optional): Content to store. If None, tries to read from the path
#         directory (str, optional): Directory to store the cache. Defaults to "/tmp/filecache"
        
#     Returns:
#         Dict[str, Any]: A dictionary containing:
#             - path (str): Path used as the cache key
#             - success (bool): Whether the caching was successful
#             - error (str, optional): Error message if caching failed
            
#     Raises:
#         FileNotFoundError: If content is None and the file doesn't exist
#     """
#     file_cache = FileCache(directory=directory)
    
#     try:
#         normalized_path = normalize_path(path)
#         if content is None:
#             # Read and cache the content from the file
#             with open(normalized_path, 'r') as f:
#                 content = f.read()
        
#         file_cache.set(normalized_path, content)
        
#         return {
#             "path": normalized_path,
#             "success": True
#         }
#     except Exception as e:
#         return {
#             "path": path,
#             "success": False,
#             "error": str(e)
#         }

# @mcp.tool()
# async def get_cached_file(path: str, directory: str = "/tmp/filecache"):
#     """
#     Retrieve a file from the cache by its path key.
    
#     Parameters:
#         path (str): Path used as the cache key when the file was cached
#         directory (str, optional): Directory where the cache is stored. Defaults to "/tmp/filecache"
        
#     Returns:
#         Dict[str, Any]: A dictionary containing:
#             - path (str): Path used as the cache key
#             - content (str): The cached content, if found
#             - success (bool): Whether the retrieval was successful
#             - error (str, optional): Error message if retrieval failed
            
#     Raises:
#         KeyError: If the path is not found in the cache
#     """
#     file_cache = FileCache(directory=directory)
    
#     try:
#         normalized_path = normalize_path(path)
#         content = file_cache.get(normalized_path)
#         return {
#             "path": normalized_path,
#             "content": content,
#             "success": True
#         }
#     except KeyError:
#         return {
#             "path": path,
#             "success": False,
#             "error": f"Path '{path}' not found in cache"
#         }

if __name__ == "__main__":
    mcp.run(transport='stdio')
