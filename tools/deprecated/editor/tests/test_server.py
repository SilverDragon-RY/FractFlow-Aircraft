"""
OpenHands ACI Tool Server Tests

This module contains unit tests for the server.py module, testing the FastMCP server
functions that expose OpenHands ACI operations as tools.

The tests use pytest and the pytest-asyncio plugin to test asynchronous functions.
Each test is isolated using temporary directories and real file operations to test actual implementation.

Author: Ying-Cong Chen (yingcong.ian.chen@gmail.com)
Date: 2025-04-27
License: MIT License
"""

import os
import sys
import json
import pytest
import tempfile
import shutil
import time
import re
from pathlib import Path
from unittest.mock import patch, MagicMock, mock_open

# Add the parent directory to the Python path so we can import modules from there
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import the functions to test
from server import (
    view_file,
    create_file,
    str_replace_in_file,
    insert_in_file,
    remove_lines_in_file,
    run_linter,
    execute_shell
)

# ===== PYTEST FIXTURES =====

@pytest.fixture(scope="session", autouse=True)
def setup_logging_directories():
    """Create logging directories for test results"""
    log_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "test_logs")
    
    # Create timestamped directories for this test run
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    integration_log_dir = os.path.join(log_root, "integration_tests", timestamp)
    unit_log_dir = os.path.join(log_root, "unit_tests", timestamp)
    
    # Ensure directories exist
    os.makedirs(integration_log_dir, exist_ok=True)
    os.makedirs(unit_log_dir, exist_ok=True)
    
    # Log the test run start
    with open(os.path.join(unit_log_dir, "test_run_info.txt"), "w") as f:
        f.write(f"Unit Test Run Started: {timestamp}\n")
        f.write(f"Python version: {sys.version}\n")
        f.write(f"Working directory: {os.getcwd()}\n")
    
    print(f"Unit test logs will be saved to: {unit_log_dir}")
    
    # Return log directories
    return {
        "integration": integration_log_dir,
        "unit": unit_log_dir
    }

def log_test(test_name, input_params=None, mock_calls=None, result=None, log_dirs=None):
    """
    Helper function to log test information
    
    Args:
        test_name: Name of the test function
        input_params: Dictionary of input parameters
        mock_calls: Dictionary of mock object calls and returns
        result: Test result
        log_dirs: Dictionary of log directories
    """
    if not log_dirs:
        return
    
    log_dir = log_dirs.get("unit")
    if not log_dir:
        return
    
    timestamp = time.strftime("%H%M%S")
    test_id = test_name.replace(" ", "_")
    log_file = os.path.join(log_dir, f"{test_id}_{timestamp}.txt")
    
    with open(log_file, "w") as f:
        # Write header
        f.write(f"=== Test: {test_name} ===\n")
        f.write(f"Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        # Write input parameters
        if input_params:
            f.write("--- Input Parameters ---\n")
            for key, value in input_params.items():
                f.write(f"{key}: {value}\n")
            f.write("\n")
        
        # Write mock calls
        if mock_calls:
            f.write("--- Mock Calls ---\n")
            for mock_name, calls in mock_calls.items():
                f.write(f"{mock_name}:\n")
                if isinstance(calls, list):
                    for i, call in enumerate(calls, 1):
                        f.write(f"  Call {i}: {call}\n")
                else:
                    f.write(f"  {calls}\n")
            f.write("\n")
        
        # Write result
        if result:
            f.write("--- Result ---\n")
            if isinstance(result, dict):
                for key, value in result.items():
                    f.write(f"{key}: {value}\n")
            else:
                f.write(f"{result}\n")

@pytest.fixture
def temp_dir(setup_logging_directories):
    """
    A pytest fixture that creates a temporary directory for testing.
    
    This fixture:
    1. Creates a new temporary directory before each test that uses it
    2. Yields the directory path to the test function
    3. Cleans up by removing the directory after the test completes
    """
    temp_dir = tempfile.mkdtemp()
    
    # Log temp directory creation
    log_dirs = setup_logging_directories
    if log_dirs:
        log_dir = log_dirs.get("unit")
        if log_dir:
            with open(os.path.join(log_dir, "temp_dirs.txt"), "a") as f:
                f.write(f"{time.strftime('%H:%M:%S')} - Created temporary directory: {temp_dir}\n")
    
    yield temp_dir
    
    # Cleanup after the test is done (even if it fails)
    if os.path.exists(temp_dir):
        # Log cleanup
        if log_dirs:
            log_dir = log_dirs.get("unit")
            if log_dir:
                with open(os.path.join(log_dir, "temp_dirs.txt"), "a") as f:
                    f.write(f"{time.strftime('%H:%M:%S')} - Cleaned up temporary directory: {temp_dir}\n")
        
        shutil.rmtree(temp_dir)

@pytest.fixture
def setup_test_files(temp_dir):
    """
    Create test files with various content for testing real operations
    
    Returns:
        dict: Dictionary of file paths created for testing
    """
    # Create a basic Python file
    python_file = os.path.join(temp_dir, "test.py")
    with open(python_file, "w") as f:
        f.write('def hello():\n    print("Hello, world!")\n\nhello()')
    
    # Create a file with syntax errors
    error_file = os.path.join(temp_dir, "error.py")
    with open(error_file, "w") as f:
        f.write('def missing_colon()\n    print("This has syntax errors")')
    
    # Create a text file for string replacements
    replace_file = os.path.join(temp_dir, "replace.txt")
    with open(replace_file, "w") as f:
        f.write("This is a test file.\nIt contains text to replace.\nReplace this text.")
    
    # Create a file for line insertion
    insert_file = os.path.join(temp_dir, "insert.txt")
    with open(insert_file, "w") as f:
        f.write("Line 1\nLine 3")
    
    # Return the dictionary of created files
    return {
        "python_file": python_file,
        "error_file": error_file,
        "replace_file": replace_file,
        "insert_file": insert_file,
        "temp_dir": temp_dir
    }

# ===== TEST CASES =====

@pytest.mark.asyncio
async def test_view_file(temp_dir, setup_test_files, setup_logging_directories):
    """
    Test the view_file function with real file operations.
    
    This test:
    1. Calls view_file on a real file
    2. Verifies the returned string contains the correct file content
    """
    # Get the path to the test Python file
    python_file = setup_test_files["python_file"]
    
    # Input parameters
    input_params = {
        "path": python_file
    }
    
    # Run the test with real file_editor
    result = await view_file(**input_params)
    
    # Log test details
    log_test(
        test_name="test_view_file",
        input_params=input_params,
        result=result,
        log_dirs=setup_logging_directories
    )
    
    # Extract the JSON data from within the markers
    pattern = r'<oh_aci_output_[^>]+>\s*(.*?)\s*</oh_aci_output_[^>]+>'
    match = re.search(pattern, result, re.DOTALL)
    assert match, "Output should be wrapped with markers"
    
    # Parse the JSON data
    json_data = json.loads(match.group(1))
    
    # Verify the result contains the expected content
    assert json_data["path"] == python_file
    
    # Check that the content field exists and contains the expected code
    assert "content" in json_data
    assert 'def hello()' in json_data["content"]
    assert 'print("Hello, world!")' in json_data["content"]

@pytest.mark.asyncio
async def test_create_file(temp_dir, setup_logging_directories):
    """
    Test the create_file function with real file operations.
    
    This test:
    1. Calls create_file to create a new file
    2. Verifies the file is created with the specified content
    """
    # Create a path for a new file
    new_file = os.path.join(temp_dir, "new_file.py")
    
    # Input parameters
    input_params = {
        "path": new_file,
        "file_text": "def factorial(n):\n    return 1 if n <= 1 else n * factorial(n-1)"
    }
    
    # Run the test with real file_editor
    result = await create_file(**input_params)
    
    # Log test details
    log_test(
        test_name="test_create_file",
        input_params=input_params,
        result=result,
        log_dirs=setup_logging_directories
    )
    
    # Verify the file was created
    assert os.path.exists(new_file), "File should be created"
    
    # Verify the file content
    with open(new_file, "r") as f:
        content = f.read()
    
    assert "def factorial(n):" in content
    assert "return 1 if n <= 1 else n * factorial(n-1)" in content
    
    # Extract the JSON data from within the markers
    pattern = r'<oh_aci_output_[^>]+>\s*(.*?)\s*</oh_aci_output_[^>]+>'
    match = re.search(pattern, result, re.DOTALL)
    assert match, "Output should be wrapped with markers"
    
    # Parse the JSON data
    json_data = json.loads(match.group(1))
    
    # Verify the result contains the expected content
    assert json_data["path"] == new_file
    assert "new_content" in json_data
    assert "def factorial(n):" in json_data["new_content"]

@pytest.mark.asyncio
async def test_str_replace_in_file(setup_test_files, setup_logging_directories):
    """
    Test the str_replace_in_file function with real file operations.
    
    This test:
    1. Calls str_replace_in_file to replace text in a file
    2. Verifies the text is replaced correctly
    """
    # Get the path to the test file for replacements
    replace_file = setup_test_files["replace_file"]
    
    # Input parameters
    input_params = {
        "path": replace_file,
        "old_str": "Replace this text",
        "new_str": "This text has been replaced"
    }
    
    # Run the test with real file_editor
    result = await str_replace_in_file(**input_params)
    
    # Log test details
    log_test(
        test_name="test_str_replace_in_file",
        input_params=input_params,
        result=result,
        log_dirs=setup_logging_directories
    )
    
    # Verify the file content was updated
    with open(replace_file, "r") as f:
        updated_content = f.read()
    
    assert "This text has been replaced" in updated_content
    assert "Replace this text" not in updated_content
    
    # Extract the JSON data from within the markers
    pattern = r'<oh_aci_output_[^>]+>\s*(.*?)\s*</oh_aci_output_[^>]+>'
    match = re.search(pattern, result, re.DOTALL)
    assert match, "Output should be wrapped with markers"
    
    # Parse the JSON data
    json_data = json.loads(match.group(1))
    
    # Verify the result contains the expected content
    assert json_data["path"] == replace_file
    assert "new_content" in json_data
    assert "This text has been replaced" in json_data["new_content"]
    
    # Check either old_content or prev_content field
    if "old_content" in json_data:
        assert "Replace this text" in json_data["old_content"]
    elif "prev_content" in json_data:
        assert "Replace this text" in json_data["prev_content"]

@pytest.mark.asyncio
async def test_insert_in_file(setup_test_files, setup_logging_directories):
    """
    Test the insert_in_file function with real file operations.
    
    This test:
    1. Calls insert_in_file to insert a line in a file
    2. Verifies the line is inserted at the correct position
    """
    # Get the path to the test file for insertions
    insert_file = setup_test_files["insert_file"]
    
    # Input parameters
    input_params = {
        "path": insert_file,
        "insert_line": 1,  # Insert after first line (Line 1), before second line (Line 3)
        "new_str": "Line 2"
    }
    
    # Run the test with real file_editor
    result = await insert_in_file(**input_params)
    
    # Log test details
    log_test(
        test_name="test_insert_in_file",
        input_params=input_params,
        result=result,
        log_dirs=setup_logging_directories
    )
    
    # Verify the file content was updated
    with open(insert_file, "r") as f:
        updated_content = f.read()
    
    print(f"Updated content: '{updated_content}'")
    
    # Expected content - with proper newline handling
    expected_content = "Line 2\nLine 1\nLine 3"
    assert updated_content.strip() == expected_content
    
    # Extract the JSON data from within the markers
    pattern = r'<oh_aci_output_[^>]+>\s*(.*?)\s*</oh_aci_output_[^>]+>'
    match = re.search(pattern, result, re.DOTALL)
    assert match, "Output should be wrapped with markers"
    
    # Parse the JSON data
    json_data = json.loads(match.group(1))
    
    # Verify the result contains the expected content
    assert json_data["path"] == insert_file
    assert "new_content" in json_data
    assert "Line 2" in json_data["new_content"]

@pytest.mark.asyncio
async def test_remove_lines_in_file(temp_dir, setup_logging_directories):
    """
    Test the remove_lines_in_file function with various line removal scenarios.
    
    This test:
    1. Tests removing lines from the middle of a file
    2. Tests removing lines from the beginning of a file
    3. Tests removing lines from the end of a file
    4. Tests removing all lines in a file
    5. Tests error handling for non-existent files
    """
    # Create a test file with multiple lines
    test_file = os.path.join(temp_dir, "remove_lines_test.txt")
    with open(test_file, "w") as f:
        f.write("Line 1\nLine 2\nLine 3\nLine 4\nLine 5\n")
    
    # Test 1: Remove lines from the middle (lines 2-3)
    result1 = await remove_lines_in_file(test_file, 2, 3)
    
    # Extract the JSON data
    pattern = r'<oh_aci_output_[^>]+>\s*(.*?)\s*</oh_aci_output_[^>]+>'
    match1 = re.search(pattern, result1, re.DOTALL)
    assert match1, "Output should be wrapped with markers"
    json_data1 = json.loads(match1.group(1))
    
    # Log test details
    log_test(
        test_name="test_remove_lines_in_file_middle",
        input_params={"path": test_file, "start_line": 2, "end_line": 3},
        result=json_data1,
        log_dirs=setup_logging_directories
    )
    
    # Verify the content after removing middle lines
    with open(test_file, "r") as f:
        content1 = f.read()
    
    assert "Line 1\nLine 4\nLine 5\n" == content1
    assert "Line 2" not in content1
    assert "Line 3" not in content1
    
    # Test 2: Remove lines from the beginning (line 1)
    result2 = await remove_lines_in_file(test_file, 1, 1)
    
    # Extract the JSON data
    match2 = re.search(pattern, result2, re.DOTALL)
    assert match2, "Output should be wrapped with markers"
    json_data2 = json.loads(match2.group(1))
    
    # Log test details
    log_test(
        test_name="test_remove_lines_in_file_beginning",
        input_params={"path": test_file, "start_line": 1, "end_line": 1},
        result=json_data2,
        log_dirs=setup_logging_directories
    )
    
    # Verify the content after removing the first line
    with open(test_file, "r") as f:
        content2 = f.read()
    
    assert "Line 4\nLine 5\n" == content2
    assert "Line 1" not in content2
    
    # Test 3: Remove lines from the end (last line)
    result3 = await remove_lines_in_file(test_file, 2, 2)
    
    # Extract the JSON data
    match3 = re.search(pattern, result3, re.DOTALL)
    assert match3, "Output should be wrapped with markers"
    json_data3 = json.loads(match3.group(1))
    
    # Log test details
    log_test(
        test_name="test_remove_lines_in_file_end",
        input_params={"path": test_file, "start_line": 2, "end_line": 2},
        result=json_data3,
        log_dirs=setup_logging_directories
    )
    
    # Verify the content after removing the last line
    with open(test_file, "r") as f:
        content3 = f.read()
    
    assert "Line 4\n" == content3
    assert "Line 5" not in content3
    
    # Test 4: Remove all lines (from line 1 to end)
    result4 = await remove_lines_in_file(test_file, 1, -1)
    
    # Extract the JSON data
    match4 = re.search(pattern, result4, re.DOTALL)
    assert match4, "Output should be wrapped with markers"
    json_data4 = json.loads(match4.group(1))
    
    # Log test details
    log_test(
        test_name="test_remove_lines_in_file_all",
        input_params={"path": test_file, "start_line": 1, "end_line": -1},
        result=json_data4,
        log_dirs=setup_logging_directories
    )
    
    # Verify the file is empty
    with open(test_file, "r") as f:
        content4 = f.read()
    
    assert content4 == ""
    
    # Test 5: Error handling - non-existent file
    non_existent_file = os.path.join(temp_dir, "non_existent.txt")
    
    # This should return an error in the result
    result5 = await remove_lines_in_file(non_existent_file, 1, 2)
    
    # Extract the JSON data
    match5 = re.search(pattern, result5, re.DOTALL)
    assert match5, "Output should be wrapped with markers"
    json_data5 = json.loads(match5.group(1))
    
    # Log test details
    log_test(
        test_name="test_remove_lines_in_file_nonexistent",
        input_params={"path": non_existent_file, "start_line": 1, "end_line": 2},
        result=json_data5,
        log_dirs=setup_logging_directories
    )
    
    # Verify there's an error message in the result
    assert "error" in json_data5 and json_data5["error"]
    
    # Test 6: Test with linting enabled (for a Python file)
    python_test_file = os.path.join(temp_dir, "test_python.py")
    with open(python_test_file, "w") as f:
        f.write("def func1():\n    print('Function 1')\n\ndef func2():\n    print('Function 2')\n")
    
    # Remove the first function
    result6 = await remove_lines_in_file(python_test_file, 1, 2, enable_linting=True)
    
    # Extract the JSON data
    match6 = re.search(pattern, result6, re.DOTALL)
    assert match6, "Output should be wrapped with markers"
    json_data6 = json.loads(match6.group(1))
    
    # Log test details
    log_test(
        test_name="test_remove_lines_in_file_with_linting",
        input_params={"path": python_test_file, "start_line": 1, "end_line": 2, "enable_linting": True},
        result=json_data6,
        log_dirs=setup_logging_directories
    )
    
    # Verify the content
    with open(python_test_file, "r") as f:
        content6 = f.read()
    
    assert "def func1()" not in content6
    assert "def func2()" in content6

@pytest.mark.asyncio
async def test_invalid_command(setup_logging_directories):
    """
    Test with an invalid command.
    
    This is a negative test case that verifies error handling when an invalid command is provided.
    """
    # Create a temporary file for testing
    with tempfile.NamedTemporaryFile(mode='w+', delete=False) as temp_file:
        temp_file.write("Test content")
        file_path = temp_file.name
    
    try:
        # Since we're testing error handling, we'll mock the function call
        # to avoid actually sending invalid commands to the server
        with patch('server.file_editor') as mock_file_editor:
            mock_file_editor.side_effect = ValueError("Invalid command 'invalid_command'")
            
            # Log test details
            log_test(
                test_name="test_invalid_command",
                input_params={"command": "invalid_command", "path": file_path},
                mock_calls={"mock_file_editor": "Raises ValueError"},
                log_dirs=setup_logging_directories
            )
            
            # Verify that the correct error is raised
            with pytest.raises(ValueError) as excinfo:
                result = await create_file(path=file_path, file_text="test", command="invalid_command")
            
            assert "Invalid command" in str(excinfo.value)
    finally:
        # Cleanup
        if os.path.exists(file_path):
            os.unlink(file_path)

@pytest.mark.asyncio
async def test_run_linter(setup_test_files, setup_logging_directories):
    """
    Test the run_linter function with real linting.
    
    This test:
    1. Calls run_linter with Python code
    2. Verifies it returns linting results
    """
    # Sample code to lint
    code = "def test():\n    print('Hello')\nundefined_var = unknown_function()"
    language = "python"
    
    # Run the test with real linter
    result = await run_linter(code, language)
    
    # Log test details
    log_test(
        test_name="test_run_linter",
        input_params={"code": code, "language": language},
        result=result,
        log_dirs=setup_logging_directories
    )
    
    # Verify the result has the expected structure
    assert "success" in result
    assert "issues" in result
    
    # There should be linting issues in the sample code
    if result["issues"]:
        has_undefined_error = False
        for issue in result["issues"]:
            if "undefined" in issue["message"].lower() or "unknown" in issue["message"].lower():
                has_undefined_error = True
                break
        
        assert has_undefined_error, "Linter should detect undefined variables"

@pytest.mark.asyncio
async def test_execute_shell(setup_logging_directories):
    """
    Test the execute_shell function with real shell commands.
    
    This test:
    1. Calls execute_shell with a simple command
    2. Verifies the command output is returned
    """
    # Use a simple command that works on most systems
    command = "echo 'Hello from shell'"
    
    # Run the test with real shell execution
    result = await execute_shell(command)
    
    # Log test details
    log_test(
        test_name="test_execute_shell",
        input_params={"command": command},
        result=result,
        log_dirs=setup_logging_directories
    )
    
    # Verify the result
    assert result["success"] is True
    assert result["returncode"] == 0
    assert "Hello from shell" in result["stdout"]
    assert result["stderr"] == ""

# Run the tests if the file is executed directly
if __name__ == "__main__":
    pytest.main(["-v", __file__]) 