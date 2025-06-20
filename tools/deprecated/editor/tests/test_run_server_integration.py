"""
Run Server Integration Tests for OpenHands ACI Editor

This module contains integration tests for the OpenHands ACI editor server.
Tests interact with the server through the command-line interface, 
sending queries and verifying that the operations are properly executed on the file system.

These tests use temporary directories to ensure they don't affect the actual file system and
clean up after themselves.

Author: Ying-Cong Chen (yingcong.ian.chen@gmail.com)
Date: 2025-04-27
License: MIT License
"""

import os
import sys
import pytest
import tempfile
import shutil
import subprocess
import json
import time
from pathlib import Path
import re

# Add project root directory to Python path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(project_root)


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
    with open(os.path.join(integration_log_dir, "test_run_info.txt"), "w") as f:
        f.write(f"Integration Test Run Started: {timestamp}\n")
        f.write(f"Python version: {sys.version}\n")
        f.write(f"Working directory: {os.getcwd()}\n")
    
    print(f"Logs will be saved to: {integration_log_dir}")
    
    # Return log directories
    return {
        "integration": integration_log_dir,
        "unit": unit_log_dir
    }


@pytest.fixture
def temp_test_dir(setup_logging_directories):
    """Create temporary test directory and clean up after the test"""
    temp_dir = tempfile.mkdtemp()
    log_dirs = setup_logging_directories
    
    # Log the temp directory creation
    with open(os.path.join(log_dirs["integration"], "temp_dirs.txt"), "a") as f:
        f.write(f"{time.strftime('%H:%M:%S')} - Created temporary directory: {temp_dir}\n")
    
    # Create some test files
    test_file = os.path.join(temp_dir, "test_file.py")
    with open(test_file, "w") as f:
        f.write("def hello():\n    print('Hello, world!')\n\nhello()\n")
    
    # Create a file with errors for linting tests
    error_file = os.path.join(temp_dir, "error_file.py")
    with open(error_file, "w") as f:
        f.write("def missing_colon()\n    print('This has syntax errors')\n")
    
    # Create a file for editing tests
    edit_file = os.path.join(temp_dir, "edit_file.py")
    with open(edit_file, "w") as f:
        f.write("def greet(name):\n    print(f'Hello, {name}!')\n\ngreet('world')\n")
    
    yield temp_dir
    
    # Log the cleanup
    with open(os.path.join(log_dirs["integration"], "temp_dirs.txt"), "a") as f:
        f.write(f"{time.strftime('%H:%M:%S')} - Cleaned up temporary directory: {temp_dir}\n")
    
    # Clean up
    shutil.rmtree(temp_dir)


def run_query(query, test_name=None, log_dirs=None):
    """
    Run a single query and return the result
    
    Args:
        query: Query string to send to run_server.py
        test_name: Name of the test being run
        log_dirs: Dictionary of log directories
        
    Returns:
        str: Command output
    """
    # Determine the path to run_server.py
    script_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "run_server.py")
    
    # Create log file path if logging is enabled
    log_file_prefix = None
    if log_dirs and test_name:
        # Use integration test log directory
        log_dir = log_dirs.get("integration")
        timestamp = time.strftime("%H%M%S")
        test_id = re.sub(r'[^a-zA-Z0-9_]', '_', test_name)
        log_file_prefix = os.path.join(log_dir, f"{test_id}_{timestamp}")
        
        # Log the input query
        with open(f"{log_file_prefix}_query.txt", "w") as f:
            f.write(f"Test: {test_name}\n")
            f.write(f"Query: {query}\n")
            f.write(f"Command: {sys.executable} {script_path} --user_query \"{query}\"\n")
            f.write(f"Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    # Use subprocess to execute the command
    process = subprocess.Popen(
        [sys.executable, script_path, "--user_query", query],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        universal_newlines=True
    )
    
    start_time = time.time()
    
    # Capture output
    stdout, stderr = process.communicate(timeout=360)
    
    execution_time = time.time() - start_time
    
    # Log the outputs if logging is enabled
    if log_file_prefix:
        with open(f"{log_file_prefix}_stdout.txt", "w") as f:
            f.write(f"Execution time: {execution_time:.2f} seconds\n")
            f.write("=== STDOUT ===\n")
            f.write(stdout)
        
        with open(f"{log_file_prefix}_stderr.txt", "w") as f:
            f.write("=== STDERR ===\n")
            f.write(stderr)
        
        # Log return code
        with open(f"{log_file_prefix}_result.txt", "w") as f:
            f.write(f"Return code: {process.returncode}\n")
            f.write(f"Execution time: {execution_time:.2f} seconds\n")
    
    # Check for errors
    if process.returncode != 0:
        print(f"Error executing command: {stderr}")
    
    # Extract the result (remove thinking and other non-result output)
    result_match = re.search(r"Agent:\s+(.*?)(?:\n\nAgent session ended\.)?$", stdout, re.DOTALL)
    if result_match:
        result = result_match.group(1).strip()
        
        # Log parsed result if logging is enabled
        if log_file_prefix:
            with open(f"{log_file_prefix}_parsed_result.txt", "w") as f:
                f.write("=== PARSED RESULT ===\n")
                f.write(result)
        
        return result
    
    return stdout


def test_view_file_query(temp_test_dir, setup_logging_directories):
    """Test query for viewing a file"""
    test_file = os.path.join(temp_test_dir, "test_file.py")
    
    # Run the query
    query = f"Show me the content of the file {test_file}"
    result = run_query(query, test_name="test_view_file_query", log_dirs=setup_logging_directories)
    
    # Verify that the result contains the file content
    assert "def hello()" in result
    assert "print('Hello, world!')" in result


def test_create_file_query(temp_test_dir, setup_logging_directories):
    """Test query for creating a new file"""
    new_file = os.path.join(temp_test_dir, "new_file.py")
    
    # Run the query
    query = f"Create a new file at {new_file} with a function that calculates the factorial of a number"
    result = run_query(query, test_name="test_create_file_query", log_dirs=setup_logging_directories)
    
    # Log the file creation result
    if setup_logging_directories:
        log_dir = setup_logging_directories.get("integration")
        with open(os.path.join(log_dir, "file_operations.txt"), "a") as f:
            f.write(f"{time.strftime('%H:%M:%S')} - File creation attempt: {new_file}\n")
            f.write(f"  - File exists: {os.path.exists(new_file)}\n")
    
    # Verify the file was created
    assert os.path.exists(new_file)
    
    # Verify the content looks like a factorial function
    with open(new_file, "r") as f:
        content = f.read()
    
    # Log the content
    if setup_logging_directories:
        log_dir = setup_logging_directories.get("integration")
        timestamp = time.strftime("%H%M%S")
        with open(os.path.join(log_dir, f"test_create_file_query_{timestamp}_file_content.txt"), "w") as f:
            f.write(f"=== File: {new_file} ===\n")
            f.write(content)
    
    assert "def factorial" in content.lower()
    assert "return" in content


def test_str_replace_query(temp_test_dir, setup_logging_directories):
    """Test query for replacing strings in a file"""
    edit_file = os.path.join(temp_test_dir, "edit_file.py")
    
    # Log the file content before modification
    if setup_logging_directories:
        log_dir = setup_logging_directories.get("integration")
        timestamp = time.strftime("%H%M%S")
        with open(os.path.join(log_dir, f"test_str_replace_query_{timestamp}_before.txt"), "w") as f:
            f.write(f"=== File before modification: {edit_file} ===\n")
            with open(edit_file, "r") as src:
                f.write(src.read())
    
    # Run the query
    query = f"In the file {edit_file}, replace 'Hello' with 'Greetings'"
    result = run_query(query, test_name="test_str_replace_query", log_dirs=setup_logging_directories)
    
    # Log the file content after modification
    if setup_logging_directories:
        log_dir = setup_logging_directories.get("integration")
        timestamp = time.strftime("%H%M%S")
        with open(os.path.join(log_dir, f"test_str_replace_query_{timestamp}_after.txt"), "w") as f:
            f.write(f"=== File after modification: {edit_file} ===\n")
            with open(edit_file, "r") as src:
                f.write(src.read())
    
    # Verify the content was changed
    with open(edit_file, "r") as f:
        content = f.read()
    
    assert "print(f'Greetings, {name}!')" in content


def test_insert_line_query(temp_test_dir, setup_logging_directories):
    """Test query for inserting a line in a file"""
    edit_file = os.path.join(temp_test_dir, "edit_file.py")
    original_line_count = len(open(edit_file).readlines())
    
    # Log the file content before modification
    if setup_logging_directories:
        log_dir = setup_logging_directories.get("integration")
        timestamp = time.strftime("%H%M%S")
        with open(os.path.join(log_dir, f"test_insert_line_query_{timestamp}_before.txt"), "w") as f:
            f.write(f"=== File before modification: {edit_file} ===\n")
            f.write(f"Line count: {original_line_count}\n")
            with open(edit_file, "r") as src:
                f.write(src.read())
    
    # Run the query
    query = f"Insert a comment at the beginning of file {edit_file} with author information"
    result = run_query(query, test_name="test_insert_line_query", log_dirs=setup_logging_directories)
    
    # Verify a line was inserted
    new_line_count = len(open(edit_file).readlines())
    
    # Log the file content after modification
    if setup_logging_directories:
        log_dir = setup_logging_directories.get("integration")
        timestamp = time.strftime("%H%M%S")
        with open(os.path.join(log_dir, f"test_insert_line_query_{timestamp}_after.txt"), "w") as f:
            f.write(f"=== File after modification: {edit_file} ===\n")
            f.write(f"Line count: {new_line_count}\n")
            f.write(f"Original line count: {original_line_count}\n")
            with open(edit_file, "r") as src:
                f.write(src.read())
    
    assert new_line_count > original_line_count
    
    # Verify the content contains a comment
    with open(edit_file, "r") as f:
        content = f.read()
    
    assert "#" in content and ("author" in content.lower() or "created" in content.lower())


def test_remove_lines_query(temp_test_dir, setup_logging_directories):
    """Test query for removing lines from a file"""
    # Create a multiline test file for this specific test
    remove_test_file = os.path.join(temp_test_dir, "remove_test_file.py")
    with open(remove_test_file, "w") as f:
        f.write("# Line 1 - Header comment\n# Line 2 - License info\n\ndef test_function():\n    print('This is a test')\n    # Debug code to remove\n    print('Debug output')\n    return True\n")
    
    # Log the file content before modification
    if setup_logging_directories:
        log_dir = setup_logging_directories.get("integration")
        timestamp = time.strftime("%H%M%S")
        with open(os.path.join(log_dir, f"test_remove_lines_query_{timestamp}_before.txt"), "w") as f:
            f.write(f"=== File before modification: {remove_test_file} ===\n")
            with open(remove_test_file, "r") as src:
                f.write(src.read())
    
    # Run the query - remove the debug line
    query = f"Remove the debug print line (line 7) from the file {remove_test_file}"
    result = run_query(query, test_name="test_remove_lines_query", log_dirs=setup_logging_directories)
    
    # Log the file content after modification
    if setup_logging_directories:
        log_dir = setup_logging_directories.get("integration")
        timestamp = time.strftime("%H%M%S")
        with open(os.path.join(log_dir, f"test_remove_lines_query_{timestamp}_after.txt"), "w") as f:
            f.write(f"=== File after modification: {remove_test_file} ===\n")
            with open(remove_test_file, "r") as src:
                f.write(src.read())
    
    # Verify the file was modified correctly
    with open(remove_test_file, "r") as f:
        content = f.read()
    
    # Check that the debug line was removed
    assert "print('Debug output')" not in content
    assert "# Debug code to remove" not in content
    assert "def test_function():" in content
    assert "print('This is a test')" in content
    assert "return True" in content
    
    # Now test removing multiple lines - remove the header comments
    query2 = f"Remove the first two comment lines from {remove_test_file}"
    result2 = run_query(query2, test_name="test_remove_lines_query_multiple", log_dirs=setup_logging_directories)
    
    # Log the file content after second modification
    if setup_logging_directories:
        log_dir = setup_logging_directories.get("integration")
        timestamp = time.strftime("%H%M%S")
        with open(os.path.join(log_dir, f"test_remove_lines_query_{timestamp}_after2.txt"), "w") as f:
            f.write(f"=== File after second modification: {remove_test_file} ===\n")
            with open(remove_test_file, "r") as src:
                f.write(src.read())
    
    # Verify the file was modified correctly
    with open(remove_test_file, "r") as f:
        content2 = f.read()
    
    # Check that the header comments were removed
    assert "# Line 1 - Header comment" not in content2
    assert "# Line 2 - License info" not in content2
    assert content2.strip().startswith("def test_function()")


def test_run_linter_query(temp_test_dir, setup_logging_directories):
    """Test query for running the linter on a file"""
    error_file = os.path.join(temp_test_dir, "error_file.py")
    
    # Run the query
    query = f"Check the file {error_file} for errors"
    result = run_query(query, test_name="test_run_linter_query", log_dirs=setup_logging_directories)
    
    # Verify that the result mentions syntax errors
    assert "error" in result.lower() or "missing" in result.lower() or "colon" in result.lower()


def test_fix_error_query(temp_test_dir, setup_logging_directories):
    """Test query for fixing errors in a file"""
    error_file = os.path.join(temp_test_dir, "error_file.py")
    
    # Log the file content before modification
    if setup_logging_directories:
        log_dir = setup_logging_directories.get("integration")
        timestamp = time.strftime("%H%M%S")
        with open(os.path.join(log_dir, f"test_fix_error_query_{timestamp}_before.txt"), "w") as f:
            f.write(f"=== File before fixing: {error_file} ===\n")
            with open(error_file, "r") as src:
                f.write(src.read())
    
    # Run the query
    query = f"Fix the syntax errors in {error_file}"
    result = run_query(query, test_name="test_fix_error_query", log_dirs=setup_logging_directories)
    
    # Verify the file was fixed
    with open(error_file, "r") as f:
        content = f.read()
    
    # Log the file content after modification
    if setup_logging_directories:
        log_dir = setup_logging_directories.get("integration")
        timestamp = time.strftime("%H%M%S")
        with open(os.path.join(log_dir, f"test_fix_error_query_{timestamp}_after.txt"), "w") as f:
            f.write(f"=== File after fixing: {error_file} ===\n")
            f.write(content)
    
    # Should have added the missing colon
    assert "def missing_colon():" in content


def test_extract_function_query(temp_test_dir, setup_logging_directories):
    """Test query for extracting code into a function"""
    # Create a file with code to extract
    extract_file = os.path.join(temp_test_dir, "extract_file.py")
    with open(extract_file, "w") as f:
        f.write("""
x = 10
y = 20
result = x + y
print(f"The sum is {result}")
        """)
    
    # Log the file content before modification
    if setup_logging_directories:
        log_dir = setup_logging_directories.get("integration")
        timestamp = time.strftime("%H%M%S")
        with open(os.path.join(log_dir, f"test_extract_function_query_{timestamp}_before.txt"), "w") as f:
            f.write(f"=== File before extraction: {extract_file} ===\n")
            with open(extract_file, "r") as src:
                f.write(src.read())
    
    # Run the query
    query = f"Extract the calculation and printing into a function called 'calculate_sum' in {extract_file}"
    result = run_query(query, test_name="test_extract_function_query", log_dirs=setup_logging_directories)
    
    # Verify a function was created
    with open(extract_file, "r") as f:
        content = f.read()
    
    # Log the file content after modification
    if setup_logging_directories:
        log_dir = setup_logging_directories.get("integration")
        timestamp = time.strftime("%H%M%S")
        with open(os.path.join(log_dir, f"test_extract_function_query_{timestamp}_after.txt"), "w") as f:
            f.write(f"=== File after extraction: {extract_file} ===\n")
            f.write(content)
    
    assert "def calculate_sum" in content
    # Should still contain the original functionality
    assert "The sum is" in content


def test_execute_shell_query(temp_test_dir, setup_logging_directories):
    """Test query for executing a shell command"""
    # Run the query to get directory listing
    query = f"Run ls command to list files in {temp_test_dir}"
    result = run_query(query, test_name="test_execute_shell_query", log_dirs=setup_logging_directories)
    
    # Verify that the result includes the test files we created
    assert "test_file.py" in result
    assert "error_file.py" in result
    assert "edit_file.py" in result


def test_cache_file_query(temp_test_dir, setup_logging_directories):
    """Test query for caching a file and retrieving its content"""
    test_file = os.path.join(temp_test_dir, "test_file.py")
    
    # First query to cache the file
    query1 = f"Cache the file {test_file} for faster access"
    result1 = run_query(query1, test_name="test_cache_file_query_1", log_dirs=setup_logging_directories)
    
    # Verify operation was mentioned in the result
    assert "cache" in result1.lower() or "stored" in result1.lower() or "saved" in result1.lower()
    
    # Second query to retrieve the cached content
    query2 = f"Get the cached content of {test_file}"
    result2 = run_query(query2, test_name="test_cache_file_query_2", log_dirs=setup_logging_directories)
    
    # Verify the cached content is returned
    assert "def hello()" in result2
    assert "print('Hello, world!')" in result2


def test_complex_multifile_query(temp_test_dir, setup_logging_directories):
    """Test complex query involving multiple files and operations"""
    # Create files for a simple module
    module_dir = os.path.join(temp_test_dir, "mymodule")
    os.makedirs(module_dir)
    
    # Create __init__.py
    init_file = os.path.join(module_dir, "__init__.py")
    with open(init_file, "w") as f:
        f.write("# Empty init file\n")
    
    # Query to implement a full module with proper imports
    query = f"""Create a Python module in {module_dir} with two files:
    1. math_utils.py with functions for addition, subtraction, multiplication, division
    2. main.py that imports math_utils and demonstrates using all the functions
    """
    
    result = run_query(query, test_name="test_complex_multifile_query", log_dirs=setup_logging_directories)
    
    # Verify the files were created
    math_utils_file = os.path.join(module_dir, "math_utils.py")
    main_file = os.path.join(module_dir, "main.py")
    
    # Log the created files
    if setup_logging_directories:
        log_dir = setup_logging_directories.get("integration")
        timestamp = time.strftime("%H%M%S")
        
        # Log math_utils.py
        if os.path.exists(math_utils_file):
            with open(os.path.join(log_dir, f"test_complex_multifile_query_{timestamp}_math_utils.txt"), "w") as f:
                f.write(f"=== Created file: {math_utils_file} ===\n")
                with open(math_utils_file, "r") as src:
                    f.write(src.read())
        else:
            with open(os.path.join(log_dir, f"test_complex_multifile_query_{timestamp}_missing_files.txt"), "w") as f:
                f.write(f"File not created: {math_utils_file}\n")
        
        # Log main.py
        if os.path.exists(main_file):
            with open(os.path.join(log_dir, f"test_complex_multifile_query_{timestamp}_main.txt"), "w") as f:
                f.write(f"=== Created file: {main_file} ===\n")
                with open(main_file, "r") as src:
                    f.write(src.read())
        else:
            with open(os.path.join(log_dir, f"test_complex_multifile_query_{timestamp}_missing_files.txt"), "a") as f:
                f.write(f"File not created: {main_file}\n")
    
    assert os.path.exists(math_utils_file)
    assert os.path.exists(main_file)
    
    # Verify math_utils.py content
    with open(math_utils_file, "r") as f:
        math_utils_content = f.read()
    
    assert "def add" in math_utils_content
    assert "def subtract" in math_utils_content 
    assert "def multiply" in math_utils_content
    assert "def divide" in math_utils_content
    
    # Verify main.py imports and uses the functions
    with open(main_file, "r") as f:
        main_content = f.read()
    
    assert "import" in main_content
    assert "math_utils" in main_content
    # Should have usage examples of each function
    assert "add" in main_content
    assert "subtract" in main_content
    assert "multiply" in main_content
    assert "divide" in main_content


if __name__ == "__main__":
    pytest.main(["-xvs", __file__]) 