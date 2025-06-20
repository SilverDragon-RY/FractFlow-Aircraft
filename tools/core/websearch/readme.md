# Web Search Tool

A comprehensive tool for web search and browsing operations, designed to be used with the FractFlow framework.

## Features

- Search the web using different search engines (DuckDuckGo, Bing, Google)
- Browse web pages and extract various types of content
- Extract full text content from web pages
- Extract only titles from web pages
- Extract all links from web pages
- Retrieve raw HTML content from web pages

## Project Structure

- `__init__.py` - Package initialization file
- `requirements.txt` - Lists required packages (requests, beautifulsoup4, mcp, pytest, pytest-asyncio)
- `run_server.py` - Main entry point for starting the tool server in interactive or single query mode
- `src/` - Source code directory
  - `__init__.py` - Package initialization for the source directory
  - `core_logic.py` - Core functions for web search and browsing operations
  - `server.py` - FastMCP server that exposes web operations as tools for the FractFlow framework
- `tests/` - Test directory
  - `__init__.py` - Package initialization for the tests directory
  - `test_core_logic.py` - Unit tests for core web search and browsing functions
  - `test_run_server_integration.py` - Integration tests for the server interface

## Environment Setup with uv

[uv](https://github.com/astral-sh/uv) is a fast Python package installer and resolver that can be used to create isolated environments. Follow these steps to set up an environment for the Web Search Tool:

```bash
# Navigate to the websearch directory
cd tools/websearch

# Create a virtual environment
uv venv

# Activate the virtual environment
# On Unix/macOS:
source .venv/bin/activate
# On Windows:
# .venv\Scripts\activate

# Install dependencies using uv
uv pip install -r requirements.txt
uv pip install -r ../../requirements.txt

# To install dev dependencies or additional packages
uv pip install pytest pytest-asyncio
```

## Environment Variables

The Web Search Tool requires certain environment variables to be set. Create a `.env` file in the `tools/websearch` directory following the file `.env.example`.


## Testing

The tool includes both unit tests and integration tests. You can run the tests using pytest:

```bash
# Run all tests
python -m pytest tools/websearch/tests/

# Run specific test file
python -m pytest tools/websearch/tests/test_core_logic.py
python -m pytest tools/websearch/tests/test_run_server_integration.py

# Run with verbose output
python -m pytest -v tools/websearch/tests/

# Run a specific test function
python -m pytest tools/websearch/tests/test_core_logic.py::test_web_search
```

## Usage

### Starting the Tool Server

```bash
# 1. Interactive chat mode
python run_server.py
# Enter your question, e.g. "Latest tech industry news"
Enter your question: 


# 2. Single query mode - process a single query and exit automatically
python run_server.py --user_query "Search for Python tutorials"

# 3. Import as an Agent tool
agent.add_tool("./tools/websearch/src/server.py")
```

#### Running Modes

1. **Interactive Chat Mode** - Default mode
   - Starts an interactive command line interface
   - Can input multiple queries and receive responses
   - Enter 'exit', 'quit', or 'bye' to exit

2. **Single Query Mode** - Run with `--user_query` parameter
   - Automatically processes one query and displays the result
   - Exits automatically after processing
   - Suitable for script integration and automation tasks

Examples:
```bash
# Search the web
python run_server.py --user_query "Search for Python tutorials"

# Browse a specific website
python run_server.py --user_query "Browse the website https://www.python.org and extract all the text"

# Extract links from a website
python run_server.py --user_query "Get all links from https://www.python.org"
```

---

# Development Guidelines

To ensure consistency and easy integration, please follow these development standards:

1. **Project Structure**
   - Follow the standard layout: `src/`, `tests/`, `requirements.txt`, `run_server.py`.
   - Keep core logic (`core_logic.py`) and server interface (`server.py`) separated.

2. **Environment Management**
   - Use `uv venv` to create a virtual environment inside each tool directory.
   - Install dependencies from `requirements.txt`.
   - Always activate the `.venv` before developing or running the tool.

3. **Testing Requirements**
   - Provide at least one unit test (`test_core_logic.py`) to verify basic functions.
   - Provide at least one integration test (`test_run_server_integration.py`) to test server interface.
   - Use `pytest` for all tests.

4. **Coding Style**
   - Use `snake_case` for all variable, function, and file names.
   - Write clear docstrings for all public functions and classes.
   - Avoid hardcoding values; use parameters.
   - Keep each function small and focused.

5. **README Requirements**
   - Document your tool's main functions.
   - Include a section on environment setup.
   - Include usage examples (minimum 3 queries). 
# Web Search Tool v1.0

A comprehensive tool for web search and browsing operations, designed to be used with the FractFlow framework.

## Features
- **Web Search and Browsing**: Multi-functional search and browse capabilities
  - Supported search engines: Google, Baidu, DuckDuckGo
  - Configurable number of search results
  - Option to search only or browse results simultaneously
  - Flexible control over number of results to browse (single, multiple, or all)

- **Web Crawling**: Retrieve and extract web page content
  - Automatic encoding handling
  - Content size control to avoid 413 errors
  - **PDF file parsing support**: Automatically identifies and extracts text from PDFs


## ⚠️ Network Requirements

- **Google Search** and **DuckDuckGo Search** require external network access. If you are in mainland China or regions with network restrictions, you may need a proxy or VPN.
- **Baidu Search** works in mainland China without special network settings.
- For restricted networks, prefer "baidu" search engine.
## Project Structure

- `__init__.py` - Package initialization file
- `requirements.txt` - Lists required packages (requests, beautifulsoup4, mcp, pytest, pytest-asyncio)
- `run_server.py` - Main entry point for starting the tool server in interactive or single query mode
- `src/` - Source code directory
  - `__init__.py` - Package initialization for the source directory
  - `core_logic.py` - Core functions for web search and browsing operations
  - `server.py` - FastMCP server that exposes web operations as tools for the FractFlow framework
- `tests/` - Test directory
  - `__init__.py` - Package initialization for the tests directory
  - `test_core_logic.py` - Unit tests for core web search and browsing functions
  - `test_run_server_integration.py` - Integration tests for the server interface

## Environment Setup with uv

[uv](https://github.com/astral-sh/uv) is a fast Python package installer and resolver that can be used to create isolated environments. Follow these steps to set up an environment for the Web Search Tool:

```bash
# Navigate to the websearch directory
cd tools/websearch

# Create a virtual environment
uv venv

# Activate the virtual environment
# On Unix/macOS:
source .venv/bin/activate
# On Windows:
# .venv\Scripts\activate

# Install dependencies using uv
uv pip install -r requirements.txt
uv pip install -r ../../requirements.txt

# To install dev dependencies or additional packages
uv pip install pytest pytest-asyncio
```

## Environment Variables

The Web Search Tool requires certain environment variables to be set. Create a `.env` file in the `tools/websearch` directory following the file `.env.example`.


## Testing

The tool includes both unit tests and integration tests. You can run the tests using pytest:

```bash
# Navigate to the websearch directory
cd tools/websearch

# Run all tests
python -m pytest ./tests/

# Run specific test file
python -m pytest ./tests/test_core_logic.py
python -m pytest ,/tests/test_run_server_integration.py

# Run with verbose output
python -m pytest -v ./tests/

# Run a specific test function
python -m pytest ./tests/test_core_logic.py::test_web_search
```

## Usage

### Starting the Tool Server

```bash
# 1. Interactive chat mode, run the following command and Enter your question, e.g. "Latest tech industry news", "https://arxiv.org/pdf/2311.11284 这篇paper讲了什么？", "ResNet网络结构介绍"
python run_server.py


# 2. Single query mode - process a single query and exit automatically
python run_server.py --user_query "Search for Python tutorials"

# 3. Import as an Agent tool
agent.add_tool("./tools/websearch/src/server.py")
```

#### Running Modes

1. **Interactive Chat Mode** - Default mode
   - Starts an interactive command line interface
   - Can input multiple queries and receive responses
   - Enter 'exit', 'quit', or 'bye' to exit

2. **Single Query Mode** - Run with `--user_query` parameter
   - Automatically processes one query and displays the result
   - Exits automatically after processing
   - Suitable for script integration and automation tasks

Examples:
```bash
# Search the web
python run_server.py --user_query "Search for Python tutorials"

# Browse a specific website
python run_server.py --user_query "Browse the website https://www.python.org and extract all the text"

# Extract links from a website
python run_server.py --user_query "Get all links from https://www.python.org"
```




## Tool Documentation
### 1. Search and Browse Tool (search_and_browse)

```python
search_and_browse(query: str, search_engine: str = "duckduckgo", num_results: int = 5, max_browse: int = 1, max_length: int = 50000) -> str
```
Parameters:
- `query`: Search query keywords
- `search_engine`: Search engine ("duckduckgo", "baidu", "google")
- `num_results`: Number of search results to return
- `max_browse`: Number of search results to automatically browse
  - Default is 1 (browse only the first result)
  - Set to 0 to browse all search results
  - Set to -1 to search only without browsing (get only search results list)
- `max_length`: Maximum length of content for each web page (when browsing multiple results, each page will automatically be allocated less length)

Features:
- Flexible "one-stop" search tool that supports multiple usage scenarios
- Search-only mode (`max_browse=-1`): Quickly get a list of search results with minimal data
- Browse mode: Get both search results and web page content for deeper information
- Automatically adjusts content length for each page to avoid excessive total content
- Suitable for various information retrieval needs, from quick overviews to in-depth research


### 2. Web Crawling Tool (web_crawl)

Parameters:
- `url`: Web page URL
- `max_length`: Maximum return content length (characters, default 40000 characters, about 50KB)

Features:
- Automatically identifies and handles PDF files
- Extracts text content from PDFs (requires PyPDF2)
- Works with both HTML web pages and PDF documents

## Content Size Limits

To avoid 413 (Request Entity Too Large) errors with model APIs, this tool limits all returned content:



## Troubleshooting

- **413 Error**: Lower `max_length` parameter (try 35000)
- **Network Issues**: Configure proxy or use Baidu in restricted networks
- **PDF Issues**: Ensure PyPDF2 is installed

