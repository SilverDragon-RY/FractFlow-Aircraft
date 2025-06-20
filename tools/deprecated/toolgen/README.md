# ToolGen - AI-Powered Tool Generator

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)

ToolGen is an intelligent code analysis and tool generation utility that automatically creates MCP (Model Context Protocol) tools from your existing Python source code. It leverages AI agents to understand your code structure and generate production-ready tool implementations with both direct function access and AI-enhanced capabilities.

## Features

### 🤖 AI-Powered Analysis
- **Intelligent Function Detection**: Automatically identifies functions that should be exposed as tools
- **Code Understanding**: Analyzes code structure, dependencies, and interfaces
- **Smart Recommendations**: Suggests tool names, descriptions, and optimal interfaces

### 🛠️ Multi-Mode Tool Generation
- **Direct Tools** (`server.py`): Standard MCP tools for direct function access
- **AI-Enhanced Tools** (`AI_server.py`): FractFlow Agent-powered tools with natural language processing
- **Interactive Runner** (`run_server.py`): Command-line interface for testing and interaction

### 📝 Template-Driven Generation
- **Jinja2 Templates**: Flexible template system for customizable output
- **Multiple Formats**: Generates servers, tests, documentation, and configuration files
- **Extensible**: Easy to add new templates and output formats

### 🔧 Advanced Code Processing
- **Import Management**: Intelligent handling of dependencies and imports
- **Decorator Application**: Automatic addition of required MCP decorators
- **Code Optimization**: Cleans and optimizes generated code for production use

## Installation

### Prerequisites

- Python 3.8 or higher
- Required API keys for AI providers (DeepSeek or Qwen)

### Dependencies

```bash
pip install jinja2 python-dotenv
# Plus FractFlow dependencies for AI-enhanced features
```

### Environment Setup

Create a `.env` file with your API credentials:

```env
DEEPSEEK_API_KEY=your_deepseek_api_key_here
QWEN_API_KEY=your_qwen_api_key_here  # Fallback option
```

## Quick Start

### Basic Usage

```bash
# Generate tools from your source code
python toolgen.py /path/to/your/source_file.py
```

This will analyze your source file and generate:
- `server.py` - Direct MCP tools
- `AI_server.py` - AI-enhanced tools with FractFlow Agent
- `run_server.py` - Interactive runner script

### Example Workflow

1. **Prepare your source code**:
```python
# my_functions.py
def process_data(data: str) -> dict:
    """Process input data and return structured results."""
    return {"processed": data.upper(), "length": len(data)}

def analyze_text(text: str, mode: str = "basic") -> dict:
    """Analyze text content with different analysis modes."""
    # Your implementation here
    pass
```

2. **Generate tools**:
```bash
python toolgen.py my_functions.py
```

3. **Use the generated tools**:
```bash
# Direct tool access
python run_server.py

# AI-enhanced mode
python run_server.py --ai

# Single query mode
python run_server.py -q "Process this data: hello world"
```

### Agent Configuration

The generated agents use the following default configuration:

```python
config = {
    'agent': {
        'provider': 'deepseek',
        'max_iterations': 5
    },
    'deepseek': {
        'model': 'deepseek-chat'
    },
    'tool_calling': {
        'version': 'turbo'
    }
}
```

## Output Structure

After running ToolGen, you'll get the following structure:

```
your_project/
├── source_file.py          # Your original source code
├── server.py               # Generated MCP tool server
├── AI_server.py           # Generated AI-enhanced server
├── run_server.py          # Interactive runner script
└── (optional test files)  # If test templates are used
```

## ⚠️ Important Notice

**ToolGen is an AI-assisted code generation tool that provides a starting point, not a final solution.** 

### Manual Review and Adjustment Required

The generated code will likely need manual refinement and adjustments:

- **Server Function Abstractions** (`server.py`)

- **System Prompts** (`AI_server.py` and `run_server.py`)
