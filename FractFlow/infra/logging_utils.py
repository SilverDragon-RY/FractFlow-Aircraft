"""
Logging utilities for the FractFlow project.

Provides standardized logging functions and formatting to ensure
consistent logging across the entire system.
"""

import inspect
import sys
import yaml
from typing import Any, Dict, Optional, Union, List

from loguru import logger

# Remove default handler
logger.remove()

# Custom formatter for YAML output of extras
def format_extra_as_yaml(record):
    """Format the extra data as YAML for better readability."""
    # Make a copy to avoid modifying the original record
    extras = record["extra"].copy()
    
    # Remove logger_name to avoid duplication
    if "logger_name" in extras:
        del extras["logger_name"]
    
    # Remove caller info as it's already displayed in the log format
    if "caller_file" in extras:
        del extras["caller_file"]
    if "caller_line" in extras:
        del extras["caller_line"]
    
    # If there are any extra fields, format them as YAML
    if extras:
        # Convert to YAML, remove the document start marker
        yaml_str = yaml.dump(extras, default_flow_style=False, sort_keys=False, allow_unicode=True).strip()
        if yaml_str.startswith('---'):
            yaml_str = yaml_str[3:].strip()
        # Indent each line for better visual separation
        yaml_lines = yaml_str.split('\n')
        yaml_str = '\n  '.join(yaml_lines)
        record["extra_yaml"] = f"\n  {yaml_str}" if yaml_str else ""
    else:
        record["extra_yaml"] = ""
    
    return record

def setup_logging(level: int = 20, use_colors: bool = True, namespace_levels: Optional[Dict[str, int]] = None):
    """
    Configure logging with standard formatting.
    
    Args:
        level: The logging level to use for root logger
        use_colors: Whether to enable colored output
        namespace_levels: Dictionary mapping logger namespaces to their log levels
    """
    # Remove any existing handlers
    logger.remove()
    
    # Define format with YAML-formatted extra data
    log_format = "<green>{time:YYYY-MM-DD HH:mm:ss}</green> <level>[{level}]</level> <cyan>{extra[logger_name]}</cyan> <blue>({extra[caller_file]}:{extra[caller_line]})</blue>: <level>{message}</level>{extra_yaml}"
    
    # Add console handler with coloring
    handler_id = logger.add(
        sys.stderr,
        format=log_format,
        level=level,
        colorize=use_colors,
        filter=format_extra_as_yaml
    )
    
    # Set namespace-specific log levels
    if namespace_levels:
        for namespace, ns_level in namespace_levels.items():
            logger.level(namespace, ns_level)
    
    # Output current log level for debugging
    print(f"Root logger level: {level}")

def get_logger(name: Optional[str] = None):
    """
    Get a logger with the given name or the caller's module name.
    
    Args:
        name: Logger name (if None, uses caller's module name)
        
    Returns:
        A logger instance
    """
    if name is None:
        # Get the caller's module name
        frame = inspect.stack()[1]
        module = inspect.getmodule(frame[0])
        name = module.__name__ if module else "unknown"
    
    return LoggerWrapper(name)

class LoggerWrapper:
    """
    Wrapper around loguru logger to provide API compatibility with the old implementation.
    """
    
    def __init__(self, name: str):
        """
        Initialize the logger wrapper.
        
        Args:
            name: Logger name
        """
        self.name = name
        self.use_colors = sys.stdout.isatty()
    
    def _format_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Process structured data for loguru."""
        return {
            k: v for k, v in data.items()
            if k not in {"logger_name", "message"} and not k.startswith("_")
        }

    def _log(self, level: str, message: str, data: Optional[Dict[str, Any]] = None):
        # Get caller's frame (2 levels up in the stack to skip this method and the calling log method)
        frame = inspect.currentframe().f_back.f_back
        file_path = frame.f_code.co_filename
        line_no = frame.f_lineno
        
        # Extract just the filename from the path
        filename = file_path.split("/")[-1]
        
        # Setup context with logger_name and caller info
        context = {
            "logger_name": self.name,
            "caller_file": filename,
            "caller_line": line_no
        }
        
        # Add any additional data
        if data:
            extra_data = self._format_data(data)
            context.update(extra_data)
        
        # Log with context bound
        logger.bind(**context).log(level, message)

    def debug(self, message: str, data: Optional[Dict[str, Any]] = None):
        self._log("DEBUG", message, data)

    def info(self, message: str, data: Optional[Dict[str, Any]] = None):
        self._log("INFO", message, data)

    def warning(self, message: str, data: Optional[Dict[str, Any]] = None):
        self._log("WARNING", message, data)

    def error(self, message: str, data: Optional[Dict[str, Any]] = None):
        self._log("ERROR", message, data)

    def critical(self, message: str, data: Optional[Dict[str, Any]] = None):
        self._log("CRITICAL", message, data)

    def highlight(self, message: str, data: Optional[Dict[str, Any]] = None):
        if "HIGHLIGHT" not in logger._core.levels:
            logger.level("HIGHLIGHT", no=25, color="<bold><white>")
        self._log("HIGHLIGHT", message, data)
        
    def result(self, message: str, data: Optional[Dict[str, Any]] = None):
        """
        Log final results in a highlighted format.
        This is an alias for the highlight method.
        
        Args:
            message: Result message
            data: Optional structured data
        """
        self.highlight(message, data) 