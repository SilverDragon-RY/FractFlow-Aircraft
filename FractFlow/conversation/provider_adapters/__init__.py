"""
Provider-specific adapters for conversation history.

This module provides adapters for different AI providers to format
conversation history in provider-specific ways.
"""

from .base_adapter import HistoryAdapter
from .deepseek_adapter import DeepSeekHistoryAdapter
from .openai_adapter import OpenAIHistoryAdapter
from .qwen_adapter import QwenHistoryAdapter

__all__ = [
    'HistoryAdapter',
    'DeepSeekHistoryAdapter',
    'OpenAIHistoryAdapter',
    'QwenHistoryAdapter',
]
