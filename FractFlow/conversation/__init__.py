"""
Provider-specific adapters for conversation history.

This module provides adapters for different AI providers to format
conversation history in provider-specific ways.
"""

from .provider_adapters.base_adapter import HistoryAdapter
from .provider_adapters.deepseek_adapter import DeepSeekHistoryAdapter
from .provider_adapters.openai_adapter import OpenAIHistoryAdapter
from .provider_adapters.qwen_adapter import QwenHistoryAdapter

__all__ = [
    'HistoryAdapter',
    'DeepSeekHistoryAdapter',
    'OpenAIHistoryAdapter',
    'QwenHistoryAdapter',
]
