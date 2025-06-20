"""
DeepSeek model implementation.

Provides implementation of the BaseModel interface for DeepSeek models.
"""

from typing import Dict, List, Any, Optional

from .orchestrator_model import OrchestratorModel
from ..infra.config import ConfigManager
from ..infra.logging_utils import get_logger
from ..conversation.provider_adapters.deepseek_adapter import DeepSeekHistoryAdapter

class DeepSeekModel(OrchestratorModel):
    """
    Implementation of OrchestratorModel for DeepSeek models.
    
    Handles user interaction, understands requirements, and generates
    high-quality tool calling instructions using DeepSeek's models.
    """
    
    def __init__(self, config: Optional[ConfigManager] = None):
        """
        Initialize the DeepSeek model with DeepSeek-specific configuration.
        
        Args:
            config: Configuration manager instance to use
        """
        if config is None:
            config = ConfigManager()
            
        # Push model type to call path
        config.push_to_call_path("deepseek")
        
        # Initialize logger
        self.logger = get_logger(config.get_call_path())
            
        history_adapter = DeepSeekHistoryAdapter()
        
        self.logger.debug("Creating DeepSeek model")
        
        super().__init__(
            base_url=config.get('deepseek.base_url', 'https://api.deepseek.com'),
            api_key=config.get('deepseek.api_key'),
            model_name=config.get('deepseek.model', 'deepseek-reasoner'),
            provider_name='deepseek',
            history_adapter=history_adapter,
            config=config
        )
        
        self.logger.debug("DeepSeek model created", {
            "model": config.get('deepseek.model', 'deepseek-reasoner'),
            "base_url": config.get('deepseek.base_url', 'https://api.deepseek.com')
        })