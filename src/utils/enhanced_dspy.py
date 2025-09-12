"""Enhanced DSPy LM wrapper with better timeout and retry handling.

This module provides a wrapper around DSPy's LM that configures the underlying
OpenAI client with better timeout, retry, and async handling.
"""
import os
import time
from typing import Optional, Dict, Any, List
import logging

import dspy
from src.utils.openai_config import configure_dspy_with_better_timeouts

logger = logging.getLogger(__name__)


class EnhancedDSPyLM(dspy.BaseLM):
    """Enhanced DSPy LM wrapper with better error handling and retries.
    
    This class inherits from dspy.BaseLM and wraps an underlying dspy.LM 
    with enhanced timeout and retry logic.
    """
    
    def __init__(
        self,
        model: str,
        api_key: str,
        api_base: Optional[str] = None,
        headers: Optional[dict] = None,
        temperature: float = 0.0,
        max_tokens: int = 2000,
        timeout: float = 120.0,  # Increased timeout
        max_retries: int = 5,
        retry_delay: float = 1.0,
        seed: Optional[int] = None,
        **kwargs
    ):
        """Initialize enhanced DSPy LM with better timeout handling.
        
        Args:
            model: Model name (e.g., "openai/gpt-5-nano")
            api_key: OpenAI API key
            api_base: Custom API base URL
            headers: Additional headers
            temperature: Temperature for generation
            max_tokens: Maximum tokens
            timeout: Request timeout in seconds
            max_retries: Maximum retries for failed requests
            retry_delay: Base delay between retries
            **kwargs: Additional arguments for DSPy LM
        """
        # Initialize the base class
        super().__init__(
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            **kwargs
        )
        
        self.timeout = timeout
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        
        # Configure global settings first
        configure_dspy_with_better_timeouts()
        
        # Set OpenAI environment variables for timeout and retries
        os.environ["OPENAI_TIMEOUT"] = str(timeout)
        os.environ["OPENAI_MAX_RETRIES"] = "2"  # Let our wrapper handle most retries
        
        # Prepare LM arguments
        lm_args = {
            "model": model,
            "api_key": api_key,
            "temperature": temperature,
            "max_tokens": max_tokens,
            **kwargs
        }
        if seed is not None:
            lm_args["seed"] = seed
        if api_base is not None:
            lm_args["api_base"] = api_base
        if headers is not None:
            lm_args["headers"] = headers
        # Create underlying DSPy LM with retry logic
        self._underlying_lm = self._create_lm_with_retry(lm_args)
        
        logger.info(
            f"✓ Enhanced DSPy LM initialized: model={model}, "
            f"timeout={timeout}s, max_retries={max_retries}"
        )
    
    def _create_lm_with_retry(self, lm_args: Dict[str, Any]) -> dspy.LM:
        """Create DSPy LM with retry logic for initialization."""
        last_error = None
        
        for attempt in range(self.max_retries):
            try:
                lm = dspy.LM(**lm_args)
                logger.info(f"✓ DSPy LM created successfully on attempt {attempt + 1}")
                return lm
            except Exception as e:
                last_error = e
                if attempt < self.max_retries - 1:
                    delay = self.retry_delay * (2 ** attempt)  # Exponential backoff
                    logger.warning(
                        f"Failed to create DSPy LM (attempt {attempt + 1}/{self.max_retries}): {e}. "
                        f"Retrying in {delay:.1f}s..."
                    )
                    time.sleep(delay)
                else:
                    logger.error(f"Failed to create DSPy LM after {self.max_retries} attempts")
        
        if last_error:
            raise last_error
        else:
            raise RuntimeError("Failed to create DSPy LM with no specific error")
    
    def forward(self, prompt=None, messages=None, **kwargs):
        """Forward method required by BaseLM interface."""
        return self._call_with_retry(prompt=prompt, messages=messages, **kwargs)
    
    def _call_with_retry(self, prompt=None, messages=None, **kwargs):
        """Call underlying DSPy LM with retry logic."""
        last_error = None
        
        for attempt in range(self.max_retries):
            try:
                return self._underlying_lm.forward(prompt=prompt, messages=messages, **kwargs)
            except Exception as e:
                last_error = e
                error_msg = str(e).lower()
                
                # Check if it's a retryable error
                retryable_errors = [
                    "timeout", "read timeout", "connection", "network", 
                    "rate limit", "server error", "503", "502", "500"
                ]
                
                if any(err in error_msg for err in retryable_errors):
                    if attempt < self.max_retries - 1:
                        delay = self.retry_delay * (2 ** attempt)  # Exponential backoff
                        logger.warning(
                            f"Retryable error (attempt {attempt + 1}/{self.max_retries}): {e}. "
                            f"Retrying in {delay:.1f}s..."
                        )
                        time.sleep(delay)
                        continue
                
                # Non-retryable error or max retries reached
                logger.error(f"LM call failed after {attempt + 1} attempts: {e}")
                break
        
        if last_error:
            raise last_error
        else:
            raise RuntimeError("LM call failed with no specific error")
    
    def __call__(self, *args, **kwargs):
        """Support direct calling like the underlying LM."""
        return self._underlying_lm(*args, **kwargs)
    
    def __getattr__(self, name):
        """Delegate other attribute access to the underlying DSPy LM."""
        return getattr(self._underlying_lm, name)


def create_enhanced_dspy_lm(config_models, api_key: str) -> EnhancedDSPyLM:
    """Create enhanced DSPy LM from config.
    
    Args:
        config_models: Models configuration from config
        api_key: OpenAI API key
        
    Returns:
        Enhanced DSPy LM instance
    """
    lm_args = {
        "model": config_models.lm_name,
        "api_key": api_key,
        "temperature": config_models.lm_temperature,
        "max_tokens": config_models.max_tokens,
        "timeout": 120.0,  # Increased from 60.0 to 120.0 seconds
        "max_retries": 5,  # 5 retries with exponential backoff
        "retry_delay": 1.0,  # Start with 1 second delay
    }
    # Pass seed if present in config_models
    if hasattr(config_models, "seed") and config_models.seed is not None:
        lm_args["seed"] = config_models.seed
    if config_models.api_base is not None:
        lm_args["api_base"] = config_models.api_base
    if config_models.headers is not None:
        lm_args["headers"] = config_models.headers
    return EnhancedDSPyLM(**lm_args)


__all__ = ["EnhancedDSPyLM", "create_enhanced_dspy_lm"]
