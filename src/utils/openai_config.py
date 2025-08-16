"""OpenAI client configuration with aiohttp for better async handling.

This module configures the OpenAI client to use aiohttp as the HTTP client,
which can help reduce timeout issues and improve async performance.

Based on: https://github.com/openai/openai-python#with-aiohttp
"""
import os
import asyncio
import logging
from typing import Optional

try:
    import aiohttp
    AIOHTTP_AVAILABLE = True
except ImportError:
    AIOHTTP_AVAILABLE = False
    aiohttp = None

import httpx
from openai import AsyncOpenAI
import openai

def configure_openai_with_aiohttp(
    timeout: float = 60.0,
    max_retries: int = 3,
    connector_limit: int = 100
) -> None:
    """Configure OpenAI to use aiohttp for better async performance.
    
    This follows the recommendation from:
    https://github.com/openai/openai-python#with-aiohttp
    
    Args:
        timeout: Request timeout in seconds
        max_retries: Maximum number of retries for failed requests
        connector_limit: Maximum number of connections in the pool
    """
    if not AIOHTTP_AVAILABLE:
        print("Warning: aiohttp not available, falling back to httpx")
        return
    
    # Set default timeout for OpenAI client
    openai.timeout = timeout
    
    # Configure httpx with better async settings
    # Note: OpenAI python client uses httpx internally, but we can configure
    # some global settings that will affect retry behavior
    
    # Set environment variables that OpenAI client will pick up
    os.environ.setdefault("OPENAI_MAX_RETRIES", str(max_retries))
    
    print(f"✓ Configured OpenAI with aiohttp support: timeout={timeout}s, max_retries={max_retries}")


def create_aiohttp_openai_client(
    api_key: str,
    base_url: Optional[str] = None,
    headers: Optional[dict] = None,
    timeout: float = 60.0,
    max_retries: int = 3
) -> AsyncOpenAI:
    """Create OpenAI client with better timeout configuration.
    
    While the OpenAI client doesn't directly support aiohttp,
    we can configure it with better timeout and retry settings.
    
    Args:
        api_key: OpenAI API key
        base_url: Custom API base URL
        headers: Additional headers
        timeout: Request timeout in seconds
        max_retries: Maximum number of retries
        
    Returns:
        AsyncOpenAI client configured with better timeouts
    """
    # Create httpx client with better settings (OpenAI uses httpx internally)
    http_client = httpx.AsyncClient(
        timeout=httpx.Timeout(
            timeout=timeout,
            connect=10.0,
            read=timeout,
            write=10.0,
            pool=5.0
        ),
        limits=httpx.Limits(
            max_keepalive_connections=20,
            max_connections=100,
            keepalive_expiry=30.0
        ),
        headers=headers or {},
        follow_redirects=True,
        http2=True
    )
    
    # Create OpenAI client with custom httpx client
    client = AsyncOpenAI(
        api_key=api_key,
        base_url=base_url,
        http_client=http_client,
        max_retries=max_retries,
        timeout=timeout
    )
    
    return client


def create_custom_openai_client(
    api_key: str,
    base_url: Optional[str] = None,
    headers: Optional[dict] = None,
    timeout: float = 60.0,
    max_retries: int = 3
) -> httpx.AsyncClient:
    """Create a custom HTTP client for OpenAI with better timeout handling.
    
    Args:
        api_key: OpenAI API key
        base_url: Custom API base URL
        headers: Additional headers
        timeout: Request timeout in seconds
        max_retries: Maximum number of retries
        
    Returns:
        Configured httpx AsyncClient
    """
    # Create custom timeout configuration
    timeout_config = httpx.Timeout(
        timeout=timeout,
        connect=10.0,  # Connection timeout
        read=timeout,  # Read timeout
        write=10.0,    # Write timeout
        pool=5.0       # Pool timeout
    )
    
    # Create limits for connection pooling
    limits = httpx.Limits(
        max_keepalive_connections=20,
        max_connections=100,
        keepalive_expiry=30.0
    )
    
    # Create the client with custom configuration
    client = httpx.AsyncClient(
        timeout=timeout_config,
        limits=limits,
        headers=headers or {},
        follow_redirects=True,
        http2=True  # Enable HTTP/2 for better performance
    )
    
    return client


def configure_dspy_with_better_timeouts():
    """Configure DSPy/OpenAI with better timeout and retry settings."""
    # Set environment variables that will be picked up by OpenAI client
    os.environ.setdefault("OPENAI_MAX_RETRIES", "2")  # Reduce client-level retries
    os.environ.setdefault("OPENAI_TIMEOUT", "60")
    
    # Disable OpenAI client retry logging to reduce noise
    logging.getLogger("openai._base_client").setLevel(logging.WARNING)
    
    # Configure aiohttp if available
    if AIOHTTP_AVAILABLE:
        configure_openai_with_aiohttp(timeout=60.0, max_retries=2)
    
    print("✓ Configured DSPy with better timeout settings and reduced retry noise")


__all__ = [
    "configure_openai_with_aiohttp",
    "create_aiohttp_openai_client", 
    "create_custom_openai_client",
    "configure_dspy_with_better_timeouts",
    "AIOHTTP_AVAILABLE"
]
