"""Synchronous wrappers for async inspect_evals components.

This module provides utilities to convert async grading functions and tools
to synchronous versions compatible with DSPy's optimization framework.

Key improvement: enforce a hard timeout on all async calls so a single
stalled network request cannot hang the entire evaluation loop. The timeout
is configurable via the env var `EVAL_ASYNC_TIMEOUT_SEC` (default: 45s).
"""
import asyncio
import functools
import inspect
import os
from typing import Any, Callable, Dict, List

from inspect_ai.model import ChatMessage
from external.inspect_evals.src.inspect_evals.agentharm.benchmark import harmful_grading_functions, benign_grading_functions


def _get_timeout_seconds(default: float = 45.0) -> float:
    """Read timeout from env with a sane default."""
    try:
        return float(os.getenv("EVAL_ASYNC_TIMEOUT_SEC", str(default)))
    except Exception:
        return default


def asyncio_run_wrapper(async_func: Callable, *, timeout: float | None = None) -> Callable:
    """Convert an async function to sync with an asyncio.wait_for timeout.

    Args:
        async_func: The coroutine function to wrap.
        timeout: Optional timeout seconds; falls back to env/default.
    """

    hard_timeout = _get_timeout_seconds() if timeout is None else timeout

    @functools.wraps(async_func)
    def sync_wrapper(*args, **kwargs):
        async def _runner():
            return await asyncio.wait_for(async_func(*args, **kwargs), timeout=hard_timeout)

        try:
            return asyncio.run(_runner())
        except RuntimeError as e:
            # In environments with a running loop (e.g., notebooks), fall back
            # to creating a task and awaiting it with timeout using the running loop.
            if "already running" in str(e).lower():
                loop = asyncio.get_running_loop()
                # Create a future and run wait_for within the running loop context
                return loop.run_until_complete(asyncio.wait_for(async_func(*args, **kwargs), timeout=hard_timeout))
            raise

    return sync_wrapper


def wrap_async_tool(async_tool_func: Callable) -> Callable:
    """Convert an async inspect_evals tool to a synchronous version.
    
    Args:
        async_tool_func: The async tool function from inspect_evals
        
    Returns:
        A synchronous version of the tool that can be used with DSPy
    """
    @functools.wraps(async_tool_func)
    def sync_tool(*args, **kwargs):
        return asyncio_run_wrapper(async_tool_func)(*args, **kwargs)
    
    return sync_tool


def wrap_async_grading_function(grading_func: Callable, *, timeout: float | None = None) -> Callable:
    """Convert an async grading function to synchronous version.
    
    Args:
        grading_func: Async grading function that takes (metadata, messages) -> Dict[str, Any]
        
    Returns:
        Synchronous version of the grading function
    """
    
    @functools.wraps(grading_func)
    def sync_grading_func(metadata: Dict[str, Any], messages: List[ChatMessage]) -> Dict[str, Any]:
        return asyncio_run_wrapper(grading_func, timeout=timeout)(metadata, messages)
    
    return sync_grading_func


def get_sync_grading_function(task_name: str, grading_function_name: str, *, timeout: float | None = None) -> Callable:
    """Get a synchronous version of a grading function by name.
    
    Args:
        task_name: Either "harmful" or "benign"
        grading_function_name: Name of the grading function (e.g., "grade_visa_fraud")
        
    Returns:
        Synchronous version of the grading function
        
    Raises:
        AttributeError: If the grading function is not found
    """
    if task_name == "harmful":
        grading_module = harmful_grading_functions
    elif task_name == "benign":
        grading_module = benign_grading_functions
    else:
        raise ValueError(f"Unknown task_name: {task_name}")
    
    if not hasattr(grading_module, grading_function_name):
        raise AttributeError(f"Grading function '{grading_function_name}' not found in {task_name} module")
    
    async_grading_func = getattr(grading_module, grading_function_name)
    return wrap_async_grading_function(async_grading_func, timeout=timeout)


def convert_tools_to_sync(tools_dict: Dict[str, Any]) -> Dict[str, Any]:
    """Convert a dictionary of async tools to synchronous versions.
    
    Args:
        tools_dict: Dictionary mapping tool names to async tool functions
        
    Returns:
        Dictionary mapping tool names to synchronous tool functions
    """
    sync_tools = {}
    
    for tool_name, tool_func in tools_dict.items():
        # Skip the finish function as it's already sync
        if tool_name == "finish":
            sync_tools[tool_name] = tool_func
            continue
            
        # Check if it's an async function
        if inspect.iscoroutinefunction(tool_func):
            sync_tools[tool_name] = wrap_async_tool(tool_func)
        else:
            # Already synchronous
            sync_tools[tool_name] = tool_func
    
    return sync_tools


__all__ = [
    "asyncio_run_wrapper",
    "wrap_async_tool", 
    "wrap_async_grading_function",
    "get_sync_grading_function",
    "convert_tools_to_sync",
]
