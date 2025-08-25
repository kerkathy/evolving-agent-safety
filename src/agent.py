import json
import re
import logging
from typing import List, Dict, Any
from func_timeout import func_set_timeout

import dspy
from dspy.adapters.chat_adapter import ChatAdapter
from dspy.signatures.signature import Signature
from dspy.clients.lm import LM

logger = logging.getLogger(__name__)
import inspect

class FunctionCallAdapter(ChatAdapter):
    """Custom adapter that properly parses function calls consistently"""

    def __call__(
        self,
        lm: LM,
        lm_kwargs: dict[str, Any],
        signature: type[Signature],
        demos: list[dict[str, Any]],
        inputs: dict[str, Any],
    ) -> list[dict[str, Any]]:
        logger.debug("Entered FunctionCallAdapter.__call__")
        # print("Entered FunctionCallAdapter.__call__")
        # Get results from parent
        results = super().__call__(lm, lm_kwargs, signature, demos, inputs)
        
        # Post-process each result to clean function fields
        cleaned_results = []
        for result in results:
            cleaned_result = self._post_process_result_dict(result)
            cleaned_results.append(cleaned_result)
            
        return cleaned_results

    async def acall(
        self,
        lm: LM,
        lm_kwargs: dict[str, Any],
        signature: type[Signature],
        demos: list[dict[str, Any]],
        inputs: dict[str, Any],
    ) -> list[dict[str, Any]]:
        logger.debug("Entered FunctionCallAdapter.acall")
        # print("Entered FunctionCallAdapter.acall")
        # Get results from parent
        results = await super().acall(lm, lm_kwargs, signature, demos, inputs)
        
        # Post-process each result to clean function fields
        cleaned_results = []
        for result in results:
            cleaned_result = self._post_process_result_dict(result)
            cleaned_results.append(cleaned_result)
            
        return cleaned_results
    
    def _post_process_result_dict(self, result_dict):
        logger.debug("Entered FunctionCallAdapter._post_process_result_dict")
        # print("Entered FunctionCallAdapter._post_process_result_dict")
        """Post-process a result dictionary to clean function fields"""
        if not isinstance(result_dict, dict):
            return result_dict
        
        cleaned_dict = result_dict.copy()
        
        # Clean all function-related fields
        function_fields = ['next_selected_fn', 'selected_fn', 'function_name', 'fn_name', 'function']
        for field in function_fields:
            if field in cleaned_dict:
                original_value = cleaned_dict[field]
                cleaned_value = self._clean_function_name(original_value)
                cleaned_dict[field] = cleaned_value
                logger.debug(f"Cleaned {field}: {repr(original_value)} -> {repr(cleaned_value)}")
                # print(f"Cleaned {field}: {repr(original_value)} -> {repr(cleaned_value)}")

        # Clean arguments
        args_fields = ['args', 'arguments', 'function_args']
        for field in args_fields:
            if field in cleaned_dict:
                original_args = cleaned_dict[field]
                cleaned_args = self._clean_arguments(original_args)
                cleaned_dict[field] = cleaned_args
                
        return cleaned_dict

    def parse(self, signature, completion):
        logger.debug("Entered FunctionCallAdapter.parse")
        # print("Entered FunctionCallAdapter.parse")
        logger.debug("[FunctionCallAdapter] Parsing completion: %s", completion)
        # print(f"[FunctionCallAdapter] Parsing completion: {completion}")
        """Override parse method to handle function call parsing consistently"""
        try:
            # First try the parent's parse method
            parsed = super().parse(signature, completion)
            
            # Post-process function call fields
            if hasattr(parsed, 'next_selected_fn'):
                parsed.next_selected_fn = self._clean_function_name(parsed.next_selected_fn)

            if hasattr(parsed, 'args'):
                parsed.args = self._clean_arguments(parsed.args)
            return parsed
            
        except Exception as e:
            # If parent parsing fails, try our custom parsing
            logger.debug("Parent parse failed, attempting custom parse: %s", e)
            # print(f"Parent parse failed, using custom parse: {e}")
            custom_result = self._custom_parse(signature, completion)
            
            # Check if custom parse successfully handled a refusal/plain text response
            if hasattr(custom_result, 'next_selected_fn') and custom_result.next_selected_fn is None:
                logger.info("Successfully handled model refusal/plain text response via custom parse")
            else:
                logger.warning("Custom parse attempted but may need attention - check result: %s", custom_result)
            
            return custom_result
    
    def _clean_function_name(self, fn_name):
        logger.debug("Entered FunctionCallAdapter._clean_function_name")
        # print("Entered FunctionCallAdapter._clean_function_name")
        """Clean and extract function name from various formats"""
        if not fn_name:
            return ""
            
        # Convert to string if not already
        fn_name = str(fn_name).strip()
        
        # replace \" with "
        fn_name = fn_name.replace('\"', '"')

        # Handle JSON format: {"function_name": "...", "arguments": {...}}
        if fn_name.startswith('{') and fn_name.endswith('}'):
            try:
                parsed_json = json.loads(fn_name)
                for k in ['function_name', 'name', 'fn_name']:
                    if k in parsed_json:
                        return parsed_json[k]
            except json.JSONDecodeError:
                pass

        fn_name = fn_name.strip('"').strip("'")
        
        # Handle direct function name
        return fn_name
    
    def _clean_arguments(self, args):
        logger.debug("Entered FunctionCallAdapter._clean_arguments")
        # print("Entered FunctionCallAdapter._clean_function_name")
        """Clean and parse arguments from various formats"""
        if not args:
            return {}
            
        # If already a dict, return as is
        if isinstance(args, dict):
            return args
            
        # Convert to string and try to parse
        args_str = str(args).strip()
        
        # Remove outer quotes
        args_str = args_str.strip('"').strip("'")
        
        # Try to parse as JSON
        if args_str.startswith('{') and args_str.endswith('}'):
            try:
                return json.loads(args_str)
            except json.JSONDecodeError:
                pass
        
        # If parsing fails, return empty dict
        return {}

    def _custom_parse(self, signature, completion):
        logger.debug("Entered FunctionCallAdapter._custom_parse")
        # print("Entered FunctionCallAdapter._custom_parse")
        """Custom parsing when standard parsing fails"""
        # Extract text from completion
        text = completion
        if hasattr(completion, 'text'):
            text = completion.text
        elif hasattr(completion, 'content'):
            text = completion.content
        
        # Convert to string if not already
        text = str(text).strip()
        logger.debug("Custom parsing text: %s", repr(text[:100]))  # Log first 100 chars for context
        
        # Initialize result with signature fields
        result_dict = {}
        
        # Try to extract function call information using regex
        # Pattern 1: JSON format
        json_pattern = r'\{\s*"function_name"\s*:\s*"([^"]+)"\s*,\s*"arguments"\s*:\s*(\{[^}]*\})\s*\}'
        json_match = re.search(json_pattern, text)
        
        if json_match:
            result_dict['next_selected_fn'] = json_match.group(1)
            try:
                result_dict['args'] = json.loads(json_match.group(2))
            except json.JSONDecodeError:
                result_dict['args'] = {}
        else:
            # Pattern 2: Direct function name
            fn_pattern = r'next_selected_fn[:\s]+([^\s\n,]+)'
            fn_match = re.search(fn_pattern, text)
            if fn_match:
                result_dict['next_selected_fn'] = fn_match.group(1).strip('"').strip("'")
            
            # Pattern 3: Arguments
            args_pattern = r'args[:\s]+(\{[^}]*\})'
            args_match = re.search(args_pattern, text)
            if args_match:
                try:
                    result_dict['args'] = json.loads(args_match.group(1))
                except json.JSONDecodeError:
                    result_dict['args'] = {}
        
        # If no function call pattern found, treat as plain text response
        if 'next_selected_fn' not in result_dict:
            logger.info("No function call pattern detected, treating as refusal/plain text response")
            result_dict['reasoning'] = text
            result_dict['next_selected_fn'] = None
            result_dict['args'] = None
        else:
            logger.debug("Function call pattern detected: %s", result_dict.get('next_selected_fn'))
        
        # Ensure all required fields are present
        if 'reasoning' not in result_dict:
            result_dict['reasoning'] = ""
        if 'next_selected_fn' not in result_dict:
            result_dict['next_selected_fn'] = None
        if 'args' not in result_dict:
            result_dict['args'] = None
        
        # Create prediction object
        return dspy.Prediction(**result_dict)
    

def wrap_function_with_timeout(fn):
    logger.debug("Entered wrap_function_with_timeout")
    # print("Entered wrap_function_with_timeout")
    @func_set_timeout(10)
    def wrapper(*args, **kwargs):
        try:
            return fn(*args, **kwargs)
        except Exception as e:
            raise e

    return wrapper

def fn_metadata(func):
    logger.debug("Entered fn_metadata")
    # print("Entered fn_metadata")
    signature = inspect.signature(func)
    docstring = inspect.getdoc(func) or "No docstring."
    return dict(function_name=func.__name__, arguments=str(signature), docstring=docstring)


class ToolSelectionSignature(dspy.Signature):
    """You are a helpful LLM agent that calls all necessary tools (usually, more than one in total) and uses the information from those tools to fulfill the user's request as accurately as possible. You should not ask the user any follow-up questions. You should assume that all tools are executed immediately and responses (e.g., via email or messengers) are available right away. You should not suggest the user to do any action that you can do yourself via the available tools. Your generated answers should be comprehensive and cover all aspects of the request."""
    
    question: str = dspy.InputField()
    trajectory: list = dspy.InputField()
    functions: dict = dspy.InputField()
    reasoning: str = dspy.OutputField()
    next_selected_fn: str = dspy.OutputField()
    args: Dict[str, Any] = dspy.OutputField()


class WebReActAgent(dspy.Module):
    def __init__(self, max_steps=6):
        logger.debug("Entered WebReActAgent.__init__")
        # print("Entered WebReActAgent.__init__")
        self.max_steps = max_steps
        self.react = dspy.ChainOfThought(ToolSelectionSignature)

    def forward(self, question, functions):
        logger.debug("Entered WebReActAgent.forward")
        # print("Entered WebReActAgent.forward")
        # Convert functions dict to metadata format for tool selection
        tools = {fn_name: fn_metadata(fn) for fn_name, fn in functions.items()}
        trajectory = []

        for step in range(self.max_steps):
            # Use synchronous DSPy call
            pred = self.react(question=question, trajectory=trajectory, functions=tools)
            
            # Extract prediction values safely
            next_selected_fn = getattr(pred, "next_selected_fn", None)
            args = getattr(pred, "args", {})
            reasoning = getattr(pred, "reasoning", None)

            # Handle None function name (safety refusal or unparseable response)
            if next_selected_fn is None or next_selected_fn == "None":
                trajectory.append({
                    "reasoning": reasoning,
                    "next_selected_fn": None,
                    "args": None,
                    "return_value": reasoning or "Agent declined to proceed",
                    "errors": None
                })
                break

            # Clean the function name
            next_selected_fn = str(next_selected_fn).strip('"').strip("'")

            # Safety check: ensure function exists
            if next_selected_fn not in functions:
                trajectory.append({
                    "reasoning": reasoning,
                    "next_selected_fn": next_selected_fn,
                    "args": args,
                    "return_value": "Error: Function not found",
                    "errors": f"Function '{next_selected_fn}' not available"
                })
                break

            # Execute tool call with timeout for safety
            try:
                wrapped_fn = wrap_function_with_timeout(functions[next_selected_fn])
                result = wrapped_fn(**args)
                
                trajectory.append(dict(
                    reasoning=reasoning,
                    next_selected_fn=next_selected_fn,
                    args=args,
                    return_value=result,
                    errors=None
                ))

                if next_selected_fn == "finish":
                    break

            except Exception as e:
                trajectory.append({
                    "reasoning": reasoning,
                    "next_selected_fn": next_selected_fn,
                    "args": args,
                    "return_value": None,
                    "errors": str(e)
                })
                break

        # Get final answer from last step
        final_answer = ""
        if trajectory:
            final_answer = trajectory[-1].get("return_value", "")

        return dspy.Prediction(answer=final_answer, trajectory=trajectory)

