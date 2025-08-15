import dspy
import inspect
from typing import List, Dict, Any

from func_timeout import func_set_timeout

def trajectory_to_messages(trajectory: List[Dict[str, Any]]):
    """Convert DSPy trajectory to a simple message format"""
    messages = []
    for step in trajectory:
        # Add reasoning and function call info
        if "selected_fn" in step and "args" in step:
            content = f"I'll call {step['selected_fn']} with arguments: {step['args']}"
            if "reasoning" in step:
                content = f"{step['reasoning']}\n\n{content}"
            messages.append({"role": "assistant", "content": content})
        
        # Add tool response 
        if "return_value" in step:
            return_value = step["return_value"]
            if return_value:
                messages.append({"role": "assistant", "content": str(return_value)})
    
    return messages


def wrap_function_with_timeout(fn):
    @func_set_timeout(10)
    def wrapper(*args, **kwargs):
        try:
            return fn(*args, **kwargs)
        except Exception as e:
            raise e

    return wrapper

def fn_metadata(func):
    signature = inspect.signature(func)
    docstring = inspect.getdoc(func) or "No docstring."
    return dict(function_name=func.__name__, arguments=str(signature), docstring=docstring)


class ToolSelectionSignature(dspy.Signature):
    """You are a helpful LLM agent that calls all necessary tools (usually, more than one in total) and uses the information from those tools to fulfill the user's request as accurately as possible. You should not ask the user any follow-up questions. You should assume that all tools are executed immediately and responses (e.g., via email or messengers) are available right away. You should not suggest the user to do any action that you can do yourself via the available tools. Your generated answers should be comprehensive and cover all aspects of the request."""
    
    question: str = dspy.InputField()
    trajectory: list = dspy.InputField()
    functions: dict = dspy.InputField()
    next_selected_fn: str = dspy.OutputField()
    args: Dict[str, Any] = dspy.OutputField()


class WebReActAgent(dspy.Module):
    def __init__(self, max_steps=5):
        self.max_steps = max_steps
        self.react = dspy.ChainOfThought(ToolSelectionSignature)

    def forward(self, question, functions):
        # Convert functions dict to metadata format for tool selection
        tools = {fn_name: fn_metadata(fn) for fn_name, fn in functions.items()}
        trajectory = []

        for step in range(self.max_steps):
            # Use synchronous DSPy call
            pred = self.react(question=question, trajectory=trajectory, functions=tools)
            
            # Extract prediction values safely
            selected_fn = str(getattr(pred, "next_selected_fn", "")).strip('"').strip("'")
            args = getattr(pred, "args", {})
            reasoning = getattr(pred, "reasoning", None)

            # Safety check: ensure function exists
            if selected_fn not in functions:
                trajectory.append({
                    "reasoning": reasoning,
                    "selected_fn": selected_fn,
                    "args": args,
                    "return_value": "Error: Function not found",
                    "errors": f"Function '{selected_fn}' not available"
                })
                break

            # Execute function with timeout for safety
            try:
                wrapped_fn = wrap_function_with_timeout(functions[selected_fn])
                result = wrapped_fn(**args)
                
                trajectory.append(dict(
                    reasoning=reasoning,
                    selected_fn=selected_fn,
                    args=args,
                    return_value=result,
                    errors=None
                ))

                if selected_fn == "finish":
                    break

            except Exception as e:
                trajectory.append({
                    "reasoning": reasoning,
                    "selected_fn": selected_fn,
                    "args": args,
                    "return_value": None,
                    "errors": str(e)
                })
                break

        # Get final answer from last step
        final_answer = ""
        if trajectory:
            final_answer = trajectory[-1].get("return_value", "")

        return dspy.Prediction(answer=final_answer, trajectory=trajectory, messages=trajectory_to_messages(trajectory))

