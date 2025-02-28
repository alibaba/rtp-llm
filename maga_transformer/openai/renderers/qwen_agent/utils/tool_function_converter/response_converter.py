from typing import Any, Dict
import secrets
import string

def _generate_random_call_id(length: int = 24) -> str:
    """生成随机调用ID"""
    characters = string.ascii_letters + string.digits
    random_string = "".join(secrets.choice(characters) for _ in range(length))
    return "call_" + random_string

def convert_function_to_tool_response(delta_msg: Dict[str, Any]) -> Dict[str, Any]:
    """
    Convert a response with function_call to one with tool_calls.
    
    Args:
        delta_msg: Original delta message dict with function_call
        
    Returns:
        dict: Converted delta message with tool_calls instead of function_call
    """
    new_delta = delta_msg.copy()
    
    if "function_call" in delta_msg:
        func_call = delta_msg["function_call"]
        new_delta["tool_calls"] = [
            {
                "id": _generate_random_call_id(),
                "type": "function",
                "index": 0,
                "function": {
                    "name": func_call["name"],
                    "arguments": func_call["arguments"]
                }
            }
        ]
        new_delta.pop("function_call", None)
        
    return new_delta

if __name__ == "__main__":
    # Sample delta message with function_call
    delta_msg = {
        "content": "让我帮你查询北京的天气信息。",
        "function_call": {
            "name": "get_weather",
            "arguments": '{"location": "北京", "unit": "celsius"}'
        }
    }

    # Convert the delta message
    converted_delta = convert_function_to_tool_response(delta_msg)
    
    # Print original and converted messages for comparison  
    import json
    print("Original Delta Message:")
    print(json.dumps(delta_msg, indent=2, ensure_ascii=False))
    print("\nConverted Delta Message:")
    print(json.dumps(converted_delta, indent=2, ensure_ascii=False))
