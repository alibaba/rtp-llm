from typing import Any, Dict
import secrets
import string
import json

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


def test_convert_function_to_tool_response():
    # Test with function_call
    delta_msg = {
        "content": "让我帮你查询北京的天气信息。",
        "function_call": {
            "name": "get_weather",
            "arguments": '{"location": "北京", "unit": "celsius"}'
        }
    }

    converted = convert_function_to_tool_response(delta_msg)

    # Replace random id with fixed one for comparison
    if "tool_calls" in converted:
        converted["tool_calls"][0]["id"] = "call_test"

    expected = {
        "content": "让我帮你查询北京的天气信息。",
        "tool_calls": [
            {
                "id": "call_test",
                "type": "function",
                "index": 0,
                "function": {
                    "name": "get_weather",
                    "arguments": '{"location": "北京", "unit": "celsius"}',
                },
            }
        ],
    }

    assert converted == expected

    # Test without function_call
    delta_msg_no_func = {"content": "这是普通消息"}
    expected_no_func = {"content": "这是普通消息"}

    converted_no_func = convert_function_to_tool_response(delta_msg_no_func)
    assert converted_no_func == expected_no_func

    print("All tests passed!")


if __name__ == "__main__":
    test_convert_function_to_tool_response()
