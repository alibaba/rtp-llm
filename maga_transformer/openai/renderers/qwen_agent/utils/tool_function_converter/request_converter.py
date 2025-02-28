from typing import Any, Dict
import json

def convert_tool_to_function_request(request: Dict[str, Any]) -> Dict[str, Any]:
    """
    Convert a request with tools to one with functions.
    Also converts:
    1. tools to functions in request
    2. tool_calls to function_call in messages
    3. role "tool" to "function" in messages

    Args:
        request: Original request dict with tools

    Returns:
        dict: Converted request with functions instead of tools
    """
    new_request = request.copy()

    if "tools" in request:
        functions = []
        for tool in request["tools"]:
            if tool["type"] == "function":
                functions.append(tool["function"])
        new_request["functions"] = functions
        new_request.pop("tools", None)

    if "messages" in request:
        new_messages = []
        for msg in request["messages"]:
            new_msg = msg.copy()

            if msg.get("role") == "tool":
                new_msg["role"] = "function"

            if "tool_calls" in msg:
                tool_call = msg["tool_calls"][0]
                new_msg["function_call"] = {
                    "name": tool_call["function"]["name"],
                    "arguments": tool_call["function"]["arguments"]
                }
                new_msg.pop("tool_calls", None)

            new_messages.append(new_msg)

        new_request["messages"] = new_messages

    return new_request


def test_convert_tool_to_function_request():
    request = {
        "model": "qwen-max",
        "messages": [
            {"role": "user", "content": "帮我查一下北京的天气"},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "call_1",
                        "type": "function",
                        "index": 0,
                        "function": {
                            "name": "get_weather",
                            "arguments": '{"location": "北京", "unit": "celsius"}',
                        },
                    }
                ],
            },
            {
                "role": "tool",
                "content": '{"temperature": 20, "weather": "晴天", "unit": "celsius"}',
            },
            {
                "role": "assistant",
                "content": "根据天气信息，北京现在是晴天，温度20摄氏度，天气不错适合出行。",
            },
        ],
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "description": "获取指定城市的天气信息",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "location": {"type": "string", "description": "城市名称"},
                            "unit": {
                                "type": "string",
                                "enum": ["celsius", "fahrenheit"],
                                "description": "温度单位",
                            },
                        },
                        "required": ["location"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "get_air_quality",
                    "description": "获取指定城市的空气质量信息",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "city": {"type": "string", "description": "城市名称"}
                        },
                        "required": ["city"],
                    },
                },
            },
        ],
        "temperature": 0.7,
        "stream": True,
    }

    expected = {
        "model": "qwen-max",
        "messages": [
            {"role": "user", "content": "帮我查一下北京的天气"},
            {
                "role": "assistant",
                "content": None,
                "function_call": {
                    "name": "get_weather",
                    "arguments": '{"location": "北京", "unit": "celsius"}',
                },
            },
            {
                "role": "function",
                "content": '{"temperature": 20, "weather": "晴天", "unit": "celsius"}',
            },
            {
                "role": "assistant",
                "content": "根据天气信息，北京现在是晴天，温度20摄氏度，天气不错适合出行。",
            },
        ],
        "functions": [
            {
                "name": "get_weather",
                "description": "获取指定城市的天气信息",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "location": {"type": "string", "description": "城市名称"},
                        "unit": {
                            "type": "string",
                            "enum": ["celsius", "fahrenheit"],
                            "description": "温度单位",
                        },
                    },
                    "required": ["location"],
                },
            },
            {
                "name": "get_air_quality",
                "description": "获取指定城市的空气质量信息",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "city": {"type": "string", "description": "城市名称"}
                    },
                    "required": ["city"],
                },
            },
        ],
        "temperature": 0.7,
        "stream": True,
    }

    converted = convert_tool_to_function_request(request)
    assert converted == expected
    print("All tests passed!")


if __name__ == "__main__":
    test_convert_tool_to_function_request()
