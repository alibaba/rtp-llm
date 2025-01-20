import os
import sys
import signal
from threading import Thread
import requests
import time
import pathlib
import logging
import json
import asyncio
import subprocess

# import openai
# from openai.types.chat import ChatCompletionMessageParam, ChatCompletionUserMessageParam
# openai.base_url = f"http://127.0.0.1:{int(os.environ['START_PORT'])}/"
# openai.api_key = "none"
from typing import List, Dict, Optional

current_file_path = pathlib.Path(__file__).parent.absolute()
sys.path.append(str(current_file_path.parent.absolute()))
sys.path.insert(0, "/home/pengshixin.psx/FasterTransformer/bazel-out/k8-opt/bin")

from maga_transformer.start_server import main as server_main

# from py_inference.benchmark.bench_args import BenchArgs
# from py_inference.benchmark.test_object import BaseTester, HttpTester
# from py_inference.benchmark.benchmark_runner import BenchRunner
# from py_inference.benchmark.analyzers.llm_analyzer import LlmAnalyzer

from uvicorn.loops.uvloop import uvloop_setup

uvloop_setup()


def wait_server_start(server_thread: Optional[Thread], port: int):
    start_time = time.time()
    while True:
        time.sleep(1)
        try:
            if server_thread and not server_thread.is_alive():
                raise SystemExit("Server thread dead!")
            requests.get(f"http://localhost:{port}/status")
            break
        except Exception as e:
            # logging.info(f"Waiting server on {port}, used {time.time() - start_time}s: {e}")
            continue


def setup_server(port):
    """设置和启动服务器"""
    pgrp_set = False
    if int(os.environ.get("TP_SIZE", 1)) > 1:
        try:
            os.setpgrp()
            pgrp_set = True
        except Exception as e:
            logging.info(f"setpgrp error: {e}")

    server_thread = Thread(target=server_main)
    server_thread.start()
    wait_server_start(None, port)
    return pgrp_set, server_thread


def send_chat_request(port, request, step=""):
    """发送聊天请求并返回响应"""
    print("\n" + "=" * 20 + f" Step {step} " + "=" * 20)
    print(json.dumps(request, indent=4, ensure_ascii=False))
    print("-" * 50)
    response = requests.post(
        f"http://localhost:{port}/v1/chat/completions", json=request
    )
    print(response.content.decode("utf-8"))
    print("=" * 50 + "\n")
    return response


def create_chat_request(messages, tools=None, functions=None):
    """创建聊天请求"""
    return {
        "temperature": 0.0,
        "top_p": 1,
        "messages": messages,
        "debug_info": False,
        "aux_info": False,
        "tools": tools,
        # "stream": True,
    }


if __name__ == "__main__":
    port = int(os.environ["START_PORT"])
    pgrp_set, server_thread = setup_server(port)

    # 初始化工具定义
    weather_tool = {
        "type": "function",
        "function": {
            "name": "get_current_weather",
            "description": "Get the current weather in a given location",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "The city and state, e.g. San Francisco, CA",
                    },
                    "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]},
                },
                "required": ["location"],
            },
        },
    }
    tools = [weather_tool]

    # Step 1: 初始对话 - 只有用户提问
    chat_messages = [
        {
            "role": "system",
            "content": "you are a helpful assistant, your name is Qwen",
        },
        {
            "role": "user",
            "content": "北京和杭州的天气怎么样",
        },
    ]
    request1 = create_chat_request(chat_messages)
    send_chat_request(port, request1, "1: Initial Question")

    # Step 2: 添加工具定义
    request2 = create_chat_request(chat_messages, tools=tools)
    send_chat_request(port, request2, "2: With Tools Definition")

    # Step 3: 添加工具调用结果
    chat_messages.extend(
        [
            # 助手的工具调用请求
            {
                "role": "assistant",
                "tool_calls": [
                    {
                        "index": 0,
                        "id": "call_dt4VfMkmDShK5DwfbiYp3zQs",
                        "function": {
                            "name": "get_current_weather",
                            "arguments": '{"location":"北京","unit":"fahrenheit"}',
                        },
                        "type": "function",
                    },
                    {
                        "id": "call_dt4VfMkmDShK5DwfbiYp3zQ2",
                        "function": {
                            "name": "get_current_weather",
                            "arguments": '{"location":"杭州","unit":"fahrenheit"}',
                        },
                        "type": "function",
                    },
                ],
            },
            # 工具返回的结果
            {
                "role": "tool",
                "content": "北京: 10 摄氏度, 天气一般",
            },
            {
                "role": "tool",
                "content": "杭州: 20 摄氏度, 天气很好",
            },
        ]
    )

    request3 = create_chat_request(chat_messages, tools=tools)
    send_chat_request(port, request3, "3: With Tool Results")

    # 清理和退出
    if pgrp_set:
        os.killpg(0, signal.SIGKILL)
        os._exit(0)
    else:
        os._exit(0)

    server_thread.join()
