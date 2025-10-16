# -*- coding: utf-8 -*-
import concurrent.futures
import requests
import threading
import sys
import random

def load_prompts_from_file(filename):
    """
    从文件中读取提示词，每行作为一个提示词元素
    """
    prompts = []
    try:
        with open(filename, 'r', encoding='utf-8') as file:
            for line in file:
                line = line.strip()  # 去除首尾空白字符
                if line:  # 只添加非空行
                    prompts.append(line)
        print(f"成功从 {filename} 加载了 {len(prompts)} 个提示词")
        return prompts
    except FileNotFoundError:
        print(f"错误：文件 {filename} 不存在")
        return []
    except Exception as e:
        print(f"读取文件时出错：{e}")
        return []

def send_request(prompt):
    url = "http://localhost:8066/v1/chat/completions"
    headers = {"Content-Type": "application/json"}
    data = {
        "temperature": 0,
        "max_tokens": 2000,
        "top_k": 1,
        "enable_thinking": False,
        "messages": [{"role": "user", "content": prompt}]
    }
    try:
        import time
        time.sleep(0.1)
        response = requests.post(url, json=data, headers=headers)

        # 解析JSON响应
        response_data = response.json()

        # 提取content和reuse_len字段
        content = response_data.get("choices", [{}])[0].get("message", {}).get("content", "N/A")
        reuse_len = response_data.get("aux_info", {}).get("reuse_len", "N/A")

        # 简洁输出
        print(f"Content: {content} | Reuse_len: {reuse_len}")

    except Exception as e:
        print(f"[Error] - {e}")

def main():
    # 检查命令行参数
    if len(sys.argv) < 2:
        print("用法: python script.py <提示词文件> [并发数]")
        print("示例: python script.py prompts.txt 64")
        return
    
    # 第一个参数是文件名
    filename = sys.argv[1]
    
    # 第二个参数是并发数（可选）
    if len(sys.argv) >= 3:
        try:
            bs = int(sys.argv[2])
        except ValueError:
            print("请输入有效的整数作为并发数")
            return
    else:
        bs = 64  # 默认并发数

    # 从文件加载提示词
    prompts = load_prompts_from_file(filename)
    if not prompts:
        print("没有可用的提示词，程序退出")
        return

    print(f"并发数: {bs}")
    print(f"提示词数量: {len(prompts)}")
    
    # 使用线程池并发执行请求
    with concurrent.futures.ThreadPoolExecutor(max_workers=bs) as executor:
        # 为每个请求分配一个提示词，如果请求数多于提示词数量则循环使用
        futures = [executor.submit(send_request, prompts[i % len(prompts)]) for i in range(bs)]
        # 等待所有任务完成
        concurrent.futures.wait(futures)

if __name__ == "__main__":
    main()
