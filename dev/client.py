import glob
import os
from time import time

import requests
import torch
from tqdm import tqdm

class RTPHttpClient:
    """
    与 RTP 推理服务器交互的轻量级客户端。
    提供对话、权重更新、KV 缓存管理、RoPE 重建等接口。
    """

    def __init__(self, server_address: str):
        """
        参数
        ----
        server_address : str
            服务器 IP 或 hostname，例如 "localhost"
        """
        self._srv = server_address
        # 各端口常量
        self._port_chat = 26000
        self._port_ops = 26006

    # ------------- 内部工具 -------------
    @staticmethod
    def _check_resp(resp: requests.Response | None) -> bool:
        """
        统一检查响应是否异常，True 表示正常。

        会在内部打印错误信息；调用方可按需再处理。
        """
        if resp is None:
            return False
        try:
            resp.raise_for_status()
            return True
        except requests.exceptions.RequestException as e:
            print(f"[RTPHttpClient] HTTP error: {e} , status={resp.status_code}")
            return False

    # ------------- 对话接口 -------------
    def send_query(self, prompt: str) -> requests.Response:
        """
        发送一条非流式对话请求。

        参数
        ----
        prompt : str
            用户输入

        返回
        ----
        requests.Response
        """
        url = f"http://{self._srv}:{self._port_chat}/v1/chat/completions"
        data = {
            "model": "your_model_name",  # 按需替换
            "messages": [
                {"role": "system", "content": "你是阿里巴巴Qwen大模型"},
                {"role": "user", "content": "你是谁"},
            ],
            "stream": False,
            "max_tokens": 128,
            "topk": 1,
        }
        try:
            resp = requests.post(url, json=data, timeout=60)
        except requests.exceptions.RequestException as e:
            print(f"[send_query] {e}")
            resp = None
        self._check_resp(resp)
        return resp

    # ------------- 权重更新 -------------
    def update_weight(self, name: str, w: torch.Tensor) -> requests.Response:
        """
        通过 CUDA-IPC 把权重张量推送到服务器。

        参数
        ----
        name : str
            权重名称
        w : torch.Tensor
            需位于 CUDA 上

        返回
        ----
        requests.Response
        """
        return None
        url = f"http://{self._srv}:{self._port_ops}/update_weights"
        w = w.cuda()
        meta: CuIpcTensorMeta = CudaIpcHelper().build_tensor_meta(w)
        payload = {
            "name": name,
            "time": time(),
            "method": "cuda_ipc",
            "desc": meta.hex(),
        }
        try:
            resp = requests.post(url, json=payload, timeout=300)
        except requests.exceptions.RequestException as e:
            print(f"[update_weight] {e}")
            resp = None
        self._check_resp(resp)
        return resp

    # ------------- 服务器启停 -------------
    def pause(self) -> requests.Response:
        """
        暂停服务器推理。
        """
        url = f"http://{self._srv}:{self._port_ops}/pause"
        try:
            resp = requests.post(url, timeout=30)
        except requests.exceptions.RequestException as e:
            print(f"[pause] {e}")
            resp = None
        self._check_resp(resp)
        return resp

    def restart(self) -> requests.Response:
        """
        重启服务器。
        """
        url = f"http://{self._srv}:{self._port_ops}/restart"
        try:
            resp = requests.post(url, timeout=30)
        except requests.exceptions.RequestException as e:
            print(f"[restart] {e}")
            resp = None
        self._check_resp(resp)
        return resp

    # ------------- KV 缓存管理 -------------
    def detach_physical_memory(self) -> requests.Response:
        """
        通知服务器卸载 KV 缓存物理内存。
        """
        url = f"http://{self._srv}:{self._port_ops}/detach_physical_memory"
        try:
            resp = requests.post(url, timeout=30)
        except requests.exceptions.RequestException as e:
            print(f"[detach_physical_memory] {e}")
            resp = None
        self._check_resp(resp)
        return resp

    def attach_physical_memory(self) -> requests.Response:
        """
        通知服务器重新挂载 KV 缓存物理内存。
        """
        url = f"http://{self._srv}:{self._port_ops}/attach_physical_memory"
        try:
            resp = requests.post(url, timeout=30)
        except requests.exceptions.RequestException as e:
            print(f"[attach_physical_memory] {e}")
            resp = None
        self._check_resp(resp)
        return resp

    # ------------- 组合重建 -------------
    def rebuild(self) -> requests.Response:
        """
        依次 attach 内存 + 重建 rope（使用默认 rescale_factor=1.0）。
        如需自定义系数，请手动调用 attach_physical_memory + rebuild_rope。
        """
        if not self._check_resp(resp):
            return resp


def print_server_response(res: requests.Response):
    if res is not None:
        print(f"Server Response Code: {res.status_code}, Content: {res.json()}")


N_SAMPLES: int = 3
if __name__ == "__main__":
    PATH = "/mnt/nas1/hf/Qwen3-30B-A3B"
    client = RTPHttpClient("localhost")

    response = client.restart()
    print_server_response(response)

    # 1. 首轮对话
    for i in range(N_SAMPLES):
        response = client.send_query("hello world, tell me who you are(in chinese)")
        print_server_response(response)

    # 2. 暂停推理
    print("---- Pause 服务器 ----")
    response = client.pause()
    print_server_response(response)

    print("---- 重启服务器 ----")
    response = client.restart()
    print_server_response(response)

    # 1. 对话
    for i in range(N_SAMPLES):
        response = client.send_query("hello world, tell me who you are(in chinese)")
        print_server_response(response)

    # 2. 暂停推理
    print("---- Pause 服务器 ----")
    response = client.pause()
    print_server_response(response)

    # 4. 卸载内存
    print("---- Detach 物理内存 ----")
    response = client.detach_physical_memory()
    print_server_response(response)

    print("---- Attach 物理内存 ----")
    response = client.attach_physical_memory()
    print_server_response(response)

    print("---- 重启服务器 ----")
    response = client.restart()
    print_server_response(response)

    # 6. 第二轮对话
    for i in range(N_SAMPLES):
        response = client.send_query("hello world, tell me who you are(in chinese)")
        print_server_response(response)
