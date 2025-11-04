# simplified_test_rtp_client.py
import asyncio
import glob
import os
import unittest
from time import time
from typing import Any, Dict

import httpx
from safetensors.torch import load_file
from tipc import CudaIpcHelper, SharedMemoryIPCHelper, TensorTransportClient
from tqdm import tqdm

PATH = "/root/hf/Qwen3-30B-A3B"
# PATH = "/root/hf/Qwen2-0.5B-Instruct"


class RtpLLMHttpClient(TensorTransportClient):
    def __init__(self, address: str, frentend_port: int, backend_port: int):
        super().__init__(url=f"http://{address}:{backend_port}/update_weight")
        self.client1 = httpx.AsyncClient(
            base_url=f"http://{address}:{frentend_port}", timeout=30.0
        )
        self.client2 = httpx.AsyncClient(
            base_url=f"http://{address}:{backend_port}", timeout=30.0
        )
        self.cuipc_helper = CudaIpcHelper()
        self.shipc_helper = SharedMemoryIPCHelper()
        self.records: Dict[str, Any] = {}

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.client1.aclose()
        await self.client2.aclose()

    def _handle_response(self, response: httpx.Response) -> Dict[str, Any]:
        response.raise_for_status()
        response = response.json()

        if response is not None and "status" in response:
            if response["status"] == "error":
                raise Exception(f"server internal error: {response}")
        return response

    async def chat_completion(self, name: str, prompt: str) -> None:
        payload = {
            "model": "qwen",
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 64,
            "temperature": 0.0,
            "topk": 1,
            "stream": False,
        }
        response = await self.client1.post("/v1/chat/completions", json=payload)
        content = self._handle_response(response)
        self.records[name] = {
            "content": content["choices"][0]["message"],
            "end_time": time(),
        }

    async def update_model_weight(self, path: str, method: str = "shm"):
        files = sorted(glob.glob(os.path.join(path, "model*.safetensors")))
        if not files:
            files = sorted(glob.glob(os.path.join(path, "*.safetensors")))

        weights = {}
        for fn in tqdm(files, desc="loading weights"):
            part = load_file(fn, device="cpu")
            weights.update(part)

        # sort weight is necessary
        weights = [(name, tensor) for name, tensor in tqdm(weights.items())]
        for name, tensor in tqdm(sorted(weights), "updating weights"):
            self.write(name, tensor)
        self.flush()

    async def pause(self) -> None:
        response = await self.client2.post("/pause")
        self._handle_response(response)

    async def restart(self) -> None:
        response = await self.client2.post("/restart")
        self._handle_response(response)


class TestRtpClient(unittest.IsolatedAsyncioTestCase):
    async def test_full_flow(self):
        async with RtpLLMHttpClient("localhost", 26000, 26006) as client:
            await client.restart()
            await client.chat_completion("chat_1", "hello qwen.")
            await client.pause()

            asyncio.gather(
                client.chat_completion("chat_2", "hello qwen."),
                client.chat_completion("chat_3", "tell me 1+1=?"),
            )

            await asyncio.sleep(3)
            timestamp = time()
            await client.restart()

            await client.chat_completion("chat_4", "tell me 1+1=?")
            await client.chat_completion(
                "chat_5", "find how many a in this word: apple"
            )

            await client.pause()
            asyncio.gather(client.chat_completion("_", "hello qwen."))
            await client.update_model_weight(path=PATH, method="cu_ipc")
            await client.restart()

            await client.chat_completion(
                "chat_6", "find how many a in this word: apple"
            )

        self.assertLess(client.records["chat_1"]["end_time"], timestamp)
        self.assertGreater(client.records["chat_2"]["end_time"], timestamp)
        self.assertGreater(client.records["chat_3"]["end_time"], timestamp)
        self.assertGreater(client.records["chat_4"]["end_time"], timestamp)

        print(client.records["chat_1"]["content"])
        print(client.records["chat_2"]["content"])
        print(client.records["chat_3"]["content"])
        print(client.records["chat_4"]["content"])
        print(client.records["chat_5"]["content"])
        print(client.records["chat_6"]["content"])

        self.assertEqual(
            str(client.records["chat_1"]["content"]),
            str(client.records["chat_2"]["content"]),
        )
        self.assertEqual(
            str(client.records["chat_3"]["content"]),
            str(client.records["chat_4"]["content"]),
        )
        self.assertEqual(
            str(client.records["chat_5"]["content"]),
            str(client.records["chat_6"]["content"]),
        )


if __name__ == "__main__":
    unittest.main()
