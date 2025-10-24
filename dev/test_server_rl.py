# simplified_test_rtp_client.py
import asyncio
import time
import unittest
from typing import Any, Dict

import httpx


class RtpLLMHttpClient:
    def __init__(self, address: str, frentend_port: int, backend_port: int):
        self.client1 = httpx.AsyncClient(
            base_url=f"http://{address}:{frentend_port}", timeout=30.0
        )
        self.client2 = httpx.AsyncClient(
            base_url=f"http://{address}:{backend_port}", timeout=30.0
        )
        self.records: Dict[str, Any] = {}

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.client1.aclose()
        await self.client2.aclose()

    async def _handle_response(self, response: httpx.Response) -> Dict[str, Any]:
        response.raise_for_status()
        response = response.json()
        if response and "error" in response:
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
        content = await self._handle_response(response)
        self.records[name] = {
            "content": content["choices"][0]["message"],
            "end_time": time.time(),
        }

    async def pause(self) -> None:
        response = await self.client2.post("/pause")
        await self._handle_response(response)

    async def restart(self) -> None:
        response = await self.client2.post("/restart")
        await self._handle_response(response)

    async def detach(self) -> None:
        response = await self.client2.post("/detach_physical_memory")
        await self._handle_response(response)

    async def attach(self) -> None:
        response = await self.client2.post("/attach_physical_memory")
        await self._handle_response(response)


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
            timestamp = time.time()
            await client.restart()

            await client.chat_completion("chat_4", "tell me 1+1=?")
            await client.chat_completion(
                "chat_5", "find how many a in this word: apple"
            )

            await client.detach()
            await client.pause()
            await client.restart()
            await client.attach()

            await client.chat_completion(
                "chat_6", "find how many a in this word: apple"
            )

        self.assertLess(client.records["chat_1"]["end_time"], timestamp)
        self.assertGreater(client.records["chat_2"]["end_time"], timestamp)
        self.assertGreater(client.records["chat_3"]["end_time"], timestamp)
        self.assertGreater(client.records["chat_4"]["end_time"], timestamp)

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
