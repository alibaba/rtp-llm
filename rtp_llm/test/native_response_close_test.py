import asyncio
from types import SimpleNamespace
from unittest import IsolatedAsyncioTestCase, main
from unittest.mock import MagicMock, patch

import torch

from rtp_llm.frontend.frontend_worker import FrontendWorker
from rtp_llm.pipeline.pipeline import Pipeline
from rtp_llm.server.backend_rpc_server_visitor import BackendRPCServerVisitor
from rtp_llm.utils.base_model_datatypes import AuxInfo, GenerateOutput, GenerateOutputs


class _Tokenizer:
    def __len__(self):
        return 32

    def encode(self, text, **kwargs):
        return [1]

    def batch_decode(self, batches, **kwargs):
        return ["ok" for _ in batches]

    def convert_tokens_to_ids(self, token):
        return 1


class _BackendStream:
    def __init__(self):
        self.first = True
        self.closed = False
        self.blocked = asyncio.Event()
        self.next_cancelled = asyncio.Event()
        self.never = asyncio.Event()
        self.close_calls = 0
        self.release_count = 0

    def __aiter__(self):
        return self

    async def __anext__(self):
        if self.closed:
            raise StopAsyncIteration
        if self.first:
            self.first = False
            return GenerateOutputs(
                [
                    GenerateOutput(
                        output_ids=torch.tensor([[7]], dtype=torch.int32),
                        finished=False,
                        aux_info=AuxInfo(),
                    )
                ]
            )
        self.blocked.set()
        try:
            await self.never.wait()
        except asyncio.CancelledError:
            self.next_cancelled.set()
            raise
        raise AssertionError("unreachable")

    async def aclose(self):
        self.close_calls += 1
        if not self.closed:
            self.closed = True
            self.release_count += 1


class _ModelClient:
    def __init__(self, source):
        self.source = source
        self.inputs = []

    def enqueue(self, generate_input):
        self.inputs.append(generate_input)
        return self.source


class NativeResponseCloseTest(IsolatedAsyncioTestCase):
    async def test_frontend_close_reaches_exact_backend_stream_once(self):
        source = _BackendStream()
        visitor = object.__new__(BackendRPCServerVisitor)
        visitor.max_seq_len = 128
        visitor.sp_config = None
        visitor.host_service = SimpleNamespace(service_available=False)
        visitor.model_rpc_client = _ModelClient(source)

        pipeline = object.__new__(Pipeline)
        pipeline.tokenizer = _Tokenizer()
        pipeline._special_tokens = SimpleNamespace(
            stop_words_id_list=[],
            stop_words_str_list=[],
            eos_token_id=0,
        )
        pipeline.backend_rpc_server_visitor = visitor

        worker = object.__new__(FrontendWorker)
        worker.pipeline = pipeline
        worker.generate_env_config = SimpleNamespace(
            think_end_token_id=-1,
            think_mode=0,
            think_end_tag="",
        )

        with patch("rtp_llm.pipeline.pipeline.kmonitor", new=MagicMock()):
            response = worker.inference(
                prompt="x",
                __request_id__=17,
                generate_config={
                    "aux_info": False,
                    "ignore_eos": True,
                    "max_new_tokens": 2,
                },
            )
            first = await response.__anext__()
            self.assertEqual(first.response, "ok")

            pending_next = asyncio.create_task(response.__anext__())
            await asyncio.wait_for(source.blocked.wait(), timeout=1)
            await asyncio.wait_for(
                asyncio.gather(response.aclose(), response.aclose()),
                timeout=1,
            )

            with self.assertRaises(asyncio.CancelledError):
                await pending_next
            await response.aclose()

        self.assertEqual(len(visitor.model_rpc_client.inputs), 1)
        self.assertEqual(visitor.model_rpc_client.inputs[0].request_id, 17)
        self.assertTrue(source.next_cancelled.is_set())
        self.assertEqual(source.close_calls, 1)
        self.assertEqual(source.release_count, 1)


if __name__ == "__main__":
    main()
