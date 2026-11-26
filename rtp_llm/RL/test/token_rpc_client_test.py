"""Unit tests for TokenRpcClient RL transport contracts."""

import math
import struct
import unittest

import grpc

from rtp_llm.cpp.model_rpc.proto import model_rpc_service_pb2 as pb
from rtp_llm.cpp.model_rpc.proto import model_rpc_service_pb2_grpc as rpc
from rtp_llm.RL.token_rpc_client import (
    TokenProtocolError,
    TokenRpcClient,
    TokenSamplingParams,
)


def _packed(values, integer=False):
    kwargs = {
        "data_type": pb.TensorPB.INT32 if integer else pb.TensorPB.FP32,
        "shape": [1, len(values)],
    }
    kwargs["int32_data" if integer else "fp32_data"] = struct.pack(
        f"<{len(values)}{'i' if integer else 'f'}", *values
    )
    return pb.TensorPB(**kwargs)


class _StubService(rpc.RpcServiceServicer):
    def __init__(self):
        self.requests = []
        self.reply_fn = None

    async def GenerateStreamCall(self, request, context):
        self.requests.append(request)
        for resp in self.reply_fn(request):
            yield resp


class TokenRpcClientTest(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.service = _StubService()
        self.server = grpc.aio.server()
        self.client = None
        try:
            rpc.add_RpcServiceServicer_to_server(self.service, self.server)
            port = self.server.add_insecure_port("127.0.0.1:0")
            await self.server.start()
            self.client = TokenRpcClient(
                f"127.0.0.1:{port}",
                vocab_size=100,
                eos_token_ids=[99],
                timeout=2.0,
            )
        except BaseException:
            await self.asyncTearDown()
            raise

    async def asyncTearDown(self):
        if self.client is not None:
            await self.client.close()
        await self.server.stop(0)

    async def test_native_raw_logprobs_and_reuse_flags(self):
        expected_lp = [-0.25, -1.5]

        def make_reply(req):
            out = pb.GenerateOutputsPB(
                request_id=req.request_id,
                model_version="stage1:3",
                global_step=3,
            )
            flat = out.flatten_output
            flat.finished.append(True)
            flat.output_ids.CopyFrom(_packed([11, 22], integer=True))
            info = flat.aux_info.add(
                input_len=len(req.token_ids),
                output_len=2,
                total_reuse_len=128,
                local_reuse_len=128,
                prefill_local_reuse_len=128,
            )
            info.softmax_probs.CopyFrom(_packed([0.5, 0.25]))
            info.raw_logprobs.CopyFrom(_packed(expected_lp))
            return [out]

        self.service.reply_fn = make_reply
        result = await self.client.generate(
            101,
            [1, 2, 3, 4],
            TokenSamplingParams(
                max_new_tokens=2,
                temperature=0.8,
                reuse_cache=True,
                enable_device_cache=True,
                require_raw_logprobs=True,
            ),
        )
        self.assertEqual(result.token_ids, [11, 22])
        self.assertTrue(result.has_raw_logprobs)
        self.assertEqual(result.model_version, "stage1:3")
        self.assertEqual(result.global_step, 3)
        self.assertEqual(result.total_reuse_len, 128)
        self.assertEqual(result.local_reuse_len, 128)
        self.assertEqual(result.prefill_local_reuse_len, 128)
        for got, exp in zip(result.log_probs, expected_lp):
            self.assertAlmostEqual(got, exp, places=6)

        req = self.service.requests[-1]
        self.assertTrue(req.generate_config.return_softmax_probs)
        self.assertTrue(req.generate_config.reuse_cache)
        self.assertTrue(req.generate_config.enable_device_cache)

    async def test_require_raw_logprobs_rejects_softmax_only_response(self):
        def make_reply(req):
            out = pb.GenerateOutputsPB(request_id=req.request_id)
            flat = out.flatten_output
            flat.finished.append(True)
            flat.output_ids.CopyFrom(_packed([7], integer=True))
            info = flat.aux_info.add(input_len=len(req.token_ids), output_len=1)
            info.softmax_probs.CopyFrom(_packed([0.5]))
            return [out]

        self.service.reply_fn = make_reply
        with self.assertRaisesRegex(TokenProtocolError, "raw_logprobs missing"):
            await self.client.generate(
                102,
                [1, 2],
                TokenSamplingParams(max_new_tokens=1, require_raw_logprobs=True),
            )

        # Fallback allowed only when require_raw_logprobs=False.
        fallback = await self.client.generate(
            103,
            [1, 2],
            TokenSamplingParams(max_new_tokens=1, require_raw_logprobs=False),
        )
        self.assertFalse(fallback.has_raw_logprobs)
        self.assertAlmostEqual(fallback.log_probs[0], math.log(0.5), places=6)


if __name__ == "__main__":
    unittest.main()
