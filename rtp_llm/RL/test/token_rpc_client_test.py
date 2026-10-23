"""Transport contract tests; real model numerics belong in GPU validation."""

import asyncio
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


def packed(values, integer=False):
    kwargs = {
        "data_type": pb.TensorPB.INT32 if integer else pb.TensorPB.FP32,
        "shape": [1, len(values)],
    }
    kwargs["int32_data" if integer else "fp32_data"] = struct.pack(
        f"<{len(values)}{'i' if integer else 'f'}", *values
    )
    return pb.TensorPB(**kwargs)


def reply(request, ids=(4, 2), probabilities=(0.5, 0.25)):
    output = pb.GenerateOutputsPB(request_id=request.request_id)
    final = output.flatten_output
    final.finished.append(True)
    final.output_ids.CopyFrom(packed(ids, integer=True))
    info = final.aux_info.add(input_len=len(request.token_ids), output_len=len(ids))
    info.softmax_probs.CopyFrom(packed(probabilities))
    return output


class Service(rpc.RpcServiceServicer):
    def __init__(self):
        self.requests = []
        self.delay = 0
        self.responses = lambda request: [reply(request)]

    async def GenerateStreamCall(self, request, context):
        self.requests.append(request)
        await asyncio.sleep(self.delay)
        for response in self.responses(request):
            yield response


class TokenClientTest(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.service = Service()
        self.server = grpc.aio.server()
        rpc.add_RpcServiceServicer_to_server(self.service, self.server)
        port = self.server.add_insecure_port("127.0.0.1:0")
        await self.server.start()
        self.client = TokenRpcClient(
            f"127.0.0.1:{port}", vocab_size=10, eos_token_ids=[2], timeout=2
        )
        self.params = TokenSamplingParams(
            max_new_tokens=4, logprobs_mode="processed_logprobs"
        )

    async def asyncTearDown(self):
        await self.client.close()
        await self.server.stop(0)

    async def test_tokens_and_seed_pass_through_without_text_or_n_expansion(self):
        result = await self.client.generate(
            19, [8, 0, 3], TokenSamplingParams(4, top_k=-1, seed=0)
        )
        self.assertEqual(result.token_ids, [4, 2])
        self.assertEqual(result.log_probs, [math.log(0.5), math.log(0.25)])
        self.assertEqual(result.finish_reason, "stop")
        req = self.service.requests[0]
        self.assertEqual(list(req.token_ids), [8, 0, 3])
        self.assertEqual(req.generate_config.num_return_sequences, 1)
        self.assertEqual(req.generate_config.top_k, 0)
        self.assertTrue(req.generate_config.HasField("random_seed"))
        self.assertEqual(req.generate_config.random_seed.value, 0)
        self.assertFalse(req.generate_config.reuse_cache)

    async def test_length_and_explicit_stop_token_are_preserved(self):
        self.service.responses = lambda req: [
            reply(req, ids=(4,), probabilities=(0.5,))
        ]
        result = await self.client.generate(1, [3], TokenSamplingParams(1))
        self.assertEqual(result.finish_reason, "length")
        result = await self.client.generate(
            2, [3], TokenSamplingParams(5, stop_token_ids=(4,))
        )
        self.assertEqual(result.token_ids, [4])
        self.assertEqual(result.finish_reason, "stop")
        self.assertEqual(
            list(
                self.service.requests[-1].generate_config.stop_words_list.rows[0].values
            ),
            [4],
        )

    async def test_greedy_processed_logprobs_preserve_full_policy_probability(self):
        result = await self.client.generate(
            20,
            [3],
            TokenSamplingParams(4, temperature=0, logprobs_mode="processed_logprobs"),
        )
        self.assertEqual(result.log_probs, [math.log(0.5), math.log(0.25)])
        self.assertEqual(result.logprobs_mode, "processed_logprobs")
        config = self.service.requests[-1].generate_config
        self.assertFalse(config.do_sample)
        self.assertEqual((config.top_k, config.top_p, config.temperature), (1, 1, 1))

    async def test_greedy_raw_mode_ignores_top_p_like_vllm(self):
        await self.client.generate(
            21, [3], TokenSamplingParams(4, temperature=0, top_p=0.7)
        )
        config = self.service.requests[-1].generate_config
        self.assertEqual((config.top_k, config.top_p, config.temperature), (1, 1, 1))

    async def test_engine_version_is_returned_and_stale_policy_rejected(self):
        def response(req):
            value = reply(req)
            value.model_version = "replica-incarnation:3"
            value.global_step = 2
            return [value]

        self.service.responses = response
        result = await self.client.generate(
            21, [3], self.params, expected_version="replica-incarnation:3"
        )
        self.assertEqual(result.model_version, "replica-incarnation:3")
        self.assertEqual(result.global_step, 2)
        with self.assertRaisesRegex(TokenProtocolError, "model version"):
            await self.client.generate(
                22, [3], self.params, expected_version="replica-incarnation:2"
            )

    async def test_missing_engine_version_cannot_satisfy_policy_check(self):
        with self.assertRaisesRegex(TokenProtocolError, "model version"):
            await self.client.generate(
                23, [3], self.params, expected_version="replica-incarnation:0"
            )

    async def test_empty_incomplete_and_wrong_request_replies_fail(self):
        def incomplete(req):
            out = reply(req)
            out.flatten_output.finished[0] = False
            return [out]

        def wrong_id(req):
            out = reply(req)
            out.request_id += 1
            return [out]

        for factory in (
            lambda req: [],
            incomplete,
            wrong_id,
            lambda req: [reply(req), reply(req)],
        ):
            with self.subTest(factory=factory):
                self.service.responses = factory
                with self.assertRaises(TokenProtocolError):
                    await self.client.generate(3, [1], self.params)

    async def test_invalid_probabilities_fail_instead_of_silent_clipping(self):
        for probabilities in ((0.0, 0.5), (math.nan, 0.5), (1.1, 0.5), (0.5,)):
            with self.subTest(probabilities=probabilities):
                self.service.responses = lambda req: [
                    reply(req, probabilities=probabilities)
                ]
                with self.assertRaises(TokenProtocolError):
                    await self.client.generate(4, [1], self.params)

    async def test_missing_probability_and_wrong_length_fail(self):
        def missing(req):
            out = reply(req)
            out.flatten_output.aux_info[0].ClearField("softmax_probs")
            return [out]

        def wrong_length(req):
            out = reply(req)
            out.flatten_output.aux_info[0].output_len = 1
            return [out]

        for factory in (missing, wrong_length):
            self.service.responses = factory
            with self.assertRaises(TokenProtocolError):
                await self.client.generate(5, [1], self.params)

    async def test_score_indexes_first_and_last_response_positions(self):
        def scoring(req):
            out = reply(req)
            cfg = req.generate_config
            scores = out.flatten_output.prompt_logits
            scores.start_pos, scores.end_pos = (
                cfg.prompt_logits_start,
                cfg.prompt_logits_end,
            )
            scores.target_logprobs.CopyFrom(packed([-1.0, -2.0, -3.0]))
            return [out]

        self.service.responses = scoring
        self.assertEqual(
            await self.client.score_tokens(6, [7, 8], [3, 4, 2]), [-1.0, -2.0, -3.0]
        )
        req = self.service.requests[-1]
        self.assertEqual(list(req.token_ids), [7, 8, 3, 4, 2])
        self.assertEqual(
            (
                req.generate_config.prompt_logits_start,
                req.generate_config.prompt_logits_end,
            ),
            (1, 4),
        )

    async def test_shifted_score_positions_fail(self):
        def scoring(req):
            out = reply(req)
            scores = out.flatten_output.prompt_logits
            scores.start_pos, scores.end_pos = 0, 1
            scores.target_logprobs.CopyFrom(packed([-1.0]))
            return [out]

        self.service.responses = scoring
        with self.assertRaises(TokenProtocolError):
            await self.client.score_tokens(7, [7, 8], [2])

    async def test_unsupported_processed_distribution_is_rejected_before_rpc(self):
        for override in (
            {"temperature": 0.7},
            {"top_p": 0.9},
            {"top_k": 20},
            {"min_new_tokens": 1},
        ):
            with self.subTest(override=override):
                with self.assertRaises(NotImplementedError):
                    await self.client.generate(
                        8,
                        [1],
                        TokenSamplingParams(
                            4, logprobs_mode="processed_logprobs", **override
                        ),
                    )
        self.assertEqual(self.service.requests, [])

    async def test_invalid_tokens_and_sampling_are_rejected_before_rpc(self):
        for prompt in ([], [True], [-1], [10], [1.5]):
            with self.subTest(prompt=prompt):
                with self.assertRaises(ValueError):
                    await self.client.generate(9, prompt, self.params)
        for params in (
            TokenSamplingParams(0),
            TokenSamplingParams(4, seed=-1),
            TokenSamplingParams(4, seed=2**31),
            TokenSamplingParams(4, top_p=0),
        ):
            with self.assertRaises(ValueError):
                await self.client.generate(9, [1], params)
        self.assertEqual(self.service.requests, [])

    async def test_timeout_propagates_and_releases_request_identity(self):
        self.service.delay = 1
        self.client.timeout = 0.02
        with self.assertRaises(grpc.aio.AioRpcError) as caught:
            await self.client.generate(10, [1], self.params)
        self.assertEqual(caught.exception.code(), grpc.StatusCode.DEADLINE_EXCEEDED)
        self.assertFalse(self.client._active)
        self.assertFalse(self.client._request_ids)

    async def test_cancellation_does_not_return_a_partial_training_sample(self):
        self.service.delay = 1
        task = asyncio.create_task(self.client.generate(11, [1], self.params))

        async def wait_for_request():
            while not self.service.requests:
                if task.done():
                    await task  # Surface schema/setup failures instead of hanging.
                await asyncio.sleep(0.001)

        await asyncio.wait_for(wait_for_request(), timeout=2)
        with self.assertRaises(ValueError):
            await self.client.generate(11, [1], self.params)
        task.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await task
        self.assertFalse(self.client._active)
        self.assertFalse(self.client._request_ids)


if __name__ == "__main__":
    unittest.main()
