"""Exercise input embeddings over model_rpc with only a real backend running."""

import argparse
import asyncio
import itertools
import json
import logging
import os
import sys
import time
import unittest
from pathlib import Path
from typing import NamedTuple

import grpc
import psutil
import torch
from safetensors import safe_open

from rtp_llm.config.exceptions import ExceptionType, FtRuntimeException
from rtp_llm.config.generate_config import GenerateConfig, RoleType
from rtp_llm.config.py_config_modules import PyEnvConfigs, ServerConfig
from rtp_llm.cpp.model_rpc.model_rpc_client import ModelRpcClient
from rtp_llm.cpp.model_rpc.proto.flexlb_schedule_service_pb2 import (
    FlexlbScheduleResponsePB,
)
from rtp_llm.cpp.model_rpc.proto.flexlb_schedule_service_pb2_grpc import (
    FlexlbServiceServicer,
    add_FlexlbServiceServicer_to_server,
)
from rtp_llm.server.backend_rpc_server_visitor import BackendRPCServerVisitor
from rtp_llm.server.host_service import HostService, HostServiceArgs, RouteSnapshot
from rtp_llm.test.utils.maga_server_manager import MagaServerManager
from rtp_llm.utils.base_model_datatypes import GenerateInput, InputEmbeddings


class BackendResult(NamedTuple):
    hidden_states: list[torch.Tensor]
    logits: list[torch.Tensor]
    output_ids: list[torch.Tensor]


class BatchMasterTripwire(FlexlbServiceServicer):
    """A reachable Master rejecting metadata-only scheduling, like BATCH mode.

    The target under test is bypassing this endpoint, not Java dispatch itself.
    """

    def __init__(self):
        self.calls = 0

    async def Schedule(self, request, context):
        self.calls += 1
        return FlexlbScheduleResponsePB(
            success=False, code=8512, error_message="BATCH_BUILD_FAILED"
        )


class BackendRpcServerManager(MagaServerManager):
    def wait_sever_done(self, timeout=600):
        # No frontend is running: readiness must use the backend gRPC listener,
        # not the launcher's default frontend HTTP /health endpoint.
        ports = ServerConfig()
        ports.start_port = self.port
        with grpc.insecure_channel(f"127.0.0.1:{ports.rpc_server_port}") as channel:
            ready = grpc.channel_ready_future(channel)
            deadline = time.monotonic() + timeout
            try:
                while self._server_process.poll() is None:
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        break
                    try:
                        ready.result(timeout=min(1, remaining))
                        return True
                    except grpc.FutureTimeoutError:
                        continue
            finally:
                ready.cancel()
        self.print_process_log()
        return False


class InputEmbeddingRpcTest(unittest.TestCase):
    expect_prefill_cuda_graph = False
    test_reuse_cache = False

    def test_real_backend_embedding_values(self):
        model_path = Path(
            os.environ.get(
                "INPUT_EMBEDDING_SMOKE_MODEL_PATH", "/mnt/nas1/hf/Qwen2.5-0.5B-Instruct"
            )
        ).resolve()
        self.assertTrue(model_path.is_dir(), f"missing checkpoint: {model_path}")
        token_ids = torch.tensor(
            [9707, 11, 358, 1079, 264, 1657, 13], dtype=torch.int32
        )
        rows = self._embedding_rows(model_path, token_ids)
        visible = os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(",")[0]
        manager = BackendRpcServerManager(
            device_ids=[visible],
            role_name="input_embedding_backend_rpc",
            env_args={
                "ENABLE_PROMPT_GENERATOR": "false",
                "ENABLE_PROMPT_GENERATOR_MPS": "false",
                "FRONTEND_SERVER_COUNT": "0",
                "FT_SERVER_TEST": "1",
                "LOG_LEVEL": "INFO",
                "SPECIAL_TOKEN_TEXT": "<|im_end|>",
            },
            smoke_args_str=os.environ.get("SMOKE_ARGS", ""),
        )
        try:
            self.assertTrue(
                manager.start_server(
                    model_path=str(model_path), model_type="qwen_2", timeout=600
                ),
                f"backend startup failed; see {manager.log_file_path}",
            )
            self._assert_backend_only(manager)
            ports = ServerConfig()
            ports.start_port = manager.port
            check = (
                self._check_cache_requests
                if self.test_reuse_cache
                else self._check_requests
            )
            asyncio.run(check(ports.rpc_server_port, token_ids, rows))
            self._assert_backend_only(manager)
        finally:
            manager.stop_server()

    def _assert_backend_only(self, manager):
        children = psutil.Process(manager.server_pid).children(recursive=True)
        titles = [" ".join(child.cmdline()) for child in children]
        self.assertTrue(
            any("rtp_llm_backend_server" in title for title in titles), titles
        )
        for title in titles:
            self.assertNotIn("rtp_llm_frontend_server", title)
            self.assertNotIn("pg_server_rank_", title)

    def _embedding_rows(self, model_path, token_ids):
        key = "model.embed_tokens.weight"
        index_path = model_path / "model.safetensors.index.json"
        if index_path.exists():
            index = json.loads(index_path.read_text())
            shard = index.get("weight_map", {}).get(key)
            self.assertIsNotNone(shard, f"{index_path} is missing embedding key {key}")
            candidates = [model_path / shard]
        else:
            candidates = sorted(model_path.glob("*.safetensors"))
        for candidate in candidates:
            with safe_open(str(candidate), framework="pt", device="cpu") as weights:
                if key in weights.keys():
                    # Slice only the required rows, without loading transformer
                    # weights or allocating a second full embedding table.
                    table = weights.get_slice(key)
                    return torch.cat(
                        [table[token : token + 1] for token in token_ids.tolist()]
                    ).to(torch.bfloat16)
        self.fail(f"{key} was not found in safetensors checkpoint {model_path}")

    async def _check_cache_requests(self, port, token_ids, rows):
        client = ModelRpcClient(
            addresses=[f"127.0.0.1:{port}"],
            client_config=PyEnvConfigs().grpc_config.get_client_config(),
            max_rpc_timeout_ms=60000,
        )
        request_ids = itertools.count(92000)
        prompt = token_ids.repeat(20)  # 140 tokens: two full 64-token cache blocks.
        full_rows = rows.repeat(20, 1)
        spans = [full_rows[48:80].clone(), full_rows[100:108].clone()]

        async def generate(
            embeddings,
            reuse,
            expected_reuse,
            full_hidden_states=False,
            reference_ids=None,
        ):
            request = GenerateInput(
                request_id=next(request_ids),
                token_ids=prompt if reference_ids is None else reference_ids,
                mm_inputs=[],
                generate_config=GenerateConfig(
                    max_new_tokens=3,
                    min_new_tokens=3,
                    top_k=1,
                    random_seed=1234,
                    timeout_ms=60000,
                    return_logits=True,
                    reuse_cache=reuse,
                    return_all_hidden_states=full_hidden_states,
                ),
                input_embeddings=(
                    InputEmbeddings(embeddings, [48, 100])
                    if embeddings is not None
                    else None
                ),
            )
            logits = []
            finished = False
            prompt_states = None
            async for response in client.enqueue(request):
                output = response.generate_outputs[0]
                self.assertEqual(output.aux_info.reuse_len, expected_reuse)
                if self.expect_prefill_cuda_graph:
                    # Generation prefill graphs currently reject nonzero prefixes,
                    # including token-only requests. Cache hits must fall back to eager.
                    self.assertEqual(
                        output.aux_info.generation_prefill_cuda_graph_status,
                        "prefix_cache_not_supported" if expected_reuse else "replayed",
                    )
                self.assertIsNotNone(output.logits)
                logits.append(output.logits.float().cpu())
                if output.all_hidden_states is not None:
                    prompt_states = output.all_hidden_states.float().cpu()
                finished = output.finished
            self.assertTrue(finished)
            if full_hidden_states:
                self.assertIsNotNone(prompt_states)
                self.assertEqual(
                    tuple(prompt_states.shape), (prompt.numel(), rows.shape[1])
                )
            return output.output_ids.cpu(), logits, prompt_states

        def assert_equivalent(actual, expected):
            torch.testing.assert_close(actual[0], expected[0], rtol=0, atol=0)
            self.assertEqual(len(actual[1]), len(expected[1]))
            for got, want in zip(actual[1], expected[1]):
                torch.testing.assert_close(got, want, rtol=0.02, atol=0.02)
            if expected[2] is not None:
                torch.testing.assert_close(actual[2], expected[2], rtol=0.02, atol=0.02)

        try:
            # Match the token-only execution shape: cached and uncached prefill
            # can differ numerically even without custom embeddings.
            baseline = await generate(spans, False, 0)
            assert_equivalent(await generate(spans, True, 0), baseline)
            warm = await generate(spans, True, 128)
            torch.testing.assert_close(warm[0], baseline[0], rtol=0, atol=0)

            # Identical placeholder IDs must not hit an embedding request's KV.
            text_reference = await generate(None, False, 0)
            assert_equivalent(baseline, text_reference)
            assert_equivalent(await generate(None, True, 0), text_reference)
            text_warm = await generate(None, True, 128)
            assert_equivalent(warm, text_warm)

            # KV alone cannot reconstruct full prompt states: explicitly recompute them.
            full_reference = await generate(spans, False, 0, full_hidden_states=True)
            assert_equivalent(
                await generate(spans, True, 0, full_hidden_states=True), full_reference
            )

            async def check_changed(
                embeddings, reference_ids, reuse_length, warm=False
            ):
                reference = await generate(None, False, 0, reference_ids=reference_ids)
                assert_equivalent(await generate(embeddings, False, 0), reference)
                for reuse in [reuse_length, 128] if warm else [reuse_length]:
                    actual = await generate(embeddings, True, reuse)
                    expected = await generate(
                        None, True, reuse, reference_ids=reference_ids
                    )
                    assert_equivalent(actual, expected)
                    torch.testing.assert_close(actual[0], reference[0], rtol=0, atol=0)

            # Use another real embedding row so each changed request has an
            # independent token-only oracle with the same cached prefix length.
            # Block two change: reuse 64 tokens, slicing inside the first span.
            changed = [span.clone() for span in spans]
            changed[0][17] = rows[0]
            changed_ids = prompt.clone()
            changed_ids[65] = token_ids[0]
            await check_changed(changed, changed_ids, 64, warm=True)

            # Change only the later span; its values must also participate in the key.
            changed_later = [span.clone() for span in spans]
            changed_later[1][0] = rows[0]
            later_ids = prompt.clone()
            later_ids[100] = token_ids[0]
            await check_changed(changed_later, later_ids, 64)

            # Changing block one invalidates all subsequent blocks; original A remains reusable.
            changed[0][0] = rows[0]
            changed_ids[48] = token_ids[0]
            await check_changed(changed, changed_ids, 0)
            assert_equivalent(await generate(spans, True, 128), text_warm)
            assert_equivalent(await generate(spans, False, 0), baseline)
        finally:
            await client.close()

    async def _check_requests(self, port, token_ids, rows):
        client = ModelRpcClient(
            addresses=[f"127.0.0.1:{port}"],
            client_config=PyEnvConfigs().grpc_config.get_client_config(),
            max_rpc_timeout_ms=60000,
        )
        master = grpc.aio.server()
        tripwire = BatchMasterTripwire()
        add_FlexlbServiceServicer_to_server(tripwire, master)
        master_port = master.add_insecure_port("127.0.0.1:0")
        await master.start()
        configs = PyEnvConfigs()
        configs.pd_separation_config.role_type = RoleType.FRONTEND
        configs.pd_separation_config.max_rpc_timeout_ms = 60000
        visitor = BackendRPCServerVisitor(
            max_seq_len=256,
            seq_size_per_block=64,
            pd_sep_config=configs.pd_separation_config,
            # Routed embedding requests must use discovered roles, never this
            # deliberately unusable default. Plain token references use client.
            addresses=["127.0.0.1:1"],
            master_config=configs.master_config,
            source_role="smoke",
        )
        host_args = HostServiceArgs(
            pdfusion_domain=f"127.0.0.1:{port - 1}", use_local=True
        )
        host_service = HostService(host_args)
        with host_service.master_service._snapshot_lock:
            # Seed service discovery; the actual HostService domain selection,
            # visitor routing and gRPC clients remain production implementations.
            host_service.master_service._route_snapshot = RouteSnapshot(
                master_addr=f"127.0.0.1:{master_port - 2}",
                slave_addr=None,
                queue_length=0,
                host_health_status={},
            )
        visitor.host_service = host_service
        visitor.master_client.host_service = host_service
        visitor.backend_role_list = visitor.get_backend_role_list(
            configs.pd_separation_config, host_args
        )
        request_ids = itertools.count(91000)

        async def generate(
            spans=None, locs=None, sequences=0, new_tokens=1, prompt_repeats=1
        ):
            prompt_ids = token_ids.repeat(prompt_repeats)
            request = GenerateInput(
                request_id=next(request_ids),
                token_ids=prompt_ids,
                mm_inputs=[],
                generate_config=GenerateConfig(
                    max_new_tokens=new_tokens,
                    min_new_tokens=new_tokens,
                    top_k=1,
                    random_seed=1234,
                    timeout_ms=60000,
                    return_all_hidden_states=True,
                    return_logits=True,
                    num_return_sequences=sequences,
                ),
                input_embeddings=(
                    InputEmbeddings(spans, locs) if spans is not None else None
                ),
            )
            if spans is not None:
                await visitor.route_ips(request)
                self.assertFalse(request.enqueued_by_master)
                self.assertEqual(request.generate_config.role_addrs[0].grpc_port, port)
                stream = visitor.model_rpc_client.enqueue(request)
            else:
                stream = client.enqueue(request)
            states = None
            logits = None
            finished = False
            async for response in stream:
                outputs = response.generate_outputs
                self.assertEqual(len(outputs), sequences or 1)
                if self.expect_prefill_cuda_graph:
                    for output in outputs:
                        status = output.aux_info.generation_prefill_cuda_graph_status
                        if prompt_ids.numel() <= 64:
                            self.assertEqual(status, "replayed")
                        else:
                            self.assertEqual(
                                status, "input_tokens_exceed_capture_limit"
                            )
                if all(output.all_hidden_states is not None for output in outputs):
                    states = [
                        output.all_hidden_states.float().cpu() for output in outputs
                    ]
                if all(output.logits is not None for output in outputs):
                    logits = [output.logits.float().cpu() for output in outputs]
                finished = all(output.finished for output in outputs)
            self.assertTrue(finished, "backend did not finish the request")
            self.assertIsNotNone(states, "backend returned no prompt hidden states")
            for state in states:
                self.assertEqual(
                    tuple(state.shape), (prompt_ids.numel(), rows.shape[1])
                )
                self.assertTrue(torch.isfinite(state).all().item())
            self.assertIsNotNone(logits, "backend returned no per-sequence logits")
            for output_logits in logits:
                self.assertGreater(output_logits.numel(), 0)
                self.assertTrue(torch.isfinite(output_logits).all().item())
            output_ids = [output.output_ids.cpu() for output in outputs]
            for ids in output_ids:
                self.assertEqual(ids.numel(), new_tokens)
            return BackendResult(states, logits, output_ids)

        def assert_equivalent(actual, expected):
            for actual_tensors, expected_tensors in zip(actual[:2], expected[:2]):
                self.assertEqual(len(actual_tensors), len(expected_tensors))
                for got, want in zip(actual_tensors, expected_tensors):
                    torch.testing.assert_close(got, want, rtol=0.02, atol=0.02)
            self.assertEqual(len(actual.output_ids), len(expected.output_ids))
            for got, want in zip(actual.output_ids, expected.output_ids):
                torch.testing.assert_close(got, want, rtol=0, atol=0)

        try:
            control = GenerateInput(
                request_id=90999,
                token_ids=token_ids,
                mm_inputs=[],
                generate_config=GenerateConfig(max_new_tokens=1, timeout_ms=60000),
            )
            with self.assertRaises(FtRuntimeException) as error:
                await visitor.master_client.get_backend_role_addrs(
                    block_cache_keys=[],
                    cache_key_block_size=64,
                    input=control,
                    request_id=control.request_id,
                    input_pb=None,
                )
            self.assertEqual(
                error.exception.exception_type, ExceptionType.BATCH_BUILD_FAILED
            )
            self.assertEqual(tripwire.calls, 1)
            baseline = await generate()
            # Matching rows must retain the token path's actual numerical result.
            single = await generate([rows[2:4].clone()], [2])
            assert_equivalent(single, baseline)
            multiple = await generate([rows[1:3].clone(), rows[5].clone()], [1, 5])
            assert_equivalent(multiple, baseline)

            # Also cover span replication into two return sequences.
            baseline_two = await generate(sequences=2)
            multiple_two = await generate(
                [rows[1:3].clone(), rows[5].clone()], [1, 5], sequences=2
            )
            assert_equivalent(multiple_two, baseline_two)

            # A nonuniform perturbation cannot disappear through RMSNorm as a
            # simple scalar rescaling could. Keep the prefix and positions fixed.
            perturbation = torch.linspace(-1.0, 1.0, rows.shape[1]).to(torch.bfloat16)
            changed = await generate([rows[2:4] + perturbation], [2])
            torch.testing.assert_close(
                changed.hidden_states[0][:2],
                baseline.hidden_states[0][:2],
                rtol=0.02,
                atol=0.02,
            )
            self.assertFalse(
                torch.allclose(
                    changed.hidden_states[0][2:],
                    baseline.hidden_states[0][2:],
                    rtol=0.02,
                    atol=0.02,
                ),
                "backend ignored changed input embedding values",
            )
            logging.info(
                "changed-span max hidden delta: %s",
                (changed.hidden_states[0] - baseline.hidden_states[0])
                .abs()
                .max()
                .item(),
            )

            # Equal rows alone cannot detect a dropped second span. Perturb only
            # that span while retaining the first, then verify its exact causal
            # position changes and all preceding positions remain unchanged.
            later_row = rows[5] - perturbation.roll(rows.shape[1] // 3)
            changed_later = await generate([rows[1:3].clone(), later_row], [1, 5])
            torch.testing.assert_close(
                changed_later.hidden_states[0][:5],
                baseline.hidden_states[0][:5],
                rtol=0.02,
                atol=0.02,
            )
            self.assertFalse(
                torch.allclose(
                    changed_later.hidden_states[0][5],
                    baseline.hidden_states[0][5],
                    rtol=0.02,
                    atol=0.02,
                ),
                "backend ignored the second input embedding span",
            )

            changed_spans = [rows[1:3] + perturbation, later_row]
            changed_multi = await generate(changed_spans, [1, 5])
            changed_multi_two = await generate(changed_spans, [1, 5], sequences=2)
            # all_hidden_states can be shared from one prefill across outputs.
            # Per-output logits test each returned sequence independently; a
            # sequence that loses custom embeddings must not match this reference.
            for index, sequence_logits in enumerate(changed_multi_two.logits):
                torch.testing.assert_close(
                    sequence_logits, changed_multi.logits[0], rtol=0.02, atol=0.02
                )
                self.assertFalse(
                    torch.allclose(
                        sequence_logits,
                        baseline_two.logits[index],
                        rtol=0.02,
                        atol=0.02,
                    ),
                    f"return sequence {index} ignored the perturbed embedding spans",
                )
            self.assertFalse(
                torch.allclose(
                    changed_multi.hidden_states[0][1],
                    baseline.hidden_states[0][1],
                    rtol=0.02,
                    atol=0.02,
                ),
                "backend ignored the first perturbed multi-span embedding",
            )

            # Width is unknown to Python serialization and must be rejected by
            # the real backend's model-aware admission, before model forward.
            with self.assertRaises(FtRuntimeException) as caught:
                await generate([torch.zeros(1, rows.shape[1] + 1)], [2])
            self.assertEqual(
                caught.exception.exception_type, ExceptionType.INVALID_PARAMS
            )
            self.assertIn("hidden size", caught.exception.message)
            assert_equivalent(await generate([rows[2:4].clone()], [2]), baseline)
            # Custom embeddings must not leak into the next token-only request.
            assert_equivalent(await generate(), baseline)
            assert_equivalent(await generate([rows[2:4].clone()], [2]), baseline)

            # Cross the prefill/decode boundary and compare every generated token.
            decode_baseline = await generate(new_tokens=3)
            assert_equivalent(
                await generate([rows[2:4].clone()], [2], new_tokens=3),
                decode_baseline,
            )
            if self.expect_prefill_cuda_graph:
                # Switch buckets, fall back above the largest bucket, then replay.
                for repeats in (3, 10, 1):
                    reference = await generate(prompt_repeats=repeats)
                    assert_equivalent(
                        await generate(
                            [rows[2:4].clone()], [2], prompt_repeats=repeats
                        ),
                        reference,
                    )
            self.assertEqual(
                tripwire.calls, 1, "embedding requests reached BATCH Master"
            )
        finally:
            await visitor.close()
            await client.close()
            await master.stop(0)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--prefill-cuda-graph", action="store_true")
    parser.add_argument("--reuse-cache", action="store_true")
    options, remaining = parser.parse_known_args()
    InputEmbeddingRpcTest.expect_prefill_cuda_graph = options.prefill_cuda_graph
    InputEmbeddingRpcTest.test_reuse_cache = options.reuse_cache
    unittest.main(argv=[sys.argv[0]] + remaining)
