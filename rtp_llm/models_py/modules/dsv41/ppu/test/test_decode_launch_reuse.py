"""Launch reuse preserves live bindings, idle bytes and replay counter totals."""

import json
import os
import unittest
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import patch

import torch

from rtp_llm.models_py.modules.dsv41.ppu import mega_attention as mega
from rtp_llm.models_py.modules.dsv41.ppu.graph_runtime import SharedGraphRuntime


class DecodeLaunchReuseContracts(unittest.TestCase):
    def make_signature_runtime(self):
        model = NS(v4=torch.nn.Module())
        model.v4.embed = torch.nn.Embedding(8, 4)
        model.v4.norm = torch.nn.LayerNorm(4)
        model.v4.projection = torch.nn.Linear(4, 4)
        weight = model.v4.projection.weight
        from rtp_llm.utils.model_weight import W

        model.weight = NS(global_weights={W.lm_head: weight})
        model.shared_weight_views = NS(
            _resident={"alias": weight, "other": model.v4.embed.weight}
        )
        runtime = SharedGraphRuntime.__new__(SharedGraphRuntime)
        runtime._model = lambda: model
        runtime._fast_storage_signature = True
        runtime.ops = NS(
            moe_engine=NS(
                _weights={0: (weight, weight)}, _processed={0: (weight, None)}
            )
        )
        return runtime, model

    def test_signature_preserves_occurrences_and_live_rebindings(self):
        runtime, model = self.make_signature_runtime()
        for mutation in ("initial", "weight", "buffer", "child", "resident"):
            if mutation == "weight":
                model.v4.projection.weight = torch.nn.Parameter(torch.ones(4, 4))
            if mutation == "buffer":
                model.v4.register_buffer("base", torch.ones(3))
            if mutation == "child":
                model.v4.other = torch.nn.Linear(4, 3)
            if mutation == "resident":
                model.shared_weight_views._resident["new"] = torch.zeros(2)
            with patch.dict(os.environ, {mega.LAUNCH_REUSE_ENV: "0"}):
                baseline = runtime._storage_signature()
            with patch.dict(os.environ, {mega.LAUNCH_REUSE_ENV: "1"}):
                actual = runtime._storage_signature()
            self.assertEqual(actual, baseline)
            if mutation == "initial":
                initial = actual
            else:
                self.assertNotEqual(actual, initial)
        # Retain duplicate occurrences; the check is not reduced to a set.
        weight = model.shared_weight_views._resident["alias"]
        key = (weight.data_ptr(), tuple(weight.shape), weight.dtype)
        self.assertGreaterEqual(actual.count(key), 4)

    def test_signature_rechecks_tensor_subclasses_and_patched_methods(self):
        runtime, model = self.make_signature_runtime()

        class ObservedTensor(torch.Tensor):
            calls = 0

            def data_ptr(self):
                type(self).calls += 1
                return super().data_ptr()

        value = torch.zeros(3).as_subclass(ObservedTensor)
        runtime.ops.moe_engine._weights = {0: (value, value, value)}
        with patch.dict(os.environ, {mega.LAUNCH_REUSE_ENV: "1"}):
            runtime._storage_signature()
        self.assertEqual(ObservedTensor.calls, 3)
        original = torch.Tensor.data_ptr
        calls = []

        def observed(tensor):
            calls.append(id(tensor))
            return original(tensor)

        with patch.object(torch.Tensor, "data_ptr", observed), patch.dict(
            os.environ, {mega.LAUNCH_REUSE_ENV: "1"}
        ):
            runtime._storage_signature()
        self.assertGreater(
            calls.count(id(model.shared_weight_views._resident["alias"])), 1
        )

    def test_verify_counter_selection_keeps_eager_and_other_modes_unchanged(self):
        runtime = SharedGraphRuntime.__new__(SharedGraphRuntime)
        native = NS(current=None, pending_verify=None, pending_idle_verify=None)
        layers = [NS(attn_hc=NS(), ffn_hc=NS(), engram=None) for _ in range(40)]
        model = NS(shared_runtime=native, kv_cache=object(), v4=NS(layers=layers))
        runtime._model = lambda: model
        runtime.geometry = NS(engram_layer_ids=())
        runtime._bulk_counters = False
        runtime._bulk_verify_counters = True
        runtime.grouped = None
        runtime.verify_moe_overlap = None
        runtime.engram = object()
        runtime.freqs = ()
        runtime.layer_calls = runtime.moe_calls = runtime.engram_calls = torch.zeros(
            40, dtype=torch.int64
        )
        runtime.routed_rows_device = torch.zeros(40, dtype=torch.int64)
        metadata = NS(
            batch_size=8, q_len_per_req=4, valid_mask=torch.ones(32, dtype=torch.bool)
        )
        impl = NS(
            runtime=runtime,
            cache=NS(native=model.kv_cache),
            mode="dense",
            packed_attention=object(),
        )
        for capture, mode, batch, width, expected in (
            (True, "dense", 8, 4, True),
            (False, "dense", 8, 4, False),
            (True, "packed", 8, 4, False),
            (True, "dense", 1, 4, False),
            (True, "dense", 8, 3, False),
        ):
            impl.mode = mode
            metadata.batch_size = batch
            metadata.q_len_per_req = width
            with patch.object(
                torch.cuda, "is_current_stream_capturing", return_value=capture
            ):
                with runtime.bind(impl, metadata) as state:
                    self.assertEqual(state.bulk_verify_counters, expected)
                    self.assertFalse(
                        state.bulk_counters,
                        "Engram mask/math policy must remain unchanged",
                    )
            self.assertIsNone(native.current)


@unittest.skipUnless(
    os.environ.get("RTP_V41_TEST_GRAPH_DEVICE") == "1" and torch.cuda.is_available(),
    "requires explicit PPU/CUDA device qualification",
)
class DecodeLaunchReuseDevice(unittest.TestCase):
    @torch.inference_mode()
    def test_tail_copy_transaction_graph_replay_exception_and_fallback(self):
        from rtp_llm.models_py.modules.dsv41.ppu.graph_packed_attention import (
            GraphPackedAttention,
            _can_fuse_tail_copies,
        )

        for rows in (1, 8, 24, 32):
            for widths in ((528,), (528, 288)):
                adapter = GraphPackedAttention.__new__(GraphPackedAttention)
                keys = [
                    (i, "swa_kv" if width == 528 else "global_kv_1")
                    for i, width in enumerate(widths)
                ]
                adapter.views = {
                    key: torch.randint(
                        256, (2, 128, 1, width), device="cuda", dtype=torch.uint8
                    )
                    for key, width in zip(keys, widths)
                }
                payloads = {
                    key: torch.randint(
                        256, (rows, width), device="cuda", dtype=torch.uint8
                    )
                    for key, width in zip(keys, widths)
                }
                adapter.journal = NS(rows=rows, payloads=payloads)
                pages = [adapter.views[key][0, :rows, 0] for key in keys]
                observed = [torch.empty_like(page) for page in pages]
                with patch.dict(os.environ, {mega.LAUNCH_REUSE_ENV: "1"}):
                    self.assertTrue(
                        _can_fuse_tail_copies(pages, list(payloads.values()))
                    )
                    before = {key: pool.clone() for key, pool in adapter.views.items()}
                    # Exercise the actual transaction wrapper, including its
                    # native-consumer failure path, before graph capture.
                    with self.assertRaisesRegex(RuntimeError, "consumer failed"):
                        with adapter._stage_verify_tail(keys):
                            for page, payload in zip(pages, payloads.values()):
                                self.assertTrue(torch.equal(page, payload))
                            raise RuntimeError("consumer failed")
                    for key, pool in adapter.views.items():
                        self.assertTrue(torch.equal(pool, before[key]))
                    graph = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(graph):
                        with adapter._stage_verify_tail(keys):
                            for out, page in zip(observed, pages):
                                out.copy_(page)
                    for _ in range(3):
                        for pool in adapter.views.values():
                            pool.random_(0, 256)
                        for payload in payloads.values():
                            payload.random_(0, 256)
                        before = {
                            key: pool.clone() for key, pool in adapter.views.items()
                        }
                        graph.replay()
                        torch.cuda.synchronize()
                        for out, payload in zip(observed, payloads.values()):
                            self.assertTrue(torch.equal(out, payload))
                        for key, pool in adapter.views.items():
                            self.assertTrue(torch.equal(pool, before[key]))
                    aliases = [*payloads.values()]
                    aliases[0] = pages[0]
                    self.assertFalse(_can_fuse_tail_copies(pages, aliases))
                    strided = [
                        torch.empty(
                            (rows, width * 2), device="cuda", dtype=torch.uint8
                        )[:, ::2]
                        for width in widths
                    ]
                    self.assertFalse(_can_fuse_tail_copies(pages, strided))
                    if rows == 32 and len(widths) == 2:
                        with torch.profiler.profile(
                            activities=[
                                torch.profiler.ProfilerActivity.CPU,
                                torch.profiler.ProfilerActivity.CUDA,
                            ]
                        ) as profile:
                            with adapter._stage_verify_tail(keys):
                                pass
                            torch.cuda.synchronize()
                        path = (
                            Path(os.environ["TEST_UNDECLARED_OUTPUTS_DIR"])
                            / "tail-copies.trace.json"
                        )
                        profile.export_chrome_trace(str(path))
                        kernels = [
                            e["name"]
                            for e in json.loads(path.read_text())["traceEvents"]
                            if e.get("cat") == "kernel"
                        ]
                        self.assertEqual(len(kernels), 2, kernels)
                        self.assertTrue(
                            any("backup_and_stage" in name for name in kernels)
                        )
                        self.assertTrue(any("restore_tail" in name for name in kernels))

    @torch.inference_mode()
    def test_owned_mask_graph_preserves_live_nan_signed_zero_and_idle_zero(self):
        for rows in (1, 8, 24, 32):
            output = torch.randn((rows, 8, 1024), device="cuda", dtype=torch.bfloat16)
            valid = torch.ones(rows, device="cuda", dtype=torch.bool)
            mega._mask_owned_output(output, valid)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                mega._mask_owned_output(output, valid)
            for pattern in ("live", "mixed", "idle"):
                source = torch.randn_like(output)
                source[:, 0, 0] = float("nan")
                source[:, 0, 1] = float("inf")
                source[:, 0, 2] = -0.0
                output.copy_(source)
                valid.fill_(pattern != "idle")
                if pattern == "mixed":
                    valid[::2] = False
                expected = torch.where(valid[:, None, None], source, 0)
                graph.replay()
                torch.cuda.synchronize()
                self.assertTrue(
                    torch.equal(output.view(torch.uint8), expected.view(torch.uint8))
                )
            if rows == 32:
                with torch.profiler.profile(
                    activities=[
                        torch.profiler.ProfilerActivity.CPU,
                        torch.profiler.ProfilerActivity.CUDA,
                    ]
                ) as profile:
                    mega._mask_owned_output(output, valid)
                    torch.cuda.synchronize()
                path = (
                    Path(os.environ["TEST_UNDECLARED_OUTPUTS_DIR"])
                    / "owned-mask.trace.json"
                )
                profile.export_chrome_trace(str(path))
                kernels = [
                    e["name"]
                    for e in json.loads(path.read_text())["traceEvents"]
                    if e.get("cat") == "kernel"
                ]
                self.assertEqual(len(kernels), 1, kernels)

    @torch.inference_mode()
    def test_batched_counter_graph_matches_per_layer_totals(self):
        from rtp_llm.models_py.modules.dsv41.ppu.graph_metadata import (
            increment_decode_counters,
        )

        layers = torch.arange(40, device="cuda", dtype=torch.int64) + 2**32
        moe = layers.clone() + 100
        engram = layers.clone() + 200
        mask = torch.zeros(40, device="cuda", dtype=torch.bool)
        mask[[0, 3, 7]] = True
        before = layers.clone(), moe.clone(), engram.clone()
        increment_decode_counters(layers, moe, engram, mask)
        layers.copy_(before[0])
        moe.copy_(before[1])
        engram.copy_(before[2])
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            increment_decode_counters(layers, moe, engram, mask)
        for _ in range(7):
            graph.replay()
        torch.cuda.synchronize()
        torch.testing.assert_close(layers, before[0] + 7, rtol=0, atol=0)
        torch.testing.assert_close(moe, before[1] + 7, rtol=0, atol=0)
        torch.testing.assert_close(engram, before[2] + 7 * mask.long(), rtol=0, atol=0)

    @torch.inference_mode()
    def test_authenticated_native_decode_position_reuse_and_masks(self):
        from rtp_llm.models_py.modules.dsv41.ppu.kv_codec import encode_kv
        from rtp_llm.models_py.modules.dsv41.ppu.linear import PPUPTPCLinear

        device = torch.device("cuda", torch.cuda.current_device())
        geometry = NS(
            n_heads=64,
            head_dim=512,
            rope_head_dim=64,
            o_groups=8,
            o_lora_rank=128,
            q_lora_rank=128,
            sm_scale=lambda: 512**-0.5,
        )
        owner = NS(
            geometry=geometry,
            wq_b=PPUPTPCLinear(
                torch.full(
                    (32768, 128), 0.01, device=device, dtype=torch.float8_e4m3fn
                ),
                torch.ones((32768, 1), device=device),
            ),
            wo_a=PPUPTPCLinear(
                torch.full(
                    (1024, 4096), 0.01, device=device, dtype=torch.float8_e4m3fn
                ),
                torch.ones((1024, 1), device=device),
            ),
            attn_sink=torch.zeros(64, device=device),
        )
        freqs = torch.polar(
            torch.ones((256, 32), device=device),
            torch.randn((256, 32), device=device) * 0.1,
        )
        with patch.dict(os.environ, {mega.ENV: "1"}):
            consumer = mega.prepare(owner, freqs)
        self.assertIsNotNone(consumer)
        self.assertTrue(
            mega.readonly_decode_positions(consumer),
            "must qualify the actual pinned const-input wrapper",
        )
        rows = 8
        q = torch.randn((rows, 64, 512), device=device, dtype=torch.bfloat16) * 0.1
        pool = torch.zeros((2, 128, 1, 528), device=device, dtype=torch.uint8)
        pool[1, :, 0] = encode_kv(
            torch.randn((128, 512), device=device, dtype=torch.bfloat16) * 0.1,
            pool_tag="swa_kv",
        )
        valid = torch.ones(rows, device=device, dtype=torch.bool)
        valid[::2] = False
        positions = torch.arange(rows, device=device, dtype=torch.int64) + 64
        positions[~valid] = -1
        indices = (
            (torch.arange(64, device=device, dtype=torch.int32) + 128)
            .expand(rows, -1)
            .contiguous()[:, None]
        )
        lengths = torch.full((rows,), 64, device=device, dtype=torch.int32)
        lengths[~valid] = 0
        plan = NS(
            valid_mask=valid,
            swa_cache=pool,
            swa_indices=indices,
            swa_lengths=lengths,
            global_cache=None,
            global_indices=None,
            global_lengths=None,
        )
        results = {}
        for enabled in ("0", "1"):
            with patch.dict(
                os.environ, {mega.ENV: "1", mega.LAUNCH_REUSE_ENV: enabled}
            ):
                # Compile and initialize the real provider outside capture.
                for _ in range(2):
                    consumer.decode(q, plan, positions, shared={})
                torch.cuda.synchronize()
                shared = {}
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    first = consumer.decode(q, plan, positions, shared=shared)
                    second = consumer.decode(q, plan, positions, shared=shared)
                graph.replay()
                torch.cuda.synchronize()
                self.assertTrue(
                    torch.equal(first.view(torch.uint8), second.view(torch.uint8))
                )
                self.assertEqual(int(second[~valid].count_nonzero()), 0)
                results[enabled] = second.clone()
                self.assertEqual(len(shared), int(enabled))
                if enabled == "1":
                    saved = positions.clone()
                    consumer.decode(q, plan, positions, shared=shared)
                    self.assertTrue(
                        torch.equal(positions, saved),
                        "native consumer must keep shared positions const",
                    )
        self.assertTrue(
            torch.equal(results["0"].view(torch.uint8), results["1"].view(torch.uint8))
        )
