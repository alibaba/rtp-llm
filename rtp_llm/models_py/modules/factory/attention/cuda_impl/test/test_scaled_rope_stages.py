"""Real prefill and native decode must produce the same Q and cached K.

The independent reference never calls either implementation. In particular,
comparing decode only against eager decode would preserve the same kernel bug.
"""

import os
import unittest
from unittest.mock import patch

import torch

from rtp_llm.device.device_type import DeviceType, get_device_type
from rtp_llm.models_py.kernels.cuda.test.scaled_rope_reference import reference_rope
from rtp_llm.models_py.modules.factory.attention.cuda_impl.flashinfer_rotary_emb import (
    MhaRotaryEmbeddingOp,
)
from rtp_llm.ops import AttentionConfigs, KvCacheDataType, RopeStyle
from rtp_llm.ops.compute_ops import (
    FlashInferMlaAttnParams,
    FusedRopeKVCacheDecodeOp,
    FusedRopeKVCachePrefillOpQOut,
    LayerKVCache,
    PyAttentionInputs,
    get_typemeta,
)


class ScaledRopeStagesTest(unittest.TestCase):
    HEADS, KV_HEADS, DIM, PAGE = 16, 8, 128, 64
    SENTINEL = 19.0

    @classmethod
    def setUpClass(cls):
        if not torch.cuda.is_available() or get_device_type() not in (
            DeviceType.Cuda,
            DeviceType.Ppu,
        ):
            raise RuntimeError("this target requires its scheduled CUDA/PPU GPU")

    def setUp(self):
        self.env = patch.dict(
            os.environ,
            {"RTP_LLM_STREAM_ASYNC": "0", "RTP_LLM_MTP_ASYNC_PREPARE": "0"},
        )
        self.env.start()
        self.addCleanup(self.env.stop)
        torch.manual_seed(1685)

    def config(self, llama3, dtype, scale=8.0):
        c = AttentionConfigs()
        c.head_num, c.kv_head_num, c.size_per_head = self.HEADS, self.KV_HEADS, self.DIM
        c.tokens_per_block = c.kernel_tokens_per_block = self.PAGE
        c.dtype, c.kv_cache_dtype = dtype, KvCacheDataType.BASE
        c.need_rope_kv_cache, c.max_seq_len = True, 256
        r = c.rope_config
        r.style = RopeStyle.Llama3 if llama3 else RopeStyle.DynamicNTK
        r.dim, r.base, r.scale, r.max_pos = self.DIM, 10000, scale, 32
        r.factor1, r.factor2, r.index_factor = 1.0, 4.0, 1
        return c

    def inputs(self, lengths, positions, table, graph, dtype):
        x = PyAttentionInputs()
        x.is_prefill, x.is_cuda_graph = False, graph
        x.dtype = get_typemeta(torch.empty(0, dtype=dtype))
        x.sequence_lengths = torch.tensor(lengths, dtype=torch.int32).pin_memory()
        x.sequence_lengths_plus_1_device = x.sequence_lengths.cuda() + 1
        if positions is not None:
            x.combo_position_ids = positions
        x.kv_cache_kernel_block_id = table.cpu().pin_memory()
        x.kv_cache_kernel_block_id_device = table
        return x

    def cache(self, batch, dtype):
        c = LayerKVCache()
        c.kv_cache_base = torch.full(
            (batch * 4 + 1, 2, self.KV_HEADS, self.PAGE, self.DIM),
            self.SENTINEL,
            dtype=dtype,
            device="cuda",
        )
        return c

    def raw(self, batch, dtype):
        return torch.randn(
            batch,
            self.HEADS + 2 * self.KV_HEADS,
            self.DIM,
            dtype=dtype,
            device="cuda",
        )

    def check(self, output, raw, cache, lengths, positions, table, config):
        batch = len(lengths)
        history = torch.tensor(lengths, dtype=torch.int32, device="cuda")
        counts = torch.ones_like(history)
        empty = torch.empty(0, dtype=torch.int32, device="cuda")
        params = FlashInferMlaAttnParams()
        params.fill_params_mha_device(history, empty, counts, table, self.PAGE, False)
        # The device planner reserves token capacity from the page table but
        # writes only the live token prefix (one Q token per request here).
        # Keep its storage and shape intact: set_params shares these buffers.
        active_positions = params.positions_d[: raw.shape[0]]
        torch.testing.assert_close(active_positions, history)
        torch.testing.assert_close(
            params.batch_indice_d[: raw.shape[0]],
            torch.arange(batch, device=history.device, dtype=history.dtype),
        )
        torch.testing.assert_close(params.kvlen_d, history + counts)
        # Override just the logical RoPE positions, not the physical KV slots.
        active_positions.copy_(positions)
        prefill = MhaRotaryEmbeddingOp(config)
        prefill.set_params(params)
        pq, pk, pv = prefill.forward(raw.clone().reshape(batch, -1))
        # Exercise the other production prefill entry, too: CUDA uses the
        # external wheel's layout/KV writer, PPU its native context_rope.
        # The PPU native CP path intentionally suppresses KV writes when
        # explicit positions are supplied. Test its ordinary cache-writing
        # path here; explicit positions are covered by FlashInfer and decode.
        native_inputs = self.inputs(lengths, None, table, False, raw.dtype)
        native_inputs.is_prefill = True
        native_inputs.sequence_lengths = torch.empty(0, dtype=torch.int32)
        native_inputs.input_lengths = torch.ones(batch, dtype=torch.int32).pin_memory()
        native_inputs.prefix_lengths = torch.tensor(
            lengths, dtype=torch.int32
        ).pin_memory()
        native_inputs.cu_seqlens_device = torch.arange(
            batch + 1, dtype=torch.int32, device="cuda"
        )
        native_inputs.cu_seqlens = native_inputs.cu_seqlens_device.cpu()
        native_inputs.cu_kv_seqlens_device = torch.cat(
            (history.new_zeros(1), (history + 1).cumsum(0).int())
        )
        native_inputs.padding_offset = torch.zeros(
            batch, dtype=torch.int32, device="cuda"
        )
        native_inputs.context_total_kv_length = sum(lengths) + batch
        native_inputs.total_tokens = batch
        native_op = FusedRopeKVCachePrefillOpQOut(config)
        native_cache = self.cache(batch, raw.dtype)
        native_q = native_op.forward(
            raw.clone().reshape(batch, -1),
            native_cache,
            native_op.prepare(native_inputs),
        )
        q, k, v = raw.split((self.HEADS, self.KV_HEADS, self.KV_HEADS), dim=1)
        r = config.rope_config

        def ref(t):
            return reference_rope(
                t,
                positions,
                history + 1,
                rotary_dim=r.dim,
                base=r.base,
                scale=r.scale,
                original_max_position=r.max_pos,
                llama3=r.style == RopeStyle.Llama3,
                low_freq_factor=r.factor1,
                high_freq_factor=r.factor2,
            )

        expected_q, expected_k = ref(q), ref(k)
        tol = 0.025 if raw.dtype == torch.bfloat16 else 0.006
        for actual in (pq, output.reshape_as(q)):
            torch.testing.assert_close(actual, expected_q, atol=tol, rtol=tol)
        torch.testing.assert_close(pk, expected_k, atol=tol, rtol=tol)
        torch.testing.assert_close(pv, v, atol=0, rtol=0)
        pages = table[
            torch.arange(batch, device="cuda"), history.long() // self.PAGE
        ].long()
        slots = history.long() % self.PAGE
        written = cache.kv_cache_base.permute(0, 3, 1, 2, 4)[pages, slots]
        native_written = native_cache.kv_cache_base.permute(0, 3, 1, 2, 4)[pages, slots]
        native_ref = dict(
            rotary_dim=r.dim,
            base=r.base,
            scale=r.scale,
            original_max_position=r.max_pos,
            llama3=r.style == RopeStyle.Llama3,
            low_freq_factor=r.factor1,
            high_freq_factor=r.factor2,
        )
        torch.testing.assert_close(
            native_q.reshape_as(q),
            reference_rope(q, history, history + 1, **native_ref),
            atol=tol,
            rtol=tol,
        )
        torch.testing.assert_close(
            native_written[:, 0],
            reference_rope(k, history, history + 1, **native_ref),
            atol=tol,
            rtol=tol,
        )
        torch.testing.assert_close(native_written[:, 1], v, atol=0, rtol=0)
        torch.testing.assert_close(written[:, 0], expected_k, atol=tol, rtol=tol)
        torch.testing.assert_close(written[:, 1], v, atol=0, rtol=0)
        untouched = torch.ones(
            (batch * 4 + 1, self.PAGE), device="cuda", dtype=torch.bool
        )
        untouched[pages, slots] = False
        self.assertTrue(
            torch.all(
                cache.kv_cache_base.permute(0, 3, 1, 2, 4)[untouched] == self.SENTINEL
            )
        )

    def test_prefill_decode_and_cache_use_one_scaling_contract(self):
        # <=16 and >16 select the two PPU non-cache native kernels.
        for batch in (4, 32):
            for llama3 in (False, True):
                for dtype in (torch.float16, torch.bfloat16):
                    for scale in (1.0, 8.0):
                        with self.subTest(
                            batch=batch, llama3=llama3, dtype=dtype, scale=scale
                        ):
                            lengths = [0, 31, 32, 127] * (batch // 4)
                            positions = torch.tensor(
                                lengths, device="cuda", dtype=torch.int32
                            )
                            positions[1::4] = 0
                            positions[3::4] += 13
                            table = torch.arange(
                                1, batch * 4 + 1, device="cuda", dtype=torch.int32
                            ).reshape(batch, 4)
                            config = self.config(llama3, dtype, scale)
                            inputs = self.inputs(lengths, positions, table, False, dtype)
                            op, cache, raw = (
                                FusedRopeKVCacheDecodeOp(config),
                                self.cache(batch, dtype),
                                self.raw(batch, dtype),
                            )
                            actual = op.forward(
                                raw.reshape(batch, -1), cache, op.prepare(inputs)
                            )
                            self.check(
                                actual, raw, cache, lengths, positions, table, config
                            )

    def test_graph_replay_crosses_ntk_boundary_and_cache_page(self):
        for batch in (4, 32):
            for llama3 in (False, True):
                with self.subTest(batch=batch, llama3=llama3):
                    config = self.config(llama3, torch.bfloat16)
                    lengths = [7] * batch
                    positions = torch.tensor(lengths, device="cuda", dtype=torch.int32)
                    table = torch.arange(
                        1, batch * 4 + 1, device="cuda", dtype=torch.int32
                    ).reshape(batch, 4)
                    inputs = self.inputs(lengths, positions, table, True, torch.bfloat16)
                    op = FusedRopeKVCacheDecodeOp(config)
                    params = op.prepare(inputs)
                    cache, raw = self.cache(batch, torch.bfloat16), self.raw(
                        batch, torch.bfloat16
                    )
                    sequence_ptr, offset_ptr = (
                        params.sequence_lengths.data_ptr(),
                        params.kv_cache_offset.data_ptr(),
                    )
                    stream = torch.cuda.Stream()
                    stream.wait_stream(torch.cuda.current_stream())
                    with torch.cuda.stream(stream):
                        for _ in range(3):
                            op.forward(raw.reshape(batch, -1), cache, params)
                    torch.cuda.current_stream().wait_stream(stream)
                    graph = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(graph):
                        output = op.forward(raw.reshape(batch, -1), cache, params)
                    for step in (31, 32, 63, 64, 127, 7):
                        lengths = [step if i % 2 else 3 for i in range(batch)]
                        positions.copy_(
                            torch.tensor(lengths, device="cuda", dtype=torch.int32)
                        )
                        positions[0] = 0
                        # Same capture pointers, different page IDs and fresh
                        # caller-owned host lengths; no per-replay recapture.
                        table.copy_(table.flip(0))
                        live = self.inputs(lengths, positions, table, True, torch.bfloat16)
                        refreshed = op.prepare(live, forbid_reallocation=True)
                        self.assertEqual(
                            sequence_ptr, refreshed.sequence_lengths.data_ptr()
                        )
                        # CUDA converts offsets afresh; copy into capture storage.
                        params.kv_cache_offset.copy_(refreshed.kv_cache_offset)
                        self.assertEqual(offset_ptr, params.kv_cache_offset.data_ptr())
                        raw.copy_(self.raw(batch, torch.bfloat16))
                        cache.kv_cache_base.fill_(self.SENTINEL)
                        graph.replay()
                        self.check(
                            output, raw, cache, lengths, positions, table, config
                        )


if __name__ == "__main__":
    unittest.main()
