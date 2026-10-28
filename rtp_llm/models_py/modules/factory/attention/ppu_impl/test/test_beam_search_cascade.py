"""PPU factory -> RoPE/KV write -> Cascade graph with online beam dimensions."""

import math
import os
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch
from flashinfer.decode import BatchDecodeWithPagedKVCacheWrapper

from rtp_llm.device.device_type import DeviceType, get_device_type
from rtp_llm.models_py.modules.factory.attention.attn_factory import AttnImplFactory
from rtp_llm.models_py.modules.factory.attention.cuda_impl.beam_search_cascade import (
    PyFlashinferCascadeDecodeImpl,
)
from rtp_llm.ops import AttentionConfigs, KvCacheDataType, ParallelismConfig, RopeStyle
from rtp_llm.ops.compute_ops import LayerKVCache, PyAttentionInputs, get_typemeta


@unittest.skipUnless(get_device_type() == DeviceType.Ppu, "requires PPU")
class PpuBeamSearchCascadeTest(unittest.TestCase):
    BATCH = 512
    ACTIVE = 500
    HEADS = 16
    KV_HEADS = 8
    DIM = 128
    PAGE = 64
    MAX_SEQ = 1536
    MAX_PAGES = MAX_SEQ // PAGE
    MAX_REQUESTS = 3
    # Five real groups also exercise the safe no-sharing fallback when the
    # configured three-request capacity plus one padding slot is exceeded.
    PREFIX_BANKS = 5
    PRIVATE_BASE = 1 + PREFIX_BANKS * MAX_PAGES
    NUM_PAGES = PRIVATE_BASE + 3 * BATCH + 1
    SENTINEL = 19.0
    CASES = (
        ([500], [265], "shared"),
        ([250, 250], [319, 319], "shared"),
        ([250, 250], [320, 320], "shared"),
        ([128, 128, 244], [265, 511, 1354], "shared"),
        ([250, 250], [1279, 1279], "partial"),
        ([250, 250], [1280, 1280], "shared"),
        ([500], [1354], "shared"),
        ([500], [265], "distinct"),
        ([100] * 5, [265] * 5, "shared"),
        ([256], [265], "shared"),
        ([250, 250], [319, 319], "reused_first_page"),
        ([128, 128, 244], [265, 511, 1354], "reused_first_page"),
        ([500], [265], "shared"),
    )

    def setUp(self):
        self.env = patch.dict(
            os.environ,
            {"RTP_LLM_STREAM_ASYNC": "0", "RTP_LLM_MTP_ASYNC_PREPARE": "0"},
        )
        self.env.start()
        self.addCleanup(self.env.stop)
        torch.manual_seed(20261028)
        self.config = AttentionConfigs()
        self.config.head_num = self.HEADS
        self.config.kv_head_num = self.KV_HEADS
        self.config.size_per_head = self.DIM
        self.config.tokens_per_block = self.PAGE
        self.config.kernel_tokens_per_block = self.PAGE
        self.config.dtype = torch.bfloat16
        self.config.kv_cache_dtype = KvCacheDataType.BASE
        self.config.need_rope_kv_cache = True
        self.config.max_seq_len = self.MAX_SEQ
        self.config.rope_config.style = RopeStyle.Base
        self.config.rope_config.dim = self.DIM
        self.config.rope_config.base = 1000000
        self.config.rope_config.max_pos = self.MAX_SEQ
        self.parallel = ParallelismConfig()
        self.parallel.tp_size = 1
        self.model = SimpleNamespace(
            getAttentionConfigs=lambda _: self.config,
            max_seq_len=self.MAX_SEQ,
            headwise_config=None,
            quant_config=None,
        )

    def _inputs(self, widths, histories, layout="shared"):
        inputs = PyAttentionInputs()
        inputs.is_prefill = False
        inputs.is_cuda_graph = True
        inputs.input_lengths = torch.ones(self.BATCH, dtype=torch.int32).pin_memory()
        inputs.prefix_lengths = torch.empty(0, dtype=torch.int32)
        inputs.sequence_lengths = torch.zeros(
            self.BATCH, dtype=torch.int32
        ).pin_memory()
        table = torch.zeros(self.BATCH, self.MAX_PAGES, dtype=torch.int32).pin_memory()
        start = 0
        for group, (width, history) in enumerate(zip(widths, histories)):
            pages = history // self.PAGE
            end = start + width
            table[start:end, :pages] = (
                torch.arange(pages, dtype=torch.int32) + 1 + group * self.MAX_PAGES
            )
            rows = torch.arange(start, end, dtype=torch.int32)
            table[start:end, pages] = self.PRIVATE_BASE + 3 * rows
            table[start:end, pages + 1] = self.PRIVATE_BASE + 3 * rows + 1
            if layout == "partial":
                table[start:end, 2] = self.PRIVATE_BASE + 3 * rows + 2
            elif layout == "distinct":
                table[start:end, 0] = self.PRIVATE_BASE + 3 * rows + 2
            elif layout == "reused_first_page":
                table[start:end, 0] = 1
            inputs.sequence_lengths[start:end] = history
            start = end
        inputs.input_lengths[start:].zero_()
        inputs.kv_cache_kernel_block_id = table
        inputs.kv_cache_kernel_block_id_device = table.cuda()
        inputs.cu_seqlens_device = torch.arange(
            self.BATCH + 1, dtype=torch.int32, device="cuda"
        )
        inputs.sequence_lengths_plus_1_device = (inputs.sequence_lengths + 1).cuda()
        inputs.dtype = get_typemeta(torch.zeros(1, dtype=torch.bfloat16))
        return inputs

    def _factory(self, inputs):
        impl = AttnImplFactory.get_fmha_impl(
            self.model,
            self.parallel,
            None,
            inputs,
            is_cuda_graph=True,
            decode_cascade_max_requests=self.MAX_REQUESTS,
        )
        self.assertIsInstance(impl, PyFlashinferCascadeDecodeImpl)
        return impl

    def _raw(self, active):
        raw = torch.randn(
            self.BATCH,
            (self.HEADS + 2 * self.KV_HEADS) * self.DIM,
            dtype=torch.bfloat16,
            device="cuda",
        )
        raw[active:].zero_()
        return raw

    def _rotate(self, value, positions):
        value = value.float().cpu()
        inv = 1.0 / (
            self.config.rope_config.base
            ** (torch.arange(0, self.DIM, 2).float() / self.DIM)
        )
        angles = positions.cpu().float()[:, None] * inv
        cosine = torch.cat((angles.cos(), angles.cos()), -1)[:, None, :]
        sine = torch.cat((angles.sin(), angles.sin()), -1)[:, None, :]
        first, second = value.chunk(2, -1)
        return (value * cosine + torch.cat((-second, first), -1) * sine).to(
            torch.bfloat16
        )

    def _rotated_qkv(self, raw, inputs):
        packed = raw.cpu().reshape(self.BATCH, self.HEADS + 2 * self.KV_HEADS, self.DIM)
        q, k, v = packed.split((self.HEADS, self.KV_HEADS, self.KV_HEADS), dim=1)
        return (
            self._rotate(q, inputs.sequence_lengths),
            self._rotate(k, inputs.sequence_lengths),
            v,
        )

    def _assert_current_writes(self, cache, inputs, expected_k, expected_v):
        positions = inputs.sequence_lengths.long()
        pages = inputs.kv_cache_kernel_block_id[
            torch.arange(self.BATCH), positions // self.PAGE
        ].long()
        actual = cache.kv_cache_base.permute(0, 3, 1, 2, 4)[
            pages.cuda(), (positions % self.PAGE).cuda()
        ].cpu()
        torch.testing.assert_close(actual[:, 0], expected_k, atol=0.02, rtol=0.02)
        torch.testing.assert_close(actual[:, 1], expected_v, atol=0, rtol=0)
        self.assertTrue(bool((cache.kv_cache_base[-1] == self.SENTINEL).all()))
        return actual

    def _assert_full_kv_read(self, output, q, cache, inputs, widths, normal):
        lengths = inputs.sequence_lengths
        counts = (lengths + self.PAGE) // self.PAGE
        ptr = torch.cat((torch.zeros(1, dtype=torch.int32), counts.cumsum(0).int()))
        table = inputs.kv_cache_kernel_block_id
        indices = table[torch.arange(self.MAX_PAGES)[None, :] < counts[:, None]]
        normal.plan(
            ptr,
            indices.cuda(),
            lengths % self.PAGE + 1,
            self.HEADS,
            self.KV_HEADS,
            self.DIM,
            self.PAGE,
            q_data_type=torch.bfloat16,
            kv_data_type=torch.bfloat16,
        )
        reference = normal.run(q.cuda(), cache.kv_cache_base)
        torch.testing.assert_close(output, reference, atol=0.006, rtol=0.03)
        # Independent CPU FP32 softmax over gathered physical KV pages for the
        # first and last beam of each real request, plus one scratch query.
        rows, start = {self.BATCH - 1}, 0
        for width in widths:
            rows.update((start, start + width - 1))
            start += width
        for row in sorted(rows):
            data = cache.kv_cache_base[table[row, : int(counts[row])].long().cuda()]
            data = data.permute(1, 2, 0, 3, 4).reshape(2, self.KV_HEADS, -1, self.DIM)
            data = data[:, :, : int(lengths[row]) + 1].float().cpu()
            k = data[0].repeat_interleave(2, 0)
            v = data[1].repeat_interleave(2, 0)
            scores = torch.einsum("hd,htd->ht", q[row].float(), k) / math.sqrt(self.DIM)
            expected = torch.einsum("ht,htd->hd", scores.softmax(-1), v)
            torch.testing.assert_close(
                output[row].float().cpu(), expected, atol=0.006, rtol=0.03
            )

    @staticmethod
    def _pointers(impl):
        cascade = impl.fmha_impl.cascade
        return (
            impl.rope_params.sequence_lengths.data_ptr(),
            impl.rope_params.kv_cache_offset.data_ptr(),
            cascade.prefix._qo_indptr_buf.data_ptr(),
            cascade.prefix._paged_kv_indptr_buf.data_ptr(),
            cascade.prefix._paged_kv_indices_buf.data_ptr(),
            cascade.tail._paged_kv_indptr_buf.data_ptr(),
            cascade.tail._paged_kv_indices_buf.data_ptr(),
        )

    def _replay(self, impl, graph, source, inputs, raw, pointers):
        # The synchronous host pipeline completes the old replay before these
        # pinned staging buffers are refreshed. Do not claim stream-async support.
        torch.cuda.synchronize()
        source.copy_(raw)
        impl.prepare_cuda_graph(inputs)
        self.assertEqual(self._pointers(impl), pointers)
        torch.cuda.synchronize()
        inputs.sequence_lengths.fill_(1)
        inputs.kv_cache_kernel_block_id.zero_()
        inputs.kv_cache_kernel_block_id_device.zero_()
        graph.replay()
        torch.cuda.synchronize()

    def test_online_shapes_changed_groups_and_next_kv_consumer(self):
        capture_inputs = self._inputs([self.ACTIVE], [265])
        impl = self._factory(capture_inputs)
        self.assertTrue(impl.support_cuda_graph())
        pointers = self._pointers(impl)
        cache = LayerKVCache()
        cache.kv_cache_base = torch.randn(
            self.NUM_PAGES,
            2,
            self.KV_HEADS,
            self.PAGE,
            self.DIM,
            dtype=torch.bfloat16,
            device="cuda",
        )
        source = self._raw(self.ACTIVE)
        for _ in range(2):
            impl.forward(source, cache)
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            output = impl.forward(source, cache)
        normal = BatchDecodeWithPagedKVCacheWrapper(
            torch.zeros(128 * 1024**2, dtype=torch.uint8, device="cuda"),
            "HND",
            use_tensor_cores=False,
        )
        for widths, histories, layout in self.CASES:
            with self.subTest(widths=widths, histories=histories, layout=layout):
                active = sum(widths)
                inputs = self._inputs(widths, histories, layout)
                reference_inputs = self._inputs(widths, histories, layout)
                raw = self._raw(active)
                cache.kv_cache_base.normal_()
                cache.kv_cache_base[-1].fill_(self.SENTINEL)
                self._replay(impl, graph, source, inputs, raw, pointers)
                q, k, v = self._rotated_qkv(raw, reference_inputs)
                current = self._assert_current_writes(cache, reference_inputs, k, v)
                self._assert_full_kv_read(
                    output, q, cache, reference_inputs, widths, normal
                )
                if layout == "distinct" or len(widths) > self.MAX_REQUESTS:
                    self.assertEqual(impl.fmha_impl.cascade.shared_tokens, 0)
                else:
                    self.assertGreater(impl.fmha_impl.cascade.shared_tokens, 0)

                if layout == "reused_first_page":
                    self.assertEqual(
                        impl.fmha_impl.cascade.shared_tokens,
                        sum(
                            w * (h // self.PAGE) * self.PAGE
                            for w, h in zip(widths, histories)
                        ),
                    )
                    self.assertEqual(
                        int(impl.fmha_impl.cascade.tail_ptr[-1]), self.BATCH
                    )

                if histories in ([319, 319], [1354]):
                    # Make the next query align strongly with the KV just written.
                    # The next two steps must read that private historical token,
                    # including the 319 -> 320 page transition. New K/V are zero.
                    for step in (1, 2):
                        advanced = self._inputs(widths, histories, layout)
                        advanced.sequence_lengths[:active] += step
                        advanced.sequence_lengths_plus_1_device = (
                            advanced.sequence_lengths + 1
                        ).cuda()
                        advanced_reference = self._inputs(widths, histories, layout)
                        advanced_reference.sequence_lengths[:active] += step
                        advanced_reference.sequence_lengths_plus_1_device = (
                            advanced_reference.sequence_lengths + 1
                        ).cuda()
                        parents = torch.arange(self.BATCH)
                        if step == 2:
                            # Reorder parents within each request on the third
                            # round, including their now-complete private page.
                            start = 0
                            for width in widths:
                                end = start + width
                                parents[start:end] = torch.arange(
                                    end - 1, start - 1, -1
                                )
                                start = end
                            for metadata in (advanced, advanced_reference):
                                metadata.kv_cache_kernel_block_id.copy_(
                                    metadata.kv_cache_kernel_block_id[parents]
                                )
                                metadata.kv_cache_kernel_block_id_device = (
                                    metadata.kv_cache_kernel_block_id.cuda()
                                )
                        target_q = current[parents, 0].repeat_interleave(2, 1) * 8
                        raw_q = self._rotate(
                            target_q, -advanced_reference.sequence_lengths
                        )
                        next_raw = torch.zeros_like(raw)
                        next_raw[:, : self.HEADS * self.DIM] = raw_q.reshape(
                            self.BATCH, -1
                        ).cuda()
                        self._replay(impl, graph, source, advanced, next_raw, pointers)
                        next_q, next_k, next_v = self._rotated_qkv(
                            next_raw, advanced_reference
                        )
                        self._assert_current_writes(
                            cache, advanced_reference, next_k, next_v
                        )
                        self._assert_full_kv_read(
                            output, next_q, cache, advanced_reference, widths, normal
                        )
                        previous_v = current[parents[:active], 1].repeat_interleave(
                            2, 1
                        )
                        torch.testing.assert_close(
                            output[:active].cpu(), previous_v, atol=0.02, rtol=0.02
                        )

    def test_invalid_host_metadata_does_not_launch_a_new_plan(self):
        inputs = self._inputs([500], [265])
        impl = self._factory(inputs)
        cascade = impl.fmha_impl.cascade
        with self.assertRaisesRegex(ValueError, "requires host"):
            cascade.plan(
                inputs.kv_cache_kernel_block_id_device, inputs.sequence_lengths
            )
        inputs.sequence_lengths[0] = -1
        with self.assertRaisesRegex(ValueError, "nonnegative"):
            cascade.plan(inputs.kv_cache_kernel_block_id, inputs.sequence_lengths)
        inputs.sequence_lengths[0] = self.MAX_SEQ
        with self.assertRaisesRegex(ValueError, "captured capacity"):
            cascade.plan(inputs.kv_cache_kernel_block_id, inputs.sequence_lengths)


class PpuBeamSearchCascadePage16Test(PpuBeamSearchCascadeTest):
    PAGE = 16
    MAX_PAGES = PpuBeamSearchCascadeTest.MAX_SEQ // PAGE
    PRIVATE_BASE = 1 + PpuBeamSearchCascadeTest.PREFIX_BANKS * MAX_PAGES
    NUM_PAGES = PRIVATE_BASE + 3 * PpuBeamSearchCascadeTest.BATCH + 1
    CASES = (
        ([500], [265], "shared"),
        ([250, 250], [319, 319], "reused_first_page"),
        ([128, 128, 244], [265, 511, 1354], "reused_first_page"),
        ([500], [265], "distinct"),
        ([100] * 5, [265] * 5, "shared"),
        ([256], [1354], "shared"),
        ([500], [265], "shared"),
    )


if __name__ == "__main__":
    unittest.main()
