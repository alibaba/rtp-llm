"""PPU decode RoPE graph replay with fresh caller-owned metadata each step."""

import os
import unittest
from unittest.mock import patch

import torch

from rtp_llm.device.device_type import DeviceType, get_device_type
from rtp_llm.models_py.modules.factory.attention.common import copy_kv_cache_offset
from rtp_llm.ops import AttentionConfigs, KvCacheDataType, RopeStyle
from rtp_llm.ops.compute_ops import (
    FusedRopeKVCacheDecodeOp,
    LayerKVCache,
    PyAttentionInputs,
)


@unittest.skipUnless(get_device_type() == DeviceType.Ppu, "requires PPU")
class PpuDecodeRopeGraphTest(unittest.TestCase):
    HEADS = 16
    KV_HEADS = 8
    DIM = 128
    PAGE = 64
    SENTINEL = 19.0

    def setUp(self):
        self.env = patch.dict(
            os.environ,
            {"RTP_LLM_STREAM_ASYNC": "0", "RTP_LLM_MTP_ASYNC_PREPARE": "0"},
        )
        self.env.start()
        self.addCleanup(self.env.stop)
        torch.manual_seed(20261027)
        self.config = AttentionConfigs()
        self.config.head_num = self.HEADS
        self.config.kv_head_num = self.KV_HEADS
        self.config.size_per_head = self.DIM
        self.config.tokens_per_block = self.PAGE
        self.config.kernel_tokens_per_block = self.PAGE
        self.config.dtype = torch.bfloat16
        self.config.kv_cache_dtype = KvCacheDataType.BASE
        self.config.need_rope_kv_cache = True
        self.config.max_seq_len = 128
        self.config.rope_config.style = RopeStyle.Base
        self.config.rope_config.dim = self.DIM
        self.config.rope_config.base = 10000
        self.config.rope_config.max_pos = 128

    def _inputs(self, lengths, *, graph=True, reverse=False):
        inputs = PyAttentionInputs()
        inputs.is_prefill = False
        inputs.is_cuda_graph = graph
        inputs.sequence_lengths = torch.tensor(lengths, dtype=torch.int32).pin_memory()
        table = torch.arange(1, len(lengths) * 2 + 1, dtype=torch.int32).reshape(-1, 2)
        if reverse:
            table = table.flip((0, 1))
        # Zero-length graph padding writes only to the reserved scratch page.
        table[torch.tensor(lengths) == 0] = 0
        inputs.kv_cache_kernel_block_id = table.pin_memory()
        inputs.kv_cache_kernel_block_id_device = table.cuda()
        return inputs

    def _cache(self, batch):
        cache = LayerKVCache()
        cache.kv_cache_base = torch.full(
            (batch * 2 + 1, 2, self.KV_HEADS, self.PAGE, self.DIM),
            self.SENTINEL,
            dtype=torch.bfloat16,
            device="cuda",
        )
        return cache

    def _qkv(self, batch, active):
        qkv = torch.randn(
            batch,
            (self.HEADS + 2 * self.KV_HEADS) * self.DIM,
            dtype=torch.bfloat16,
            device="cuda",
        )
        qkv[active:].zero_()
        return qkv

    def _rope_reference(self, tensor, lengths):
        # Independent FP32 base-RoPE reference, evaluated on the CPU.
        value = tensor.float().cpu()
        inv_freq = 1.0 / (10000 ** (torch.arange(0, self.DIM, 2).float() / self.DIM))
        angles = torch.tensor(lengths).float()[:, None] * inv_freq
        cosine = torch.cat((angles.cos(), angles.cos()), -1)[:, None, :]
        sine = torch.cat((angles.sin(), angles.sin()), -1)[:, None, :]
        first, second = value.chunk(2, -1)
        return (value * cosine + torch.cat((-second, first), -1) * sine).to(
            tensor.dtype
        )

    def _assert_reference_and_sentinels(self, output, qkv, cache, inputs):
        batch = qkv.size(0)
        lengths = inputs.sequence_lengths.tolist()
        packed = qkv.reshape(batch, self.HEADS + 2 * self.KV_HEADS, self.DIM)
        q, k, v = packed.split((self.HEADS, self.KV_HEADS, self.KV_HEADS), dim=1)
        torch.testing.assert_close(
            output.cpu(), self._rope_reference(q, lengths), atol=0.02, rtol=0.02
        )
        positions = torch.tensor(lengths, dtype=torch.long)
        pages = inputs.kv_cache_kernel_block_id[
            torch.arange(batch), positions // self.PAGE
        ].long()
        slots = positions % self.PAGE
        data = cache.kv_cache_base.permute(0, 3, 1, 2, 4)
        written = data[pages.cuda(), slots.cuda()].cpu()
        torch.testing.assert_close(
            written[:, 0], self._rope_reference(k, lengths), atol=0.02, rtol=0.02
        )
        torch.testing.assert_close(written[:, 1], v.cpu(), atol=0, rtol=0)
        expected_untouched = torch.ones((batch * 2 + 1, self.PAGE), dtype=torch.bool)
        expected_untouched[pages, slots] = False
        actual_untouched = (cache.kv_cache_base == self.SENTINEL).all(dim=(1, 2, 4))
        torch.testing.assert_close(actual_untouched.cpu(), expected_untouched)

    def test_changed_input_replay_matches_eager_and_preserves_unwritten_cache(self):
        # <=16 and >16 select different PPU decode kernels. 512 with 500 live
        # rows exercises padding; exact multiples of 500 cover the observed
        # beam groups and the workload-specific capture buckets up to 2000.
        for batch, active in (
            (4, 4),
            (512, 500),
            (500, 500),
            (1000, 1000),
            (1500, 1500),
            (2000, 2000),
        ):
            with self.subTest(batch=batch, active=active):
                capture_inputs = self._inputs([3] * active + [0] * (batch - active))
                op = FusedRopeKVCacheDecodeOp(self.config)
                params = op.prepare(capture_inputs)
                length_ptr = params.sequence_lengths.data_ptr()
                offset_ptr = params.kv_cache_offset.data_ptr()
                self.assertNotEqual(
                    length_ptr, capture_inputs.sequence_lengths.data_ptr()
                )
                qkv = self._qkv(batch, active)
                cache = self._cache(batch)
                # Resolve lazy rope-cache allocation before capture.
                for _ in range(2):
                    op.forward(qkv, cache, params)
                torch.cuda.synchronize()
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    result = op.forward(qkv, cache, params)

                # Every step replaces input tensors, QKV source, and page ids.
                # 63 -> 64 forces the write to move into another cache page.
                for step, history in enumerate((3, 63, 64, 7)):
                    torch.cuda.synchronize()
                    lengths = [history] * active + [0] * (batch - active)
                    live_inputs = self._inputs(lengths, reverse=bool(step % 2))
                    eager_inputs = self._inputs(
                        lengths, graph=False, reverse=bool(step % 2)
                    )
                    live_qkv = self._qkv(batch, active)
                    qkv.copy_(live_qkv)
                    cache.kv_cache_base.fill_(self.SENTINEL)
                    new_params = op.prepare(live_inputs, forbid_reallocation=True)
                    copy_kv_cache_offset(
                        params.kv_cache_offset, new_params.kv_cache_offset
                    )
                    params.sequence_lengths = new_params.sequence_lengths
                    self.assertEqual(params.sequence_lengths.data_ptr(), length_ptr)
                    self.assertEqual(params.kv_cache_offset.data_ptr(), offset_ptr)
                    self.assertTrue(params.sequence_lengths.is_pinned())
                    # The H2H length copy is finished; changing caller storage
                    # must not change values consumed by the captured pointer.
                    live_inputs.sequence_lengths.fill_(99)
                    torch.cuda.synchronize()
                    # Conversion/copies are complete before poisoning old
                    # source tables. Only the retained capture offsets may win.
                    live_inputs.kv_cache_kernel_block_id.fill_(0)
                    live_inputs.kv_cache_kernel_block_id_device.zero_()
                    graph.replay()
                    torch.cuda.synchronize()

                    eager_op = FusedRopeKVCacheDecodeOp(self.config)
                    eager_cache = self._cache(batch)
                    eager_params = eager_op.prepare(eager_inputs)
                    eager_output = eager_op.forward(live_qkv, eager_cache, eager_params)
                    torch.cuda.synchronize()
                    torch.testing.assert_close(result, eager_output, atol=0, rtol=0)
                    torch.testing.assert_close(
                        cache.kv_cache_base, eager_cache.kv_cache_base, atol=0, rtol=0
                    )
                    self._assert_reference_and_sentinels(
                        result, live_qkv, cache, eager_inputs
                    )

    def test_fa3_decode_replay_refreshes_rope_and_kv_positions(self):
        from rtp_llm.models_py.modules.factory.attention.ppu_impl.fa3_mha import (
            FA3DecodeImpl,
        )

        batch = 4
        captured = self._inputs([3] * batch)
        impl = FA3DecodeImpl(self.config, captured)
        length_ptr = impl.rope_params.sequence_lengths.data_ptr()
        qkv = self._qkv(batch, batch)
        cache = self._cache(batch)
        for _ in range(2):
            impl.forward(qkv, cache, 0)
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            result = impl.forward(qkv, cache, 0)
        for step, history in enumerate((7, 63, 64, 11)):
            torch.cuda.synchronize()
            live = self._inputs([history] * batch, graph=False, reverse=bool(step % 2))
            # The runner updates its captured device page-table allocation in place,
            # but passes fresh host lengths with is_cuda_graph=False on replay.
            captured.kv_cache_kernel_block_id_device.copy_(
                live.kv_cache_kernel_block_id_device
            )
            live.kv_cache_kernel_block_id_device = (
                captured.kv_cache_kernel_block_id_device
            )
            source = self._qkv(batch, batch)
            qkv.copy_(source)
            cache.kv_cache_base.fill_(self.SENTINEL)
            impl.prepare_cuda_graph(live)
            self.assertEqual(impl.rope_params.sequence_lengths.data_ptr(), length_ptr)
            torch.testing.assert_close(
                impl.rope_params.sequence_lengths, live.sequence_lengths
            )
            graph.replay()
            torch.cuda.synchronize()
            eager_inputs = self._inputs(
                [history] * batch, graph=False, reverse=bool(step % 2)
            )
            eager_cache = self._cache(batch)
            expected = FA3DecodeImpl(self.config, eager_inputs).forward(
                source.clone(), eager_cache, 0
            )
            torch.cuda.synchronize()
            torch.testing.assert_close(result, expected, atol=0, rtol=0)
            torch.testing.assert_close(
                cache.kv_cache_base, eager_cache.kv_cache_base, atol=0, rtol=0
            )
            packed = source.reshape(batch, self.HEADS + 2 * self.KV_HEADS, self.DIM)
            key = packed[:, self.HEADS : self.HEADS + self.KV_HEADS]
            pages = (
                eager_inputs.kv_cache_kernel_block_id[:, history // self.PAGE]
                .long()
                .cuda()
            )
            torch.testing.assert_close(
                cache.kv_cache_base[pages, 0, :, history % self.PAGE].cpu(),
                self._rope_reference(key, [history] * batch),
                atol=0.02,
                rtol=0.02,
            )

    def test_eager_retains_caller_pinned_lengths_even_with_async_flags(self):
        with patch.dict(os.environ, {"RTP_LLM_STREAM_ASYNC": "1"}):
            inputs = self._inputs([3, 7], graph=False)
            params = FusedRopeKVCacheDecodeOp(self.config).prepare(inputs)
            self.assertEqual(
                params.sequence_lengths.data_ptr(), inputs.sequence_lengths.data_ptr()
            )

    def test_graph_rejects_resize_invalid_storage_and_async_prepare(self):
        op = FusedRopeKVCacheDecodeOp(self.config)
        params = op.prepare(self._inputs([3, 7]))
        before = params.sequence_lengths.clone()
        with self.assertRaisesRegex(RuntimeError, "cannot resize"):
            op.prepare(self._inputs([9]), forbid_reallocation=True)
        invalid_sources = (
            torch.tensor([3, 7], dtype=torch.int32),  # pageable CPU
            torch.tensor([3, 7], dtype=torch.int64).pin_memory(),
            torch.zeros(4, dtype=torch.int32, pin_memory=True)[::2],
            torch.tensor([3, 7], dtype=torch.int32, device="cuda"),
        )
        for source in invalid_sources:
            inputs = self._inputs([3, 7])
            inputs.sequence_lengths = source
            with self.assertRaisesRegex(RuntimeError, "contiguous pinned CPU int32"):
                op.prepare(inputs, forbid_reallocation=True)
        for name in ("RTP_LLM_STREAM_ASYNC", "RTP_LLM_MTP_ASYNC_PREPARE"):
            with patch.dict(os.environ, {name: "1"}):
                with self.assertRaisesRegex(RuntimeError, name + "=1 is not supported"):
                    op.prepare(self._inputs([9, 11]), forbid_reallocation=True)
        torch.testing.assert_close(params.sequence_lengths, before)


if __name__ == "__main__":
    unittest.main()
