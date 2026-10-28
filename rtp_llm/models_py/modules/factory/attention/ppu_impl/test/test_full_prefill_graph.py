"""Real PPU ragged-prefill graph replay and next-decode KV consumer tests."""

import math
import unittest
from unittest.mock import patch

import torch

from rtp_llm.device.device_type import DeviceType, get_device_type
from rtp_llm.models_py.modules.factory.attention import attn_factory
from rtp_llm.models_py.modules.factory.attention.cuda_impl.py_flashinfer_mha import (
    PyFlashinferDecodeImpl,
    PyFlashinferPrefillImpl,
)
from rtp_llm.models_py.modules.factory.attention.fmha_impl_base import (
    PrefillCudaGraphCapability,
)
from rtp_llm.models_py.modules.factory.attention.ppu_impl.full_prefill import (
    PpuFlashinferFullPrefillGraphImpl,
)
from rtp_llm.ops import (
    AttentionConfigs,
    FMHAConfig,
    KvCacheDataType,
    ParallelismConfig,
    RopeStyle,
)
from rtp_llm.ops.compute_ops import LayerKVCache, PyAttentionInputs, get_typemeta


@unittest.skipUnless(get_device_type() == DeviceType.Ppu, "requires PPU")
class PpuFullPrefillGraphTest(unittest.TestCase):
    HEADS = 16
    KV_HEADS = 8
    DIM = 128
    PAGE = 64
    CAPACITY = 192
    SENTINEL = 19.0

    def setUp(self):
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
        self.config.max_seq_len = 256
        self.config.rope_config.style = RopeStyle.Base
        self.config.rope_config.dim = self.DIM
        self.config.rope_config.base = 10000
        self.config.rope_config.max_pos = 256
        self.width = self.CAPACITY // self.PAGE + 1
        self.num_pages = 4 * self.width + 1

    def _inputs(self, real_tokens, *, graph=True, alternate=False):
        inputs = PyAttentionInputs()
        inputs.is_prefill = True
        inputs.is_cuda_graph = graph
        inputs.input_lengths = torch.tensor(
            [real_tokens, self.CAPACITY - real_tokens], dtype=torch.int32
        ).pin_memory()
        inputs.prefix_lengths = torch.zeros(2, dtype=torch.int32, pin_memory=True)
        inputs.sequence_lengths = torch.empty(0, dtype=torch.int32)
        inputs.cu_seqlens = torch.tensor(
            [0, real_tokens, self.CAPACITY], dtype=torch.int32
        ).pin_memory()
        inputs.cu_seqlens_device = inputs.cu_seqlens.cuda()
        inputs.cu_kv_seqlens_device = inputs.cu_seqlens.cuda()
        # input_lengths_device/prefix_lengths_device are C++ runner-owned,
        # read-only properties. Both this adapter and the eager reference plan
        # from host lengths, so standalone inputs leave those mirrors unset.
        # Two disjoint banks for real/scratch, with another bank on odd steps.
        start = 1 + (2 * self.width if alternate else 0)
        table = torch.arange(start, start + 2 * self.width, dtype=torch.int32)
        table = table.reshape(2, self.width)
        if alternate:
            table = table.flip(1)
        inputs.kv_cache_kernel_block_id = table.pin_memory()
        inputs.kv_cache_kernel_block_id_device = table.cuda()
        inputs.context_total_kv_length = self.CAPACITY
        inputs.dtype = get_typemeta(torch.zeros(1, dtype=torch.bfloat16))
        return inputs

    def _cache(self):
        cache = LayerKVCache()
        cache.kv_cache_base = torch.full(
            (self.num_pages, 2, self.KV_HEADS, self.PAGE, self.DIM),
            self.SENTINEL,
            dtype=torch.bfloat16,
            device="cuda",
        )
        return cache

    def _qkv(self, tokens):
        return torch.randn(
            tokens,
            (self.HEADS + 2 * self.KV_HEADS) * self.DIM,
            dtype=torch.bfloat16,
            device="cuda",
        )

    def _rotate(self, tensor, positions):
        value = tensor.float().cpu()
        inv_freq = 1.0 / (10000 ** (torch.arange(0, self.DIM, 2).float() / self.DIM))
        angles = torch.tensor(positions).float()[:, None] * inv_freq
        cosine = torch.cat((angles.cos(), angles.cos()), -1)[:, None, :]
        sine = torch.cat((angles.sin(), angles.sin()), -1)[:, None, :]
        first, second = value.chunk(2, -1)
        return (value * cosine + torch.cat((-second, first), -1) * sine).to(
            torch.bfloat16
        )

    def _reference(self, raw_qkv, inputs):
        lengths = inputs.input_lengths.tolist()
        positions = [p for length in lengths for p in range(length)]
        packed = raw_qkv.cpu().reshape(-1, self.HEADS + 2 * self.KV_HEADS, self.DIM)
        raw_q, raw_k, v = packed.split((self.HEADS, self.KV_HEADS, self.KV_HEADS), 1)
        q, k = self._rotate(raw_q, positions), self._rotate(raw_k, positions)
        output = torch.empty_like(q)
        offset = 0
        for length in lengths:
            if length:
                qs = q[offset : offset + length].float().transpose(0, 1)
                ks = k[offset : offset + length].repeat_interleave(2, 1).float()
                vs = v[offset : offset + length].repeat_interleave(2, 1).float()
                scores = qs @ ks.permute(1, 2, 0) / math.sqrt(self.DIM)
                future = torch.ones(length, length, dtype=torch.bool).triu(1)
                scores.masked_fill_(future, float("-inf"))
                result = scores.softmax(-1) @ vs.transpose(0, 1)
                output[offset : offset + length] = result.transpose(0, 1).to(
                    output.dtype
                )
            offset += length
        return output, k, v

    def _assert_cache(self, cache, inputs, expected_k, expected_v):
        untouched = torch.ones(self.num_pages, self.PAGE, dtype=torch.bool)
        data = cache.kv_cache_base.permute(0, 3, 1, 2, 4)
        offset = 0
        for slot, length in enumerate(inputs.input_lengths.tolist()):
            positions = torch.arange(length)
            pages = inputs.kv_cache_kernel_block_id[slot, positions // self.PAGE].long()
            slots = positions % self.PAGE
            actual = data[pages.cuda(), slots.cuda()].cpu()
            torch.testing.assert_close(
                actual[:, 0], expected_k[offset : offset + length], atol=0.02, rtol=0.02
            )
            torch.testing.assert_close(
                actual[:, 1], expected_v[offset : offset + length], atol=0, rtol=0
            )
            untouched[pages, slots] = False
            offset += length
        actual_untouched = (cache.kv_cache_base == self.SENTINEL).all(dim=(1, 2, 4))
        torch.testing.assert_close(actual_untouched.cpu(), untouched)

    def _assert_next_decode(self, graph_cache, eager_cache, prefill_inputs, k, v):
        length = int(prefill_inputs.input_lengths[0])
        if not length:
            return
        inputs = PyAttentionInputs()
        inputs.is_prefill = False
        inputs.is_cuda_graph = False
        inputs.input_lengths = torch.ones(1, dtype=torch.int32).pin_memory()
        inputs.prefix_lengths = torch.empty(0, dtype=torch.int32)
        inputs.sequence_lengths = torch.tensor([length], dtype=torch.int32).pin_memory()
        inputs.sequence_lengths_plus_1_device = (inputs.sequence_lengths + 1).cuda()
        inputs.cu_seqlens_device = torch.tensor(
            [0, 1], dtype=torch.int32, device="cuda"
        )
        inputs.kv_cache_kernel_block_id = prefill_inputs.kv_cache_kernel_block_id[
            :1
        ].clone()
        inputs.kv_cache_kernel_block_id_device = inputs.kv_cache_kernel_block_id.cuda()
        inputs.dtype = prefill_inputs.dtype
        decode = PyFlashinferDecodeImpl(self.config, inputs)
        raw = self._qkv(1)
        actual = decode.forward(raw.clone(), graph_cache)
        eager = decode.forward(raw.clone(), eager_cache)
        torch.cuda.synchronize()
        torch.testing.assert_close(actual, eager, atol=0, rtol=0)
        packed = raw.cpu().reshape(1, self.HEADS + 2 * self.KV_HEADS, self.DIM)
        dq, dk, dv = packed.split((self.HEADS, self.KV_HEADS, self.KV_HEADS), 1)
        dq, dk = self._rotate(dq, [length]), self._rotate(dk, [length])
        all_k = torch.cat((k[:length], dk), 0).repeat_interleave(2, 1).float()
        all_v = torch.cat((v[:length], dv), 0).repeat_interleave(2, 1).float()
        scores = torch.einsum("hd,thd->ht", dq[0].float(), all_k) / math.sqrt(self.DIM)
        expected = torch.einsum("ht,thd->hd", scores.softmax(-1), all_v)
        torch.testing.assert_close(
            actual.reshape(self.HEADS, self.DIM).cpu().float(),
            expected,
            atol=0.02,
            rtol=0.02,
        )

    @staticmethod
    def _pointers(impl):
        return tuple(
            tensor.data_ptr()
            for tensor in (
                impl.qo_indptr,
                impl.kv_indptr,
                impl.output,
                impl.params.positions_d,
                impl.params.batch_indice_d,
                impl.params.page_indice_d,
                impl.params.decode_page_indptr_d,
                impl.params.paged_kv_last_page_len_d,
            )
        )

    def test_changed_tokens_pages_slots_and_next_decode(self):
        capture_inputs = self._inputs(0)
        impl = PpuFlashinferFullPrefillGraphImpl(self.config, capture_inputs)
        self.assertTrue(impl.wrapper.is_cuda_graph_enabled)
        self.assertEqual(
            impl.prefill_cuda_graph_capability(),
            PrefillCudaGraphCapability.FULL_NO_PREFIX,
        )
        pointers = self._pointers(impl)
        source, work = self._qkv(self.CAPACITY), self._qkv(self.CAPACITY)
        graph_cache = self._cache()
        for _ in range(2):
            work.copy_(source)
            impl.forward(work, graph_cache)
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            # Model QKV projection overwrites its output each replay; mimic it
            # because the reused RoPE operator intentionally updates Q/K in place.
            work.copy_(source)
            output = impl.forward(work, graph_cache)

        # Start with scratch-only capture, then fill real slot 0, cross pages,
        # empty the scratch slot, and return all tokens to it. Both slot identity
        # and the real/scratch boundary change while the same graph is reused.
        for step, real_tokens in enumerate((0, 63, 64, 129, 192, 7, 0)):
            with self.subTest(real_tokens=real_tokens):
                torch.cuda.synchronize()
                # CudaGraphRunner passes its original C++ metadata on replay.
                # Only the Python copy used by the factory had its graph flag
                # set; runtime inputs keep the default False.
                inputs = self._inputs(
                    real_tokens, graph=False, alternate=bool(step % 2)
                )
                self.assertFalse(impl.support(self.config, inputs))
                reference_inputs = self._inputs(
                    real_tokens, graph=False, alternate=bool(step % 2)
                )
                raw = self._qkv(self.CAPACITY)
                source.copy_(raw)
                graph_cache.kv_cache_base.fill_(self.SENTINEL)
                impl.prepare_cuda_graph(inputs)
                self.assertFalse(inputs.is_cuda_graph)
                self.assertEqual(self._pointers(impl), pointers)
                torch.cuda.synchronize()
                # Metadata belongs to the adapter after prepare; fresh caller
                # storage may be overwritten without changing captured readers.
                inputs.input_lengths.zero_()
                inputs.cu_seqlens.zero_()
                inputs.kv_cache_kernel_block_id.zero_()
                inputs.kv_cache_kernel_block_id_device.zero_()
                graph.replay()
                torch.cuda.synchronize()
                eager_cache = self._cache()
                eager = PyFlashinferPrefillImpl(self.config, reference_inputs)
                eager_output = eager.forward(raw.clone(), eager_cache)
                torch.cuda.synchronize()
                expected, k, v = self._reference(raw, reference_inputs)
                torch.testing.assert_close(output, eager_output, atol=0.02, rtol=0.02)
                torch.testing.assert_close(output.cpu(), expected, atol=0.02, rtol=0.02)
                torch.testing.assert_close(
                    graph_cache.kv_cache_base, eager_cache.kv_cache_base, atol=0, rtol=0
                )
                self._assert_cache(graph_cache, reference_inputs, k, v)
                self._assert_next_decode(
                    graph_cache, eager_cache, reference_inputs, k, v
                )

    def test_factory_selects_only_ppu_full_prefill_graph(self):
        inputs = self._inputs(0)
        # Exercise the exact factory contract even before production registration.
        # Restore the original list on exit so this cannot change eager routing.
        impls = [PpuFlashinferFullPrefillGraphImpl] + [
            impl
            for impl in attn_factory.PREFILL_MHA_IMPS
            if impl is not PpuFlashinferFullPrefillGraphImpl
        ]
        with patch.object(attn_factory, "PREFILL_MHA_IMPS", impls):
            impl = attn_factory.get_fmha_impl(
                self.config,
                None,
                inputs,
                is_cuda_graph=True,
                cuda_graph_selection_mode="full_prefill_no_prefix_graph",
            )
            self.assertIsInstance(impl, PpuFlashinferFullPrefillGraphImpl)
            disabled = FMHAConfig()
            disabled.disable_flashinfer_native = True
            with self.assertRaisesRegex(RuntimeError, "unsupported_backend"):
                attn_factory.get_fmha_impl(
                    self.config,
                    None,
                    self._inputs(0),
                    fmha_config=disabled,
                    is_cuda_graph=True,
                    cuda_graph_selection_mode="full_prefill_no_prefix_graph",
                )
        self.assertFalse(
            PpuFlashinferFullPrefillGraphImpl.support(
                self.config, self._inputs(0, graph=False)
            )
        )
        with self.assertRaisesRegex(ValueError, "requires BF16"):
            PpuFlashinferFullPrefillGraphImpl(self.config, self._inputs(0, graph=False))
        inputs.prefix_lengths[0] = 64
        self.assertFalse(PpuFlashinferFullPrefillGraphImpl.support(self.config, inputs))
        inputs = self._inputs(0)
        inputs.input_lengths = torch.tensor([0, 0, self.CAPACITY], dtype=torch.int32)
        self.assertFalse(PpuFlashinferFullPrefillGraphImpl.support(self.config, inputs))
        parallel = ParallelismConfig()
        parallel.tp_size = 2
        self.assertFalse(
            PpuFlashinferFullPrefillGraphImpl.support_parallelism_config(parallel)
        )
        self.config.dtype = torch.float16
        self.assertFalse(
            PpuFlashinferFullPrefillGraphImpl.support(self.config, self._inputs(0))
        )

    def test_replay_rejects_capacity_change_and_inconsistent_metadata(self):
        impl = PpuFlashinferFullPrefillGraphImpl(self.config, self._inputs(0))
        inputs = self._inputs(64, graph=False)
        inputs.input_lengths[1] -= 1
        inputs.cu_seqlens[2] -= 1
        with self.assertRaisesRegex(ValueError, "cannot change captured"):
            impl.prepare_cuda_graph(inputs)
        inputs = self._inputs(64, graph=False)
        inputs.cu_seqlens[1] = 63
        with self.assertRaisesRegex(ValueError, "cu_seqlens disagrees"):
            impl.prepare_cuda_graph(inputs)
        inputs = self._inputs(64, graph=False)
        inputs.kv_cache_kernel_block_id = inputs.kv_cache_kernel_block_id[
            :, :1
        ].contiguous()
        with self.assertRaisesRegex(ValueError, "page table"):
            impl.prepare_cuda_graph(inputs)

        # Accepting the runner's False selection marker must not weaken the
        # no-prefix, prefill-only, or implicit-position semantic constraints.
        invalid_metadata = (
            ("prefix_lengths", torch.tensor([1, 0], dtype=torch.int32)),
            ("is_prefill", False),
            ("is_target_verify", True),
            ("combo_position_ids", torch.arange(self.CAPACITY, dtype=torch.int32)),
        )
        for field, value in invalid_metadata:
            with self.subTest(field=field):
                inputs = self._inputs(64, graph=False)
                setattr(inputs, field, value)
                with self.assertRaisesRegex(ValueError, "unsupported PPU full prefill"):
                    impl.prepare_cuda_graph(inputs)


if __name__ == "__main__":
    unittest.main()
