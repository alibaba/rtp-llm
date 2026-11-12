"""Exact local Q/output in-place RoPE over the existing signed-zero kernel.

The device body fixture uses actual codecs/workspace/FlashMLA and fixtures for
fresh projections plus read-only source banks. It is not a model benchmark.
"""

from contextlib import ExitStack
import os
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch

import torch
from torch import nn

from rtp_llm.models_py.modules.dsv41.ppu import attention_layer, graph_swa_write, prefill
from rtp_llm.models_py.modules.dsv41.ppu._quant import is_ppu
from rtp_llm.models_py.modules.dsv41.ppu.kv_codec import decode_kv, encode_kv
from rtp_llm.models_py.modules.dsv41.ppu.test.test_local_prefill_fused import assert_bytes
from rtp_llm.models_py.modules.dsv41.ppu.test.test_local_prefill_operator_flags import (
    _rope_inputs,
    _trace_events,
)


ENV = attention_layer.LOCAL_PREFILL_GROUPED_QO_ROPE_ENV


def assert_same_storage_view(case, actual, expected):
    """A reshape wrapper may differ; its exact storage and view must not."""
    case.assertTrue(torch._C._is_alias_of(actual, expected))
    case.assertEqual(actual.untyped_storage().data_ptr(), expected.untyped_storage().data_ptr())
    case.assertEqual(actual.data_ptr(), expected.data_ptr())
    case.assertEqual(actual.storage_offset(), expected.storage_offset())
    case.assertEqual(actual.shape, expected.shape)
    case.assertEqual(actual.stride(), expected.stride())
    case.assertEqual(actual.dtype, expected.dtype)
    case.assertEqual(actual.device, expected.device)


def _state():
    return NS(prefill=True, idle=False, cache=NS(sharded=False), attn=NS(is_cuda_graph=False),
              runtime=NS(model=NS(_is_speculative=False)))


class LocalQORopeCPU(unittest.TestCase):
    @torch.inference_mode()
    def test_default_off_local_cp1_and_ineligible_phases(self):
        with patch.dict(os.environ), patch.object(attention_layer, "is_ppu", return_value=True), patch(
            "torch.cuda.is_current_stream_capturing", return_value=False,
        ):
            os.environ.pop(ENV, None)
            self.assertFalse(attention_layer._local_prefill_grouped_qo_rope_enabled(_state(), None))
            os.environ[ENV] = "1"
            self.assertTrue(attention_layer._local_prefill_grouped_qo_rope_enabled(_state(), None))
            self.assertTrue(attention_layer._local_prefill_grouped_qo_rope_enabled(
                _state(), NS(cp_size=1, kv_cache_sharded=False),
            ))
            for cp in (NS(cp_size=8, kv_cache_sharded=True), NS(cp_size=1, kv_cache_sharded=True)):
                self.assertFalse(attention_layer._local_prefill_grouped_qo_rope_enabled(_state(), cp))
            for flag in ("idle", "is_graph", "is_graph_commit", "verify", "proposal"):
                state = _state()
                setattr(state, flag, True)
                self.assertFalse(attention_layer._local_prefill_grouped_qo_rope_enabled(state, None))
            for change in ("decode", "sharded", "speculative", "capture"):
                state = _state()
                if change == "decode":
                    state.prefill = False
                elif change == "sharded":
                    state.cache.sharded = True
                elif change == "speculative":
                    state.runtime.model._is_speculative = True
                else:
                    state.attn.is_cuda_graph = True
                self.assertFalse(attention_layer._local_prefill_grouped_qo_rope_enabled(state, None))
            self.assertFalse(attention_layer._local_prefill_grouped_qo_rope_enabled(
                _state(), None, diagnostic={},
            ))
            with patch("torch.cuda.is_current_stream_capturing", return_value=True):
                self.assertFalse(attention_layer._local_prefill_grouped_qo_rope_enabled(_state(), None))

    def test_grad_device_and_unowned_tensors_do_not_admit(self):
        with patch.dict(os.environ, {ENV: "1"}):
            self.assertFalse(attention_layer._local_prefill_grouped_qo_rope_enabled(_state(), None))
        with torch.inference_mode(), patch.dict(os.environ, {ENV: "1"}), patch.object(
            attention_layer, "is_ppu", return_value=False,
        ):
            self.assertFalse(attention_layer._local_prefill_grouped_qo_rope_enabled(_state(), None))
        # Metadata-only ownership checks never clone an alias into admission.
        storage = lambda address: NS(data_ptr=lambda: address)
        value = NS(is_cuda=True, dtype=torch.bfloat16, ndim=3, shape=(140, 64, 512),
                   is_contiguous=lambda: True, requires_grad=False, untyped_storage=lambda: storage(1234))
        source = NS(untyped_storage=lambda: storage(5678))
        self.assertTrue(attention_layer._local_qo_rope_owned(value, source))
        source.untyped_storage = lambda: storage(1234)
        self.assertFalse(attention_layer._local_qo_rope_owned(value, source))
        for tensor in (torch.zeros(2, 64, 512, dtype=torch.bfloat16), torch.zeros(2, 8, 128)):
            self.assertFalse(attention_layer._local_qo_rope_owned(tensor))


class _SourceBanks:
    sharded = False

    def __init__(self, global_values, swa_values, receipt):
        self.lengths, self.receipt = (global_values.shape[0],), receipt
        self.global_bytes = encode_kv(global_values, pool_tag="global_kv_1")
        self.index_bytes = torch.full((global_values.shape[0], 68), 0xA5,
                                      device=global_values.device, dtype=torch.uint8)
        self.swa_bytes = encode_kv(swa_values, pool_tag="swa_kv")

    def stage_swa(self, layer, positions, requests, packed, prefixes, **kwargs):
        self.receipt.staged.append((positions.clone(), packed.clone()))
        self.swa_bytes.index_copy_(0, positions.long(), packed)

    def descriptors(self, layer):
        return {"global_kv_1": None, "swa_kv": None}

    def prepare_source(self, source):
        self.receipt.sources.append(source)

    def request_length(self, request):
        return self.lengths[request]

    def read_source(self, source, request, *, global_entries):
        return self.global_bytes[:global_entries], self.index_bytes[:global_entries]

    def read_swa(self, layer, request, positions):
        return self.swa_bytes.index_select(0, positions.long())

    def release_swa(self, layer):
        pass


def _body_fixture(rows, q_seed, kv_seed, frequencies, global_values, prefix_values, *, alias_qr=False):
    """A local consumer with actual codec/workspace/native attention execution."""
    device, prefix = q_seed.device, prefix_values.shape[0]
    receipt = NS(staged=[], sources=[], q=None, output=None)
    cache = _SourceBanks(global_values, torch.cat((prefix_values, kv_seed)), receipt)
    attn = attention_layer.SharedPPUAttention.__new__(attention_layer.SharedPPUAttention)
    nn.Module.__init__(attn)
    geometry = NS(hidden_size=8, n_heads=64, head_dim=512, window=128,
                  o_groups=1, o_lora_rank=8, ratio=lambda layer: 1,
                  kv_source_for=lambda layer: 20, sm_scale=lambda: 512 ** -0.5)
    attn.geometry, attn.layer_id, attn.kv_source_layer_id = geometry, 21, 20
    attn.compress_ratio, attn.is_index_source, attn.is_kv_source = 1, False, False
    attn.ops = NS(encode_kv=encode_kv, decode_kv=decode_kv)
    # Seed copies are allocated before body profiling and represent fresh owned
    # projection outputs. The body must not create copies to admit aliases.
    q_projection, kv_projection = q_seed.clone(), kv_seed.clone()
    receipt.q = q_projection
    attn.project_query_kv = lambda value, **kwargs: (
        q_projection.flatten(1) if alias_qr else value, q_projection, kv_projection, None,
    )
    attn.wo_a = nn.Linear(64 * 512, 8, bias=False, device=device, dtype=torch.bfloat16)
    attn.wo_a.weight.data.fill_(1.0 / (64 * 512))
    attn.wo_b = nn.Identity()
    attn.attn_sink = torch.zeros(64, device=device)
    positions = torch.arange(prefix, prefix + rows, device=device, dtype=torch.int64)
    requests = torch.zeros(rows, device=device, dtype=torch.int64)
    cu = torch.tensor([0, rows], device=device, dtype=torch.int32)
    valid = torch.ones(rows, device=device, dtype=torch.bool)
    state = _state()
    state.cache, state.context, state.shared, state.compacted = cache, None, {}, False
    state.query_prefixes, state.lengths = (prefix,), (rows,)
    state.rows = lambda cp=None: (positions, requests, cu, valid)
    state.full_rows = lambda cp, **kwargs: (positions, requests, slice(None))
    state.runtime.policy = NS(mode="full")
    attn.runtime = NS(current=state, policy=state.runtime.policy, freqs_for=lambda layer: frequencies,
                      qkv_streams=None)
    attn._cp_ctx = None
    selected = torch.full((rows, 512), -1, device=device, dtype=torch.int32)
    selected[:, :16] = torch.arange(16, device=device, dtype=torch.int32)
    attn._select = lambda *args, **kwargs: selected
    return attn, state, receipt


@unittest.skipUnless(os.environ.get("RTP_V41_TEST_LOCAL_WORKSPACE_DEVICE") == "1", "explicit PPU device test")
class LocalQORopeDevice(unittest.TestCase):
    def setUp(self):
        self.assertTrue(is_ppu(), "in-place RoPE qualification requires a ZW PPU")
        self.environment = patch.dict(os.environ, {
            ENV: "1", "RTP_V41_PPU_FUSED_BF16_ROPE": "1",
            "RTP_V41_PPU_LOCAL_PREFILL_WORKSPACE": "1", "RTP_V41_PPU_LOCAL_PREFILL_WORKSPACE_FUSED": "0",
            attention_layer.FUSION_ENV: "0", attention_layer.STREAMS_ENV: "0",
        })
        self.environment.start()
        self.addCleanup(self.environment.stop)

    @torch.inference_mode()
    def test_actual_rows_edges_signed_zeros_and_nonrope_prefix_are_bit_exact(self):
        for rows in (4096, 4097, 140):
            for dimension, heads in ((512, 64), (128, 64), (512, None), (128, None)):
                with self.subTest(rows=rows, dimension=dimension, heads=heads):
                    value, frequencies = _rope_inputs(rows, dimension, heads=heads, half=32)
                    frozen = value.clone(), frequencies.clone()
                    for inverse in (False, True):
                        expected = prefill._rope_tail(value, frequencies, inverse)
                        owned = value.clone()
                        result = graph_swa_write.try_rope_(
                            owned, frequencies, inverse=inverse, preserve_signed_zero=True,
                        )
                        self.assertIs(result, owned)
                        self.assertEqual(result.data_ptr(), owned.data_ptr())
                        assert_bytes(self, result, expected)
                        assert_bytes(self, result[..., :-64], value[..., :-64])
                    assert_bytes(self, value, frozen[0])
                    assert_bytes(self, frequencies, frozen[1])

    @torch.inference_mode()
    def test_unsupported_layouts_and_live_aliases_retain_original_fallback(self):
        value, frequencies = _rope_inputs(140, 512, heads=64)
        strided = torch.empty(140, 64, 1024, device="cuda", dtype=torch.bfloat16)[..., ::2]
        strided.copy_(value)
        for operand, table in ((strided, frequencies), (value.float(), frequencies),
                               (value, frequencies.conj()), (value, frequencies[:, :16]),
                               (value[..., :256].contiguous(), frequencies),
                               (value[:0], frequencies[:0])):
            saved = operand.clone()
            self.assertIsNone(graph_swa_write.try_rope_(operand, table, preserve_signed_zero=True))
            assert_bytes(self, operand, saved)
            for inverse in (False, True):
                expected = prefill._rope_tail(operand, table, inverse)
                actual = graph_swa_write.try_rope_(operand, table, inverse=inverse, preserve_signed_zero=True)
                if actual is None:
                    actual = prefill._rope_tail(operand, table, inverse)
                assert_bytes(self, actual, expected)
        alias = value.view_as(value)
        self.assertFalse(attention_layer._local_qo_rope_owned(alias, value))
        output_base = torch.empty_like(value)
        self.assertTrue(attention_layer._local_qo_rope_owned(output_base[:, :64], value))

    @torch.inference_mode()
    def test_inplace_profile_has_one_existing_kernel_and_no_cloned_prefix(self):
        value, frequencies = _rope_inputs(140, 512, heads=64)
        expected = prefill._rope_tail(value, frequencies, inverse=True)
        owned = value.clone()
        graph_swa_write.try_rope_(owned, frequencies, inverse=True, preserve_signed_zero=True)
        owned.copy_(value)
        torch.cuda.synchronize()
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                              torch.profiler.ProfilerActivity.CUDA]) as profile:
            actual = graph_swa_write.try_rope_(owned, frequencies, inverse=True, preserve_signed_zero=True)
            torch.cuda.synchronize()
        self.assertIs(actual, owned)
        assert_bytes(self, actual, expected)
        events = _trace_events(profile, "local-grouped-qo-inplace")
        kernels = [event["name"] for event in events if event.get("cat") == "kernel"]
        self.assertEqual(sum("rope_inplace" in name for name in kernels), 1)
        self.assertFalse(any("fused_rope_tail" in name for name in kernels), kernels)
        cpu = [event["name"] for event in events if event.get("cat") == "cpu_op"]
        self.assertNotIn("aten::clone", cpu)
        self.assertNotIn("aten::copy_", cpu)
        self.assertNotIn("aten::empty_like", cpu)

    @torch.inference_mode()
    def test_actual_body_native_attention_and_source_cache_bytes_are_exact(self):
        rows, prefix = 140, 17
        q = torch.randn(rows, 64, 512, device="cuda", dtype=torch.bfloat16) * 0.1
        kv = torch.randn(rows, 512, device="cuda", dtype=torch.bfloat16) * 0.1
        old_swa = torch.randn(prefix, 512, device="cuda", dtype=torch.bfloat16) * 0.1
        globals_ = torch.randn(prefix + rows, 512, device="cuda", dtype=torch.bfloat16) * 0.1
        angles = torch.randn(prefix + rows, 32, device="cuda")
        frequencies = torch.polar(torch.ones_like(angles), angles).contiguous()
        input_ = torch.randn(rows, 8, device="cuda", dtype=torch.bfloat16)
        outputs, receipts = [], []
        native = attention_layer.flash_mla_sparse_fwd_ppu
        for flag, alias_qr in (("0", False), ("1", False), ("1", True)):
            attn, state, receipt = _body_fixture(rows, q, kv, frequencies, globals_, old_swa, alias_qr=alias_qr)
            saved = tuple(value.clone() for value in (input_, receipt.q, state.cache.global_bytes, state.cache.index_bytes))
            calls = []

            def attention(*args, **kwargs):
                result = native(*args, **kwargs)
                receipt.output = result[0]
                calls.append((args[0].data_ptr(), result[0].data_ptr()))
                return result

            with ExitStack() as stack:
                stack.enter_context(patch.dict(os.environ, {ENV: flag}))
                stack.enter_context(patch.object(attention_layer, "prepare_packed_attention", return_value=(None, "disabled")))
                stack.enter_context(patch.object(attention_layer, "optimized_diagnostics_enabled", return_value=False))
                stack.enter_context(patch.object(attention_layer, "flash_mla_sparse_fwd_ppu", side_effect=attention))
                with patch.object(attention_layer, "try_rope_", wraps=attention_layer.try_rope_) as rotate:
                    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                                          torch.profiler.ProfilerActivity.CUDA]) as profile:
                        outputs.append(attn._body(input_))
                        torch.cuda.synchronize()
                events = _trace_events(profile, f"local-grouped-qo-body-{flag}-{int(alias_qr)}")
                kernels = [event["name"] for event in events if event.get("cat") == "kernel"]
                expected_inplace = (1 if alias_qr else 2) if flag == "1" else 0
                self.assertEqual(sum("rope_inplace" in name for name in kernels), expected_inplace, kernels)
                self.assertEqual(sum("fused_rope_tail" in name for name in kernels), 3 - expected_inplace, kernels)
                self.assertTrue(any("flash_sparse_prefill" in name for name in kernels), kernels)
                self.assertEqual(rotate.call_count, expected_inplace)
                if flag == "1" and not alias_qr:
                    assert_same_storage_view(self, rotate.call_args_list[0].args[0], receipt.q)
                    assert_same_storage_view(self, rotate.call_args_list[1].args[0], receipt.output)
                    self.assertEqual(calls[0][0], receipt.q.data_ptr())
                    self.assertNotEqual(calls[0][0], calls[0][1])
                    self.assertTrue(rotate.call_args_list[0].kwargs["preserve_signed_zero"])
                    self.assertTrue(rotate.call_args_list[1].kwargs["preserve_signed_zero"])
                if alias_qr:
                    assert_bytes(self, receipt.q, saved[1])
            assert_bytes(self, input_, saved[0])
            assert_bytes(self, state.cache.global_bytes, saved[2])
            assert_bytes(self, state.cache.index_bytes, saved[3])
            receipts.append(receipt)
        for arm in (1, 2):
            assert_bytes(self, outputs[arm], outputs[0])
            for candidate, control in zip(receipts[arm].staged[0], receipts[0].staged[0]):
                assert_bytes(self, candidate, control)
            self.assertEqual(receipts[0].sources, receipts[arm].sources)


if __name__ == "__main__":
    unittest.main()
