"""Qualify existing local Prefill bounds and BF16 RoPE switches, exactly.

Registration data must include this file, test_local_prefill_fused.py (the
byte comparator), test_fused_ops.py (the frozen RoPE edge operands), and the
existing v41_ppu_shared_sources. The current local-workspace contracts/device
wrapper may invoke this entire module; only the explicit PPU class uses a GPU.
This is not a new selector, math implementation, or model performance claim.
"""

import json
import os
from pathlib import Path
import tempfile
import unittest
from types import SimpleNamespace as NS
from unittest.mock import patch

import torch

from rtp_llm.models_py.modules.dsv41.backend import V41QuantizedTensor
from rtp_llm.models_py.modules.dsv41.fp8 import _v41_prefill_indexer as native_indexer
from rtp_llm.models_py.modules.dsv41.fp8 import _v41_prefill_metadata as metadata
from rtp_llm.models_py.modules.dsv41.fp8.attention_v41 import select_candidate_blocks
from rtp_llm.models_py.modules.dsv41.ppu import _quant, fused_ops, indexer, prefill
from rtp_llm.models_py.modules.dsv41.ppu.test.test_fused_ops import _ROPE_EDGES
from rtp_llm.models_py.modules.dsv41.ppu.test.test_local_prefill_fused import assert_bytes


BOUNDS_ENV = "RTP_V41_PPU_CP8_DENSE_INDEX_BOUNDS"
ROPE_ENV = "RTP_V41_PPU_FUSED_BF16_ROPE"


def _score(q, k, weights, positions, ratio, flag):
    with patch.dict(os.environ, {BOUNDS_ENV: str(int(flag))}):
        return indexer.dense_fp4_index_score(q, k, weights, positions, ratio)


def _rotate(value, frequencies, inverse, flag):
    with patch.dict(os.environ, {ROPE_ENV: str(int(flag))}):
        return prefill._rope_tail(value, frequencies, inverse)


def _selection(logits, positions, ratio):
    # Deliberately keep the original Torch algorithm, ordering and tie behavior.
    scores, ids = logits.topk(min(512, logits.shape[1]), dim=-1)
    tokens = torch.where(torch.isfinite(scores), ids, -1).int()
    # The candidate helper expects a physical visible count. The bounds
    # canaries also include synthetic positions beyond the allocated bank.
    visible = ((positions + 1) // ratio).clamp(0, logits.shape[1])
    blocks = select_candidate_blocks(logits, visible, 8, 2048)
    return tokens, blocks


def _native_inputs(rows, heads, width, ratio):
    query = torch.randn(rows, heads, 128, device="cuda", dtype=torch.bfloat16)
    keys = torch.randn(width, 128, device="cuda", dtype=torch.bfloat16)
    query[0].zero_()  # Many exact BF16 ties: preserve Torch's selected slots.
    qp, qs = _quant.quantize_indexer_q(query)
    kp, ks = _quant.quantize_indexer_k_rows(keys)
    edge = [-3, -1, 0, 1, ratio * 511 - 1, ratio * 512 - 1,
            ratio * width - 1, 2**40]
    # Cached full-domain rows interleaved with short/empty and clamped bounds.
    values = (edge + [126976 + row * 37 for row in range(rows)])[:rows]
    backing = torch.empty(rows * 2, device="cuda", dtype=torch.int64)
    positions = backing[::2]
    positions.copy_(torch.tensor(values, device="cuda", dtype=torch.int64))
    weights = torch.randn(rows, heads, device="cuda") / (heads * 128) ** 0.5
    return (V41QuantizedTensor(qp, qs, "fp4_ue8m0"),
            V41QuantizedTensor(kp, ks, "fp4_ue8m0"), weights, positions)


def _rope_inputs(rows, dimension, *, heads=None, half=32):
    shape = (rows, dimension) if heads is None else (rows, heads, dimension)
    value = torch.randn(shape, device="cuda", dtype=torch.bfloat16)
    angles = torch.randn(rows, half, device="cuda")
    frequencies = torch.polar(torch.ones_like(angles), angles).contiguous()
    parts = torch.view_as_real(frequencies)
    start = dimension - half * 2
    for pair, (row, head, real, imag, cosine, sine, _, _) in enumerate(_ROPE_EDGES):
        if pair >= half or row >= rows:
            continue
        address = (row, slice(start + pair * 2, start + pair * 2 + 2))
        if heads is not None:
            address = (row, head % heads, address[-1])
        value[address] = torch.tensor([real, imag], device="cuda", dtype=torch.bfloat16)
        parts[row, pair] = torch.tensor([cosine, sine], device="cuda")
    # Every sign combination of zero complex operands, including NoPE channels.
    if rows >= 16:
        for signs in range(16):
            row = rows - 16 + signs
            operands = [-0.0 if signs & (1 << bit) else 0.0 for bit in range(4)]
            value[row, ..., start:start + 2] = torch.tensor(operands[:2], device="cuda", dtype=torch.bfloat16)
            parts[row, 0] = torch.tensor(operands[2:], device="cuda")
            if start:
                value[row, ..., 0] = operands[0]
    return value, frequencies


def _trace_events(profile, label):
    artifact_root = os.environ.get("TEST_UNDECLARED_OUTPUTS_DIR")
    with tempfile.TemporaryDirectory(prefix="local-prefill-operators-") as temporary:
        directory = Path(artifact_root) if artifact_root else Path(temporary)
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / f"local-prefill-{label}.trace.json"
        profile.export_chrome_trace(str(path))
        return json.loads(path.read_text())["traceEvents"]


class LocalPrefillOperatorFlagsCPU(unittest.TestCase):
    def test_bounds_flag_cpu_metadata_retains_original_fallback_and_topk(self):
        positions = torch.tensor([-3, -1, 0, 1, 1023, 2047, 2**40])
        q, k = NS(values=torch.empty(7, 32, 64), scales=None), NS(values=torch.empty(2048, 64), scales=None)
        raw = torch.arange(2048, dtype=torch.float32).remainder(17).bfloat16().expand(7, -1).clone()
        for ratio in (1, 2):
            with self.subTest(ratio=ratio), patch.object(native_indexer, "score_indexer_chunk", side_effect=lambda *a, **kw: raw.clone()):
                with patch.object(metadata, "try_score_bounds", wraps=metadata.try_score_bounds) as admission:
                    old = _score(q, k, None, positions, ratio, False)
                    admission.assert_not_called()
                    actual = _score(q, k, None, positions, ratio, True)
                    admission.assert_called_once()
                assert_bytes(self, actual, old)
                for left, right in zip(_selection(actual, positions, ratio), _selection(old, positions, ratio)):
                    assert_bytes(self, left, right)
        self.assertIsNone(metadata.try_score_bounds(positions, 2048, 2))

    def test_rope_cpu_unsupported_and_disabled_flag_retain_exact_reference(self):
        for dtype in (torch.bfloat16, torch.float32):
            value = torch.randn(17, 128, dtype=dtype)
            value[-1, -2:] = torch.tensor([-0.0, 0.0], dtype=dtype)
            angles = torch.randn(17, 32)
            frequencies = torch.polar(torch.ones_like(angles), angles)
            for inverse in (False, True):
                with self.subTest(dtype=dtype, inverse=inverse), patch.object(
                    fused_ops, "try_rope_tail", wraps=fused_ops.try_rope_tail,
                ) as admission:
                    expected = _rotate(value, frequencies, inverse, False)
                    admission.assert_not_called()
                    actual = _rotate(value, frequencies, inverse, True)
                    admission.assert_called_once()
                    assert_bytes(self, actual, expected)
                    self.assertNotEqual(actual.data_ptr(), value.data_ptr())

    def test_existing_shape_errors_and_selected_runtime_errors_propagate(self):
        value, frequencies = torch.empty(2, 128), torch.empty(3, 32, dtype=torch.complex64)
        for flag in (False, True):
            with self.assertRaisesRegex(ValueError, "freqs rows"):
                _rotate(value, frequencies, False, flag)
        with patch.object(fused_ops, "try_rope_tail", side_effect=RuntimeError("selected rope launch")):
            with self.assertRaisesRegex(RuntimeError, "selected rope launch"):
                _rotate(value, frequencies[:2], False, True)
        with patch.object(native_indexer, "score_indexer_chunk", side_effect=RuntimeError("selected score launch")):
            with self.assertRaisesRegex(RuntimeError, "selected score launch"):
                _score(NS(values=torch.empty(2, 32, 64), scales=None),
                       NS(values=torch.empty(2048, 64), scales=None), None,
                       torch.arange(2), 2, True)


@unittest.skipUnless(os.environ.get("RTP_V41_TEST_LOCAL_WORKSPACE_DEVICE") == "1", "explicit PPU device test")
class LocalPrefillOperatorFlagsDevice(unittest.TestCase):
    def setUp(self):
        self.assertTrue(_quant.is_ppu(), "operator qualification requires an actual ZW PPU")
        torch.manual_seed(20261112)

    @torch.inference_mode()
    def test_native_dense_bounds_exact_wide_logits_poisoned_tails_and_original_topk(self):
        original = native_indexer.score_indexer_chunk
        for ratio, heads, width, padded in ((2, 64, 65536, False), (1, 32, 131072, False),
                                             (1, 64, 65536, False), (2, 32, 131072, False),
                                             (1, 32, 2048, True), (2, 64, 2048, True)):
            with self.subTest(ratio=ratio, heads=heads, width=width, padded=padded):
                q, k, weights, positions = _native_inputs(32, heads, width, ratio)
                seen = []
                padded_outputs = []

                def poison(*args, **kwargs):
                    logits = original(*args, **kwargs)
                    self.assertEqual(logits.dtype, torch.bfloat16)
                    if padded:
                        backing = logits.new_full((logits.shape[0], width + 17), 173)
                        backing[:, :width].copy_(logits)
                        logits = backing[:, :width]
                        padded_outputs.append(backing)
                    ends = kwargs["bounds"][1]
                    tail = torch.arange(width, device="cuda")[None] >= ends[:, None]
                    logits.masked_fill_(tail, float("nan"))
                    logits[:, ::2].masked_fill_(tail[:, ::2], float("inf"))
                    seen.append((logits.stride(), logits.untyped_storage()))
                    return logits

                with patch.object(native_indexer, "score_indexer_chunk", new=poison):
                    old = _score(q, k, weights, positions, ratio, False)
                    actual = _score(q, k, weights, positions, ratio, True)
                self.assertEqual(len(seen), 2)
                self.assertEqual(old.shape, (32, width))
                if padded:
                    self.assertEqual(actual.stride(0), width + 17)
                    for backing in padded_outputs:
                        assert_bytes(self, backing[:, width:], torch.full_like(backing[:, width:], 173))
                assert_bytes(self, actual, old)
                ends = ((positions + 1) // ratio).clamp(0, width)
                tail = torch.arange(width, device="cuda")[None] >= ends[:, None]
                self.assertTrue(bool(torch.isneginf(actual[tail]).all()))
                for left, right in zip(_selection(actual, positions, ratio), _selection(old, positions, ratio)):
                    assert_bytes(self, left, right)

    @torch.inference_mode()
    def test_actual_dense_profile_executes_bounds_and_tail_mask_without_full_causal_mask(self):
        q, k, weights, positions = _native_inputs(32, 64, 65536, 2)
        expected = _score(q, k, weights, positions, 2, False)
        _score(q, k, weights, positions, 2, True)
        torch.cuda.synchronize()
        with patch.object(metadata, "try_score_bounds", wraps=metadata.try_score_bounds) as admission:
            with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                                   torch.profiler.ProfilerActivity.CUDA]) as profile:
                actual = _score(q, k, weights, positions, 2, True)
                torch.cuda.synchronize()
        admission.assert_called_once()
        assert_bytes(self, actual, expected)
        events = _trace_events(profile, "dense-bounds")
        kernels = [event["name"] for event in events if event.get("cat") == "kernel"]
        for name in ("_prefill_bounds_kernel", "_mask_tail_kernel"):
            self.assertEqual(sum(name in kernel for kernel in kernels), 1, name)
        self.assertTrue(any("attention_mqa_logits_fp4" in name for name in kernels), "native FP4 scorer must execute")
        cpu = [event["name"] for event in events if event.get("cat") == "cpu_op"]
        self.assertNotIn("aten::ge", cpu)
        self.assertNotIn("aten::masked_fill_", cpu)

    @torch.inference_mode()
    def test_rope_forward_inverse_actual_local_shapes_edges_and_signed_zero_are_bit_exact(self):
        for rows in (4096, 140):
            for dimension, heads, half in ((128, 64, 32), (512, 64, 32),
                                            (128, None, 64), (512, None, 32)):
                with self.subTest(rows=rows, dimension=dimension, heads=heads, half=half):
                    value, frequencies = _rope_inputs(rows, dimension, heads=heads, half=half)
                    saved = value.clone(), frequencies.clone()
                    for inverse in (False, True):
                        expected = _rotate(value, frequencies, inverse, False)
                        with patch.object(fused_ops, "try_rope_tail", wraps=fused_ops.try_rope_tail) as admission:
                            actual = _rotate(value, frequencies, inverse, True)
                        admission.assert_called_once()
                        assert_bytes(self, actual, expected)
                        self.assertNotEqual(actual.data_ptr(), value.data_ptr())
                        del actual, expected
                    assert_bytes(self, value, saved[0])
                    assert_bytes(self, frequencies, saved[1])
        value = torch.empty(0, 64, 512, device="cuda", dtype=torch.bfloat16)
        frequencies = torch.empty(0, 32, device="cuda", dtype=torch.complex64)
        for inverse in (False, True):
            assert_bytes(self, _rotate(value, frequencies, inverse, True),
                         _rotate(value, frequencies, inverse, False))

    @torch.inference_mode()
    def test_rope_unsupported_layouts_use_exact_existing_fallback(self):
        value, frequencies = _rope_inputs(140, 128)
        strided = torch.empty(140, 256, device="cuda", dtype=torch.bfloat16)[:, ::2]
        strided.copy_(value)
        strided_freq = torch.empty(140, 64, device="cuda", dtype=torch.complex64)[:, ::2]
        strided_freq.copy_(frequencies)
        cases = ((strided, frequencies), (value.float(), frequencies),
                 (value, strided_freq), (value, frequencies.conj()),
                 (torch.randn(140, 256, device="cuda", dtype=torch.bfloat16), frequencies))
        for values, freqs in cases:
            for inverse in (False, True):
                with self.subTest(shape=tuple(values.shape), dtype=values.dtype,
                                  stride=values.stride(), conjugate=freqs.is_conj(), inverse=inverse):
                    self.assertIsNone(fused_ops.try_rope_tail(values, freqs, inverse))
                    assert_bytes(self, _rotate(values, freqs, inverse, True),
                                 _rotate(values, freqs, inverse, False))

    @torch.inference_mode()
    def test_actual_rope_profile_executes_single_existing_tail_kernel(self):
        value, frequencies = _rope_inputs(140, 512, heads=64)
        expected = _rotate(value, frequencies, True, False)
        _rotate(value, frequencies, True, True)
        torch.cuda.synchronize()
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                              torch.profiler.ProfilerActivity.CUDA]) as profile:
            actual = _rotate(value, frequencies, True, True)
            torch.cuda.synchronize()
        assert_bytes(self, actual, expected)
        events = _trace_events(profile, "bf16-rope")
        kernels = [event["name"] for event in events if event.get("cat") == "kernel"]
        self.assertEqual(len(kernels), 1, kernels)
        self.assertIn("fused_rope_tail", kernels[0])


if __name__ == "__main__":
    unittest.main()
