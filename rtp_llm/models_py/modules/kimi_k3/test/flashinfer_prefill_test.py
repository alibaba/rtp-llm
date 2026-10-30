"""Real FP8 GPU coverage for K3's pinned FlashInfer ragged prefill API."""

import builtins
import math
import unittest
from importlib.metadata import PackageNotFoundError
from types import SimpleNamespace
from unittest import mock

import torch

from rtp_llm.models_py.modules.kimi_k3 import flashinfer_prefill
from rtp_llm.models_py.modules.kimi_k3.flashinfer_prefill import (
    FlashinferMlaPrefill,
    _get_workspace,
    _workspace_size,
)
from rtp_llm.models_py.triton_kernels.common.activation import quantize_qkv_fp8
from rtp_llm.models_py.triton_kernels.mla.lse_merge import (
    _merge_states,
    merge_attention_states_in_place,
)


def _indptr(lengths):
    return torch.tensor(
        [0, *torch.tensor(lengths).cumsum(0).tolist()],
        dtype=torch.int32,
        device="cuda",
    )


def _reference(
    q, k, v, q_lens, kv_lens, scale, causal, *, fp8_probability_rounding=True
):
    """Independent math on actual E4M3 operands, with an explicit P contract.

    The small-case oracle includes the pinned GEN kernel's *measured* rounding:
    SM100 and SM103 basis-V probes (H12/H24, dense/causal, three scales) uniquely
    matched E4M3(448 * exp(score - max)) / 448 with the unrounded FP32 sum as
    denominator, followed by BF16 output. The public fmhaReduction.cu reducer
    supports the 448 scale; it does not expose the cubin's main-kernel math.
    This empirical single-tile contract is deliberately confined to the small
    accuracy cases. Multi-tile coverage uses the independent FP32 ideal path.
    """
    if fp8_probability_rounding and (max(q_lens) > 4 or max(kv_lens) > 9):
        raise ValueError("The measured FP8 P oracle is limited to Q<=4 and K<=9")
    outputs, lses = [], []
    q_start = kv_start = 0
    for q_len, kv_len in zip(q_lens, kv_lens):
        qr = q[q_start : q_start + q_len].float().transpose(0, 1)
        kr = k[kv_start : kv_start + kv_len].float().transpose(0, 1)
        vr = v[kv_start : kv_start + kv_len].float().transpose(0, 1)
        if not kv_len:
            outputs.append(torch.zeros_like(qr[..., :128]).transpose(0, 1))
            lses.append(torch.full((q_len, q.shape[1]), -math.inf, device=q.device))
        else:
            scores = torch.matmul(qr, kr.transpose(1, 2)) * scale
            if causal:
                rows = torch.arange(q_len, device=q.device) + kv_len - q_len
                columns = torch.arange(kv_len, device=q.device)
                scores.masked_fill_(columns[None, :] > rows[:, None], -math.inf)
            if fp8_probability_rounding:
                weights = (scores - scores.amax(-1, keepdim=True)).exp()
                denominator = weights.sum(-1, keepdim=True)
                rounded_weights = (448.0 * weights).to(torch.float8_e4m3fn).float()
                output = torch.matmul(rounded_weights / 448.0, vr) / denominator
                # Each native partial crosses this BF16 boundary before merge.
                output = output.to(torch.bfloat16).float()
            else:
                output = torch.matmul(scores.softmax(-1), vr)
            outputs.append(output.transpose(0, 1))
            lses.append(scores.logsumexp(-1).transpose(0, 1))
        q_start += q_len
        kv_start += kv_len
    return torch.cat(outputs), torch.cat(lses)


def _merge_reference_states(first, second):
    """Merge independently rounded partials, including the final BF16 store."""
    first_output, first_lse = first
    second_output, second_lse = second
    combined_lse = torch.logaddexp(first_lse, second_lse)
    first_weight = (first_lse - combined_lse).exp()[..., None]
    second_weight = (second_lse - combined_lse).exp()[..., None]
    output = first_output * first_weight + second_output * second_weight
    return output.to(torch.bfloat16).float(), combined_lse


class FlashinferPrefillTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not torch.cuda.is_available():
            raise RuntimeError(
                "This test requires a real CUDA GPU; CPU mocks are not a GPU gate"
            )
        if torch.cuda.get_device_capability() not in ((10, 0), (10, 3)):
            raise RuntimeError("Pinned TRTLLM prefill requires SM100 or SM103")

    def setUp(self):
        torch.manual_seed(314159)
        self.scale = 192**-0.5
        previous = torch.backends.cuda.matmul.allow_tf32
        self.addCleanup(setattr, torch.backends.cuda.matmul, "allow_tf32", previous)
        torch.backends.cuda.matmul.allow_tf32 = False

    def _operands(self, heads, q_lens, kv_lens):
        q = (
            torch.randn(sum(q_lens), heads, 192, device="cuda", dtype=torch.bfloat16)
            * 0.5
        )
        k = (
            torch.randn(sum(kv_lens), heads, 192, device="cuda", dtype=torch.bfloat16)
            * 0.5
        )
        v = torch.randn(sum(kv_lens), heads, 128, device="cuda", dtype=torch.bfloat16)
        return quantize_qkv_fp8(q, k, v)

    def _check_output(self, actual, expected):
        self.assertEqual(actual.dtype, torch.bfloat16)
        self.assertTrue(torch.isfinite(actual).all().item())
        # Small-case references include the measured intermediate FP8 P cast;
        # multi-tile references retain ideal FP32 math with the same tolerances.
        torch.testing.assert_close(actual.float(), expected, atol=0.025, rtol=0.05)

    def test_causal_ragged_strided_value(self):
        for heads in (12, 24):
            with self.subTest(heads=heads):
                q_lens, kv_lens = (3, 4), (7, 9)
                q, k, value = self._operands(heads, q_lens, kv_lens)
                packed = torch.empty(
                    (sum(kv_lens), heads, 320), device="cuda", dtype=value.dtype
                )
                v = packed[..., 192:]
                v.copy_(value)
                self.assertFalse(v.is_contiguous())
                params = SimpleNamespace(
                    q_tokens=sum(q_lens),
                    kv_tokens=sum(kv_lens),
                    q_lens_host=q_lens,
                    kv_lens_d=torch.tensor(kv_lens, dtype=torch.int32, device="cuda"),
                    qo_indptr_d=_indptr(q_lens),
                    prefill_ragged_kv_len_indptr_d=_indptr(kv_lens),
                    max_q=max(q_lens),
                    max_k=max(kv_lens),
                )
                output = FlashinferMlaPrefill(heads, self.scale).run(q, k, v, params)
                expected, _ = _reference(q, k, v, q_lens, kv_lens, self.scale, True)
                self._check_output(output, expected)

    def test_partial_natural_lse_and_prefix_merge(self):
        for heads in (12, 24):
            with self.subTest(heads=heads):
                q_lens, kv_lens = (2, 3), (7, 8)
                first_lens = (3, 4)
                second_lens = tuple(k - a for k, a in zip(kv_lens, first_lens))
                q, k, v = self._operands(heads, q_lens, kv_lens)
                k_parts, v_parts = [[], []], [[], []]
                start = 0
                for length, first in zip(kv_lens, first_lens):
                    for index, section in enumerate(
                        (
                            slice(start, start + first),
                            slice(start + first, start + length),
                        )
                    ):
                        k_parts[index].append(k[section])
                        v_parts[index].append(v[section])
                    start += length
                wrapper = FlashinferMlaPrefill(heads, self.scale)
                partials = []
                reference_partials = []
                for index, lengths in enumerate((first_lens, second_lens)):
                    ki, vi = torch.cat(k_parts[index]), torch.cat(v_parts[index])
                    output, lse = wrapper.run_partial(
                        q,
                        ki,
                        vi,
                        qo_indptr=_indptr(q_lens),
                        kv_indptr=_indptr(lengths),
                        max_q=max(q_lens),
                        max_k=max(lengths),
                        causal=False,
                    )
                    expected, expected_lse = _reference(
                        q, ki, vi, q_lens, lengths, self.scale, False
                    )
                    self._check_output(output, expected)
                    self.assertEqual(lse.dtype, torch.float32)
                    torch.testing.assert_close(lse, expected_lse, atol=0.01, rtol=0.005)
                    partials.append((output, lse))
                    reference_partials.append((expected, expected_lse))
                output, lse = partials[0]
                merge_attention_states_in_place(output, lse, *partials[1])
                # Quantizing each segment's P is not equivalent to quantizing
                # the concatenated KV once. Preserve both native BF16 stores.
                expected, expected_lse = _merge_reference_states(*reference_partials)
                self._check_output(output, expected)
                torch.testing.assert_close(lse, expected_lse, atol=0.01, rtol=0.005)

    def test_transposed_partial_lse_reuses_kernel_across_query_lengths(self):
        compiled = None
        for rows in (5, 17, 33):
            output = torch.randn(rows, 12, 128, device="cuda")
            partial = torch.randn_like(output, dtype=torch.bfloat16)
            lse = torch.randn(rows, 12, device="cuda")
            partial_lse = torch.randn(12, rows, device="cuda").T
            expected_lse = torch.logaddexp(lse, partial_lse)
            expected = output * (lse - expected_lse).exp()[:, :, None]
            expected += partial.float() * (partial_lse - expected_lse).exp()[:, :, None]
            merge_attention_states_in_place(output, lse, partial, partial_lse)
            torch.testing.assert_close(output, expected, atol=1e-5, rtol=1e-5)
            torch.testing.assert_close(lse, expected_lse, atol=1e-5, rtol=1e-5)
            count = len(_merge_states.device_caches[torch.cuda.current_device()][0])
            if compiled is not None:
                self.assertEqual(count, compiled)
            compiled = count

    def test_empty_segment_identity(self):
        heads = 12
        q, k, v = self._operands(heads, (2, 3), (0, 0))
        wrapper = FlashinferMlaPrefill(heads, self.scale)
        output, lse = wrapper.run_partial(
            q,
            k,
            v,
            qo_indptr=_indptr((2, 3)),
            kv_indptr=_indptr((0, 0)),
            max_q=3,
            max_k=0,
            causal=False,
        )
        self.assertEqual(tuple(output.shape), (5, heads, 128))
        self.assertEqual(torch.count_nonzero(output).item(), 0)
        self.assertTrue(torch.isneginf(lse).all().item())
        output, lse = wrapper.run_partial(
            q[:0],
            k,
            v,
            qo_indptr=_indptr((0, 0)),
            kv_indptr=_indptr((0, 0)),
            max_q=0,
            max_k=0,
            causal=False,
        )
        self.assertEqual(tuple(output.shape), (0, heads, 128))
        self.assertEqual(tuple(lse.shape), (0, heads))

    def test_mixed_empty_ragged_kv(self):
        q_lens, kv_lens, heads = (0, 2, 0, 3), (0, 0, 0, 5), 12
        q, k, v = self._operands(heads, q_lens, kv_lens)
        output, lse = FlashinferMlaPrefill(heads, self.scale).run_partial(
            q,
            k,
            v,
            qo_indptr=_indptr(q_lens),
            kv_indptr=_indptr(kv_lens),
            max_q=3,
            max_k=5,
            causal=False,
        )
        expected, expected_lse = _reference(q, k, v, q_lens, kv_lens, self.scale, False)
        self.assertEqual(torch.count_nonzero(output[:2]).item(), 0)
        self.assertTrue(torch.isneginf(lse[:2]).all().item())
        self._check_output(output, expected)
        torch.testing.assert_close(lse, expected_lse, atol=0.01, rtol=0.005)
        params = SimpleNamespace(
            q_tokens=sum(q_lens),
            kv_tokens=sum(kv_lens),
            q_lens_host=q_lens,
            kv_lens_d=torch.tensor(kv_lens, dtype=torch.int32, device="cuda"),
            qo_indptr_d=_indptr(q_lens),
            prefill_ragged_kv_len_indptr_d=_indptr(kv_lens),
            max_q=max(q_lens),
            max_k=max(kv_lens),
        )
        output = FlashinferMlaPrefill(heads, self.scale).run(q, k, v, params)
        self.assertEqual(torch.count_nonzero(output[:2]).item(), 0)
        expected, _ = _reference(q, k, v, q_lens, kv_lens, self.scale, True)
        self._check_output(output, expected)

    def test_multi_tile_ragged_tail_ideal_reference(self):
        """Check ragged tails spanning multiple native KV tiles."""
        q_lens, kv_lens = (129, 257), (257, 513)
        for heads in (12, 24):
            with self.subTest(heads=heads):
                q, k, v = self._operands(heads, q_lens, kv_lens)
                qo_indptr, kv_indptr = _indptr(q_lens), _indptr(kv_lens)
                params = SimpleNamespace(
                    q_tokens=sum(q_lens),
                    kv_tokens=sum(kv_lens),
                    q_lens_host=q_lens,
                    kv_lens_d=torch.tensor(kv_lens, dtype=torch.int32, device="cuda"),
                    qo_indptr_d=qo_indptr,
                    prefill_ragged_kv_len_indptr_d=kv_indptr,
                    max_q=max(q_lens),
                    max_k=max(kv_lens),
                )
                partial_args = dict(
                    qo_indptr=qo_indptr,
                    kv_indptr=kv_indptr,
                    max_q=max(q_lens),
                    max_k=max(kv_lens),
                    causal=True,
                )
                wrapper = FlashinferMlaPrefill(heads, self.scale)
                output = wrapper.run(q, k, v, params)
                partial, lse = wrapper.run_partial(q, k, v, **partial_args)
                expected, expected_lse = _reference(
                    q,
                    k,
                    v,
                    q_lens,
                    kv_lens,
                    self.scale,
                    True,
                    fp8_probability_rounding=False,
                )
                self._check_output(output, expected)
                self._check_output(partial, expected)
                self.assertEqual(lse.dtype, torch.float32)
                self.assertTrue(torch.isfinite(lse).all().item())
                torch.testing.assert_close(lse, expected_lse, atol=0.01, rtol=0.005)

    def test_shared_workspace_and_capture_growth_rejection(self):
        device = torch.device("cuda", torch.cuda.current_device())
        first = _get_workspace(device, 7, 12, batch_size=1, max_q=7, return_lse=False)
        self.assertIs(
            first,
            _get_workspace(device, 3, 24, batch_size=1, max_q=3, return_lse=False),
        )
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            with self.assertRaisesRegex(RuntimeError, "grow workspace.*capture"):
                _get_workspace(
                    device,
                    first.numel(),
                    24,
                    batch_size=1,
                    max_q=first.numel(),
                    return_lse=False,
                )

    def test_ragged_lse_workspace_uses_tile_padded_query_rows(self):
        # One long request plus three one-token requests. Each request reserves
        # the longest query rounded up to two 256-token tiles.
        self.assertEqual(
            _workspace_size(260, 12, batch_size=4, max_q=257, return_lse=True),
            129 * 1024**2 + 4 * 512 * 12 * 8,
        )
        self.assertEqual(
            _workspace_size(260, 12, batch_size=4, max_q=257, return_lse=False),
            128 * 1024**2 + 25_088,
        )

    def test_importable_backend_without_distribution_metadata_runs_prefill(self):
        flashinfer_prefill._load_backend.cache_clear()
        self.addCleanup(flashinfer_prefill._load_backend.cache_clear)
        with mock.patch.object(
            flashinfer_prefill,
            "version",
            side_effect=PackageNotFoundError("flashinfer-python"),
        ):
            adapter = FlashinferMlaPrefill(12, self.scale)
        indptr = torch.tensor([0, 1], dtype=torch.int32, device="cuda")
        params = SimpleNamespace(
            q_tokens=1,
            kv_tokens=1,
            qo_indptr_d=indptr,
            kv_lens_d=torch.ones(1, dtype=torch.int32, device="cuda"),
            prefill_ragged_kv_len_indptr_d=indptr,
            q_lens_host=[1],
            max_q=1,
            max_k=1,
        )
        q = torch.zeros((1, 12, 192), dtype=torch.float8_e4m3fn, device="cuda")
        k = torch.zeros_like(q)
        v = torch.full((1, 12, 128), 2, dtype=torch.float8_e4m3fn, device="cuda")
        actual = adapter.run(q, k, v, params)
        self.assertEqual(actual.shape, (1, 12, 128))
        self.assertEqual(actual.dtype, torch.bfloat16)
        torch.testing.assert_close(actual, torch.full_like(actual, 2), rtol=0, atol=0)

    def test_backend_import_error_still_fails_before_version_lookup(self):
        flashinfer_prefill._load_backend.cache_clear()
        self.addCleanup(flashinfer_prefill._load_backend.cache_clear)
        original_import = builtins.__import__
        failure = ImportError("backend dependency is unavailable")

        def importing(name, *args, **kwargs):
            if name == "flashinfer.prefill":
                raise failure
            return original_import(name, *args, **kwargs)

        with mock.patch("builtins.__import__", side_effect=importing):
            with mock.patch.object(flashinfer_prefill, "version") as lookup:
                with self.assertRaisesRegex(
                    ImportError, "pinned FlashInfer ragged TRTLLM API"
                ) as error:
                    flashinfer_prefill.FlashinferMlaPrefill(12, 192**-0.5)
        self.assertIs(error.exception.__cause__, failure)
        lookup.assert_not_called()


if __name__ == "__main__":
    unittest.main()
