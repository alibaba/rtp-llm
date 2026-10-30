"""Real FP8 GPU coverage for K3's pinned FlashInfer ragged prefill API."""

import hashlib
import json
import math
import os
import unittest
from importlib.metadata import version
from pathlib import Path
from types import SimpleNamespace

import torch

from rtp_llm.models_py.modules.kimi_k3.flashinfer_prefill import (
    FlashinferMlaPrefill,
    _get_workspace,
)
from rtp_llm.models_py.triton_kernels.common.activation import quantize_qkv_fp8
from rtp_llm.models_py.triton_kernels.mla.lse_merge import (
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
        """Cover multiple native KV tiles without extrapolating the P probe."""
        directory = Path(os.environ["TEST_UNDECLARED_OUTPUTS_DIR"])
        directory.mkdir(parents=True, exist_ok=True)
        q_lens, kv_lens = (129, 257), (257, 513)
        records = []
        properties = torch.cuda.get_device_properties()
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
                wrapper.run(q, k, v, params)
                wrapper.run_partial(q, k, v, **partial_args)
                torch.cuda.synchronize()
                name = f"prefill_multi_tile_h{heads}"
                with torch.profiler.profile(
                    activities=[
                        torch.profiler.ProfilerActivity.CPU,
                        torch.profiler.ProfilerActivity.CUDA,
                    ]
                ) as profiler:
                    output = wrapper.run(q, k, v, params)
                    partial, lse = wrapper.run_partial(q, k, v, **partial_args)
                    torch.cuda.synchronize()
                trace_file = directory / (name + ".trace.json")
                profiler.export_chrome_trace(str(trace_file))
                kernels = [
                    event.name
                    for event in profiler.events()
                    if event.device_type == torch.autograd.DeviceType.CUDA
                ]
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
                tensor_file = directory / (name + ".pt")
                torch.save(
                    {
                        "q_codes": q.view(torch.uint8).cpu(),
                        "k_codes": k.view(torch.uint8).cpu(),
                        "v_codes": v.view(torch.uint8).cpu(),
                        "operand_dtype": str(torch.float8_e4m3fn),
                        "output": output.cpu(),
                        "partial_output": partial.cpu(),
                        "natural_lse": lse.cpu(),
                        "ideal_fp32_output": expected.cpu(),
                        "ideal_natural_lse": expected_lse.cpu(),
                    },
                    tensor_file,
                )
                records.append(
                    {
                        "heads": heads,
                        "q_lens": q_lens,
                        "kv_lens": kv_lens,
                        "scale": self.scale,
                        "causal": True,
                        "min_attended_keys_per_row": 129,
                        "reference": (
                            "FP32 ideal on actual unit-scale E4M3 inputs; "
                            "no single-tile P oracle"
                        ),
                        "output_atol": 0.025,
                        "output_rtol": 0.05,
                        "natural_lse_atol": 0.01,
                        "natural_lse_rtol": 0.005,
                        "output_max_abs": float(
                            (output.float() - expected).abs().max()
                        ),
                        "partial_output_max_abs": float(
                            (partial.float() - expected).abs().max()
                        ),
                        "natural_lse_max_abs": float((lse - expected_lse).abs().max()),
                        "output_finite": bool(torch.isfinite(output).all()),
                        "partial_output_finite": bool(torch.isfinite(partial).all()),
                        "natural_lse_finite": bool(torch.isfinite(lse).all()),
                        "trace": trace_file.name,
                        "trace_sha256": hashlib.sha256(
                            trace_file.read_bytes()
                        ).hexdigest(),
                        "tensor_file": tensor_file.name,
                        "tensor_sha256": hashlib.sha256(
                            tensor_file.read_bytes()
                        ).hexdigest(),
                        "kernel_names": kernels,
                    }
                )
                # Persist before assertions so a failed numerical gate retains
                # its full native inputs, outputs, ideal reference, and trace.
                (directory / "prefill-multi-tile-accuracy.json").write_text(
                    json.dumps(
                        {
                            "flashinfer_version": version("flashinfer-python"),
                            "hardware": {
                                "name": properties.name,
                                "compute_capability": (
                                    torch.cuda.get_device_capability()
                                ),
                                "sm_count": properties.multi_processor_count,
                                "torch": torch.__version__,
                                "cuda_runtime": torch.version.cuda,
                            },
                            "records": records,
                        },
                        indent=2,
                    )
                    + "\n"
                )
                self.assertTrue(kernels)
                self._check_output(output, expected)
                self._check_output(partial, expected)
                self.assertEqual(lse.dtype, torch.float32)
                self.assertTrue(torch.isfinite(lse).all().item())
                torch.testing.assert_close(lse, expected_lse, atol=0.01, rtol=0.005)

    def test_probability_contract_probe(self):
        """Record candidate P contracts separately from formal accuracy gates.

        The pinned GEN main kernel is a cubin. Its public reducer documents a
        448 quantization scale but does not expose the main kernel's rounding
        point/denominator. Basis V makes those alternatives observable. R4
        SM100/SM103 runs uniquely matched P448 with the true FP32 denominator;
        the small-case reference records that measured single-tile contract.
        This diagnostic still saves every candidate rather than selecting one
        automatically or making claims about multi-tile/CTA kernels.
        """
        directory = Path(os.environ["TEST_UNDECLARED_OUTPUTS_DIR"])
        directory.mkdir(parents=True, exist_ok=True)
        from flashinfer.artifacts import ArtifactPath, CheckSumHash

        keys = [0, 2, 4, 8, 16, 32, 48, 64]
        records = []
        for heads in (12, 24):
            q = torch.zeros((4, heads, 192), dtype=torch.bfloat16, device="cuda")
            k = torch.zeros((8, heads, 192), dtype=torch.bfloat16, device="cuda")
            v = torch.zeros((8, heads, 128), dtype=torch.bfloat16, device="cuda")
            q[..., 0] = 1
            k[..., 0] = torch.tensor(keys, device="cuda")[:, None]
            v[:, :, :8] = torch.eye(8, device="cuda")[:, None, :]
            v[..., 8] = 1
            q, k, v = quantize_qkv_fp8(q, k, v)
            for multiplier in (0.5, 1.0, 2.0):
                scale = self.scale * multiplier
                wrapper = FlashinferMlaPrefill(heads, scale)
                for causal in (False, True):
                    args = dict(
                        qo_indptr=_indptr((4,)),
                        kv_indptr=_indptr((8,)),
                        max_q=4,
                        max_k=8,
                        causal=causal,
                    )
                    wrapper.run_partial(q, k, v, **args)
                    torch.cuda.synchronize()
                    name = f"prefill_p_round_h{heads}_scale{multiplier}_causal{int(causal)}"
                    with torch.profiler.profile(
                        activities=[
                            torch.profiler.ProfilerActivity.CPU,
                            torch.profiler.ProfilerActivity.CUDA,
                        ]
                    ) as profiler:
                        output, lse = wrapper.run_partial(q, k, v, **args)
                        torch.cuda.synchronize()
                    trace = name + ".trace.json"
                    profiler.export_chrome_trace(str(directory / trace))
                    kernels = [
                        event.name
                        for event in profiler.events()
                        if event.device_type == torch.autograd.DeviceType.CUDA
                    ]
                    self.assertTrue(kernels)
                    self.assertTrue(torch.isfinite(output).all().item())
                    scores = (
                        torch.matmul(
                            q.float().transpose(0, 1),
                            k.float().transpose(0, 1).transpose(1, 2),
                        )
                        * scale
                    )
                    if causal:
                        mask = torch.arange(8, device="cuda")[None, :] > (
                            torch.arange(4, device="cuda")[:, None] + 4
                        )
                        scores.masked_fill_(mask, -math.inf)
                    maximum = scores.max(-1, keepdim=True).values
                    weights = (scores - maximum).exp()
                    denominator = weights.sum(-1, keepdim=True)
                    values = v.float().transpose(0, 1)
                    candidates = {}
                    candidate_lses = {}
                    for p_scale in (1.0, 448.0):
                        rounded = (weights * p_scale).to(
                            torch.float8_e4m3fn
                        ).float() / p_scale
                        for quantized_sum in (False, True):
                            normalization = (
                                rounded.sum(-1, keepdim=True)
                                if quantized_sum
                                else denominator
                            )
                            candidate = torch.matmul(rounded, values) / normalization
                            key = f"exp_p_scale{p_scale}_sum_{'rounded' if quantized_sum else 'fp32'}"
                            candidates[key] = candidate
                            candidate_lses[key] = (
                                maximum.squeeze(-1) + normalization.squeeze(-1).log()
                            )
                    probabilities = weights / denominator
                    candidates["normalized_p_e4m3"] = torch.matmul(
                        probabilities.to(torch.float8_e4m3fn).float(), values
                    )
                    candidates["ideal_fp32"] = torch.matmul(probabilities, values)
                    ideal_lse = scores.logsumexp(-1)
                    candidate_lses["normalized_p_e4m3"] = ideal_lse
                    candidate_lses["ideal_fp32"] = ideal_lse
                    diagnostics = {}
                    rounded_candidates = {}
                    for key, candidate in candidates.items():
                        candidate = candidate.transpose(0, 1).to(torch.bfloat16)
                        rounded_candidates[key] = candidate.cpu()
                        difference = (candidate.float() - output.float()).abs()
                        diagnostics[key] = {
                            "max_abs": float(difference.max()),
                            "mean_abs": float(difference.mean()),
                            "natural_lse_max_abs": float(
                                (candidate_lses[key].transpose(0, 1) - lse).abs().max()
                            ),
                            "observed_weight_columns": candidate[..., :9]
                            .float()
                            .cpu()
                            .tolist(),
                        }
                    tensor_file = directory / (name + ".pt")
                    torch.save(
                        {
                            "q_codes": q.view(torch.uint8).cpu(),
                            "k_codes": k.view(torch.uint8).cpu(),
                            "v_codes": v.view(torch.uint8).cpu(),
                            "operand_dtype": str(torch.float8_e4m3fn),
                            "output": output.cpu(),
                            "natural_lse": lse.cpu(),
                            "scores_fp32": scores.cpu(),
                            "candidate_bf16_outputs": rounded_candidates,
                            "candidate_natural_lse": {
                                key: value.transpose(0, 1).cpu()
                                for key, value in candidate_lses.items()
                            },
                        },
                        tensor_file,
                    )
                    records.append(
                        {
                            "heads": heads,
                            "scale": scale,
                            "scale_multiplier": multiplier,
                            "causal": causal,
                            "Q": 4,
                            "K": 8,
                            "trace": trace,
                            "tensor_file": tensor_file.name,
                            "tensor_sha256": hashlib.sha256(
                                tensor_file.read_bytes()
                            ).hexdigest(),
                            "trace_sha256": hashlib.sha256(
                                (directory / trace).read_bytes()
                            ).hexdigest(),
                            "parameters": {
                                "q_shape": list(q.shape),
                                "k_shape": list(k.shape),
                                "v_shape": list(v.shape),
                                "qo_indptr": [0, 4],
                                "kv_indptr": [0, 8],
                                "bmm1_scale": scale,
                                "bmm2_scale": 1.0,
                                "o_sf_scale": 1.0,
                                "window_left": -1,
                                "enable_pdl": False,
                                "max_q_len": 4,
                                "max_kv_len": 8,
                                "return_lse": True,
                                "is_causal": causal,
                            },
                            "kernel_names": kernels,
                            "observed_weight_columns": output[..., :9]
                            .float()
                            .cpu()
                            .tolist(),
                            "natural_lse": lse.cpu().tolist(),
                            "ideal_natural_lse": scores.logsumexp(-1)
                            .transpose(0, 1)
                            .cpu()
                            .tolist(),
                            "candidates": diagnostics,
                        }
                    )
        properties = torch.cuda.get_device_properties()
        (directory / "prefill-probability-rounding-probe.json").write_text(
            json.dumps(
                {
                    "diagnostic_only_not_an_accuracy_gate": True,
                    "scope": "Q4/K8 observations only; no multi-tile or multi-CTA contract is inferred",
                    "flashinfer_version": version("flashinfer-python"),
                    "trtllm_gen_fmha_artifact": ArtifactPath.TRTLLM_GEN_FMHA,
                    "trtllm_gen_fmha_checksums_sha256": CheckSumHash.TRTLLM_GEN_FMHA,
                    "hardware": {
                        "name": properties.name,
                        "compute_capability": torch.cuda.get_device_capability(),
                        "sm_count": properties.multi_processor_count,
                        "torch": torch.__version__,
                        "cuda_runtime": torch.version.cuda,
                    },
                    "inputs": {
                        "q_feature0": 1,
                        "k_feature0": keys,
                        "v_columns_0_to_7": "identity8",
                        "v_column8": "ones",
                        "dtype": "unit-scale E4M3 Q/K/V; BF16 output",
                    },
                    "records": records,
                },
                indent=2,
            )
            + "\n"
        )

    def test_shared_workspace_and_capture_growth_rejection(self):
        device = torch.device("cuda", torch.cuda.current_device())
        first = _get_workspace(device, 7, 12)
        self.assertIs(first, _get_workspace(device, 3, 24))
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            with self.assertRaisesRegex(RuntimeError, "grow workspace.*capture"):
                _get_workspace(device, first.numel(), 24)


if __name__ == "__main__":
    unittest.main()
