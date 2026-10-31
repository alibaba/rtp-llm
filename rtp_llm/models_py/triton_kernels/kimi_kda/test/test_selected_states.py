"""KDA snapshot regression tests and opt-in actual CUDA kernel timing.

RTP_KDA_ORIGINAL_SOURCE selects the frozen pre-change low-level source.
PERF_JSON runs the Kimi-Linear-48B TP1 benchmark after correctness tests.
"""

import hashlib
import importlib.util
import json
import math
import os
import random
import statistics
import sys
import unittest
from pathlib import Path
from unittest import mock

import torch
import torch.nn.functional as F

from rtp_llm.models_py.triton_kernels.fla.cumsum import chunk_local_cumsum
from rtp_llm.models_py.triton_kernels.fla.utils import RCP_LN2
from rtp_llm.models_py.triton_kernels.kimi_kda.chunk import chunk_kda
from rtp_llm.models_py.triton_kernels.kimi_kda.chunk_delta_h import (
    chunk_gated_delta_rule_fwd_h,
)
from rtp_llm.models_py.triton_kernels.kimi_kda.chunk_intra import chunk_kda_fwd_intra


def load_baseline(
    env_name="RTP_KDA_ORIGINAL_SOURCE",
    module_name="kda_frozen_original",
    optional=False,
):
    source = os.environ.get(env_name)
    if not source:
        if optional:
            return None, None
        return chunk_gated_delta_rule_fwd_h, {"kind": "current_no_request_path"}
    path = Path(source).resolve(strict=True)
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module.chunk_gated_delta_rule_fwd_h, {
        "kind": (
            "frozen_previous_snapshot_source"
            if optional
            else "frozen_pre_change_source"
        ),
        "source": str(path),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }


BASELINE, BASELINE_METADATA = load_baseline()
PREVIOUS, PREVIOUS_METADATA = load_baseline(
    "RTP_KDA_PREVIOUS_SOURCE", "kda_frozen_previous", optional=True
)


def prepare(raw, cu_seqlens=None, initial_state=None):
    gk = chunk_local_cumsum(
        raw["g"], scale=RCP_LN2, chunk_size=64, cu_seqlens=cu_seqlens
    )
    w, u, _, kg, _, _ = chunk_kda_fwd_intra(
        q=raw["q"],
        k=raw["k"],
        v=raw["v"],
        gk=gk,
        beta=raw["beta"],
        scale=raw["k"].shape[-1] ** -0.5,
        cu_seqlens=cu_seqlens,
    )
    return dict(
        k=kg,
        w=w,
        u=u,
        gk=gk,
        initial_state=initial_state,
        output_final_state=True,
        cu_seqlens=cu_seqlens,
    )


def inputs(
    lengths,
    heads=4,
    dim=64,
    dtype=torch.bfloat16,
    packed=False,
    initial=True,
    decay=0.03,
    seed=1729,
):
    torch.manual_seed(seed)
    batch, tokens = (1, sum(lengths)) if packed else (len(lengths), lengths[0])
    assert packed or len(set(lengths)) == 1
    shape = (batch, tokens, heads, dim)
    raw = {
        "q": F.normalize(torch.randn(shape, device="cuda"), dim=-1).to(dtype),
        "k": F.normalize(torch.randn(shape, device="cuda"), dim=-1).to(dtype),
        "v": torch.randn(shape, device="cuda", dtype=dtype),
        "g": -torch.rand(shape, device="cuda") * decay,
        "beta": torch.rand(shape[:-1], device="cuda", dtype=dtype),
    }
    offsets = [0]
    for length in lengths:
        offsets.append(offsets[-1] + length)
    cu = torch.tensor(offsets, device="cuda", dtype=torch.int64) if packed else None
    h0 = (
        torch.randn(len(lengths), heads, dim, dim, device="cuda") * 0.1
        if initial
        else None
    )
    return prepare(raw, cu, h0), raw


def request_tensor(lengths):
    return torch.tensor(
        [
            [length, 0, 1, 31, 63, 64, 65, 127, 128, 129, 64, -1, length + 1, 1 << 40]
            for length in lengths
        ],
        device="cuda",
        dtype=torch.int64,
    )


def output_buffer(args, positions):
    state_dims = (args["k"].shape[-1], args["u"].shape[-1])
    if args.get("transpose_state_layout", False):
        state_dims = state_dims[::-1]
    return torch.full(
        (
            *positions.shape,
            args["u"].shape[-2],
            *state_dims,
        ),
        float("nan"),
        device="cuda",
        dtype=torch.float32,
    )


def selected_call(args, raw_k, positions, output):
    return chunk_gated_delta_rule_fwd_h(
        **args, raw_k=raw_k, state_positions=positions, selected_states=output
    )


def prefix_oracle(args, raw, lengths, requests):
    """Frozen prefix on identical prepared w/u/gk and identical state layout.

    Rebuild only kg from raw k at the requested endpoint. Freshly recomputing
    the entire prefix pipeline can change w/u rounding even in the original
    implementation; the optional oracle diagnostic records that control.
    """
    result = output_buffer(args, torch.tensor(requests, device="cuda"))
    result.zero_()
    packed = args["cu_seqlens"] is not None
    offset = 0
    for n, length in enumerate(lengths):
        cache = {}
        h0 = None if args["initial_state"] is None else args["initial_state"][n : n + 1]
        for slot, prefix in enumerate(requests[n]):
            if prefix < 0 or prefix > length:
                continue
            if prefix == 0:
                if h0 is not None:
                    result[n, slot].copy_(h0[0])
                continue
            if prefix not in cache:
                sliced = {
                    key: (
                        value[:, offset : offset + prefix]
                        if packed
                        else value[n : n + 1, :prefix]
                    ).contiguous()
                    for key, value in raw.items()
                }
                retained = {
                    key: (
                        args[key][:, offset : offset + prefix]
                        if packed
                        else args[key][n : n + 1, :prefix]
                    ).contiguous()
                    for key in ("w", "u", "gk")
                }
                cu = (
                    torch.tensor([0, prefix], device="cuda", dtype=torch.int64)
                    if packed
                    else None
                )
                _, _, _, kg, _, _ = chunk_kda_fwd_intra(
                    q=sliced["q"],
                    k=sliced["k"],
                    v=sliced["v"],
                    gk=retained["gk"],
                    beta=sliced["beta"],
                    scale=sliced["k"].shape[-1] ** -0.5,
                    cu_seqlens=cu,
                )
                cache[prefix] = BASELINE(
                    **retained,
                    k=kg,
                    initial_state=h0,
                    output_final_state=True,
                    cu_seqlens=cu,
                    transpose_state_layout=args.get("transpose_state_layout", False),
                )[2][0]
            result[n, slot].copy_(cache[prefix])
        offset += length
    return result


def recurrence(args, raw, lengths, requests):
    """Independent FP32 tokenwise KDA recurrence in K-first layout."""
    k, v, g, beta = (raw[key].float() for key in ("k", "v", "g", "beta"))
    result = output_buffer(args, torch.tensor(requests, device="cuda")).zero_()
    offset = 0
    for n, length in enumerate(lengths):
        batch, start = (0, offset) if args["cu_seqlens"] is not None else (n, 0)
        state = (
            torch.zeros_like(result[n, 0])
            if args["initial_state"] is None
            else args["initial_state"][n].clone()
        )
        for prefix in range(length + 1):
            for slot, position in enumerate(requests[n]):
                if prefix == position:
                    result[n, slot].copy_(state)
            if prefix == length:
                break
            t = start + prefix
            kt = k[batch, t, :, :, None]
            state = state * g[batch, t].exp()[:, :, None]
            residual = v[batch, t] - (state * kt).sum(-2)
            state = state + kt * (beta[batch, t, :, None] * residual)[:, None, :]
        offset += length
    return result


@unittest.skipUnless(
    torch.cuda.is_available() and torch.version.hip is None,
    "selected KDA states require NVIDIA CUDA",
)
class SelectedStatesTest(unittest.TestCase):
    def outputs_equal(self, baseline, candidate):
        for before, after in zip(baseline, candidate):
            if before is None:
                self.assertIsNone(after)
            else:
                torch.testing.assert_close(after, before, rtol=0, atol=0)

    def check_case(self, lengths, **kwargs):
        args, raw = inputs(lengths, **kwargs)
        positions = request_tensor(lengths)
        selected = output_buffer(args, positions)
        old = BASELINE(**args)
        self.outputs_equal(old, chunk_gated_delta_rule_fwd_h(**args))
        self.outputs_equal(old, selected_call(args, raw["k"], positions, selected))
        expected = prefix_oracle(args, raw, lengths, positions.tolist())
        torch.testing.assert_close(selected, expected, rtol=0, atol=0)
        reference = recurrence(args, raw, lengths, positions.tolist())
        scale = reference.square().mean().sqrt().clamp_min(1e-8)
        old_error = (expected - reference).square().mean().sqrt() / scale
        new_error = (selected - reference).square().mean().sqrt() / scale
        self.assertLess(float(old_error), 0.01)
        self.assertLessEqual(float(new_error), float(old_error))
        self.assertTrue(torch.isfinite(selected).all().item())

    def test_fixed_precision_and_boundaries(self):
        for dtype in (torch.float16, torch.bfloat16):
            for dim in (64, 128):
                with self.subTest(dtype=dtype, dim=dim):
                    self.check_case([129, 129], dtype=dtype, dim=dim)

    def test_packed_and_no_initial_state(self):
        self.check_case([1, 63, 64, 65, 129, 257], packed=True)
        self.check_case([31, 128], packed=True, initial=False, dtype=torch.float16)

    def test_strong_decay(self):
        # Full-chunk kg underflows for early tokens; raw k must be used for
        # interior prefixes. Inverting the full-chunk weight cannot recover it.
        self.check_case([129], decay=4.0, dim=128)

    def test_capture_request_count_specialization_boundary(self):
        args, raw = inputs([129], heads=2, dim=64)
        requests = [
            129,
            0,
            65,
            -1,
            63,
            64,
            1,
            128,
            65,
            130,
            31,
            127,
            1 << 40,
            64,
            2,
            129,
            0,
        ]
        original = BASELINE(**args)
        requests = requests * 2
        for count in (4, 5, 15, 16, 17, 32, 33):
            with self.subTest(request_count=count):
                positions = torch.tensor(
                    [requests[:count]], device="cuda", dtype=torch.int64
                )
                selected = output_buffer(args, positions)
                self.outputs_equal(
                    original, selected_call(args, raw["k"], positions, selected)
                )
                torch.testing.assert_close(
                    selected,
                    prefix_oracle(args, raw, [129], [requests[:count]]),
                    rtol=0,
                    atol=0,
                )

    def test_linear_tail_relative_positions(self):
        args, raw = inputs([512], heads=2, dim=64)
        requests = [236, 492]
        positions = torch.tensor([requests], device="cuda", dtype=torch.int64)
        selected = output_buffer(args, positions)
        self.outputs_equal(
            BASELINE(**args), selected_call(args, raw["k"], positions, selected)
        )
        torch.testing.assert_close(
            selected, prefix_oracle(args, raw, [512], [requests]), rtol=0, atol=0
        )

    def test_interior_boundary_reuses_main_state(self):
        # The interior 64/128 requests can use main-recurrence chunk starts;
        # the final 192 request must still replay its preceding full chunk.
        lengths = [192, 129]
        requests = [[128, 64, 192, 64], [64, 128, 129, 64]]
        for save_value in (False, True):
            with self.subTest(save_new_value=save_value):
                args, raw = inputs(lengths, packed=True, heads=2, dim=64)
                args["save_new_value"] = save_value
                positions = torch.tensor(requests, device="cuda", dtype=torch.int64)
                selected = output_buffer(args, positions)
                self.outputs_equal(
                    BASELINE(**args),
                    selected_call(args, raw["k"], positions, selected),
                )
                torch.testing.assert_close(
                    selected,
                    prefix_oracle(args, raw, lengths, requests),
                    rtol=0,
                    atol=0,
                )

    def test_dense_complete_boundary_requests(self):
        for slots in (16, 32, 64, 128):
            length = slots * 64
            with self.subTest(slots=slots, length=length):
                args, raw = inputs([length], heads=2, dim=64)
                requests = [boundary for boundary in range(64, length + 1, 64)]
                positions = torch.tensor([requests], device="cuda", dtype=torch.int64)
                selected = output_buffer(args, positions)
                self.outputs_equal(
                    BASELINE(**args), selected_call(args, raw["k"], positions, selected)
                )
                torch.testing.assert_close(
                    selected,
                    prefix_oracle(args, raw, [length], [requests]),
                    rtol=0,
                    atol=0,
                )

    def test_saved_value_reuse_and_recompute_fallback(self):
        for packed in (False, True):
            lengths = [65, 129] if packed else [129]
            base_args, raw = inputs(lengths, heads=2, dim=64, packed=packed)
            requests = [[63, 1, 64, 65, 63, 0, length, -1] for length in lengths]
            positions = torch.tensor(requests, device="cuda", dtype=torch.int64)
            for transposed in (False, True):
                args = dict(base_args, transpose_state_layout=transposed)
                if transposed:
                    args["initial_state"] = (
                        base_args["initial_state"].transpose(-1, -2).contiguous()
                    )
                expected = prefix_oracle(args, raw, lengths, requests)
                for save_value in (True, False):
                    with self.subTest(
                        packed=packed, transposed=transposed, save_new_value=save_value
                    ):
                        args["save_new_value"] = save_value
                        selected = output_buffer(args, positions)
                        self.outputs_equal(
                            BASELINE(**args),
                            selected_call(args, raw["k"], positions, selected),
                        )
                        torch.testing.assert_close(selected, expected, rtol=0, atol=0)

    def test_capture_map_empty_sequences(self):
        # Prepare nonempty sequences independently: the existing upstream
        # packed intra pipeline does not preserve empty sequence IDs.
        for lengths in ([0], [0, 0], [0, 65], [65, 0, 1]):
            with self.subTest(lengths=lengths):
                parts = [
                    inputs([length], heads=2, dim=64, seed=1729 + i)
                    for i, length in enumerate(lengths)
                    if length
                ]
                raw = {}
                args = {}
                for key in ("q", "k", "v", "g", "beta"):
                    shape = (1, 0, 2) if key == "beta" else (1, 0, 2, 64)
                    dtype = torch.float32 if key == "g" else torch.bfloat16
                    raw[key] = (
                        torch.cat([part[1][key] for part in parts], dim=1)
                        if parts
                        else torch.empty(shape, device="cuda", dtype=dtype)
                    )
                for key in ("k", "w", "u", "gk"):
                    dtype = torch.float32 if key == "gk" else torch.bfloat16
                    args[key] = (
                        torch.cat([part[0][key] for part in parts], dim=1)
                        if parts
                        else torch.empty(1, 0, 2, 64, device="cuda", dtype=dtype)
                    )
                offsets = [0]
                for length in lengths:
                    offsets.append(offsets[-1] + length)
                args.update(
                    cu_seqlens=torch.tensor(offsets, device="cuda", dtype=torch.int64),
                    initial_state=torch.randn(len(lengths), 2, 64, 64, device="cuda"),
                    output_final_state=True,
                )
                requests = [
                    [0, length, 1, 63, 64, 65, -1, 1 << 40] for length in lengths
                ]
                positions = torch.tensor(requests, device="cuda", dtype=torch.int64)
                selected = output_buffer(args, positions)
                self.outputs_equal(
                    BASELINE(**args), selected_call(args, raw["k"], positions, selected)
                )
                torch.testing.assert_close(
                    selected,
                    prefix_oracle(args, raw, lengths, requests),
                    rtol=0,
                    atol=0,
                )

    def test_transposed_state_layout(self):
        args, raw = inputs([129])
        positions = request_tensor([129])
        args["initial_state"] = args["initial_state"].transpose(-1, -2).contiguous()
        args["transpose_state_layout"] = True
        expected = prefix_oracle(args, raw, [129], positions.tolist())
        selected = output_buffer(args, positions)
        self.outputs_equal(
            BASELINE(**args), selected_call(args, raw["k"], positions, selected)
        )
        torch.testing.assert_close(selected, expected, rtol=0, atol=0)

    def test_empty_requests_optional_outputs_and_alias(self):
        args, raw = inputs([65])
        positions = torch.empty(1, 0, device="cuda", dtype=torch.int32)
        selected = output_buffer(args, positions)
        self.outputs_equal(
            BASELINE(**args), selected_call(args, raw["k"], positions, selected)
        )
        args.update(output_final_state=False, save_new_value=False)
        positions = torch.tensor([[0, 1, 64, 65, -1]], device="cuda", dtype=torch.int32)
        selected = output_buffer(args, positions)
        self.outputs_equal(
            BASELINE(**args), selected_call(args, raw["k"], positions, selected)
        )
        torch.testing.assert_close(
            selected, prefix_oracle(args, raw, [65], positions.tolist()), rtol=0, atol=0
        )
        with self.assertRaises(ValueError):
            selected_call(
                args,
                raw["k"],
                positions[:, :1].contiguous(),
                args["initial_state"].unsqueeze(1),
            )

    def test_public_pipeline(self):
        args, raw = inputs([129], dim=128)
        public = dict(
            raw,
            initial_state=args["initial_state"],
            output_final_state=True,
            return_intermediate_states=True,
        )
        positions = request_tensor([129])
        selected = output_buffer(args, positions)
        self.outputs_equal(
            chunk_kda(**public),
            chunk_kda(**public, state_positions=positions, selected_states=selected),
        )
        expected = prefix_oracle(args, raw, [129], positions.tolist())
        torch.testing.assert_close(selected, expected, rtol=0, atol=0)
        for slot, prefix in enumerate(positions[0].tolist()):
            if not 0 < prefix <= 129:
                continue
            truncated = {
                key: value[:, :prefix].contiguous() for key, value in raw.items()
            }
            final = chunk_kda(
                **truncated,
                initial_state=args["initial_state"],
                output_final_state=True,
            )[1]
            torch.testing.assert_close(selected[:, slot], final, rtol=0, atol=0)
        with self.assertRaises(ValueError):
            chunk_kda(
                **public,
                state_positions=positions,
                selected_states=selected.transpose(-1, -2),
            )

    def test_public_production_normalization_and_gate(self):
        torch.manual_seed(1741)
        shape = (1, 129, 4, 128)
        raw = {
            name: torch.randn(shape, device="cuda", dtype=torch.bfloat16)
            for name in ("q", "k", "v", "g")
        }
        raw["beta"] = torch.rand(shape[:-1], device="cuda", dtype=torch.bfloat16)
        h0 = torch.randn(1, 4, 128, 128, device="cuda") * 0.1
        options = dict(
            initial_state=h0,
            output_final_state=True,
            return_intermediate_states=True,
            use_qk_l2norm_in_kernel=True,
            use_gate_in_kernel=True,
            A_log=torch.randn(4, device="cuda") * 0.1,
            dt_bias=torch.randn(4 * 128, device="cuda") * 0.01,
        )
        positions = torch.tensor([[0, 1, 63, 64, 65, 129]], device="cuda")
        selected = torch.full((1, 6, 4, 128, 128), float("nan"), device="cuda")
        candidate = chunk_kda(
            **raw, **options, state_positions=positions, selected_states=selected
        )
        self.outputs_equal(chunk_kda(**raw, **options), candidate)
        forward_module = importlib.import_module(
            "rtp_llm.models_py.triton_kernels.kimi_kda.chunk_fwd"
        )

        def original_adapter(**kwargs):
            for key in ("state_positions", "selected_states", "raw_k"):
                self.assertIsNone(kwargs.pop(key, None))
            return BASELINE(**kwargs)

        with mock.patch.object(
            forward_module, "chunk_gated_delta_rule_fwd_h", original_adapter
        ):
            self.outputs_equal(chunk_kda(**raw, **options), candidate)
            for slot, prefix in enumerate(positions[0].tolist()):
                reference = h0
                if prefix:
                    truncated = {
                        name: value[:, :prefix].contiguous()
                        for name, value in raw.items()
                    }
                    reference = chunk_kda(**truncated, **options)[1]
                torch.testing.assert_close(selected[:, slot], reference, rtol=0, atol=0)

    def test_cuda_graph_changes_all_inputs(self):
        for slots in (4, 8):
            args, raw = inputs([129], dim=128)
            positions = torch.tensor(
                [[0, 1, 31, 63, 1, 31, -1, 0][:slots]], device="cuda"
            )
            selected = output_buffer(args, positions)
            stream = torch.cuda.Stream()
            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                for _ in range(3):
                    selected_call(args, raw["k"], positions, selected)
            torch.cuda.current_stream().wait_stream(stream)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                captured = selected_call(args, raw["k"], positions, selected)
            replacement, new_raw = inputs([129], dim=128, seed=1730)
            for key in ("k", "w", "u", "gk", "initial_state"):
                args[key].copy_(replacement[key])
            raw["k"].copy_(new_raw["k"])
            positions.copy_(
                torch.tensor(
                    [[1, 65, 128, 1 << 40, 129, 65, 64, -1][:slots]], device="cuda"
                )
            )
            graph.replay()
            torch.cuda.synchronize()
            replayed = selected.clone()
            eager_selected = output_buffer(args, positions)
            self.outputs_equal(
                selected_call(args, raw["k"], positions, eager_selected), captured
            )
            self.outputs_equal(BASELINE(**args), captured)
            torch.testing.assert_close(replayed, eager_selected, rtol=0, atol=0)
            torch.testing.assert_close(
                replayed,
                prefix_oracle(args, new_raw, [129], positions.tolist()),
                rtol=0,
                atol=0,
            )


def tensor_difference(actual, expected):
    difference = (actual.float() - expected.float()).abs()
    mismatches = actual != expected
    first = mismatches.nonzero()
    return {
        "exact_equal": bool(torch.equal(actual, expected)),
        "max_abs_error": float(difference.max()),
        "mismatched_elements": int(mismatches.sum()),
        "elements": actual.numel(),
        "first_mismatch_index": first[0].tolist() if first.numel() else None,
    }


def diagnose_oracles(output_path):
    """Record the failing oracle alternatives without relaxing assertions."""
    lengths = [1, 63, 64, 65, 129, 257]
    args, raw = inputs(lengths, packed=True)
    positions = request_tensor(lengths)
    snapshots = output_buffer(args, positions)
    selected_call(args, raw["k"], positions, snapshots)
    n, prefix, slot = 2, 63, 4
    start = sum(lengths[:n])
    sliced_raw = {
        name: value[:, start : start + prefix].contiguous()
        for name, value in raw.items()
    }
    h0 = args["initial_state"][n : n + 1].contiguous()
    independent = prepare(sliced_raw, initial_state=h0)
    kept = {
        name: args[name][:, start : start + prefix].contiguous()
        for name in ("w", "u", "gk")
    }
    _, _, _, rebuilt_kg, _, _ = chunk_kda_fwd_intra(
        q=sliced_raw["q"],
        k=sliced_raw["k"],
        v=sliced_raw["v"],
        gk=kept["gk"],
        beta=sliced_raw["beta"],
        scale=64**-0.5,
    )
    retained = dict(kept, k=rebuilt_kg, initial_state=h0, output_final_state=True)
    single_cu = torch.tensor([0, prefix], device="cuda", dtype=torch.int64)
    retained_varlen = dict(retained, cu_seqlens=single_cu)
    chosen = snapshots[n, slot]
    packed_report = {
        "sequence": n,
        "original_length": lengths[n],
        "prefix": prefix,
        "stages_original_packed_slice_vs_independent_prefix": {
            name: tensor_difference(kept[name], independent[name])
            for name in ("gk", "w", "u")
        },
        "kg_rebuilt_using_original_gk_vs_independent_prefix": tensor_difference(
            rebuilt_kg, independent["k"]
        ),
        "snapshot_vs_original_independent_prefix": tensor_difference(
            chosen, BASELINE(**independent)[2][0]
        ),
        "snapshot_vs_original_retained_prepared_fixed_prefix": tensor_difference(
            chosen, BASELINE(**retained)[2][0]
        ),
        "snapshot_vs_original_retained_prepared_varlen_prefix": tensor_difference(
            chosen, BASELINE(**retained_varlen)[2][0]
        ),
        "original_independent_prefix_vs_original_retained_prepared_prefix": tensor_difference(
            BASELINE(**independent)[2], BASELINE(**retained)[2]
        ),
        "original_retained_fixed_vs_original_retained_varlen": tensor_difference(
            BASELINE(**retained)[2], BASELINE(**retained_varlen)[2]
        ),
    }
    args, raw = inputs([129])
    positions = request_tensor([129])
    canonical = prefix_oracle(args, raw, [129], positions.tolist())
    transposed_args = dict(
        args,
        initial_state=args["initial_state"].transpose(-1, -2).contiguous(),
        transpose_state_layout=True,
    )
    snapshots = output_buffer(args, positions)
    selected_call(transposed_args, raw["k"], positions, snapshots)
    transposed = torch.zeros_like(snapshots)
    for slot, prefix in enumerate(positions[0].tolist()):
        if prefix < 0 or prefix > 129:
            continue
        if prefix == 0:
            transposed[:, slot].copy_(transposed_args["initial_state"])
            continue
        truncated = {
            name: value[:, :prefix].contiguous() for name, value in raw.items()
        }
        prepared = prepare(truncated, initial_state=transposed_args["initial_state"])
        prepared["transpose_state_layout"] = True
        transposed[:, slot].copy_(BASELINE(**prepared)[2])
    report = {
        "baseline": BASELINE_METADATA,
        "packed": packed_report,
        "transpose": {
            "snapshot_vs_original_same_layout": tensor_difference(
                snapshots, transposed
            ),
            "snapshot_vs_original_default_layout_transposed": tensor_difference(
                snapshots, canonical.transpose(-1, -2)
            ),
            "original_same_layout_vs_default_layout_transposed": tensor_difference(
                transposed, canonical.transpose(-1, -2)
            ),
        },
    }
    Path(output_path).write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)


def duration_stats(samples):
    ordered = sorted(samples)
    return dict(
        kernel_sum_us_per_iter=statistics.mean(samples),
        median_us=statistics.median(samples),
        p90_us=ordered[math.ceil(len(samples) * 0.9) - 1],
        min_us=min(samples),
        max_us=max(samples),
        samples_us=samples,
    )


def profile_kernels(invoke, path, roles, warmup=30, iterations=100):
    for _ in range(warmup):
        invoke()
    torch.cuda.synchronize()
    with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ]
    ) as profiler:
        for _ in range(iterations):
            invoke()
        torch.cuda.synchronize()
    profiler.export_chrome_trace(str(path))
    events = sorted(
        (
            event
            for event in json.loads(path.read_text())["traceEvents"]
            if event.get("cat") == "kernel"
        ),
        key=lambda event: event["ts"],
    )
    state_events, fill_events = [], []
    for event in events:
        if (
            "_chunk_gated_delta_rule_state_kernel_base" in event["name"]
            or event["name"] == "_prepare_selected_state_capture"
        ):
            state_events.append(event)
        elif "elementwise_kernel" in event["name"] and "FillFunctor" in event["name"]:
            fill_events.append(event)
        else:
            raise AssertionError(f"Unexpected CUDA kernel: {event['name']}")
    launches = len(roles)
    if len(state_events) != iterations * launches:
        raise AssertionError(
            f"Expected {iterations * launches} state kernels, got {len(state_events)}"
        )
    if len(fill_events) not in (0, iterations):
        raise AssertionError(
            f"Expected zero or one final-state zero-fill per iteration; got {len(fill_events)}"
        )
    components = {}
    for index, role in enumerate(roles):
        selected = state_events[index::launches]
        expected_name = (
            "_prepare_selected_state_capture"
            if role == "request_map"
            else "_chunk_gated_delta_rule_state_kernel_base"
        )
        if any(event["name"] != expected_name for event in selected):
            raise AssertionError(
                f"Unexpected kernel for role {role}: {[event['name'] for event in selected]}"
            )
        components[role] = dict(
            duration_stats([event["dur"] for event in selected]),
            kernel_names=sorted({event["name"] for event in selected}),
            launch_index_in_iteration=index,
        )
    samples = [
        sum(event["dur"] for event in state_events[index : index + launches])
        for index in range(0, len(state_events), launches)
    ]
    fill_samples = [event["dur"] for event in fill_events] or [0.0] * iterations
    return dict(
        duration_stats(samples),
        components=components,
        launches_per_iter=launches,
        expected_names_and_counts_pass=True,
        kernel_names=sorted({event["name"] for event in state_events}),
        allocation_fill=dict(
            duration_stats(fill_samples),
            launches_per_iter=len(fill_events) // iterations,
            kernel_names=sorted({event["name"] for event in fill_events}),
        ),
        all_cuda_kernel_sum_us_per_iter=sum(event["dur"] for event in events)
        / iterations,
        trace=str(path),
    )


def benchmark(output_path):
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    position_mode = os.environ.get("RTP_KDA_PERF_POSITION_MODE", "random")
    if position_mode not in ("random", "interior_boundary"):
        raise ValueError(f"Unknown KDA position mode: {position_mode}")
    tokens_values = tuple(
        int(value)
        for value in os.environ.get(
            "RTP_KDA_PERF_TOKENS", "1024,2048,4096,8192,16384,32768"
        ).split(",")
        if value.strip()
    )
    slot_values = tuple(
        int(value)
        for value in os.environ.get("RTP_KDA_PERF_SLOTS", "0,1,2,4,8").split(",")
        if value.strip()
    )
    cases = [(tokens, slots) for tokens in tokens_values for slots in slot_values]
    result = {
        "model_shape": "Kimi-Linear-48B-A3B-Instruct TP1: B1 H32 K128 V128",
        "model_config_sha256": "a6ac3c2c4b5aa72370f9727f49ffa4432715d20061889acdb37c688be853096e",
        "baseline": BASELINE_METADATA,
        "previous_snapshot_implementation": PREVIOUS_METADATA,
        "metric": "sum of actual CUDA state and request-map kernel durations, excluding CPU gaps and separately reported allocation zero-fill",
        "accuracy_comparison": "bitwise frozen original under identical prepared w/u/gk and state layout; only prefix-end kg is rebuilt from raw k",
        "accuracy_limit": "fresh full-pipeline prefix preparation can differ in w/u rounding even in the original implementation; not asserted equivalent for packed inputs",
        "torch_compile": False,
        "warmup": 30,
        "iterations": 100,
        "compilation_excluded": True,
        "input_allocation_outside_measurement": True,
        "position_sampling": position_mode,
        "gpu": torch.cuda.get_device_name(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "cases": [],
    }
    for index, (tokens, slots) in enumerate(cases):
        args, raw = inputs([tokens], heads=32, dim=128)
        seed = 20261016 + tokens
        if position_mode == "interior_boundary":
            boundaries = range(64, tokens, 64)
            if slots > len(boundaries):
                raise ValueError("Not enough interior boundaries for requested slots")
            requests = random.Random(seed).sample(boundaries, slots)
        else:
            sample_size = max(8, slots)
            requests = random.Random(seed).sample(range(1, tokens), sample_size)[:slots]
        positions = torch.tensor([requests], device="cuda", dtype=torch.int64)
        selected = output_buffer(args, positions)
        torch.cuda.reset_peak_memory_stats()
        baseline_outputs = BASELINE(**args)
        candidate_outputs = selected_call(args, raw["k"], positions, selected)
        for before, after in zip(baseline_outputs, candidate_outputs):
            torch.testing.assert_close(after, before, rtol=0, atol=0)
        if slots:
            expected_selected = prefix_oracle(args, raw, [tokens], [requests])
            torch.testing.assert_close(selected, expected_selected, rtol=0, atol=0)
        if PREVIOUS is not None:
            previous_outputs = PREVIOUS(
                **args,
                raw_k=raw["k"],
                state_positions=positions,
                selected_states=selected,
            )
            for before, previous in zip(baseline_outputs, previous_outputs):
                torch.testing.assert_close(previous, before, rtol=0, atol=0)
            if slots:
                torch.testing.assert_close(selected, expected_selected, rtol=0, atol=0)
            del previous_outputs
        peak_allocated_bytes = torch.cuda.max_memory_allocated()
        del baseline_outputs, candidate_outputs
        variants = (
            ("baseline", "previous", "candidate")
            if PREVIOUS is not None
            else ("baseline", "candidate")
        )
        offset = index % len(variants)
        order = variants[offset:] + variants[:offset]
        measured = {}
        for variant in order:
            if variant == "baseline":
                invoke = lambda: BASELINE(**args)
            elif variant == "previous":
                invoke = lambda: PREVIOUS(
                    **args,
                    raw_k=raw["k"],
                    state_positions=positions,
                    selected_states=selected,
                )
            else:
                invoke = lambda: selected_call(args, raw["k"], positions, selected)
            roles = (
                ["original"]
                if variant == "baseline"
                else (["main", "snapshot"] if slots else ["main"])
            )
            if variant != "baseline" and 4 < slots <= 32:
                roles = ["request_map", "main", "snapshot"]
            measured[variant] = profile_kernels(
                invoke, path.with_name(f"{path.stem}_{index}_{variant}.json"), roles
            )
        before = measured["baseline"]["kernel_sum_us_per_iter"]
        after = measured["candidate"]["kernel_sum_us_per_iter"]
        row = dict(
            B=1,
            T=tokens,
            H=32,
            K=128,
            V=128,
            S=slots,
            dtype="bfloat16",
            positions=requests,
            position_seed=seed,
            measurement_order=order,
            selected_states_bytes=selected.numel() * selected.element_size(),
            capture_workspace_bytes=(
                selected.numel() * selected.element_size() + ((tokens + 63) // 64) * 4
                if 4 < slots <= 32
                else 0
            ),
            peak_cuda_memory_allocated_during_correctness_bytes=peak_allocated_bytes,
            correctness={
                "original_regular_outputs_exact": True,
                "original_prefix_snapshots_exact": bool(slots),
                "max_abs_error": 0.0,
                "normalized_rms_error": 0.0,
                "previous_regular_and_selected_outputs_exact": PREVIOUS is not None,
            },
            baseline=measured["baseline"],
            candidate=measured["candidate"],
            speedup=before / after,
            additional_kernel_us=after - before,
            overhead_percent=(after / before - 1) * 100,
        )
        if PREVIOUS is not None:
            previous = measured["previous"]
            ratio = after / previous["kernel_sum_us_per_iter"]
            row.update(
                previous=previous,
                candidate_vs_previous_ratio=ratio,
                candidate_vs_previous_speedup=1 / ratio,
                candidate_vs_previous_overhead_percent=(ratio - 1) * 100,
            )
        result["cases"].append(row)
        path.write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps({key: row[key] for key in ("T", "S", "speedup")}), flush=True)


if __name__ == "__main__":
    if os.environ.get("RTP_KDA_DIAGNOSTIC_JSON"):
        diagnose_oracles(os.environ["RTP_KDA_DIAGNOSTIC_JSON"])
        sys.exit(0)
    filters = os.environ.get("RTP_KDA_TEST_FILTER", "").split(",")
    tests = unittest.defaultTestLoader.loadTestsFromTestCase(SelectedStatesTest)
    if any(filters):
        tests = unittest.TestSuite(
            test
            for test in tests
            if any(token in test.id() for token in filters if token)
        )
        if tests.countTestCases() == 0:
            raise ValueError("RTP_KDA_TEST_FILTER matched no tests")
    outcome = unittest.TextTestRunner(verbosity=2).run(tests)
    if not outcome.wasSuccessful():
        sys.exit(1)
    if os.environ.get("PERF_JSON"):
        if not torch.cuda.is_available() or torch.version.hip is not None:
            raise RuntimeError("KDA kernel benchmark requires NVIDIA CUDA")
        benchmark(os.environ["PERF_JSON"])
