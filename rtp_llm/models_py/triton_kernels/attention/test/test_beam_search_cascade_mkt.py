"""mkt-gr-shaped decode FMHA A/B, not an end-to-end model accuracy test.

Cascade-off is the production AiterDecodeAttnOpTriton.forward entry point:
USE_ASM_PA=1, USE_TRITON_PA=1 selects pa_decode_gluon, not ASM PA.
Inputs are deterministic synthetic BF16 post-RoPE queries and shuffled KV.
The recorded prompt lengths are history lengths; attention sees history + 1.
The graph max sequence length is 4096, matching the local mkt-gr reproduction.

Correctness and performance are separate Bazel targets. Run correctness first.
MKT_CASCADE_REQUESTS=1 is the quick 500-row reproduction; default is 1,2,4,8.
MKT_CASCADE_OUTPUT_DIR saves machine-readable metrics and raw timing samples.
MKT_CASCADE_HIGH_ADDRESS_GIB=2/4 selects the high-address boundary (default 2).
MKT_CASCADE_TAIL_VARIANT=common/layout/layout_only/layout_w1/layout_w4/shared_k_w1 selects an experimental tail kernel in
tests only; production is the default. Each test restores the original symbol.
MKT_CASCADE_COMPARE_TAILS=production,common,layout,layout_only enables a same-process fair
performance comparison; leave MKT_CASCADE_TAIL_VARIANT at production for it.
"""

from __future__ import annotations

import hashlib
import json
import os
import statistics
import unittest
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import torch
import triton.testing

from rtp_llm.models_py.triton_kernels.attention import (
    beam_search_cascade as cascade_mod,
)

PRODUCTION_TAIL_KERNEL = cascade_mod._private_tail_merge
TAIL_VARIANTS = (
    "production",
    "common",
    "layout",
    "layout_only",
    "layout_w1",
    "layout_w4",
    "shared_k_w1",
)
HEADS, KV_HEADS, DIM, PAGE, BEAMS = 16, 8, 128, 16, 500
MAX_SEQ_LEN = 4096
HISTORIES = (714, 836, 710, 707, 584, 759, 862, 737)
# Fixed before measurements. No epsilon based on observed error or output scale.
ATOL, RTOL, RELATIVE_L2_LIMIT = 0.005, 0.03, 0.01
AB_ATOL, AB_RTOL, AB_RELATIVE_L2_LIMIT = 0.01, 0.06, 0.02
REFERENCE_ATOL, REFERENCE_RTOL = 2e-5, 2e-5
SEED = 20261028
# RTP_MEASURED: runs/nan_precision_dispatch_01, tails 1/8/9/15: the actual
# pa_decode_gluon baseline returns 1,024,000 NaNs; production/common/layout
# Cascade remain finite and unchanged by inactive-V poison. The original
# MktCascadeTailDispatchTest preserves that baseline failure. A separate
# Cascade-only safety gate below must not turn the off result into a PASS.
OFF_NAN_COUNTEREXAMPLE = "nan_precision_dispatch_01: off pa_decode_gluon fails inactive-V NaN; original A/B poison test retained"


def high_address_gib():
    value = int(os.getenv("MKT_CASCADE_HIGH_ADDRESS_GIB", "2"))
    if value not in (2, 4):
        raise ValueError("MKT_CASCADE_HIGH_ADDRESS_GIB must be 2 or 4")
    return value


def tail_variant():
    value = os.getenv("MKT_CASCADE_TAIL_VARIANT", "production")
    if value not in TAIL_VARIANTS:
        raise ValueError(
            "MKT_CASCADE_TAIL_VARIANT must be one of " + ", ".join(TAIL_VARIANTS)
        )
    if (
        value in ("layout_w1", "layout_w4", "shared_k_w1")
        and os.getenv("MKT_CASCADE_NATIVE_FP32_O") != "1"
    ):
        raise ValueError(
            "Wave geometry experiments require MKT_CASCADE_NATIVE_FP32_O=1"
        )
    return value


class WarpOverrideKernel:
    """Test-only launch adapter; preserve the JIT kernel and every other option."""

    def __init__(self, kernel, num_warps):
        if num_warps not in (1, 4):
            raise ValueError("Only the predeclared 1/4-wave candidates are supported")
        self.kernel, self.num_warps = kernel, num_warps
        self.arg_names, self.__name__ = kernel.arg_names, kernel.__name__

    def launch_kwargs(self, kwargs):
        return {**kwargs, "num_warps": self.num_warps}

    def __getitem__(self, grid):
        launch = self.kernel[grid]

        def run(*args, **kwargs):
            return launch(*args, **self.launch_kwargs(kwargs))

        return run


def tail_kernels():
    from rtp_llm.models_py.triton_kernels.attention import (
        beam_search_cascade_tail_candidate as candidate,
    )
    from rtp_llm.models_py.triton_kernels.attention import (
        beam_search_cascade_tail_layout_only as layout_only,
    )
    from rtp_llm.models_py.triton_kernels.attention import (
        beam_search_cascade_tail_shared_k as shared_k,
    )

    return {
        "production": PRODUCTION_TAIL_KERNEL,
        "common": candidate._private_tail_merge_common,
        "layout": candidate._private_tail_merge_layout,
        "layout_only": layout_only._private_tail_merge_layout_only,
        "layout_w1": WarpOverrideKernel(candidate._private_tail_merge_layout, 1),
        "layout_w4": WarpOverrideKernel(candidate._private_tail_merge_layout, 4),
        "shared_k_w1": WarpOverrideKernel(shared_k._private_tail_merge_shared_k, 1),
    }


def comparison_variant_orders(variants):
    """Future runs: balance pair order as well as each within-group position."""
    variants = tuple(variants)
    if len(variants) < 2 or len(set(variants)) != len(variants):
        raise ValueError("Comparison requires at least two unique variants")
    rotations = [variants[index:] + variants[:index] for index in range(len(variants))]
    return rotations + [tuple(reversed(order)) for order in rotations]


def _sha_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def loaded_native_modules():
    paths = set()
    for line in Path("/proc/self/maps").read_text().splitlines():
        if "mha_batch_prefill" in line and ".so" in line:
            fields = line.split(maxsplit=5)
            if len(fields) == 6:
                paths.add(fields[5])
    return [
        {
            "path": path,
            "sha256": _sha_file(Path(path)) if Path(path).is_file() else None,
        }
        for path in sorted(paths)
    ]


def requested_batches():
    batches = tuple(
        int(n) for n in os.getenv("MKT_CASCADE_REQUESTS", "1,2,4,8").split(",")
    )
    if not batches or any(n not in (1, 2, 4, 8) for n in batches):
        raise ValueError("MKT_CASCADE_REQUESTS must contain only 1,2,4,8")
    return batches


def emit(name, result):
    payload = {"evidence": "RTP_MEASURED", "name": name, **result}
    print(json.dumps(payload, sort_keys=True), flush=True)
    output = os.getenv(
        "MKT_CASCADE_OUTPUT_DIR", os.getenv("TEST_UNDECLARED_OUTPUTS_DIR")
    )
    if output:
        Path(output).mkdir(parents=True, exist_ok=True)
        (Path(output) / (name + ".json")).write_text(
            json.dumps(payload, indent=2) + "\n"
        )


@dataclass
class Case:
    q: torch.Tensor
    kv: torch.Tensor
    table: torch.Tensor
    history: torch.Tensor
    k: torch.Tensor
    v: torch.Tensor
    logical_table: torch.Tensor
    physical_ids: torch.Tensor
    packed: torch.Tensor
    boundaries: tuple
    request_histories: tuple
    high_address: bool
    boundary_address: bool
    seed: int

    def describe(self):
        return {
            "fixture": "shape-aligned synthetic deterministic; not captured model tensors",
            "seed": self.seed,
            "requests": len(self.request_histories),
            "rows": self.q.shape[0],
            "heads": HEADS,
            "kv_heads": KV_HEADS,
            "head_dim": DIM,
            "dtype": str(self.q.dtype),
            "page": PAGE,
            "history_lengths": self.request_histories,
            "attended_lengths": tuple(n + 1 for n in self.request_histories),
            "query_stride": self.q.stride(),
            "kv_span_bytes": self.kv.numel() * self.kv.element_size(),
            "minimum_live_byte_offset": int(self.physical_ids.min().item())
            * self.kv.stride(0)
            * self.kv.element_size(),
            "maximum_live_byte_offset": int(self.physical_ids.max().item())
            * self.kv.stride(0)
            * self.kv.element_size(),
            "high_address_boundary_gib": (
                high_address_gib() if self.high_address else None
            ),
            "physical_pages_straddle_boundary": self.boundary_address,
            "tail_variant": tail_variant(),
            "graph_max_seq_len": MAX_SEQ_LEN,
            "shared_complete_pages_only": True,
        }


def make_case(
    requests,
    *,
    seed=SEED,
    high_address=False,
    histories=None,
    request_rows=None,
    boundary_address=False,
):
    """Canonical logical pages are separate from randomized physical page IDs.

    Every request shares its full prefix pages. Its partial last prefix page is
    copied into each beam's private page; only the current token is beam-specific.
    Physical pages are permuted, have holes, and can start above 2/4 GiB or
    straddle the selected boundary. No physical-page contiguity is assumed.
    Q is a noncontiguous view with the real packed-QKV row stride.
    """
    histories = tuple(histories or HISTORIES[:requests])
    request_rows = tuple(request_rows or (BEAMS,) * requests)
    rows = sum(request_rows)
    if len(histories) != requests or len(request_rows) != requests:
        raise ValueError("request metadata mismatch")
    if any(h < 128 or h >= 1024 for h in histories):
        raise ValueError("synthetic prefix reserve covers histories [128, 1024)")
    generator = torch.Generator(device="cuda").manual_seed(seed)
    cpu_generator = torch.Generator().manual_seed(seed + 31)
    local_blocks = requests * 64 + rows + 1
    k = torch.randn(
        (local_blocks, KV_HEADS, PAGE, DIM),
        generator=generator,
        device="cuda",
        dtype=torch.bfloat16,
    )
    v = torch.randn(k.shape, generator=generator, device="cuda", dtype=torch.bfloat16)
    qkv = torch.randn(
        (rows, HEADS + 2 * KV_HEADS, DIM),
        generator=generator,
        device="cuda",
        dtype=torch.bfloat16,
    )
    q = qkv[:, :HEADS]
    logical_table = torch.zeros((rows, MAX_SEQ_LEN // PAGE), dtype=torch.int32)
    history = torch.empty(rows, dtype=torch.int32)
    first = 0
    boundaries = [0]
    for request, (count, h) in enumerate(zip(request_rows, histories)):
        last = first + count
        shared_pages, remainder = divmod(h, PAGE)
        logical_table[first:last, :shared_pages] = torch.arange(
            request * 64, request * 64 + shared_pages, dtype=torch.int32
        )
        private = requests * 64 + torch.arange(
            first, last, dtype=torch.int64, device="cuda"
        )
        if remainder:
            prototype = request * 64 + shared_pages
            k[private, :, :remainder] = k[prototype, :, :remainder]
            v[private, :, :remainder] = v[prototype, :, :remainder]
        logical_table[first:last, shared_pages] = torch.arange(
            requests * 64 + first, requests * 64 + last, dtype=torch.int32
        )
        history[first:last] = h
        first = last
        boundaries.append(last)
    packed_k = k.reshape(local_blocks, KV_HEADS, PAGE, DIM // 8, 8).permute(
        0, 1, 3, 2, 4
    )
    packed_v = v.reshape(local_blocks, KV_HEADS, PAGE // 8, 8, DIM).permute(
        0, 1, 2, 4, 3
    )
    packed = torch.stack(
        (packed_k.contiguous().view_as(k), packed_v.contiguous().view_as(v)), 1
    )
    block_bytes = 2 * KV_HEADS * PAGE * DIM * 2
    if boundary_address and not high_address:
        raise ValueError("boundary_address requires high_address")
    boundary_page = high_address_gib() * (1 << 30) // block_bytes
    if boundary_address:
        # Includes pages exactly at and two pages either side of the boundary.
        offset = boundary_page - 2 * (local_blocks // 2) - 17
    else:
        offset = boundary_page + 1 if high_address else 0
    physical_ids = (
        torch.randperm(local_blocks, generator=cpu_generator).to(torch.int32) * 2
        + offset
        + 17
    )
    table = physical_ids[logical_table.to(torch.long)].contiguous()
    # Fixed extent for graph replay, regardless of the random permutation/history.
    kv = torch.empty(
        (offset + 2 * local_blocks + 18, 2, KV_HEADS, PAGE, DIM),
        device="cuda",
        dtype=torch.bfloat16,
    )
    kv.index_copy_(0, physical_ids.to(device="cuda", dtype=torch.long), packed)
    return Case(
        q,
        kv,
        table,
        history,
        k,
        v,
        logical_table,
        physical_ids,
        packed,
        tuple(boundaries),
        histories,
        high_address,
        boundary_address,
        seed,
    )


class AttentionAB:
    """Use the production off wrapper without RoPE or cache insertion on either side."""

    def __init__(self, case):
        from rtp_llm.models_py.modules.factory.attention.rocm_impl.aiter import (
            AiterDecodeAttnOpTriton,
        )
        from rtp_llm.ops import AttentionConfigs, KvCacheDataType
        from rtp_llm.ops.compute_ops import LayerKVCache

        config = AttentionConfigs()
        config.head_num, config.kv_head_num, config.size_per_head = HEADS, KV_HEADS, DIM
        config.tokens_per_block = config.kernel_tokens_per_block = PAGE
        config.max_seq_len = MAX_SEQ_LEN
        config.kv_cache_dtype = KvCacheDataType.BASE
        config.dtype = torch.bfloat16
        config.need_rope_kv_cache = False
        self.case = case
        self.off_op = AiterDecodeAttnOpTriton(config, linear_v=False)
        self.off_op.enable_cuda_graph = True
        self.cache = LayerKVCache()
        self.cache.kv_cache_base = case.kv
        self.cache.kv_scale_base = torch.empty(0, device="cuda")
        self.params = SimpleNamespace(
            seq_lens=(case.history + 1).cuda(),
            kv_cache_block_id_device=case.table.cuda(),
            max_seq_len=MAX_SEQ_LEN,
            kv_scale=None,
        )
        self.on_op = cascade_mod.RocmBeamSearchCascadeAttention(
            case.q.shape[0],
            HEADS,
            KV_HEADS,
            DIM,
            PAGE,
            MAX_SEQ_LEN // PAGE,
            case.q.dtype,
            case.q.device,
            max_requests=len(case.request_histories),
        )
        self.native_prefix_fp32_requested = (
            os.getenv("MKT_CASCADE_NATIVE_FP32_O", "0") == "1"
        )
        if self.native_prefix_fp32_requested:
            if not self.on_op.native_prefix:
                raise AssertionError(
                    "FP32 native prefix output requires a native-capable fixture"
                )
            # Only the isolated FP32-O AITER build accepts this output buffer.
            self.on_op.prefix_output = torch.empty_like(
                self.on_op.prefix_output, dtype=torch.float32
            )
        if os.getenv("MKT_CASCADE_FORCE_NATIVE_LARGE", "0") == "1":
            # An explicitly labelled experiment, never a production guard change.
            self.on_op.NATIVE_PREFIX_BYTE_LIMIT = 1 << 63
        self.on_op.plan(case.table, case.history)

    def prefix_path(self):
        span = self.case.kv.numel() * self.case.kv.element_size()
        return (
            "aiter_native"
            if self.on_op.native_prefix and span < self.on_op.NATIVE_PREFIX_BYTE_LIMIT
            else "triton_64bit"
        )

    def off(self):
        return self.off_op.forward(self.case.q, self.cache, self.params).view_as(
            self.case.q
        )

    def on(self):
        return self.on_op.run(self.case.q, self.case.kv)

    def update(self, new):
        case = self.case
        case.q.copy_(new.q)
        case.kv.index_copy_(
            0, new.physical_ids.to(device="cuda", dtype=torch.long), new.packed
        )
        case.table.copy_(new.table)
        case.history.copy_(new.history)
        case.k, case.v = new.k, new.v
        case.logical_table, case.physical_ids = new.logical_table, new.physical_ids
        case.request_histories, case.boundaries = new.request_histories, new.boundaries
        case.seed = new.seed
        self.params.seq_lens.copy_((case.history + 1).cuda())
        self.params.kv_cache_block_id_device.copy_(case.table.cuda())
        self.on_op.plan(case.table, case.history)


def reference(case, selected=None, dtype=torch.float32):
    """Independent dense QK/softmax/PV on canonical pages, no production helper."""
    selected = list(range(case.q.shape[0])) if selected is None else list(selected)
    result = []
    for start in range(0, len(selected), 16):
        row_ids = selected[start : start + 16]
        # Reference chunks may straddle requests with different lengths.
        lengths = case.history[row_ids].to(device="cuda", dtype=torch.long) + 1
        max_length = int(lengths.max().item())
        page_ids = (
            case.logical_table[row_ids, : (max_length + PAGE - 1) // PAGE].cuda().long()
        )
        k = case.k[page_ids].permute(0, 2, 1, 3, 4).flatten(2, 3)[:, :, :max_length]
        v = case.v[page_ids].permute(0, 2, 1, 3, 4).flatten(2, 3)[:, :, :max_length]
        k = k.repeat_interleave(HEADS // KV_HEADS, dim=1).to(dtype)
        v = v.repeat_interleave(HEADS // KV_HEADS, dim=1).to(dtype)
        q = case.q[row_ids].to(dtype)
        logits = torch.einsum("bhd,bhld->bhl", q, k) * DIM**-0.5
        mask = torch.arange(max_length, device="cuda")[None, :] >= lengths[:, None]
        logits.masked_fill_(mask[:, None, :], -float("inf"))
        result.append(torch.einsum("bhl,bhld->bhd", logits.softmax(-1), v))
    return torch.cat(result)


def error_metrics(actual, expected):
    difference = actual.double() - expected.double()
    denom = expected.double().norm().clamp_min(1e-30)
    return {
        "max_abs": difference.abs().max().item(),
        "relative_l2": (difference.norm() / denom).item(),
        "rms": difference.square().mean().sqrt().item(),
    }


def finite_summary(tensor):
    finite = torch.isfinite(tensor)
    nonfinite = int((~finite).sum().item())
    return {
        "numel": tensor.numel(),
        "nonfinite": nonfinite,
        "nan": int(torch.isnan(tensor).sum().item()),
        "inf": int(torch.isinf(tensor).sum().item()),
        "first_nonfinite_flat_indices": (
            (~finite).flatten().nonzero()[:8].flatten().cpu().tolist()
            if nonfinite
            else []
        ),
    }


def assert_accuracy(test, ab, off, on, *, all_rows=True):
    case = ab.case
    selected = sorted(
        set(r for b in case.boundaries for r in (b - 1, b) if 0 <= r < case.q.shape[0])
    )
    selected += [
        b + (e - b) // 2 for b, e in zip(case.boundaries[:-1], case.boundaries[1:])
    ]
    selected = sorted(set(selected))
    rows = None if all_rows else selected
    expected = reference(case, rows)
    actual_off = off if rows is None else off[rows]
    actual_on = on if rows is None else on[rows]
    tensors = {"reference_fp32": expected, "off": actual_off, "on": actual_on}
    if any(not bool(torch.isfinite(tensor).all()) for tensor in tensors.values()):
        emit(
            f"nonfinite_r{len(case.request_histories)}_h{case.request_histories[0]}_high{int(case.high_address)}",
            {
                **case.describe(),
                "classification": "pre_assert_failure_diagnostic",
                "finite": {
                    name: finite_summary(tensor) for name, tensor in tensors.items()
                },
                "cascade_prefix": ab.prefix_path(),
            },
        )
    metrics = {}
    for name, actual in (("off_vs_fp32", actual_off), ("on_vs_fp32", actual_on)):
        torch.testing.assert_close(
            actual.float(),
            expected,
            atol=ATOL,
            rtol=RTOL,
            msg=lambda detail: name + ": " + detail,
        )
        metrics[name] = error_metrics(actual, expected)
        test.assertLessEqual(metrics[name]["relative_l2"], RELATIVE_L2_LIMIT)
    torch.testing.assert_close(on.float(), off.float(), atol=AB_ATOL, rtol=AB_RTOL)
    metrics["on_vs_off"] = error_metrics(on, off)
    test.assertLessEqual(metrics["on_vs_off"]["relative_l2"], AB_RELATIVE_L2_LIMIT)
    fp64 = reference(case, selected, torch.float64)
    fp32 = expected[selected] if rows is None else expected
    torch.testing.assert_close(
        fp32.double(), fp64, atol=REFERENCE_ATOL, rtol=REFERENCE_RTOL
    )
    metrics["fp32_vs_fp64"] = error_metrics(fp32, fp64)
    metrics["fp64_rows"] = selected
    metrics["fp32_rows"] = case.q.shape[0] if rows is None else len(rows)
    return metrics


class MktCascadeTestCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if torch.version.hip is None or not torch.cuda.is_available():
            raise RuntimeError("MI308 ROCm GPU required; missing hardware is not PASS")
        torch.set_float32_matmul_precision("highest")
        torch.backends.cuda.matmul.allow_tf32 = False

    def setUp(self):
        variant = tail_variant()
        if variant != "production":
            # A test-scoped patch also lets the stage recorder wrap the selected
            # kernel. Missing experimental code is an error, never a fallback.
            kernel = tail_kernels()[variant]
            patcher = mock.patch.object(cascade_mod, "_private_tail_merge", kernel)
            patcher.start()
            self.addCleanup(patcher.stop)


class MktCascadeAccuracyTest(MktCascadeTestCase):
    def test_mkt_accuracy_matrix(self):
        for requests in requested_batches():
            for high_address in (False, True):
                with self.subTest(requests=requests, high_address=high_address):
                    case = make_case(requests, high_address=high_address)
                    self.assertFalse(case.q.is_contiguous())
                    ab = AttentionAB(case)
                    off, on = ab.off(), ab.on()
                    expected_shared = sum(
                        (h // PAGE) * PAGE * BEAMS for h in case.request_histories
                    )
                    self.assertEqual(ab.on_op.shared_tokens, expected_shared)
                    for boundary in case.boundaries[1:-1]:
                        self.assertIn(boundary, ab.on_op.qptr_host.tolist())
                    metrics = assert_accuracy(self, ab, off, on)
                    emit(
                        f"accuracy_r{requests}_high{int(high_address)}",
                        {
                            **case.describe(),
                            "metrics": metrics,
                            "cascade_off": "AiterDecodeAttnOpTriton.forward/pa_decode_gluon",
                            "cascade_prefix": ab.prefix_path(),
                            "experimental_native_large": os.getenv(
                                "MKT_CASCADE_FORCE_NATIVE_LARGE", "0"
                            )
                            == "1",
                            "reference_matmul_precision": torch.get_float32_matmul_precision(),
                            "reference_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
                            "threshold_source": "existing test_beam_search_cascade.py: .005/.03; AB doubled as triangle-bound diagnostic, not model TopK acceptance",
                            "thresholds": {
                                "atol": ATOL,
                                "rtol": RTOL,
                                "relative_l2": RELATIVE_L2_LIMIT,
                                "ab_atol": AB_ATOL,
                                "ab_rtol": AB_RTOL,
                                "ab_relative_l2": AB_RELATIVE_L2_LIMIT,
                            },
                        },
                    )
                    del ab, case, off, on

    def test_graph_replay_changes_all_inputs_and_metadata(self):
        for high_address in (False, True):
            with self.subTest(high_address=high_address):
                case = make_case(2, high_address=high_address)
                ab = AttentionAB(case)
                off_graph, off = capture(ab.off)
                on_graph, on = capture(ab.on)
                pointers = (
                    off.data_ptr(),
                    on.data_ptr(),
                    ab.on_op.table.data_ptr(),
                    ab.on_op.prefix_qptr.data_ptr(),
                    ab.on_op.row_cuts.data_ptr(),
                )
                previous = on.clone()
                for step, (boundary, histories) in enumerate(
                    ((503, (698, 820)), (497, (710, 832)), (500, HISTORIES[:2]))
                ):
                    new = make_case(
                        2,
                        seed=SEED + step + 1,
                        high_address=high_address,
                        histories=histories,
                        request_rows=(boundary, 1000 - boundary),
                    )
                    self.assertFalse(torch.equal(case.q, new.q))
                    self.assertFalse(torch.equal(case.table, new.table))
                    self.assertFalse(torch.equal(case.history, new.history))
                    ab.update(new)
                    self.assertIn(boundary, ab.on_op.qptr_host.tolist())
                    # Poison outputs outside graph capture to reject stale-output passes.
                    off.fill_(float("nan"))
                    on.fill_(float("nan"))
                    off_graph.replay()
                    on_graph.replay()
                    # run() returns a persistent workspace: clone replay outputs
                    # BEFORE eager execution can overwrite their underlying data.
                    replay_off, replay_on = off.clone(), on.clone()
                    eager_off, eager_on = ab.off().clone(), ab.on().clone()
                    torch.testing.assert_close(replay_off, eager_off, atol=0, rtol=0)
                    torch.testing.assert_close(replay_on, eager_on, atol=0, rtol=0)
                    own_eager_metrics = {
                        "off_replay_vs_own_eager": error_metrics(replay_off, eager_off),
                        "on_replay_vs_own_eager": error_metrics(replay_on, eager_on),
                    }
                    metrics = assert_accuracy(self, ab, replay_off, replay_on)
                    self.assertFalse(torch.equal(previous, replay_on))
                    self.assertEqual(
                        pointers,
                        (
                            off.data_ptr(),
                            on.data_ptr(),
                            ab.on_op.table.data_ptr(),
                            ab.on_op.prefix_qptr.data_ptr(),
                            ab.on_op.row_cuts.data_ptr(),
                        ),
                    )
                    previous = replay_on
                    emit(
                        f"graph_high{int(high_address)}_step{step}",
                        {
                            **case.describe(),
                            "metrics": metrics,
                            "request_boundary": boundary,
                            "cascade_prefix": ab.prefix_path(),
                            "stable_addresses": True,
                            "graph_vs_own_eager": own_eager_metrics,
                            "graph_vs_own_eager_thresholds": {"atol": 0, "rtol": 0},
                        },
                    )
                    del new
                del ab, case, off, on, off_graph, on_graph


class MktCascadeAddressBoundaryTest(MktCascadeTestCase):
    """Manual coverage of physical pages below, at and above 2/4 GiB."""

    def test_pages_straddle_address_boundary(self):
        case = make_case(2, high_address=True, boundary_address=True)
        block_bytes = case.kv.stride(0) * case.kv.element_size()
        boundary = high_address_gib() * (1 << 30)
        offsets = case.physical_ids.long() * block_bytes
        for offset in (
            boundary - 2 * block_bytes,
            boundary,
            boundary + 2 * block_bytes,
        ):
            self.assertTrue(bool((offsets == offset).any()))
        # Both shared-prefix and private-tail page sets must cross the boundary.
        for ids in (case.physical_ids[:128], case.physical_ids[128:-1]):
            self.assertLess(int(ids.min()) * block_bytes, boundary)
            self.assertGreater(int(ids.max()) * block_bytes, boundary)
        ab = AttentionAB(case)
        self.assertIn(BEAMS, ab.on_op.qptr_host.tolist())
        off, on = ab.off().clone(), ab.on().clone()
        emit(
            "accuracy_address_boundary",
            {
                **case.describe(),
                "metrics": assert_accuracy(self, ab, off, on),
                "cascade_prefix": ab.prefix_path(),
                "experimental_native_large": os.getenv(
                    "MKT_CASCADE_FORCE_NATIVE_LARGE", "0"
                )
                == "1",
            },
        )


def make_dispatch_case(requests, mode, *, seed=SEED, high_address=False, tail=11):
    case = make_case(
        requests,
        seed=seed,
        high_address=high_address,
        histories=(704 + tail - 1,) * requests,
    )
    if mode != "short":
        page = {"long": 40, "unshared": 0}[mode]
        case.logical_table[:, page] = requests * 64 + torch.arange(
            case.q.shape[0], dtype=torch.int32
        )
        case.table = case.physical_ids[case.logical_table.long()].contiguous()
    return case


def poison_inactive_private_v(case, tail):
    """Expose inactive-V loads multiplied by zero: IEEE NaN * 0 is still NaN."""
    first_private = len(case.request_histories) * 64
    case.v[first_private : first_private + case.q.shape[0], :, tail:, :] = float("nan")
    blocks = case.v.shape[0]
    packed_v = case.v.reshape(blocks, KV_HEADS, PAGE // 8, 8, DIM).permute(
        0, 1, 2, 4, 3
    )
    case.packed[:, 1].copy_(packed_v.contiguous().view_as(case.v))
    case.kv.index_copy_(0, case.physical_ids.cuda().long(), case.packed)
    return int(torch.isnan(case.v).sum().item())


class MktCascadeTailDispatchTest(MktCascadeTestCase):
    """Exercise the candidate's runtime guard with the real off entry and oracle."""

    def test_short_tail_lengths_one_through_sixteen(self):
        for high_address in (False, True):
            for tail in range(1, PAGE + 1):
                with self.subTest(high_address=high_address, tail=tail):
                    case = make_dispatch_case(
                        1, "short", high_address=high_address, tail=tail
                    )
                    poisoned = poison_inactive_private_v(case, tail)
                    self.assertEqual(poisoned, BEAMS * KV_HEADS * (PAGE - tail) * DIM)
                    ab = AttentionAB(case)
                    self.assertTrue(bool((ab.on_op.row_cuts_host == 704).all()))
                    self.assertTrue(
                        bool(
                            (
                                ab.on_op.lengths_host - ab.on_op.row_cuts_host == tail
                            ).all()
                        )
                    )
                    off, on = ab.off().clone(), ab.on().clone()
                    emit(
                        f"short_tail_{tail}_high{int(high_address)}",
                        {
                            **case.describe(),
                            "tail_tokens": tail,
                            "cascade_prefix": ab.prefix_path(),
                            "inactive_private_v_nan_elements": poisoned,
                            "metrics": assert_accuracy(self, ab, off, on),
                        },
                    )
                    del ab, case, off, on

    def test_graph_short_long_unshared_and_back(self):
        for high_address in (False, True):
            case = make_dispatch_case(2, "short", high_address=high_address)
            ab = AttentionAB(case)
            off_graph, off = capture(ab.off)
            on_graph, on = capture(ab.on)
            addresses = (off.data_ptr(), on.data_ptr(), ab.on_op.row_cuts.data_ptr())
            for step, (mode, tail) in enumerate(
                (("short", 1), ("long", 11), ("unshared", 16), ("short", 16))
            ):
                new = make_dispatch_case(
                    2, mode, seed=SEED + 20 + step, high_address=high_address, tail=tail
                )
                self.assertFalse(torch.equal(case.q, new.q))
                self.assertFalse(torch.equal(case.table, new.table))
                self.assertFalse(torch.equal(case.k, new.k))
                ab.update(new)
                expected_cut = {"short": 704, "long": 640, "unshared": 0}[mode]
                self.assertTrue(bool((ab.on_op.row_cuts_host == expected_cut).all()))
                off.fill_(float("nan"))
                on.fill_(float("nan"))
                off_graph.replay()
                on_graph.replay()
                replay_off, replay_on = off.clone(), on.clone()
                eager_off, eager_on = ab.off().clone(), ab.on().clone()
                torch.testing.assert_close(replay_off, eager_off, atol=0, rtol=0)
                torch.testing.assert_close(replay_on, eager_on, atol=0, rtol=0)
                self.assertEqual(
                    addresses,
                    (off.data_ptr(), on.data_ptr(), ab.on_op.row_cuts.data_ptr()),
                )
                emit(
                    f"dispatch_graph_high{int(high_address)}_step{step}_{mode}",
                    {
                        **case.describe(),
                        "dispatch_mode": mode,
                        "shared_cut": expected_cut,
                        "actual_tail_tokens": 704 + tail - expected_cut,
                        "cascade_prefix": ab.prefix_path(),
                        "stable_addresses": True,
                        "metrics": assert_accuracy(self, ab, replay_off, replay_on),
                        "graph_vs_own_eager": {
                            "off": error_metrics(replay_off, eager_off),
                            "on": error_metrics(replay_on, eager_on),
                        },
                        "graph_vs_own_eager_thresholds": {"atol": 0, "rtol": 0},
                    },
                )
                del new
            del ab, case, off, on, off_graph, on_graph


class MktCascadeTailSafetyTest(MktCascadeTestCase):
    """Cascade acceptance independent of the separately retained off failure."""

    def test_inactive_v_nan_safety_low_and_four_gib(self):
        self.assertEqual(
            high_address_gib(),
            4,
            "This target specifically requires the 4 GiB boundary",
        )
        for high_address in (False, True):
            for tail in range(1, PAGE + 1):
                with self.subTest(high_address=high_address, tail=tail):
                    case = make_dispatch_case(
                        1, "short", high_address=high_address, tail=tail
                    )
                    ab = AttentionAB(case)
                    clean = ab.on().clone()
                    poisoned = poison_inactive_private_v(case, tail)
                    self.assertEqual(poisoned, BEAMS * KV_HEADS * (PAGE - tail) * DIM)
                    output = ab.on().clone()
                    self.assertTrue(bool(torch.isfinite(output).all()))
                    # This exact gate compares the selected backend to itself;
                    # inactive cache values cannot change its valid result.
                    torch.testing.assert_close(output, clean, atol=0, rtol=0)
                    expected = reference(case)
                    selected = (0, 249, 499)
                    fp64 = reference(case, selected, torch.float64)
                    torch.testing.assert_close(
                        expected[list(selected)].double(),
                        fp64,
                        atol=REFERENCE_ATOL,
                        rtol=REFERENCE_RTOL,
                    )
                    torch.testing.assert_close(
                        output.float(), expected, atol=ATOL, rtol=RTOL
                    )
                    measured = error_metrics(output, expected)
                    self.assertLessEqual(measured["relative_l2"], RELATIVE_L2_LIMIT)
                    emit(
                        f"tail_safety_{tail}_high{int(high_address)}",
                        {
                            **case.describe(),
                            "tail_tokens": tail,
                            "cascade_prefix": ab.prefix_path(),
                            "inactive_private_v_nan_elements": poisoned,
                            "poisoned_vs_own_clean": error_metrics(output, clean),
                            "on_vs_fp32": measured,
                            "fp32_rows": case.q.shape[0],
                            "fp32_vs_fp64": error_metrics(
                                expected[list(selected)], fp64
                            ),
                            "fp64_rows": selected,
                            "thresholds": {
                                "atol": ATOL,
                                "rtol": RTOL,
                                "relative_l2": RELATIVE_L2_LIMIT,
                                "own_clean_atol": 0,
                                "own_clean_rtol": 0,
                            },
                            "threshold_source": "existing kernel UT .005/.03 and predeclared .01 relative L2; no adjustment after measurements",
                            "cascade_off_status": "KNOWN_FAIL_SEPARATELY_RETAINED_NOT_PASS",
                            "cascade_off_counterexample": OFF_NAN_COUNTEREXAMPLE,
                        },
                    )
                    del ab, case, clean, output


class MktCascadeNanPoisonDiagnosisTest(MktCascadeTestCase):
    """Collect each backend's counterexample without disguising it as acceptance.

    Original poison acceptance tests remain unchanged and can still fail. This
    target asserts fixture integrity, then records all backend results even if
    one is nonfinite. Default: one request, small pool, tails 1 and 16.
    """

    def test_locate_first_nonfinite_backend(self):
        tails = tuple(
            int(value)
            for value in os.getenv("MKT_CASCADE_POISON_TAILS", "1,16").split(",")
        )
        if not tails or any(tail < 1 or tail > PAGE for tail in tails):
            raise ValueError(
                "MKT_CASCADE_POISON_TAILS must contain lengths 1 through 16"
            )
        high_address = os.getenv("MKT_CASCADE_POISON_HIGH_ADDRESS", "0") == "1"
        registry = tail_kernels()
        # Keep this historical locator's arms; wave safety uses the selected
        # candidate acceptance targets under the explicit FP32 requirement.
        kernels = {
            name: registry[name]
            for name in ("production", "common", "layout", "layout_only")
        }
        for tail in tails:
            case = make_dispatch_case(1, "short", high_address=high_address, tail=tail)
            ab = AttentionAB(case)
            clean_reference = reference(case)
            clean_outputs = {"off": ab.off().clone()}
            for name, kernel in kernels.items():
                with mock.patch.object(cascade_mod, "_private_tail_merge", kernel):
                    clean_outputs[name] = ab.on().clone()
            poisoned = poison_inactive_private_v(case, tail)
            # Independently decode the actual shuffled V bytes copied to GPU;
            # the packer must not have moved NaNs into any attended position.
            packed_live = case.kv.index_select(0, case.physical_ids.cuda().long())
            unpacked_v = packed_live[:, 1].reshape(-1, KV_HEADS, PAGE // 8, DIM, 8)
            unpacked_v = unpacked_v.permute(0, 1, 2, 4, 3).reshape_as(case.v)
            torch.testing.assert_close(
                unpacked_v, case.v, atol=0, rtol=0, equal_nan=True
            )
            self.assertTrue(bool(torch.isfinite(unpacked_v[:64]).all()))
            self.assertTrue(
                bool(torch.isfinite(unpacked_v[64 : 64 + BEAMS, :, :tail]).all())
            )
            self.assertEqual(int(torch.isnan(unpacked_v).sum()), poisoned)
            poisoned_reference = reference(case)
            torch.testing.assert_close(
                poisoned_reference, clean_reference, atol=0, rtol=0
            )
            fp64 = reference(case, (0, 249, 499), torch.float64)
            torch.testing.assert_close(
                poisoned_reference[[0, 249, 499]].double(),
                fp64,
                atol=REFERENCE_ATOL,
                rtol=REFERENCE_RTOL,
            )
            poisoned_outputs = {"off": ab.off().clone()}
            prefix_finite = {}
            for name, kernel in kernels.items():
                with mock.patch.object(cascade_mod, "_private_tail_merge", kernel):
                    poisoned_outputs[name] = ab.on().clone()
                prefix = (
                    ab.on_op.prefix_output
                    if ab.prefix_path() == "aiter_native"
                    else ab.on_op.partial_out
                )
                prefix_finite[name] = finite_summary(prefix)
            result = {}
            for name in poisoned_outputs:
                clean, poisoned_output = clean_outputs[name], poisoned_outputs[name]
                summary = finite_summary(poisoned_output)
                result[name] = {
                    "clean_finite": finite_summary(clean),
                    "poisoned_finite": summary,
                }
                if not summary["nonfinite"]:
                    result[name]["poisoned_vs_clean"] = error_metrics(
                        poisoned_output, clean
                    )
                    result[name]["poisoned_vs_fp32"] = error_metrics(
                        poisoned_output, poisoned_reference
                    )
                else:
                    result[name][
                        "poisoned_vs_fp32"
                    ] = "NOT_VALID: nonfinite backend output"
            emit(
                f"nan_poison_diagnosis_tail{tail}_high{int(high_address)}",
                {
                    **case.describe(),
                    "classification": "controlled_counterexample_not_acceptance",
                    "tail_tokens": tail,
                    "inactive_private_v_nan_elements": poisoned,
                    "packed_v_matches_semantic_v": True,
                    "all_attended_v_finite": True,
                    "reference_unchanged_by_poison": True,
                    "reference_fp32_finite": finite_summary(poisoned_reference),
                    "reference_fp64_finite": finite_summary(fp64),
                    "cascade_prefix": ab.prefix_path(),
                    "prefix_finite": prefix_finite,
                    "backend_results": result,
                    "warning": "Target success certifies diagnostic execution and fixture integrity; original poison acceptance failures remain failures",
                },
            )
            del ab, case, clean_outputs, poisoned_outputs, packed_live, unpacked_v


class MktCascadeGenericRegressionTest(unittest.TestCase):
    def test_existing_generic_suite_with_selected_tail(self):
        # Import only for this manual target, whose srcs include the unchanged
        # generic suite. Inherit its cases and use our per-case patch/cleanup.
        from rtp_llm.models_py.triton_kernels.attention.test.test_beam_search_cascade import (
            RocmBeamCascadeTest,
        )

        class SelectedTailRegression(MktCascadeTestCase, RocmBeamCascadeTest):
            pass

        suite = unittest.defaultTestLoader.loadTestsFromTestCase(SelectedTailRegression)
        names = [test.id() for test in suite]
        result = unittest.TestResult()
        suite.run(result)
        emit(
            "generic_regression",
            {
                "tail_variant": tail_variant(),
                "test_names": names,
                "tests_run": result.testsRun,
                "failures": [(test.id(), detail) for test, detail in result.failures],
                "errors": [(test.id(), detail) for test, detail in result.errors],
                "skipped": [(test.id(), reason) for test, reason in result.skipped],
                "threshold_source": "unchanged existing RocmBeamCascadeTest assertions",
            },
        )
        self.assertEqual(result.testsRun, len(names))
        self.assertFalse(
            result.skipped, "A skipped generic case is not candidate validation"
        )
        self.assertTrue(result.wasSuccessful(), str(result.failures + result.errors))


def capture(fn):
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            fn()
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        output = fn()
    graph.replay()
    torch.cuda.synchronize()
    return graph, output


class LaunchRecorder:
    """Record the exact production Triton launch without reproducing its geometry."""

    def __init__(self, kernel, destination, metadata=None):
        self.kernel, self.destination = kernel, destination
        self.metadata = metadata

    def __getitem__(self, grid):
        launch = self.kernel[grid]

        def run(*args, **kwargs):
            if hasattr(self.kernel, "launch_kwargs"):
                kwargs = self.kernel.launch_kwargs(kwargs)
            self.destination.append(lambda: launch(*args, **kwargs))
            if self.metadata is not None:
                parameters = dict(zip(self.kernel.arg_names, args))
                parameters.update(kwargs)
                self.metadata.append(
                    {
                        "symbol": self.kernel.__name__,
                        "grid": tuple(grid),
                        "index64": bool(parameters["INDEX64"]),
                        "native_prefix": bool(parameters.get("NATIVE_PREFIX", False)),
                        "row_tile": int(parameters.get("ROW_TILE", 1)),
                        "N": int(parameters["N"]),
                        "G": int(parameters["G"]),
                        "num_warps": int(parameters["num_warps"]),
                        "part_out_dtype": str(parameters["PART_OUT"].dtype),
                    }
                )
            return launch(*args, **kwargs)

        return run


def cascade_stage_calls(ab, tail_metadata=None):
    """Diagnostic stage timings retain actual run() arguments and workspace.

    Stages are timed independently on fixed inputs; their cache state differs
    from a full forward, so stage medians must not be added to claim latency.
    """
    import aiter

    prefixes, tails = [], []
    original_native = aiter.mha_batch_prefill_func

    def native(*args, **kwargs):
        prefixes.append(lambda: original_native(*args, **kwargs))
        return original_native(*args, **kwargs)

    with mock.patch.object(
        aiter, "mha_batch_prefill_func", side_effect=native
    ), mock.patch.object(
        cascade_mod,
        "_shared_prefix",
        LaunchRecorder(cascade_mod._shared_prefix, prefixes),
    ), mock.patch.object(
        cascade_mod,
        "_private_tail_merge",
        LaunchRecorder(cascade_mod._private_tail_merge, tails, tail_metadata),
    ):
        ab.on()
    if len(prefixes) != 1 or len(tails) != 1:
        raise AssertionError(
            "Production launch decomposition changed; update the recorder"
        )
    return {"cascade_prefix_only": prefixes[0], "cascade_tail_merge_only": tails[0]}


def measure(fn):
    # Both paths are captured once, then benchmark graph replay with the same
    # event implementation. Cold-cache clearing lies outside timed events.
    # do_bench_cudagraph gives repeated-replay hot-cache lower-bound throughput.
    graph, output = capture(fn)
    cold = triton.testing.do_bench(graph.replay, warmup=20, rep=50, return_mode="all")
    hot = triton.testing.do_bench_cudagraph(fn, rep=20, return_mode="all")
    # Keep graph output alive until all timings complete.
    del output
    return {"cold_graph_ms": list(cold), "hot_graph_ms": list(hot)}


class MktCascadeProfileTest(MktCascadeTestCase):
    """Manual kernel-dispatch evidence; never use profiler durations as timing."""

    def test_profile_actual_attention_scopes(self):
        mode = os.getenv("MKT_CASCADE_PROFILE_MODE", "torch")
        if mode not in ("torch", "external"):
            raise ValueError("MKT_CASCADE_PROFILE_MODE must be torch or external")
        output = Path(
            os.getenv(
                "MKT_CASCADE_OUTPUT_DIR",
                os.getenv("TEST_UNDECLARED_OUTPUTS_DIR", "cascade_profile_results"),
            )
        )
        output.mkdir(parents=True, exist_ok=True)
        for requests in requested_batches():
            for high_address in (False, True):
                case = make_case(requests, high_address=high_address)
                ab = AttentionAB(case)
                ab.off()
                ab.on()
                tail_metadata = []
                functions = {
                    "off": ab.off,
                    "on": ab.on,
                    **cascade_stage_calls(ab, tail_metadata),
                }
                self.assertEqual(len(tail_metadata), 1)
                expected_warps = {"layout_w1": 1, "layout_w4": 4, "shared_k_w1": 1}.get(
                    tail_variant(), 2
                )
                self.assertEqual(tail_metadata[0]["num_warps"], expected_warps)
                native_before = loaded_native_modules()
                expected_native = os.getenv("MKT_CASCADE_EXPECTED_NATIVE_SHA256")
                fp32_requested = os.getenv("MKT_CASCADE_NATIVE_FP32_O") == "1"
                if fp32_requested:
                    self.assertTrue(
                        expected_native,
                        "FP32 profile requires an expected native library hash",
                    )
                    self.assertEqual(len(native_before), 1)
                    self.assertEqual(native_before[0]["sha256"], expected_native)
                    self.assertEqual(ab.on_op.prefix_output.dtype, torch.float32)
                    self.assertEqual(
                        tail_metadata[0]["part_out_dtype"], "torch.float32"
                    )
                scopes = {}
                for name, fn in functions.items():
                    for _ in range(3):
                        fn()
                    torch.cuda.synchronize()
                    trace_name = f"profile_r{requests}_high{int(high_address)}_{name}"
                    if mode == "external":
                        # ROCprof owns tracing/counters in this mode. Starting
                        # Kineto concurrently could conflict with its SDK.
                        for _ in range(3):
                            fn()
                        torch.cuda.synchronize()
                        scopes[name] = {
                            "measured_repetitions": 3,
                            "kernel_counts": "external collector",
                        }
                    else:
                        with torch.profiler.profile(
                            activities=[
                                torch.profiler.ProfilerActivity.CPU,
                                torch.profiler.ProfilerActivity.CUDA,
                            ]
                        ) as profile:
                            for _ in range(3):
                                with torch.profiler.record_function(trace_name):
                                    fn()
                            torch.cuda.synchronize()
                        trace_path = output / (trace_name + ".json")
                        profile.export_chrome_trace(str(trace_path))
                        trace = json.loads(trace_path.read_text())
                        # Count kernel-category events, excluding GPU memcpy and
                        # runtime calls. Preserve the raw trace for correlation.
                        counts = Counter(
                            event["name"]
                            for event in trace["traceEvents"]
                            if "kernel" in event.get("cat", "").lower()
                        )
                        self.assertTrue(
                            counts, "Missing GPU kernel events is not dispatch evidence"
                        )
                        scopes[name] = {
                            "measured_repetitions": 3,
                            "kernel_counts": dict(counts),
                            "trace": trace_name + ".json",
                        }
                native_after = loaded_native_modules()
                self.assertEqual(
                    native_before,
                    native_after,
                    "Native library changed during profiling",
                )
                emit(
                    f"dispatch_profile_r{requests}_high{int(high_address)}",
                    {
                        **case.describe(),
                        "mode": mode,
                        "scopes": scopes,
                        "cascade_prefix": ab.prefix_path(),
                        "tail_launch_metadata": tail_metadata,
                        "FP32_O_requested": fp32_requested,
                        "prefix_output_dtype": str(ab.on_op.prefix_output.dtype),
                        "loaded_native_modules_before": native_before,
                        "loaded_native_modules_after": native_after,
                        "expected_native_sha256": expected_native,
                        "classification": "dispatch_evidence_not_performance_acceptance",
                        "external_launch_note": "initial off/on plus one on for decomposition; each scope has 3 warmup and 3 collected calls",
                        "experimental_native_large": os.getenv(
                            "MKT_CASCADE_FORCE_NATIVE_LARGE", "0"
                        )
                        == "1",
                    },
                )
                del functions, ab, case


@unittest.skipUnless(
    os.getenv("MKT_CASCADE_RUN_PERF", "0") == "1",
    "Performance target is opt-in; not measured",
)
class MktCascadePerformanceTest(MktCascadeTestCase):
    def test_clean_event_matrix(self):
        if os.getenv("MKT_CASCADE_COMPARE_TAILS"):
            self.compare_tail_variants()
            return
        for requests in requested_batches():
            for high_address in (False, True):
                case = make_case(requests, high_address=high_address)
                ab = AttentionAB(case)
                # Compile/warm each actual entry before timing. Accuracy is run
                # in a separate process/target, not interleaved with samples.
                for fn in (ab.off, ab.on):
                    for _ in range(3):
                        fn()
                torch.cuda.synchronize()
                functions = {
                    "cascade_off_complete": ab.off,
                    "cascade_on_complete": ab.on,
                    **cascade_stage_calls(ab),
                }
                samples = {
                    key: {"cold_graph_ms": [], "hot_graph_ms": []} for key in functions
                }
                for round_index in range(3):
                    # Alternate order to expose drift instead of always timing
                    # the candidate after the warmed baseline.
                    names = list(functions)
                    if round_index % 2:
                        names.reverse()
                    for name in names:
                        measured = measure(functions[name])
                        for cache_mode, values in measured.items():
                            samples[name][cache_mode].append(values)
                summary = {}
                for name, modes in samples.items():
                    summary[name] = {}
                    for mode, rounds in modes.items():
                        flat = [sample for batch in rounds for sample in batch]
                        summary[name][mode] = {
                            "median_us": statistics.median(flat) * 1000,
                            "min_us": min(flat) * 1000,
                            "max_us": max(flat) * 1000,
                            "round_median_us": [
                                statistics.median(batch) * 1000 for batch in rounds
                            ],
                        }
                emit(
                    f"performance_r{requests}_high{int(high_address)}",
                    {
                        **case.describe(),
                        "cascade_prefix": ab.prefix_path(),
                        "summary": summary,
                        "raw_samples": samples,
                        "device": torch.cuda.get_device_name(),
                        "torch": torch.__version__,
                        "hip": torch.version.hip,
                        "timer": "triton.testing.do_bench / do_bench_cudagraph, GPU events",
                        "scope": "whole decode attention after RoPE/cache write; CPU plan, RoPE and model excluded",
                        "cold_cache": "Triton driver cache-clear before every replay, outside timed events; best-effort eviction",
                        "hot_cache": "same inputs replayed repeatedly; lower-bound latency, not service QPS",
                        "stage_warning": "independent stage timings are diagnostic; do not sum them",
                        "experimental_native_large": os.getenv(
                            "MKT_CASCADE_FORCE_NATIVE_LARGE", "0"
                        )
                        == "1",
                    },
                )
                del functions, ab, case

    def compare_tail_variants(self):
        from rtp_llm.models_py.triton_kernels.attention import (
            beam_search_cascade_tail_candidate as candidate,
        )
        from rtp_llm.models_py.triton_kernels.attention import (
            beam_search_cascade_tail_layout_only as layout_only,
        )
        from rtp_llm.models_py.triton_kernels.attention import (
            beam_search_cascade_tail_shared_k as shared_k,
        )

        self.assertEqual(
            tail_variant(),
            "production",
            "MKT_CASCADE_COMPARE_TAILS cannot be combined with an outer candidate patch",
        )
        variants = tuple(
            value.strip()
            for value in os.environ["MKT_CASCADE_COMPARE_TAILS"].split(",")
        )
        available = tail_kernels()
        if (
            len(variants) < 2
            or len(set(variants)) != len(variants)
            or "production" not in variants
            or any(name not in available for name in variants)
        ):
            raise ValueError(
                "MKT_CASCADE_COMPARE_TAILS requires unique production plus registered candidates"
            )
        fp32_requested = os.getenv("MKT_CASCADE_NATIVE_FP32_O") == "1"
        if (
            any(name in ("layout_w1", "layout_w4", "shared_k_w1") for name in variants)
            and not fp32_requested
        ):
            raise ValueError(
                "Wave geometry comparisons require MKT_CASCADE_NATIVE_FP32_O=1"
            )
        expected_native = os.getenv("MKT_CASCADE_EXPECTED_NATIVE_SHA256")
        if fp32_requested and not expected_native:
            raise ValueError(
                "FP32 performance comparisons require MKT_CASCADE_EXPECTED_NATIVE_SHA256"
            )
        variant_orders = comparison_variant_orders(variants)
        source_paths = (
            Path(__file__),
            Path(cascade_mod.__file__),
            Path(candidate.__file__),
            Path(layout_only.__file__),
            Path(shared_k.__file__),
        )

        def source_hashes():
            return {
                str(path): hashlib.sha256(path.read_bytes()).hexdigest()
                for path in source_paths
            }

        source_before = source_hashes()
        for requests in requested_batches():
            for high_address in (False, True):
                case = make_case(requests, high_address=high_address)
                # Immutable Q/KV/metadata are common, while each arm owns all
                # output, prefix, planning and Graph workspaces independently.
                operators, stages, prepared, launch_metadata = {}, {}, {}, {}
                for variant in variants:
                    self.assertIs(
                        cascade_mod._private_tail_merge, PRODUCTION_TAIL_KERNEL
                    )
                    ab = AttentionAB(case)
                    operators[variant] = ab
                    metadata = []
                    with mock.patch.object(
                        cascade_mod, "_private_tail_merge", available[variant]
                    ):
                        for _ in range(3):
                            ab.on()
                        # The recorder is inside the variant patch. It wraps
                        # the selected JIT kernel and restores that same symbol
                        # before the outer context restores production.
                        stages[variant] = cascade_stage_calls(ab, metadata)
                        for scope, fn in (
                            ("on", ab.on),
                            ("tail", stages[variant]["cascade_tail_merge_only"]),
                        ):
                            graph, output = capture(fn)
                            prepared[f"{scope}_{variant}"] = {
                                "fn": fn,
                                "graph": graph,
                                "output": output,
                                "variant": variant,
                            }
                    self.assertEqual(len(metadata), 1)
                    launch_metadata[variant] = metadata[0]
                    self.assertEqual(
                        metadata[0]["num_warps"],
                        {"layout_w1": 1, "layout_w4": 4, "shared_k_w1": 1}.get(
                            variant, 2
                        ),
                    )
                    self.assertEqual(
                        metadata[0]["index64"],
                        case.kv.numel() * case.kv.element_size() >= (1 << 31),
                    )
                self.assertIs(cascade_mod._private_tail_merge, PRODUCTION_TAIL_KERNEL)
                self.assertEqual(
                    len({ab.on_op.output.data_ptr() for ab in operators.values()}),
                    len(variants),
                )
                self.assertEqual(
                    len(
                        {ab.on_op.prefix_output.data_ptr() for ab in operators.values()}
                    ),
                    len(variants),
                )
                self.assertEqual(
                    len({ab.prefix_path() for ab in operators.values()}), 1
                )
                self.assertEqual(
                    len({meta["index64"] for meta in launch_metadata.values()}), 1
                )
                self.assertEqual(
                    len({meta["native_prefix"] for meta in launch_metadata.values()}), 1
                )
                control = operators["production"]
                for name, fn in (
                    ("off", control.off),
                    ("prefix", stages["production"]["cascade_prefix_only"]),
                ):
                    graph, output = capture(fn)
                    prepared[name] = {
                        "fn": fn,
                        "graph": graph,
                        "output": output,
                        "variant": "production",
                    }
                torch.cuda.synchronize()
                self.assertEqual(
                    source_before, source_hashes(), "Sources changed during capture"
                )
                native_before = loaded_native_modules()
                prefix_dtypes = {
                    name: {
                        "workspace": str(ab.on_op.prefix_output.dtype),
                        "actual_part_out": launch_metadata[name]["part_out_dtype"],
                    }
                    for name, ab in operators.items()
                }
                emit(
                    f"performance_compare_preflight_r{requests}_high{int(high_address)}",
                    {
                        "tail_variants": variants,
                        "FP32_O_requested": fp32_requested,
                        "prefix_dtypes": prefix_dtypes,
                        "tail_launch_metadata": launch_metadata,
                        "loaded_native_modules": native_before,
                        "expected_native_sha256": expected_native,
                        "timer_has_started": False,
                        "scope": "dtype and actual library identity checks; separate numerical gate required",
                    },
                )
                if fp32_requested:
                    self.assertEqual(
                        len(native_before),
                        1,
                        "Require exactly one actual native prefix library",
                    )
                    self.assertEqual(native_before[0]["sha256"], expected_native)
                    for dtypes in prefix_dtypes.values():
                        self.assertEqual(
                            dtypes,
                            {
                                "workspace": "torch.float32",
                                "actual_part_out": "torch.float32",
                            },
                        )
                samples = {
                    name: {"cold_graph_ms": [], "hot_graph_ms": []} for name in prepared
                }
                orders = []
                # New runs reverse each arm rotation in the second half as
                # well as group order. This balances pair order; historical
                # rotation-only runs are not retroactively relabeled balanced.
                for round_index, ordered in enumerate(variant_orders):
                    groups = [
                        ["off", "prefix"],
                        ["on_" + name for name in ordered],
                        ["tail_" + name for name in ordered],
                    ]
                    if round_index >= len(variants):
                        groups.reverse()
                    order = [name for group in groups for name in group]
                    orders.append(order)
                    for name in order:
                        item = prepared[name]
                        # Captured Graph replay no longer reads the global
                        # symbol. For the hot benchmark, its library captures
                        # the original eager entry under one outer patch; no
                        # per-call patch overhead or nested Graph capture.
                        cold = triton.testing.do_bench(
                            item["graph"].replay, warmup=20, rep=50, return_mode="all"
                        )
                        with mock.patch.object(
                            cascade_mod,
                            "_private_tail_merge",
                            available[item["variant"]],
                        ):
                            hot = triton.testing.do_bench_cudagraph(
                                item["fn"], rep=20, return_mode="all"
                            )
                        samples[name]["cold_graph_ms"].append(list(cold))
                        samples[name]["hot_graph_ms"].append(list(hot))
                self.assertIs(cascade_mod._private_tail_merge, PRODUCTION_TAIL_KERNEL)
                self.assertEqual(
                    source_before, source_hashes(), "Sources changed during measurement"
                )
                native_after = loaded_native_modules()
                self.assertEqual(
                    native_before,
                    native_after,
                    "Native library changed during measurement",
                )
                summary = {}
                for name, modes in samples.items():
                    summary[name] = {}
                    for mode, rounds in modes.items():
                        flat = [sample for batch in rounds for sample in batch]
                        round_medians = [
                            statistics.median(batch) * 1000 for batch in rounds
                        ]
                        summary[name][mode] = {
                            "median_us": statistics.median(flat) * 1000,
                            "min_us": min(flat) * 1000,
                            "max_us": max(flat) * 1000,
                            "round_median_us": round_medians,
                            "round_median_max_over_min": max(round_medians)
                            / min(round_medians),
                        }
                ratios = {}
                for variant in variants:
                    ratios[variant] = {}
                    for scope in ("on", "tail"):
                        ratios[variant][scope] = {}
                        for mode in ("cold_graph_ms", "hot_graph_ms"):
                            baseline = summary[scope + "_production"][mode]
                            measured = summary[scope + "_" + variant][mode]
                            ratios[variant][scope][mode] = {
                                "production_over_variant": baseline["median_us"]
                                / measured["median_us"],
                                "paired_round_ratios": [
                                    p / c
                                    for p, c in zip(
                                        baseline["round_median_us"],
                                        measured["round_median_us"],
                                    )
                                ],
                            }
                emit(
                    f"performance_compare_r{requests}_high{int(high_address)}",
                    {
                        **case.describe(),
                        "tail_variant": "same_process_comparison",
                        "tail_variants": variants,
                        "cascade_prefix": control.prefix_path(),
                        "tail_launch_metadata": launch_metadata,
                        "source_sha256": source_before,
                        "sources_unchanged_throughout_run": True,
                        "loaded_native_modules_before": native_before,
                        "loaded_native_modules_after": native_after,
                        "expected_native_sha256": expected_native,
                        "prefix_dtypes": prefix_dtypes,
                        "FP32_O_requested": fp32_requested,
                        "shared_input_addresses": {
                            "q": case.q.data_ptr(),
                            "kv": case.kv.data_ptr(),
                        },
                        "independent_workspace_addresses": {
                            name: {
                                "output": ab.on_op.output.data_ptr(),
                                "prefix": ab.on_op.prefix_output.data_ptr(),
                            }
                            for name, ab in operators.items()
                        },
                        "round_orders": orders,
                        "raw_samples": samples,
                        "summary": summary,
                        "round_order_protocol": "v2: rotations then reversed rotations; each pair n:n and each position twice",
                        "production_over_variant_ratios": ratios,
                        "device": torch.cuda.get_device_name(),
                        "torch": torch.__version__,
                        "hip": torch.version.hip,
                        "timer": "triton.testing.do_bench / do_bench_cudagraph, GPU events",
                        "scope": "whole decode attention after RoPE/cache write; separate accuracy gate required",
                        "cold_cache": "driver cache-clear before timed Graph replay; best-effort eviction",
                        "hot_cache": "repeated calls in a library-captured Graph; lower-bound latency, not service QPS",
                        "stage_warning": "independent stage timings are diagnostic; do not sum them",
                        "historical_comparison_warning": "Compare arms within this record; older guard-dependent INDEX64 runs are not a strict baseline",
                        "experimental_native_large": os.getenv(
                            "MKT_CASCADE_FORCE_NATIVE_LARGE", "0"
                        )
                        == "1",
                    },
                )
                del (
                    item,
                    graph,
                    output,
                    fn,
                    ab,
                    control,
                    operators,
                    stages,
                    prepared,
                    case,
                )


if __name__ == "__main__":
    unittest.main()
