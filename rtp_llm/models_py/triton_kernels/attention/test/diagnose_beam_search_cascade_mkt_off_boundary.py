"""Test-only draft: real MKT Gluon OFF boundaries with optional stages 3/4.

NOT_RUN. No CLI, kernel definition, kernel clone, or production integration.
Call only from a separately registered/locked diagnostic Bazel test. The return
value contains metadata, aggregate errors, hashes and failed coordinates only.
Diagnostic controls passing does not mean OFF passes its accuracy thresholds.
"""

import hashlib
import importlib
import inspect
from pathlib import Path
from unittest import mock

import torch

SOURCE_PINS = {
    "rtp_llm.models_py.modules.factory.attention.rocm_impl.aiter": "e47197d59e52a58b5826f5980b08a955eaaf3769efac0cbeb84b82d8957d4ae9",
    "aiter.ops.triton.gluon.pa_decode_gluon": "995267a0513bbc7e1dac1fe6810517219fed8a0efaa7cb754eec3aa9f36ee781",
    "aiter.ops.attention": "f9ff8408a09709c019dc25c08f42f6e9a35dbda37b87de2bfd050c73d1c54c3c",
}
DOT_NAME = "paged_attention_decode_v2_gluon_dot_kernel"
REDUCE_NAME = "paged_attention_decode_v2_reduce_kernel"
ATOL, RTOL, L2_LIMIT = 0.005, 0.03, 0.01
ORACLE_ATOL, ORACLE_RTOL = 2e-5, 2e-5
PARTITIONS = ((0, 256), (256, 512), (512, 715))
PRECAST_SOURCE_SHA256 = (
    "808140c42e54cc2dba1b8a07fe59330473d15d4f8ac9281c742355c34ecb7481"
)
PRECAST_POINTER = "actual_partial_fp32_ptr"
PROBABILITY_SOURCE_SHA256 = (
    "bd99ed6025676521c68d7df5311bc912b71437785eee802877fb0642dfa0e203"
)
PROBABILITY_POINTERS = (
    PRECAST_POINTER,
    "actual_probs_fp32_ptr",
    "actual_probs_bf16_ptr",
    "actual_scores_fp32_ptr",
)
PROBABILITY_ROWS = (321, 0, 250, 499)
PROBABILITY_HEADS = (14, 15)


def _require(condition, message):
    if not condition:
        raise AssertionError(message)


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _hash(tensor):
    data = tensor.detach().contiguous().view(torch.uint8).cpu().numpy()
    return hashlib.sha256(data.tobytes()).hexdigest()


def _tensor_record(tensor, *, with_hash=False):
    result = {
        "shape": list(tensor.shape),
        "stride": list(tensor.stride()),
        "dtype": str(tensor.dtype),
        "device": str(tensor.device),
        "storage_offset": tensor.storage_offset(),
        "storage_nbytes": tensor.untyped_storage().nbytes(),
    }
    if with_hash:
        result["sha256"] = _hash(tensor)
        result["nonfinite"] = int((~torch.isfinite(tensor)).sum())
    return result


def _exact(left, right):
    _require(
        left.shape == right.shape and left.dtype == right.dtype,
        "byte comparison requires identical shape/dtype",
    )
    lhs = (
        left.detach()
        .contiguous()
        .view(torch.uint8)
        .reshape(*left.shape, left.element_size())
    )
    rhs = (
        right.detach()
        .contiguous()
        .view(torch.uint8)
        .reshape(*right.shape, right.element_size())
    )
    mismatch = (lhs != rhs).any(dim=-1)
    indices = mismatch.nonzero().cpu().tolist()
    return {
        "byte_exact": not indices,
        "mismatch_count": len(indices),
        "failed_coordinates": indices[:64],
        "coordinates_truncated": len(indices) > 64,
        "left_sha256": _hash(left),
        "right_sha256": _hash(right),
    }


def _comparison(compare, actual, expected, *, oracle=False):
    """Preserve the caller's gates, but discard raw value/details from its report."""
    report = compare(
        actual,
        expected,
        atol=ORACLE_ATOL if oracle else ATOL,
        rtol=ORACLE_RTOL if oracle else RTOL,
        l2_limit=None if oracle else L2_LIMIT,
        mismatch_limit=0,
    )
    # Compute the complete coordinate list without exposing tensor values.
    a = actual.to(expected.dtype)
    bad = ~(
        torch.isclose(
            a,
            expected,
            atol=ORACLE_ATOL if oracle else ATOL,
            rtol=ORACLE_RTOL if oracle else RTOL,
        )
        & torch.isfinite(a)
        & torch.isfinite(expected)
    )
    coordinates = bad.nonzero().cpu().tolist()
    _require(len(coordinates) == report["mismatch_count"], "comparison count changed")
    keys = (
        "metrics",
        "actual_finite",
        "expected_finite",
        "thresholds",
        "elementwise_pass",
        "relative_l2_pass",
        "thresholds_pass",
        "mismatch_count",
        "mismatch_rows",
    )
    return {**{key: report[key] for key in keys}, "failed_coordinates": coordinates}


def _aggregate_error(actual, expected):
    """Descriptive M/E error only; it is not an extra acceptance threshold."""
    finite = bool(torch.isfinite(actual).all() and torch.isfinite(expected).all())
    result = {
        "all_finite": finite,
        "actual_sha256": _hash(actual),
        "expected_sha256": _hash(expected),
    }
    if finite:
        difference = actual.double() - expected.double()
        result.update(
            max_abs=float(difference.abs().max()),
            relative_l2=float(
                difference.norm() / expected.double().norm().clamp_min(1e-30)
            ),
            rms=float(difference.square().mean().sqrt()),
        )
    return result


class _LaunchRecorder:
    """Delegate the original launch unchanged, retaining its real argument tuple."""

    def __init__(self, kernel):
        self.kernel = kernel
        self.arg_names = tuple(kernel.arg_names)
        self.records = []

    def __getitem__(self, grid):
        _require(
            not callable(grid), "this diagnostic requires an explicit observed grid"
        )
        launch = self.kernel[grid]

        def run(*args, **kwargs):
            compiled = launch(*args, **kwargs)
            metadata = getattr(compiled, "metadata", None)
            self.records.append(
                {
                    "grid": tuple(grid),
                    "args": args,
                    "kwargs": dict(kwargs),
                    "launch": launch,
                    "compiled_name": getattr(compiled, "name", None),
                    "compiled_num_warps": getattr(metadata, "num_warps", None),
                }
            )
            return compiled

        return run

    def bound(self):
        _require(len(self.records) == 1, "expected exactly one actual kernel launch")
        record = self.records[0]
        return {**dict(zip(self.arg_names, record["args"])), **record["kwargs"]}

    def metadata(self):
        parameters = self.bound()
        record = self.records[0]
        encoded = {}
        for name, value in parameters.items():
            if isinstance(value, torch.Tensor):
                encoded[name] = _tensor_record(value)
            elif value is None or isinstance(value, (bool, int, float, str)):
                encoded[name] = value
            else:
                encoded[name] = str(
                    value
                )  # Triton constexpr dtype, never tensor content.
        return {
            "symbol": self.kernel.__name__,
            "grid": list(record["grid"]),
            "compiled_name": record["compiled_name"],
            "compiled_num_warps": record["compiled_num_warps"],
            "arguments": encoded,
        }

    def reduce_with(self, partial, dtype):
        """Only replace output_ptr/logits_ptr; preserve all recorded strides/flags."""
        bound = self.bound()
        original = bound["logits_ptr"]
        _require(
            partial.shape == original.shape and partial.stride() == original.stride(),
            "partial O layout differs from the recorded reduce contract",
        )
        _require(partial.device == original.device, "partial O device differs")
        output = torch.empty_like(bound["output_ptr"], dtype=dtype)
        _require(
            output.stride() == bound["output_ptr"].stride(), "output stride changed"
        )
        record = self.records[0]
        args, kwargs = list(record["args"]), dict(record["kwargs"])
        for name, tensor in (("output_ptr", output), ("logits_ptr", partial)):
            index = self.arg_names.index(name)
            if index < len(args):
                args[index] = tensor
            else:
                _require(name in kwargs, "recorded pointer argument is absent")
                kwargs[name] = tensor
        record["launch"](*args, **kwargs)
        # Clone immediately, before any other launch can reuse an output workspace.
        return output.reshape(500, 16, 128).clone()


def _partition_oracle(case, dtype):
    """Independent canonical-page QK/softmax/PV; no captured P or kernel math."""
    outputs, maxima, sums = [], [], []
    for begin, end in PARTITIONS:
        part_o, part_m, part_e = [], [], []
        for start in range(0, 500, 16):
            rows = list(range(start, min(start + 16, 500)))
            pages = case.logical_table[rows, begin // 16 : (end + 15) // 16]
            pages = pages.to(device=case.q.device, dtype=torch.long)
            key = (
                case.k[pages].permute(0, 2, 1, 3, 4).flatten(2, 3)[:, :, : end - begin]
            )
            value = (
                case.v[pages].permute(0, 2, 1, 3, 4).flatten(2, 3)[:, :, : end - begin]
            )
            key = key.repeat_interleave(2, dim=1).to(dtype)
            value = value.repeat_interleave(2, dim=1).to(dtype)
            logits = (
                torch.einsum("bhd,bhld->bhl", case.q[rows].to(dtype), key) * 128**-0.5
            )
            maximum = logits.amax(-1)
            part_m.append(maximum)
            part_e.append((logits - maximum[..., None]).exp().sum(-1))
            part_o.append(torch.einsum("bhl,bhld->bhd", logits.softmax(-1), value))
        outputs.append(torch.cat(part_o))
        maxima.append(torch.cat(part_m))
        sums.append(torch.cat(part_e))
    # [row, Q-head, partition, dim] -> [row, KV-head, partition, group, dim].
    output = (
        torch.stack(outputs, dim=2)
        .reshape(500, 8, 2, 3, 128)
        .permute(0, 1, 3, 2, 4)
        .contiguous()
    )
    maximum = (
        torch.stack(maxima, dim=2)
        .reshape(500, 8, 2, 3)
        .permute(0, 1, 3, 2)
        .contiguous()
    )
    exp_sum = (
        torch.stack(sums, dim=2).reshape(500, 8, 2, 3).permute(0, 1, 3, 2).contiguous()
    )
    return output, maximum, exp_sum


def diagnose_off_boundaries(
    ab,
    *,
    captured_off,
    full_reference,
    comparison_report,
    emit=None,
    precast_kernel=None,
    probability_kernel=None,
):
    """Return stages0--2 and optional stages3/4 evidence for one real layer.

    ab: existing RealAttentionAB; captured_off: untouched captured BF16 output.
    full_reference: existing reference(case, dtype=...) independent full oracle.
    comparison_report: existing real-fixture _comparison_report function.
    emit: optional callable(dict), invoked before final control assertions.
    precast_kernel: optional pinned test-only Gluon JIT, adding only
        actual_partial_fp32_ptr and a pre-BF16-cast observation store.
        None preserves the stages0--2 calls, computations and outputs.
    probability_kernel: optional pinned P32/P16/S32 observer; requires
        precast_kernel as the independent stage3 O32 fidelity baseline.
    No standalone execution or test registration is included in this draft.
    """
    _require(
        probability_kernel is None or precast_kernel is not None,
        "stage4 requires the stage3 observation baseline",
    )
    modules = {name: importlib.import_module(name) for name in SOURCE_PINS}
    sources = {}
    for name, module in modules.items():
        path = Path(inspect.getsourcefile(module)).resolve()
        digest = _sha(path)
        _require(digest == SOURCE_PINS[name], f"source pin changed: {name}")
        sources[name] = {"path": str(path), "sha256": digest}
    rtp, gluon, registration = [modules[name] for name in SOURCE_PINS]
    _require(
        registration.pa_decode_gluon is gluon.pa_decode_gluon,
        "registered Python entry is not the pinned Gluon function",
    )
    _require(
        gluon.pa_decode_gluon.__globals__ is vars(gluon), "unexpected launch globals"
    )
    case = ab.case
    _require(
        case.q.shape == (500, 16, 128) and case.q.dtype == torch.bfloat16,
        "only the captured BF16 500x16x128 case is supported",
    )
    _require(
        case.variant == "original", "stage0 requires the original captured mapping"
    )
    _require(
        case.k.shape[1:] == (8, 16, 128) and case.v.shape == case.k.shape,
        "canonical K/V layout differs",
    )
    _require(
        case.k.dtype == case.v.dtype == case.kv.dtype == torch.bfloat16,
        "BF16 canonical/cache contract changed",
    )
    _require(
        bool((case.history == 714).all()) and bool((ab.params.seq_lens == 715).all()),
        "the three-partition boundary is only established for length715",
    )
    _require(
        ab.params.max_seq_len == 715 and ab.off_op.context_partition_size == 256,
        "the actual off partition capacity changed",
    )
    _require(
        not ab.off_op.linear_v and not ab.off_op.enable_cuda_graph,
        "expected captured eager vectorized-V path",
    )
    _require(
        captured_off.shape == case.q.shape and captured_off.dtype == torch.bfloat16,
        "captured off shape/dtype differs",
    )
    _require(not torch.cuda.is_current_stream_capturing(), "diagnostic is eager only")

    previous_precision = torch.get_float32_matmul_precision()
    previous_tf32 = torch.backends.cuda.matmul.allow_tf32
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    try:
        return _run(
            ab,
            captured_off,
            full_reference,
            comparison_report,
            emit,
            rtp,
            gluon,
            sources,
            precast_kernel,
            probability_kernel,
        )
    finally:
        torch.set_float32_matmul_precision(previous_precision)
        torch.backends.cuda.matmul.allow_tf32 = previous_tf32


def _observe_actual_precast(
    dot,
    reduce,
    workspace,
    original_off,
    original_precast,
    kernel,
    report,
    emit,
    *,
    return_observation=False,
):
    """Observe actual P/PV history; never reconstruct or modify that history."""
    names = tuple(kernel.arg_names)
    _require(
        names.count(PRECAST_POINTER) == 1
        and tuple(name for name in names if name != PRECAST_POINTER) == dot.arg_names,
        "stage3 signature must add only the agreed observation pointer",
    )
    source_path = Path(inspect.getsourcefile(kernel.fn)).resolve()
    source_sha = _sha(source_path)
    _require(source_sha == PRECAST_SOURCE_SHA256, "stage3 candidate source pin changed")
    # Keep the actual original buffers/pointers. Snapshots are used only for
    # comparison, never fed back as replacement M/E or partial O to the clone.
    snapshots = {
        key: workspace[key].clone()
        for key in ("max_logits", "exp_sums", "temporary_output")
    }
    actual_fp32 = torch.full_like(
        workspace["temporary_output"], float("nan"), dtype=torch.float32
    )
    _require(
        actual_fp32.dtype == torch.float32
        and actual_fp32.shape == workspace["temporary_output"].shape
        and actual_fp32.stride() == workspace["temporary_output"].stride(),
        "stage3 observation buffer must preserve element strides",
    )
    independent_storage = all(
        actual_fp32.untyped_storage().data_ptr() != value.untyped_storage().data_ptr()
        for value in dot.bound().values()
        if isinstance(value, torch.Tensor)
    )
    _require(
        independent_storage,
        "stage3 observation buffer aliases an original input/output",
    )
    original_record = dot.records[0]
    # Mapping by names handles the extra pointer at argindex40 without moving
    # any original value to a different parameter. Launch options stay intact.
    launch_kwargs = dict(original_record["kwargs"])
    for name, value in zip(dot.arg_names, original_record["args"]):
        _require(name not in launch_kwargs, "duplicate recorded dot parameter")
        launch_kwargs[name] = value
    _require(
        PRECAST_POINTER not in launch_kwargs, "observation pointer already present"
    )
    launch_kwargs[PRECAST_POINTER] = actual_fp32
    observer = _LaunchRecorder(kernel)
    compiled = observer[original_record["grid"]](**launch_kwargs)
    observed = observer.bound()
    for name, value in dot.bound().items():
        _require(observed[name] is value, "stage3 changed an original parameter object")

    controls = {
        "M_byte_exact": _exact(workspace["max_logits"], snapshots["max_logits"]),
        "E_byte_exact": _exact(workspace["exp_sums"], snapshots["exp_sums"]),
        "partial_bf16_byte_exact": _exact(
            workspace["temporary_output"], snapshots["temporary_output"]
        ),
        "precast_rounds_to_original_partial": _exact(
            actual_fp32.bfloat16(), snapshots["temporary_output"]
        ),
        "clone_final_vs_original": _exact(
            reduce.reduce_with(workspace["temporary_output"], torch.bfloat16),
            original_off,
        ),
    }
    binary = getattr(compiled, "asm", {}).get("hsaco")
    metadata = getattr(compiled, "metadata", None)
    stage = {
        "source": {"path": str(source_path), "sha256": source_sha},
        "actual_launch": observer.metadata(),
        "compiled_hsaco_sha256": (
            hashlib.sha256(binary).hexdigest() if isinstance(binary, bytes) else None
        ),
        "compiled_num_stages": getattr(metadata, "num_stages", None),
        "new_pointer": PRECAST_POINTER,
        "observation_storage_independent": independent_storage,
        "actual_normalized_partial_fp32": _tensor_record(actual_fp32, with_hash=True),
        "controls": controls,
        "same_launch_geometry": observer.records[0]["grid"] == original_record["grid"]
        and observer.records[0]["compiled_num_warps"]
        == original_record["compiled_num_warps"],
        "candidate_source_unchanged": _sha(source_path) == source_sha,
        "interpretation_limit": (
            "The observed normalized FP32 partial O retains the copied actual P/PV math. "
            "Original M/E/BF16 partial/final byte controls establish observational fidelity, "
            "not equality to unseen pre-cast bits of the uninstrumented kernel. "
            "An extra store can change register allocation or scheduling; inspect source/ISA evidence."
        ),
    }
    report["stage3_actual_precast"] = stage

    def require_controls():
        stage["candidate_source_unchanged"] = _sha(source_path) == source_sha
        stage["controls_pass"] = (
            all(item["byte_exact"] for item in controls.values())
            and stage["same_launch_geometry"]
            and stage["candidate_source_unchanged"]
            and stage["actual_normalized_partial_fp32"]["nonfinite"] == 0
        )
        if not stage["controls_pass"]:
            report["diagnostic_controls_pass"] = False
            if emit is not None:
                emit(report)
            raise AssertionError(
                "stage3 observation controls failed; stop before attribution"
            )

    require_controls()
    # Only after exact restoration is established, replay the ORIGINAL reduce
    # with ORIGINAL actual M/E and the captured, not oracle, pre-cast O.
    rounded = actual_fp32.bfloat16().float()
    actual_final32 = reduce.reduce_with(actual_fp32, torch.float32)
    actual_final16 = reduce.reduce_with(actual_fp32, torch.bfloat16)
    rounded_final32 = reduce.reduce_with(rounded, torch.float32)
    rounded_final16 = reduce.reduce_with(rounded, torch.bfloat16)
    controls.update(
        {
            "actual_precast_final_cast": _exact(
                actual_final32.bfloat16(), actual_final16
            ),
            "rounded_actual_final_vs_original": _exact(rounded_final16, original_off),
            "rounded_actual_precast_vs_original_precast": _exact(
                rounded_final32, original_precast
            ),
        }
    )
    for key in snapshots:
        controls[key + "_unchanged_after_reduce"] = _exact(
            workspace[key], snapshots[key]
        )
    require_controls()
    outputs = {
        "actual_gluon_precast_partial_same_actual_ME_final_fp32": actual_final32,
        "actual_gluon_precast_partial_same_actual_ME_final_bf16": actual_final16,
        "actual_gluon_precast_bf16_rounded_partial_same_actual_ME_final_fp32": rounded_final32,
        "actual_gluon_precast_bf16_rounded_partial_same_actual_ME_final_bf16": rounded_final16,
    }
    return (outputs, actual_fp32) if return_observation else outputs


def _local_comparison(compare, actual, expected):
    """Existing thresholds with explicit local-to-real row/head coordinates."""
    result = _comparison(compare, actual, expected)
    result["scope"] = (
        "LOCAL_MODEL; selected 4 rows and heads14/15, not full implementation acceptance"
    )
    result["failed_coordinates"] = [
        [PROBABILITY_ROWS[row], PROBABILITY_HEADS[head], dim]
        for row, head, dim in result["failed_coordinates"]
    ]
    result["mismatch_rows"] = sorted(
        PROBABILITY_ROWS[row] for row in result["mismatch_rows"]
    )
    return result


def _probability_models(
    case, p32, p16, scores, maximum, exp_sum, actual_o32, full32, full64, compare
):
    """Common FP64 PV and actual-E normalization; no P renormalization."""
    rows = list(PROBABILITY_ROWS)
    model32, model16, per_partition = [], [], []
    for index, (begin, end) in enumerate(PARTITIONS):
        length = end - begin
        page_ids = case.logical_table[rows, begin // 16 : (end + 15) // 16]
        page_ids = page_ids.to(device=case.q.device, dtype=torch.long)
        # Same actual KV head7 for Q heads14/15. Only valid logical tokens.
        value = (
            case.v[page_ids][:, :, 7].flatten(1, 2)[:, :length].double().contiguous()
        )
        key = case.k[page_ids][:, :, 7].flatten(1, 2)[:, :length].double().contiguous()
        score_reference = (
            torch.einsum("rgd,rld->rgl", case.q[rows][:, 14:16].double(), key)
            * 128**-0.5
        )
        denominator = exp_sum[:, index].double()[..., None]
        products = []
        for probability in (p32, p16):
            promoted = probability[:, index, :, :length].double().contiguous()
            products.append(torch.einsum("rgl,rld->rgd", promoted, value) / denominator)
        model32.append(products[0])
        model16.append(products[1])
        observed = actual_o32[:, index].double()
        per_partition.append(
            {
                "range": [begin, end],
                "dimensions_per_head": 128,
                "P16_model_vs_P32_model": _aggregate_error(products[1], products[0]),
                "actual_O32_vs_P16_model": _aggregate_error(observed, products[1]),
                "actual_O32_vs_P32_model": _aggregate_error(observed, products[0]),
                "sum64_actual_P32_vs_actual_E": _aggregate_error(
                    p32[:, index, :, :length].double().sum(-1),
                    exp_sum[:, index].double(),
                ),
                "actual_scaled_score_vs_independent_FP64_QK_times_oracle_scale": _aggregate_error(
                    scores[:, index, :, :length], score_reference
                ),
            }
        )
    partial32 = torch.stack(model32, dim=1)
    partial16 = torch.stack(model16, dim=1)
    actual = actual_o32.double()
    # M/E are shared across every model. Reduction stays FP64 for all three.
    m64, e64 = maximum.double(), exp_sum.double()
    weights = e64 * (m64 - m64.amax(dim=1, keepdim=True)).exp()
    weights = weights / weights.sum(dim=1, keepdim=True)
    partials = {"P32_FP64PV": partial32, "P16_FP64PV": partial16, "actual_O32": actual}
    combined = {
        name: (value * weights[..., None]).sum(dim=1)
        for name, value in partials.items()
    }
    ref32, ref64 = full32[rows][:, 14:16], full64[rows][:, 14:16]
    combinations = {
        name: {
            "tensor": _tensor_record(value, with_hash=True),
            "vs_full_fp32": _local_comparison(compare, value, ref32),
            "vs_full_fp64": _local_comparison(compare, value, ref64),
        }
        for name, value in combined.items()
    }
    target = (0, 0, 58)  # selected slot0=row321, group0=head14.
    target_errors = {
        name: {
            "signed_error_vs_full_fp32": float(value[target] - ref32.double()[target]),
            "signed_error_vs_full_fp64": float(value[target] - ref64[target]),
            "absolute_error_vs_full_fp64": float((value[target] - ref64[target]).abs()),
        }
        for name, value in combined.items()
    }
    target_errors.update(
        {
            "coordinate": [321, 14, 58],
            "allowed_error_vs_full_fp64": float(ATOL + RTOL * ref64[target].abs()),
            "P_quantization_effect": float(
                combined["P16_FP64PV"][target] - combined["P32_FP64PV"][target]
            ),
            "actual_PV_normalization_residual": float(
                combined["actual_O32"][target] - combined["P16_FP64PV"][target]
            ),
            "weighted_partition_error_contributions": [
                {
                    "range": list(PARTITIONS[index]),
                    "P_quantization_effect": float(
                        weights[0, index, 0]
                        * (partial16[0, index, 0, 58] - partial32[0, index, 0, 58])
                    ),
                    "actual_PV_normalization_residual": float(
                        weights[0, index, 0]
                        * (actual[0, index, 0, 58] - partial16[0, index, 0, 58])
                    ),
                }
                for index in range(3)
            ],
        }
    )
    auxiliary = []
    for index, (begin, end) in enumerate(PARTITIONS):
        length = end - begin
        exp64 = (scores[:, index, :, :length].double() - m64[:, index, :, None]).exp()
        auxiliary.append(
            {
                "range": [begin, end],
                "actual_P32_vs_exp64_actual_scaled_score_minus_actual_M": _aggregate_error(
                    p32[:, index, :, :length].double(), exp64
                ),
            }
        )
    return {
        "scope": "LOCAL_MODEL; no selected-partial replacement into full output",
        "rows": rows,
        "heads": list(PROBABILITY_HEADS),
        "all_dimensions": 128,
        "probabilities": "actual unnormalized P32 versus captured actual P16",
        "denominator": "same actual Gluon E for both FP64 PV models; not recomputed",
        "merge": "same FP64 weights from actual M/E for all models; separate from original FP32 reduce",
        "weights": _tensor_record(weights, with_hash=True),
        "partitions": per_partition,
        "combined": combinations,
        "combined_P16_vs_P32": _aggregate_error(
            combined["P16_FP64PV"], combined["P32_FP64PV"]
        ),
        "combined_actual_vs_P16": _aggregate_error(
            combined["actual_O32"], combined["P16_FP64PV"]
        ),
        "target_error_scalars": target_errors,
        "auxiliary_score_evidence": auxiliary,
        "interpretation_limit": (
            "P16-P32 isolates P quantization within common FP64 PV/merge models. "
            "Actual O32-P16 retains PV MFMA and FP32 normalization effects together. "
            "Scaled score-FP64 QK includes QK accumulation and scale rounding. "
            "P32-exp64(S-M) includes argument rounding/LOG2_E/exp2 effects; not exp2 alone. "
            "Selected model threshold status does not constitute full implementation acceptance."
        ),
    }


def _observe_probabilities(
    dot,
    reduce,
    workspace,
    original_off,
    stage3_o32,
    kernel,
    case,
    full32,
    full64,
    compare,
    report,
    emit,
):
    names = tuple(kernel.arg_names)
    _require(
        all(names.count(name) == 1 for name in PROBABILITY_POINTERS)
        and tuple(name for name in names if name not in PROBABILITY_POINTERS)
        == dot.arg_names,
        "stage4 signature must add only the four agreed observation pointers",
    )
    d = dot.bound()
    _require(
        d["CONTEXT_PARTITION_SIZE"] == d["KV_COMPUTE_BLOCK_SIZE"] == 256
        and d["SLIDING_WINDOW"] == -1
        and d["QUERY_QUANT_MODE"] == d["KV_QUANT_MODE"] == -1
        and d["QUERY_SEQ_LEN"] == 1
        and d["ONE_QUERY_GROUP_SIZE"] == 2,
        "stage4 single-iteration/unquantized scope changed",
    )
    source_path = Path(inspect.getsourcefile(kernel.fn)).resolve()
    source_sha = _sha(source_path)
    _require(
        source_sha == PROBABILITY_SOURCE_SHA256, "stage4 candidate source pin changed"
    )
    snapshots = {
        key: workspace[key].clone()
        for key in ("max_logits", "exp_sums", "temporary_output")
    }
    new_o32 = torch.full_like(stage3_o32, float("nan"))
    p32 = torch.full(
        (4, 3, 2, 256), float("nan"), dtype=torch.float32, device=case.q.device
    )
    p16 = torch.full_like(p32, float("nan"), dtype=torch.bfloat16)
    scores = torch.full_like(p32, float("nan"))
    buffers = dict(zip(PROBABILITY_POINTERS, (new_o32, p32, p16, scores)))
    for name, value in buffers.items():
        expected_shape = (
            (500, 8, 3, 2, 128) if name == PRECAST_POINTER else (4, 3, 2, 256)
        )
        expected_stride = (
            (6144, 768, 256, 128, 1) if name == PRECAST_POINTER else (1536, 512, 256, 1)
        )
        expected_dtype = (
            torch.bfloat16 if name == "actual_probs_bf16_ptr" else torch.float32
        )
        _require(
            tuple(value.shape) == expected_shape
            and value.stride() == expected_stride
            and value.dtype == expected_dtype,
            "stage4 observation layout changed",
        )
    original_storage = {
        value.untyped_storage().data_ptr()
        for value in d.values()
        if isinstance(value, torch.Tensor)
    } | {stage3_o32.untyped_storage().data_ptr()}
    new_storage = {value.untyped_storage().data_ptr() for value in buffers.values()}
    _require(
        len(new_storage) == 4 and not (new_storage & original_storage),
        "stage4 observation buffers must be independent of each other and all prior tensors",
    )
    original_record = dot.records[0]
    launch_kwargs = dict(original_record["kwargs"])
    for name, value in zip(dot.arg_names, original_record["args"]):
        _require(name not in launch_kwargs, "duplicate original dot parameter")
        launch_kwargs[name] = value
    launch_kwargs.update(buffers)
    observer = _LaunchRecorder(kernel)
    compiled = observer[original_record["grid"]](**launch_kwargs)
    observed = observer.bound()
    for name, value in d.items():
        _require(observed[name] is value, "stage4 changed an original parameter object")
    selected_m = workspace["max_logits"][list(PROBABILITY_ROWS), 7]
    selected_e = workspace["exp_sums"][list(PROBABILITY_ROWS), 7]
    controls = {
        "M_byte_exact": _exact(workspace["max_logits"], snapshots["max_logits"]),
        "E_byte_exact": _exact(workspace["exp_sums"], snapshots["exp_sums"]),
        "partial_bf16_byte_exact": _exact(
            workspace["temporary_output"], snapshots["temporary_output"]
        ),
        "all500_O32_vs_stage3_byte_exact": _exact(new_o32, stage3_o32),
        "O32_rounds_to_original_partial": _exact(
            new_o32.bfloat16(), snapshots["temporary_output"]
        ),
        "final_off_vs_original": _exact(
            reduce.reduce_with(workspace["temporary_output"], torch.bfloat16),
            original_off,
        ),
        "actual_P16_vs_RN_P32": _exact(p16, p32.bfloat16()),
        "invalid_P32_zero": _exact(
            p32[:, 2, :, 203:], torch.zeros_like(p32[:, 2, :, 203:])
        ),
        "invalid_P16_zero": _exact(
            p16[:, 2, :, 203:], torch.zeros_like(p16[:, 2, :, 203:])
        ),
        "invalid_scaled_scores_sentinel": _exact(
            scores[:, 2, :, 203:], torch.full_like(scores[:, 2, :, 203:], -3.4e38)
        ),
        "selected_M_vs_actual_score_max": _exact(selected_m, scores.amax(-1)),
    }
    binary = getattr(compiled, "asm", {}).get("hsaco")
    metadata = getattr(compiled, "metadata", None)
    stage = {
        "source": {"path": str(source_path), "sha256": source_sha},
        "actual_launch": observer.metadata(),
        "compiled_hsaco_sha256": (
            hashlib.sha256(binary).hexdigest() if isinstance(binary, bytes) else None
        ),
        "compiled_num_stages": getattr(metadata, "num_stages", None),
        "rows": list(PROBABILITY_ROWS),
        "heads": list(PROBABILITY_HEADS),
        "observation_buffers": {
            name: _tensor_record(value, with_hash=True)
            for name, value in buffers.items()
        },
        "buffers_independent": True,
        "single_iteration": True,
        "selected_actual_E_positive": bool((selected_e > 0).all()),
        "same_launch_geometry": observer.records[0]["grid"] == original_record["grid"]
        and observer.records[0]["compiled_num_warps"]
        == original_record["compiled_num_warps"],
        "candidate_source_unchanged": _sha(source_path) == source_sha,
        "stage3_source_unchanged": _sha(
            report["stage3_actual_precast"]["source"]["path"]
        )
        == PRECAST_SOURCE_SHA256,
        "controls": controls,
        "interpretation_limit": "Observation must restore stage3 all-row FP32 O and original bytes before local P/PV models; hidden uninstrumented intermediates remain unobserved.",
    }
    report["stage4_probability_diagnostic"] = stage

    def require_controls():
        stage["candidate_source_unchanged"] = _sha(source_path) == source_sha
        stage["stage3_source_unchanged"] = (
            _sha(report["stage3_actual_precast"]["source"]["path"])
            == PRECAST_SOURCE_SHA256
        )
        stage["controls_pass"] = (
            all(item["byte_exact"] for item in controls.values())
            and all(
                item["nonfinite"] == 0 for item in stage["observation_buffers"].values()
            )
            and stage["selected_actual_E_positive"]
            and stage["same_launch_geometry"]
            and stage["candidate_source_unchanged"]
            and stage["stage3_source_unchanged"]
        )
        if not stage["controls_pass"]:
            report["diagnostic_controls_pass"] = False
            if emit is not None:
                emit(report)
            raise AssertionError(
                "stage4 observation controls failed; stop before model attribution"
            )

    require_controls()
    stage["models"] = _probability_models(
        case,
        p32,
        p16,
        scores,
        selected_m,
        selected_e,
        new_o32[list(PROBABILITY_ROWS), 7],
        full32,
        full64,
        compare,
    )
    for key, snapshot in snapshots.items():
        controls[key + "_unchanged_after_models"] = _exact(workspace[key], snapshot)
    require_controls()


def _run(
    ab,
    captured_off,
    full_reference,
    compare,
    emit,
    rtp,
    gluon,
    sources,
    precast_kernel,
    probability_kernel,
):
    case = ab.case
    original_off = ab.off().clone()
    workspace = {
        "output": torch.empty_like(case.q),
        "max_logits": torch.empty(
            (500, 8, 3, 2), device=case.q.device, dtype=torch.float32
        ),
        "exp_sums": torch.empty(
            (500, 8, 3, 2), device=case.q.device, dtype=torch.float32
        ),
        "temporary_output": torch.empty(
            (500, 8, 3, 2, 128), device=case.q.device, dtype=torch.bfloat16
        ),
    }
    dot, reduce = _LaunchRecorder(getattr(gluon, DOT_NAME)), _LaunchRecorder(
        getattr(gluon, REDUCE_NAME)
    )
    run_original = rtp._run_triton_paged_attention
    signature = inspect.signature(run_original)
    wrapper_calls = []

    def with_workspace(*args, **kwargs):
        bound = signature.bind(*args, **kwargs)
        bound.apply_defaults()
        p = bound.arguments
        _require(p["workspace"] is None, "unexpected existing eager workspace")
        _require(
            p["query_length"] == 1
            and p["num_seqs"] == 500
            and p["num_kv_heads"] == 8
            and p["max_seq_len"] == 715
            and p["context_partition_size"] == 256
            and not p["linear_v"],
            "observed production wrapper geometry changed",
        )
        _require(
            p["query"].data_ptr() == case.q.data_ptr()
            and p["seq_lens"] is ab.params.seq_lens
            and p["block_tables_id_device"] is ab.params.kv_cache_block_id_device,
            "production wrapper did not receive the original query/metadata",
        )
        wrapper_calls.append(True)
        bound.arguments["workspace"] = workspace
        return run_original(*bound.args, **bound.kwargs)

    with mock.patch.object(
        rtp, "_run_triton_paged_attention", new=with_workspace
    ), mock.patch.object(gluon, DOT_NAME, new=dot), mock.patch.object(
        gluon, REDUCE_NAME, new=reduce
    ):
        observed_off = ab.off().clone()
    _require(
        len(wrapper_calls) == 1, "production workspace hook did not run exactly once"
    )
    d, r = dot.bound(), reduce.bound()
    _require(
        dot.records[0]["grid"] == (500, 8, 3)
        and reduce.records[0]["grid"] == (500, 8, 1),
        "actual grid differs",
    )
    _require(
        d["CONTEXT_PARTITION_SIZE"] == 256
        and d["KV_BLOCK_SIZE"] == 16
        and d["QUERY_SEQ_LEN"] == 1
        and d["ONE_QUERY_GROUP_SIZE"] == 2
        and d["VALUE_TRANSPOSED"]
        and not d["IS_CAUSAL"]
        and d["SLIDING_WINDOW"] == -1
        and d["QUERY_QUANT_MODE"] == -1
        and d["KV_QUANT_MODE"] == -1
        and d["COMPUTE_TYPE"] == gluon.tl.bfloat16
        and d["CDNA_VERSION"] == 3,
        "actual dot branch/flags differ",
    )
    _require(
        r["CONTEXT_PARTITION_SIZE"] == 256
        and r["OUTPUT_SEQ_LEN"] == 1
        and r["ONE_OUTPUT_GROUP_SIZE"] == 2
        and not r["USE_SINKS"],
        "actual reduce branch/flags differ",
    )
    for name, key in (("max_logits_ptr", "max_logits"), ("exp_sums_ptr", "exp_sums")):
        _require(
            d[name] is workspace[key] and r[name] is workspace[key],
            "M/E workspace identity changed",
        )
    partial = workspace["temporary_output"]
    _require(
        d["output_ptr"] is partial and r["logits_ptr"] is partial,
        "partial O workspace identity changed",
    )
    _require(
        r["output_ptr"].data_ptr() == workspace["output"].data_ptr(),
        "final output workspace identity changed",
    )
    before = {
        key: _hash(workspace[key])
        for key in ("max_logits", "exp_sums", "temporary_output")
    }
    controls = {
        "original_vs_captured": _exact(original_off, captured_off),
        "workspace_hook_vs_original": _exact(observed_off, original_off),
    }
    report = {
        "scope": "OFF_GLUON_NUMERICAL_PLAN stages0-2; controlled diagnostic, not accuracy acceptance",
        "sources": sources,
        "partitions": [list(pair) for pair in PARTITIONS],
        "all_rows": 500,
        "oracle_precision": {
            "matmul_precision": torch.get_float32_matmul_precision(),
            "allow_tf32": torch.backends.cuda.matmul.allow_tf32,
        },
        "input_hashes": {
            key: _hash(value)
            for key, value in {
                "q": case.q,
                "canonical_k": case.k,
                "canonical_v": case.v,
                "logical_table": case.logical_table,
                "physical_table": ab.params.kv_cache_block_id_device,
                "seq_lens": ab.params.seq_lens,
            }.items()
        },
        "actual_launches": {"dot": dot.metadata(), "reduce": reduce.metadata()},
        "workspace": {
            key: _tensor_record(value, with_hash=True)
            for key, value in workspace.items()
        },
        "controls": controls,
    }
    report["workspace_all_finite"] = all(
        item["nonfinite"] == 0 for item in report["workspace"].values()
    )
    if (
        not all(item["byte_exact"] for item in controls.values())
        or not report["workspace_all_finite"]
    ):
        report["diagnostic_controls_pass"] = False
        if emit is not None:
            emit(report)
        raise AssertionError(
            "stage0 byte-exact replay failed; counterfactuals were not run"
        )

    # Stage1: replay the existing reduce. Stage2 dtype-only storage control.
    replay_bf16 = reduce.reduce_with(partial, torch.bfloat16)
    precast = reduce.reduce_with(partial, torch.float32)
    actual_partial_fp32 = partial.float()
    dtype_only = reduce.reduce_with(actual_partial_fp32, torch.bfloat16)
    dtype_only_precast = reduce.reduce_with(actual_partial_fp32, torch.float32)
    controls.update(
        {
            "low_level_bf16_vs_original": _exact(replay_bf16, original_off),
            "fp32_reduce_then_bf16_vs_original": _exact(
                precast.bfloat16(), original_off
            ),
            "actual_bf16_values_in_fp32_storage_vs_original": _exact(
                dtype_only, original_off
            ),
            "dtype_only_precast_vs_actual_precast": _exact(dtype_only_precast, precast),
        }
    )
    if not all(item["byte_exact"] for item in controls.values()):
        report["diagnostic_controls_pass"] = False
        if emit is not None:
            emit(report)
        raise AssertionError(
            "reduce/dtype-only exact control failed; stop before oracle attribution"
        )

    full32 = full_reference(case, dtype=torch.float32)
    full64 = full_reference(case, dtype=torch.float64)
    oracle_agreement = _comparison(compare, full32, full64, oracle=True)
    report["full_oracle_fp32_vs_fp64"] = oracle_agreement
    outputs = {"original_off_bf16": original_off, "actual_reduce_precast_fp32": precast}
    report["partition_oracles"] = {}
    partition_fp32 = None
    for label, dtype in (("fp32", torch.float32), ("fp64", torch.float64)):
        oracle, maximum, exp_sum = _partition_oracle(case, dtype)
        if dtype == torch.float32:
            partition_fp32 = oracle
        else:
            report["partition_oracle_fp32_vs_fp64"] = [
                {
                    "range": list(PARTITIONS[index]),
                    "comparison": _comparison(
                        compare,
                        partition_fp32[:, :, index].reshape(500, 16, 128),
                        oracle[:, :, index].reshape(500, 16, 128),
                        oracle=True,
                    ),
                }
                for index in range(3)
            ]
        per_partition = []
        for index in range(3):
            per_partition.append(
                {
                    "range": list(PARTITIONS[index]),
                    "actual_partial_vs_oracle": _comparison(
                        compare,
                        partial[:, :, index].reshape(500, 16, 128),
                        oracle[:, :, index].reshape(500, 16, 128),
                    ),
                    "actual_M_vs_oracle": _aggregate_error(
                        workspace["max_logits"][:, :, index], maximum[:, :, index]
                    ),
                    "actual_E_vs_oracle": _aggregate_error(
                        workspace["exp_sums"][:, :, index], exp_sum[:, :, index]
                    ),
                }
            )
        report["partition_oracles"][label] = {
            "description": "independent oracle, not captured pre-rounding Gluon partial O",
            "O_sha256": _hash(oracle),
            "M_sha256": _hash(maximum),
            "E_sha256": _hash(exp_sum),
            "partitions": per_partition,
        }
        for rounding, replacement in (
            ("float", oracle.float()),
            ("bf16_rounded", oracle.bfloat16().float()),
        ):
            name = f"oracle_{label}_{rounding}_partial_same_actual_ME"
            final32 = reduce.reduce_with(replacement, torch.float32)
            final16 = reduce.reduce_with(replacement, torch.bfloat16)
            outputs[name + "_final_fp32"] = final32
            outputs[name + "_final_bf16"] = final16
            controls[name + "_final_cast_control"] = _exact(final32.bfloat16(), final16)

    if precast_kernel is not None:
        # Do not interpret actual-precast interventions against invalid oracles.
        valid_oracles = oracle_agreement["thresholds_pass"] and all(
            item["comparison"]["thresholds_pass"]
            for item in report["partition_oracle_fp32_vs_fp64"]
        )
        if not valid_oracles or not all(
            item["byte_exact"] for item in controls.values()
        ):
            report["diagnostic_controls_pass"] = False
            if emit is not None:
                emit(report)
            raise AssertionError(
                "stage3 requires valid prior controls and full/partition oracles"
            )
        report["scope"] = (
            "OFF_GLUON_NUMERICAL_PLAN stages0-3; controlled diagnostic, not accuracy acceptance"
        )
        if probability_kernel is None:
            outputs.update(
                _observe_actual_precast(
                    dot,
                    reduce,
                    workspace,
                    original_off,
                    precast,
                    precast_kernel,
                    report,
                    emit,
                )
            )
        else:
            stage3_outputs, stage3_o32 = _observe_actual_precast(
                dot,
                reduce,
                workspace,
                original_off,
                precast,
                precast_kernel,
                report,
                emit,
                return_observation=True,
            )
            outputs.update(stage3_outputs)
            report["scope"] = (
                "OFF_GLUON_NUMERICAL_PLAN stages0-4; local probability models, not accuracy acceptance"
            )
            _observe_probabilities(
                dot,
                reduce,
                workspace,
                original_off,
                stage3_o32,
                probability_kernel,
                case,
                full32,
                full64,
                compare,
                report,
                emit,
            )

    report["outputs"] = {
        name: {
            "tensor": _tensor_record(output, with_hash=True),
            "vs_full_fp32": _comparison(compare, output, full32),
            "vs_full_fp64": _comparison(compare, output, full64),
        }
        for name, output in outputs.items()
    }
    report["actual_ME_and_partial_unchanged"] = all(
        _hash(workspace[key]) == digest for key, digest in before.items()
    )
    report["inputs_unchanged"] = all(
        _hash(value) == report["input_hashes"][key]
        for key, value in {
            "q": case.q,
            "canonical_k": case.k,
            "canonical_v": case.v,
            "logical_table": case.logical_table,
            "physical_table": ab.params.kv_cache_block_id_device,
            "seq_lens": ab.params.seq_lens,
        }.items()
    )
    report["source_pins_unchanged"] = all(
        _sha(item["path"]) == item["sha256"] for item in sources.values()
    )
    report["off_accuracy_pass"] = all(
        report["outputs"]["original_off_bf16"][name]["thresholds_pass"]
        for name in ("vs_full_fp32", "vs_full_fp64")
    )
    report["diagnostic_controls_pass"] = (
        all(item["byte_exact"] for item in controls.values())
        and oracle_agreement["thresholds_pass"]
        and all(
            item["comparison"]["thresholds_pass"]
            for item in report["partition_oracle_fp32_vs_fp64"]
        )
        and (precast_kernel is None or report["stage3_actual_precast"]["controls_pass"])
        and (
            probability_kernel is None
            or report["stage4_probability_diagnostic"]["controls_pass"]
        )
        and report["workspace_all_finite"]
        and report["inputs_unchanged"]
        and report["actual_ME_and_partial_unchanged"]
        and report["source_pins_unchanged"]
    )
    report["interpretation_limit"] = (
        "Exit success verifies diagnostic controls only. OFF accuracy remains separately reported. "
        "Oracle replacement removes internal QK/P/PV/exp error as well as partial rounding; "
        "stages0-2 cannot uniquely attribute actual Gluon error to partial BF16 rounding."
    )
    if emit is not None:
        emit(report)
    _require(
        report["diagnostic_controls_pass"], "OFF boundary diagnostic controls failed"
    )
    return report
