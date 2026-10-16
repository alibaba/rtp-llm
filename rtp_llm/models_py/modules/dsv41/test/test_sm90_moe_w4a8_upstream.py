"""Isolated upstream d113632 W4A8 test; never imports two Humming versions.

Required: DSV41_UPSTREAM_HUMMING points to the exact checked-out source tree.
Root owns GPU/Bazel scheduling. No production or installed dependency changes.
"""

import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import subprocess
import sys

_REVISION = "d113632a2bbcee5f25fcc8a3a7d92ccdfbac62a8"
_ROOT = Path(os.environ["DSV41_UPSTREAM_HUMMING"]).resolve()
assert (_ROOT / "humming" / "__init__.py").is_file(), _ROOT
assert not any(name == "humming" or name.startswith("humming.") for name in sys.modules), (
    "Run this target alone: Humming was imported before isolation"
)
assert subprocess.check_output(["git", "-C", str(_ROOT), "rev-parse", "HEAD"], text=True).strip() == _REVISION
sys.path.insert(0, str(_ROOT))

import pytest
import torch
import humming
from humming.forward import humming_forward
from humming.schema import HummingInputSchema, HummingWeightSchema
from humming.transform import prepare_layer_config, transform_humming_tensors
from humming.tune import get_heuristics_config

from rtp_llm.models_py.modules.dsv41.moe._sm90_humming_linear import (
    SM90HummingGroupedLinear, unpack_activation_scales,
)
from rtp_llm.models_py.modules.dsv41.test.test_sm90_moe_gemm_candidates import (
    _independent_quantized_input, _independent_weight, _metrics, _quantize,
)


def _source():
    paths = {}
    jit_modules = {}
    cache_root = Path(os.environ["HUMMING_CACHE_DIR"]).resolve()
    for name, module in tuple(sys.modules.items()):
        if (name == "humming" or name.startswith("humming.")) and getattr(module, "__file__", None):
            path = Path(module.__file__).resolve()
            if path.is_relative_to(_ROOT):
                paths[name] = str(path)
            else:
                # This exact native extension is built by upstream utils/device.py.
                # No Python module or arbitrary shared library may bypass isolation.
                assert (name == "humming._device_info" and path.name == "_device_info.abi3.so"
                        and path.is_relative_to(cache_root)), (name, path, _ROOT, cache_root)
                jit_modules[name] = {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
    return {"revision": _REVISION, "root": str(_ROOT), "module_paths": paths,
            "jit_modules": jit_modules, "jit_cache_root": str(cache_root),
            "policy_sha256": hashlib.sha256((_ROOT / "humming/tune/sm90_policies.py").read_bytes()).hexdigest(),
            "baseline_scope": "W4A16 control on SAME upstream revision, not installed 0.1.16"}


pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 9,
    reason="requires SM90; root owns GPU lock",
)


@pytest.fixture(autouse=True)
def _fp32_oracle():
    previous = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    yield
    torch.backends.cuda.matmul.allow_tf32 = previous


def _weights(e, n, k):
    w = torch.randint(-128, 128, (e, n, k // 2), device="cuda", dtype=torch.int8)
    s = torch.randint(117, 128, (e, n, k // 32), device="cuda", dtype=torch.uint8)
    s[..., 0], s[..., 1] = 117, 127
    return w, s.view(torch.float8_e8m0fnu)


def _oracle_x(x):
    q, packed = _quantize(x)
    qref, sf = _independent_quantized_input(x)
    assert torch.equal(q.view(torch.uint8), qref.view(torch.uint8))
    assert torch.equal(unpack_activation_scales(packed, x.shape[1] // 128), sf)
    return (qref.float().reshape(x.shape[0], -1, 128) * sf[..., None]).reshape_as(x)


class _UpstreamW4A8(torch.nn.Module):
    def __init__(self, weight, scales):
        super().__init__()
        e, n, half_k = weight.shape
        raw = scales.view(torch.uint8).flatten(1).to(torch.int16)
        assert bool(((raw.amax(1) - raw.amin(1)) <= 11).all()), "No lossy exponent clipping allowed"
        self.config = prepare_layer_config(
            shape_n=n, shape_k=half_k * 2, num_experts=e,
            weight_schema=HummingWeightSchema(b_dtype="float4e2m1", bs_dtype="float8e8m0",
                                            weight_scale_group_size=32, weight_scale_type="group"),
            input_schema=HummingInputSchema(a_dtype="float8e4m3", input_scale_dtype="float32",
                                          input_scale_group_size=128),
            torch_dtype=torch.bfloat16, device=weight.device,
        )
        assert self.config.use_fused_e8m0_scale and self.config.use_packed_k_layout
        tensors = transform_humming_tensors(self.config, {
            "weight": weight.clone().contiguous().view(torch.int32),
            "weight_scale": scales.clone().contiguous(),
        })
        for name, tensor in tensors.items():
            self.register_buffer(name, tensor)
        self.register_buffer("locks", torch.zeros(1024, device=weight.device, dtype=torch.int32))
        self.compute_config = dict(gemm_type="grouped_contiguous", use_f16_accum=False,
                                   use_batch_invariant=True, use_m_major_input_scale=True)

    def tuning(self, m):
        config = dict(get_heuristics_config(self.config, shape_m=m, device=self.weight.device,
                                            **self.compute_config))
        assert config["wgmma_use_late_as"] and config["wgmma_split_issue_wait"]
        assert config["warp_shape"][1:] == (16, 128)
        assert not config["use_stream_k"]
        # Explicit test adapter override: upstream W4A8 policy replaces the tile
        # but leaves H20's inherited CTA occupancy. 384 threads * 3 CTAs yields
        # only 56 registers/thread, below this packed WGMMA's required 106.
        # One CTA allows 168 launch registers; upstream still validates all
        # accumulator/math-thread/shared-memory budgets without bypasses.
        config["num_ctas_per_sm"] = 1
        return config

    def prepared(self, q, sf_major, offsets, tuning=None):
        return humming_forward(
            self.config, inputs=q, input_scale=sf_major, weight=self.weight,
            weight_scale=self.weight_scale, weight_scale_2=self.weight_scale_2,
            expert_layout=offsets, valid_shape_m=q.shape[0], locks=self.locks,
            compute_config=self.compute_config, tuning_config=self.tuning(q.shape[0]) if tuning is None else tuning,
        )

    def forward(self, x, offsets):
        if not x.shape[0]:
            return x.new_empty((0, self.config.shape_n))
        q, packed = _quantize(x)
        scales = unpack_activation_scales(packed, x.shape[1] // 128)
        # Physical M-major layout: [K/128, round_up(M,4)]. Values unchanged.
        major = torch.zeros((scales.shape[1], (x.shape[0] + 3) // 4 * 4),
                            device=x.device, dtype=torch.float32)
        major[:, :x.shape[0]].copy_(scales.T)
        return self.prepared(q, major, offsets)


def _metrics_bounded(actual, reference):
    parts = [_metrics(actual[i:i + 1024], reference[i:i + 1024])
             for i in range(0, actual.shape[0], 1024)]
    elements = sum(p["elements"] for p in parts)
    return {"elements": elements, "unequal": sum(p["unequal"] for p in parts),
            "outside_existing_tolerance": sum(p["outside_existing_tolerance"] for p in parts),
            "max_abs": max(p["max_abs"] for p in parts),
            "rms_abs": (sum(p["rms_abs"] ** 2 * p["elements"] for p in parts) / elements) ** .5,
            "finite": all(p["finite"] for p in parts)}


def _write(name, record):
    directory = Path(os.environ["TEST_UNDECLARED_OUTPUTS_DIR"]) / name
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "result.json").write_text(json.dumps(record, indent=2))
    print("UPSTREAM_W4A8 " + json.dumps({k: v for k, v in record.items() if k != "sources"}), flush=True)
    return directory


def _graph_measure(baseline_fn, candidate_fn, directory, small, x, offsets):
    # Complete operator calls captured after JIT/warmup; memory allocation and
    # capture are outside timing. No eager-model-latency claim is made.
    graphs, held_outputs = {}, {}
    for name, fn in (("upstream_w4a16", baseline_fn), ("upstream_w4a8", candidate_fn)):
        for _ in range(5):
            fn()
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            held_outputs[name] = fn()
        graphs[name] = graph
    # Prove replay consumes changed activation and routing metadata, rather than
    # returning capture-time output. Weights are immutable parameters, not inputs.
    saved_x, saved_offsets = x.clone(), offsets.clone()
    alternate_x = (x.float().flip(0) * .5 + .125).bfloat16()
    counts = offsets[1:] - offsets[:-1]
    alternate_offsets = torch.cat((offsets.new_zeros(1), counts.roll(1).cumsum(0).to(offsets.dtype)))
    assert not torch.equal(saved_x, alternate_x)
    assert not torch.equal(saved_offsets, alternate_offsets)
    replay_validation = {}
    try:
        x.copy_(alternate_x)
        offsets.copy_(alternate_offsets)
        for name, fn in (("upstream_w4a16", baseline_fn), ("upstream_w4a8", candidate_fn)):
            eager = fn().clone()
            graphs[name].replay()
            torch.cuda.synchronize()
            assert torch.equal(held_outputs[name].view(torch.int16), eager.view(torch.int16)), name
            replay_validation[name] = "changed activation+expert offsets: bitwise eager/replay PASS"
    finally:
        x.copy_(saved_x)
        offsets.copy_(saved_offsets)
        torch.cuda.synchronize()
    evict = torch.empty(256 * 1024 * 1024, dtype=torch.uint8, device="cuda") if small else None
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    samples = {name: [] for name in graphs}
    for repeat in range(15):
        names = list(graphs) if repeat % 2 == 0 else list(reversed(graphs))
        for name in names:
            if evict is not None:
                evict.zero_()
                torch.cuda.synchronize()
            start.record()
            graphs[name].replay()
            end.record()
            end.synchronize()
            samples[name].append(start.elapsed_time(end))
    profiles = {}
    for name, graph in graphs.items():
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                                 torch.profiler.ProfilerActivity.CUDA]) as prof:
            for _ in range(3):
                with torch.profiler.record_function(name + "/full_projection_graph_replay"):
                    graph.replay()
            torch.cuda.synchronize()
        path = directory / (name + ".trace.json")
        prof.export_chrome_trace(str(path))
        kernels = [e for e in json.loads(path.read_text())["traceEvents"]
                   if e.get("cat") == "kernel" and e.get("ph") == "X"]
        histogram = {}
        for event in kernels:
            entry = histogram.setdefault(event["name"], {"count": 0, "sum_us": 0.})
            entry["count"] += 1
            entry["sum_us"] += event["dur"]
        assert sum(v["count"] for k, v in histogram.items() if "void humming<" in k) == 3, histogram
        assert sum(v["count"] for k, v in histogram.items() if "per_token_group_quant" in k) == 3, histogram
        profiles[name] = {"iterations": 3, "kernel_sum_us_per_iter": sum(e["dur"] for e in kernels) / 3,
                          "kernel_count_per_iter": len(kernels) / 3, "kernels": histogram}
    return {"graph_stream_span_ms": samples,
            "graph_stream_span_median_ms": {k: statistics.median(v) for k, v in samples.items()},
            "profile": profiles, "small_m_flush_bytes": 0 if evict is None else evict.numel(),
            "replay_input_validation": replay_validation,
            "scope": "full projection quantize+scale preparation+GEMM CUDA graph replay; not eager model latency"}


def _projection(m, n, k, e, benchmark):
    torch.manual_seed(20261015)
    weight, scales = _weights(e, n, k)
    saved_w, saved_s = weight.clone(), scales.view(torch.uint8).clone()
    candidate = _UpstreamW4A8(weight, scales)
    baseline = SM90HummingGroupedLinear(weight, scales)
    assert torch.equal(weight, saved_w) and torch.equal(scales.view(torch.uint8), saved_s)
    del saved_w, saved_s
    counts = [m // e + (i < m % e) for i in range(e)]
    host_offsets = [0]
    for count in counts:
        host_offsets.append(host_offsets[-1] + count)
    offsets = torch.tensor(host_offsets, device="cuda", dtype=torch.int32)
    x = (torch.randn(m, k, device="cuda") * 12).bfloat16()
    reference = torch.empty((m, n), dtype=torch.bfloat16, device="cuda")
    for expert, count in enumerate(counts):
        if count:
            begin, finish = host_offsets[expert:expert + 2]
            w = _independent_weight(weight[expert], scales[expert])
            for first in range(begin, finish, 1024):
                last = min(first + 1024, finish)
                reference[first:last] = (_oracle_x(x[first:last]) @ w.T).bfloat16()
            del w
    actual, control = candidate(x, offsets), baseline(x, offsets)
    record = {"M_routed": m, "N": n, "K": k, "E": e, "expert_counts": counts,
              "sources": _source(), "config": candidate.config.to_str(), "tuning": candidate.tuning(m),
              "adapter_override": {"num_ctas_per_sm": 1, "reason": "H20 inherited occupancy exceeds packed WGMMA register budget; not unmodified upstream defaults"},
              "oracle": _metrics_bounded(actual, reference), "control_oracle": _metrics_bounded(control, reference),
              "vs_upstream_w4a16": _metrics_bounded(actual, control),
              "bitwise_vs_upstream_w4a16": torch.equal(actual.view(torch.int16), control.view(torch.int16)),
              "deterministic": all(torch.equal(candidate(x, offsets), actual) for _ in range(3))}
    valid = record["deterministic"] and all(record[key]["outside_existing_tolerance"] == 0
                                           for key in ("oracle", "control_oracle", "vs_upstream_w4a16"))
    record["decision"] = "COMPONENT_PASS_NOT_PROMOTED" if valid else "REJECT_NUMERICAL"
    name = f"projection_M{m}_N{n}_K{k}_E{e}"
    directory = _write(name, record)
    assert valid, record
    if benchmark:
        record["benchmark"] = _graph_measure(lambda: baseline(x, offsets), lambda: candidate(x, offsets), directory, m < 32, x, offsets)
        _write(name, record)


def test_upstream_small_projection():
    _projection(17, 128, 128, 8, benchmark=False)


def test_upstream_small_w13_diagnosis():
    """Same failed full-MoE w13 inputs; record all controls before strict gate."""
    from rtp_llm.models_py.modules.dsv41.moe.strategies.sm90_humming import group_local_routes

    torch.manual_seed(20261013)
    weight, scales = _weights(8, 256, 128)
    _weights(8, 128, 128)  # Preserve the failing full-MoE fixture RNG sequence.
    candidate = _UpstreamW4A8(weight, scales)
    baseline = SM90HummingGroupedLinear(weight, scales)
    x = (torch.randn(4, 128, device="cuda") * 12).bfloat16()
    routes = torch.tensor([[0, 1, 2, 3, 4, 5], [5, 4, 3, 2, 1, 0],
                           [2, 0, -1, -1, -1, -1], [-1] * 6], device="cuda")
    slots, offsets = group_local_routes(routes, 0, 8)
    grouped = x.index_select(0, slots // 6)
    q, packed = _quantize(grouped)
    qref, sfref = _independent_quantized_input(grouped)
    sf = unpack_activation_scales(packed, 1)
    assert torch.equal(q.view(torch.uint8), qref.view(torch.uint8))
    assert torch.equal(sf, sfref)
    major = torch.zeros((1, (q.shape[0] + 3) // 4 * 4), device="cuda", dtype=torch.float32)
    major[:, :q.shape[0]].copy_(sf.T)
    assert torch.equal(major[:, :q.shape[0]].T, sfref)
    q_before, major_before = q.view(torch.uint8).clone(), major.clone()
    decoded_x = q.float() * sf
    decoded_w = _independent_weight(weight, scales)
    expert_ids = routes.flatten().index_select(0, slots).tolist()
    fp32 = torch.cat([decoded_x[i:i + 1] @ decoded_w[e].T for i, e in enumerate(expert_ids)])
    fp64 = torch.cat([decoded_x[i:i + 1].double() @ decoded_w[e].double().T for i, e in enumerate(expert_ids)])
    reference = fp64.bfloat16()
    control = baseline(grouped, offsets)
    basis = torch.eye(128, device="cuda", dtype=torch.bfloat16).repeat(8, 1)
    basis_offsets = torch.arange(9, device="cuda", dtype=torch.int32) * 128
    # Independent identity GEMM sets the zero sign mathematically; do not compare
    # the raw E2M1 negative-zero representation against an accumulated output.
    basis_reference = torch.cat([torch.eye(128, device="cuda") @ w.T for w in decoded_w]).bfloat16()
    bq, bp = _quantize(basis)
    bs = unpack_activation_scales(bp, 1)
    bm = bs.T.contiguous()
    results, outputs = {}, {}
    for late, split in ((True, True), (False, False), (True, False), (False, True)):
        name = f"late{int(late)}_split{int(split)}"
        tuning = candidate.tuning(grouped.shape[0])
        tuning.update(wgmma_use_late_as=late, wgmma_split_issue_wait=split)
        output = candidate.prepared(q, major, offsets, tuning=tuning)
        basis_tuning = dict(tuning)
        basis_tuning["num_sms"] = candidate.tuning(basis.shape[0])["num_sms"]
        basis_output = candidate.prepared(bq, bm, basis_offsets, tuning=basis_tuning)
        repeat = candidate.prepared(q, major, offsets, tuning=tuning)
        outputs[name] = output
        results[name] = {"tuning": tuning, "basis": _metrics(basis_output, basis_reference),
                         "vs_fp64_bf16": _metrics(output, reference), "vs_w4a16": _metrics(output, control),
                         "repeat_bitwise": torch.equal(output.view(torch.int16), repeat.view(torch.int16)),
                         "decision": "DIAGNOSTIC_ONLY_NOT_PROMOTED"}
    anchors = []
    for row, column in (outputs["late1_split1"] != reference).nonzero()[:8].tolist():
        expert = expert_ids[row]
        products = (decoded_x[row].cpu().double() * decoded_w[expert, column].cpu().double()).tolist()
        exact = math.fsum(products)
        rounded = torch.tensor(exact, dtype=torch.float64).bfloat16().float().item()
        anchors.append({"row": row, "column": column, "expert": expert,
                        "cpu_fsum": exact, "cpu_fsum_bf16": rounded,
                        "fp32": fp32[row, column].item(), "fp64": fp64[row, column].item(),
                        "w4a16": control[row, column].item(),
                        "modes": {name: value[row, column].item() for name, value in outputs.items()}})
    payload_unchanged = torch.equal(q.view(torch.uint8), q_before) and torch.equal(major, major_before)
    record = {"sources": _source(), "fixture": "exact RNG/input of new failed small_full_moe w13",
              "payload_and_fp32_m_major_exact": payload_unchanged,
              "fp32_vs_fp64_bf16": _metrics(fp32.bfloat16(), reference),
              "w4a16_vs_fp64_bf16": _metrics(control, reference),
              "modes": results, "first_default_mismatch_anchors": anchors,
              "decision": "DIAGNOSTIC_ONLY; original exact-stage gate remains enforced below"}
    _write("small_w13_diagnosis", record)
    assert payload_unchanged, record
    assert record["fp32_vs_fp64_bf16"]["unequal"] == 0, record
    assert record["w4a16_vs_fp64_bf16"]["unequal"] == 0, record
    assert all(mode["basis"]["unequal"] == 0 and mode["repeat_bitwise"] for mode in results.values()), record
    # This intentionally fails if any control still violates the existing exact
    # intermediate gate; an informative completed diagnosis is not a gate PASS.
    assert all(mode["vs_fp64_bf16"]["unequal"] == 0 for mode in results.values()), record


@pytest.mark.parametrize("n,k", [(4608, 5120), (5120, 2304)], ids=["w13", "down"])
@pytest.mark.parametrize("m", [4096, 32768], ids=["m4096", "m32768"])
def test_upstream_real_projection(m, n, k):
    _projection(m, n, k, 48, benchmark=True)


def test_upstream_small_full_moe():
    from rtp_llm.models_py.modules.dsv41.moe.strategies.base import MoeCfg
    from rtp_llm.models_py.modules.dsv41.moe.strategies.sm90_humming import _GroupedLocalExperts, group_local_routes

    torch.manual_seed(20261013)
    cfg = MoeCfg(layer_id=0, dim=128, moe_inter_dim=128, n_routed_experts=8,
                 n_activated_experts=6, swiglu_limit=10., ep_size=1, ep_rank=0,
                 n_local_experts=8, local_expert_start=0, local_expert_end=8,
                 max_tokens_per_rank=32, shared_fp8_block_size=32)
    w13, s13 = _weights(8, 256, 128)
    w2, s2 = _weights(8, 128, 128)
    local = _GroupedLocalExperts(cfg)
    local.w13, local.w2 = _UpstreamW4A8(w13, s13), _UpstreamW4A8(w2, s2)
    x = (torch.randn(4, 128, device="cuda") * 12).bfloat16()
    routes = torch.tensor([[0, 1, 2, 3, 4, 5], [5, 4, 3, 2, 1, 0],
                           [2, 0, -1, -1, -1, -1], [-1] * 6], device="cuda")
    rw = torch.tensor([[.05, .1, .2, .25, .4, .5], [.5, .4, .25, .2, .1, .05],
                       [.6, .9, 0, 0, 0, 0], [0] * 6], device="cuda", dtype=torch.float32)
    captured = {}
    def capture13(module, args, output):
        captured["w13"] = output.detach().clone()
    def capture2(module, args, output):
        captured["hidden"] = args[0].detach().clone()
        captured["down"] = output.detach().clone()
    handles = [local.w13.register_forward_hook(capture13), local.w2.register_forward_hook(capture2)]
    try:
        actual = local._forward_into_buf(x, rw, routes, 0, 8)
    finally:
        for handle in handles:
            handle.remove()
    candidate_captured = captured
    captured = {}
    control_local = _GroupedLocalExperts(cfg)
    control_local.w13 = SM90HummingGroupedLinear(w13, s13)
    control_local.w2 = SM90HummingGroupedLinear(w2, s2)
    handles = [control_local.w13.register_forward_hook(capture13), control_local.w2.register_forward_hook(capture2)]
    try:
        control = control_local._forward_into_buf(x, rw, routes, 0, 8)
    finally:
        for handle in handles:
            handle.remove()
    control_captured, captured = captured, candidate_captured
    slots, _ = group_local_routes(routes, 0, 8)
    expected = torch.zeros_like(actual)
    stage = {key: [] for key in ("w13", "hidden", "down")}
    for slot in slots.tolist():
        row, choice = divmod(slot, 6)
        expert = int(routes[row, choice])
        gate_up = (_oracle_x(x[row:row + 1]) @ _independent_weight(w13[expert], s13[expert]).T).bfloat16()
        gate, up = gate_up.float().chunk(2, -1)
        hidden = (torch.nn.functional.silu(gate.clamp(max=10)) * up.clamp(-10, 10) * rw[row, choice]).bfloat16()
        down = (_oracle_x(hidden) @ _independent_weight(w2[expert], s2[expert]).T).bfloat16()
        for key, value in (("w13", gate_up), ("hidden", hidden), ("down", down)):
            stage[key].append(value)
    slot_output = torch.zeros((routes.numel(), 128), device="cuda", dtype=torch.float32)
    slot_output.index_copy_(0, slots, torch.cat(stage["down"]).float())
    expected = slot_output.view(4, 6, 128).sum(1)
    record = {"sources": _source(), "stage": {key: _metrics(captured[key], torch.cat(value)) for key, value in stage.items()},
              "control_stage_oracle": {key: _metrics(control_captured[key], torch.cat(value)) for key, value in stage.items()},
              "stage_vs_control": {key: _metrics(captured[key], control_captured[key]) for key in stage},
              "control_final_oracle": _metrics(control, expected), "final_vs_control": _metrics(actual, control),
              "final": _metrics(actual, expected),
              "fixture": "new random span10 diagnostic, NOT the original smoke fixture",
              "scope": "local MoE incl route grouping/gate/pre-down route weight/quantize/combine; no EP transport"}
    # Preserve exact-stage nature of the existing small MoE oracle, alongside
    # original final tolerance. No promotion on an intermediate mismatch.
    valid = all(value["unequal"] == 0 for value in record["stage"].values()) and record["final"]["outside_existing_tolerance"] == 0
    record["decision"] = "LOCAL_MOE_PASS_NOT_PROMOTED" if valid else "REJECT_EXACT_STAGE"
    _write("small_full_moe", record)
    assert valid, record
