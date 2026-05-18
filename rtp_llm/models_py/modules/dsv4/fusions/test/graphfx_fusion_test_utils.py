from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Callable, Iterable, Sequence

import torch

from rtp_llm.models_py.modules.dsv4.test.dsv4_kernel_perf_utils import (
    device_payload,
    env_flag,
    iters_for_m,
    measure_kernel,
    measurement_payload,
    parse_int_list,
    report_path_from_env,
    trace_dir_from_report,
    write_json_report,
)


DSV4_GRAPHFX_M_SWEEP = [
    1, 2, 3, 5, 7, 8, 11, 16, 17, 31, 32, 37, 61, 64,
    67, 97, 127, 128, 191, 251, 256, 257, 509, 512, 769, 1021, 1024,
    1531, 2039, 2048, 3079, 4093, 4096, 6151, 8191,
    8192, 12289, 16381, 16384, 24593, 32749, 32768,
    49157, 65521, 65536,
]

DSV4_GRAPHFX_CORRECTNESS_M = [1, 8, 17, 64, 257, 1024]

DSV4_RMSNORM_QUANT_SHAPES = [
    {
        "model_profile": "flash",
        "shape_group": "main",
        "role": "q_lora_norm",
        "K": 1024,
        "downstream_N": [8192, 32768],
    },
    {
        "model_profile": "flash",
        "shape_group": "main",
        "role": "hidden_norm",
        "K": 4096,
        "downstream_N": [512, 1024, 2048],
    },
    {
        "model_profile": "pro",
        "shape_group": "main",
        "role": "q_lora_norm",
        "K": 1536,
        "downstream_N": [8192, 65536],
    },
    {
        "model_profile": "pro",
        "shape_group": "main",
        "role": "hidden_norm",
        "K": 7168,
        "downstream_N": [512, 1536, 3072],
    },
    {
        "model_profile": "flash",
        "shape_group": "guard",
        "role": "moe_intermediate",
        "K": 2048,
        "downstream_N": [4096],
    },
    {
        "model_profile": "pro",
        "shape_group": "guard",
        "role": "moe_intermediate",
        "K": 3072,
        "downstream_N": [7168],
    },
]

DSV4_ROPE_SHAPES = [
    {
        "model_profile": "guard/legacy",
        "shape_group": "guard",
        "role": "q_rope_d128_guard",
        "H": 64,
        "D": 128,
        "rope_dim": 64,
    },
    {
        "model_profile": "flash",
        "shape_group": "main",
        "role": "q_rope_d512",
        "H": 64,
        "D": 512,
        "rope_dim": 64,
    },
    {
        "model_profile": "flash",
        "shape_group": "main",
        "role": "kv_rope",
        "D": 512,
        "rope_dim": 64,
    },
]


@dataclass(frozen=True)
class GraphPair:
    baseline: torch.fx.GraphModule
    candidate: torch.fx.GraphModule
    target_names: list[str]


def target_names(gm: torch.fx.GraphModule) -> list[str]:
    return [getattr(node.target, "__name__", str(node.target)) for node in gm.graph.nodes]


def make_fx_pair(
    module_factory: Callable[[], torch.nn.Module] | type[torch.nn.Module],
    pass_fn: Callable[[torch.fx.GraphModule], torch.fx.GraphModule],
    *,
    required_targets: Sequence[str] = (),
    forbidden_targets: Sequence[str] = (),
) -> GraphPair:
    baseline = torch.fx.symbolic_trace(module_factory())
    candidate = torch.fx.symbolic_trace(module_factory())
    candidate = pass_fn(candidate)
    candidate.recompile()
    names = target_names(candidate)
    missing = [name for name in required_targets if name not in names]
    if missing:
        raise AssertionError(f"GraphFX pass missed required targets {missing}: {names}")
    present = [name for name in forbidden_targets if name in names]
    if present:
        raise AssertionError(f"GraphFX pass kept forbidden targets {present}: {names}")
    return GraphPair(baseline=baseline, candidate=candidate, target_names=names)


def assert_fp8_quant_close(
    ref: tuple[torch.Tensor, torch.Tensor],
    cand: tuple[torch.Tensor, torch.Tensor],
    label: str,
    *,
    q_exact_min: float = 0.94,
    q_max_ulp: int = 2,
    s_exact_min: float = 0.98,
    s_max_byte: int = 1,
) -> None:
    def _byte_view(tensor: torch.Tensor) -> torch.Tensor:
        # dtype reinterpret requires the last stride to be 1. Flatten first so
        # column-major/TMA-aligned scale tensors with size-1 last dimensions do
        # not keep a legal contiguous layout whose last stride is still > 1.
        return tensor.detach().reshape(-1).contiguous().view(torch.uint8)

    q_ref, s_ref = ref
    q_cand, s_cand = cand
    q_diff = (_byte_view(q_ref).to(torch.int16) - _byte_view(q_cand).to(torch.int16)).abs()
    s_diff = (_byte_view(s_ref).to(torch.int16) - _byte_view(s_cand).to(torch.int16)).abs()
    q_exact = (q_diff == 0).float().mean().item()
    s_exact = (s_diff == 0).float().mean().item()
    print(
        f"[{label}] q_exact={q_exact * 100:.2f}% q_max_ulp={q_diff.max().item()} "
        f"s_exact={s_exact * 100:.2f}% s_max_byte={s_diff.max().item()}"
    )
    assert q_exact >= q_exact_min
    assert q_diff.max().item() <= q_max_ulp
    assert s_exact >= s_exact_min
    assert s_diff.max().item() <= s_max_byte


def assert_bf16_close(ref: torch.Tensor, cand: torch.Tensor, label: str, *, atol: float = 1e-3) -> None:
    max_abs = float((cand.float() - ref.float()).abs().max().item())
    print(f"[{label}] bf16_max_abs={max_abs}")
    torch.testing.assert_close(cand, ref, rtol=0, atol=atol)


def graphfx_perf_enabled() -> bool:
    return env_flag("DSV4_GRAPHFX_RUN_PERF_IN_UT", False) or bool(os.environ.get("PERF_JSON"))


def graphfx_m_sweep(env_name: str) -> list[int]:
    return parse_int_list(env_name, DSV4_GRAPHFX_M_SWEEP)


def measured_graph_pair_row(
    *,
    op: str,
    label: str,
    shape_meta: dict,
    baseline_fn: Callable[[], object],
    candidate_fn: Callable[[], object],
    trace_dir: str,
    kernel_regex: str | None = None,
    warmup: int = 20,
    iters: int | None = None,
    profile_enabled: bool = True,
) -> dict:
    m = int(shape_meta["M"])
    measure_iters = iters if iters is not None else iters_for_m(m)
    base = measure_kernel(
        baseline_fn,
        label=f"{label}_baseline",
        trace_dir=trace_dir,
        kernel_regex=kernel_regex,
        warmup=warmup,
        iters=measure_iters,
        profile_enabled=profile_enabled,
        profile_iters=max(1, min(3, measure_iters)),
    )
    cand = measure_kernel(
        candidate_fn,
        label=f"{label}_candidate_graphfx",
        trace_dir=trace_dir,
        kernel_regex=kernel_regex,
        warmup=warmup,
        iters=measure_iters,
        profile_enabled=profile_enabled,
        profile_iters=max(1, min(3, measure_iters)),
    )
    return {
        "op": op,
        **shape_meta,
        "baseline": measurement_payload(base),
        "candidate": measurement_payload(cand),
        "baseline_launches": base.kernel_count,
        "candidate_launches": cand.kernel_count,
        "speedup_kernel_sum": base.kernel_sum_us / max(cand.kernel_sum_us, 1.0e-9),
    }


def write_graphfx_perf_report(
    *,
    json_env: str,
    default_json: str,
    rows: list[dict],
    metadata: dict,
) -> str:
    report_path = os.environ.get("PERF_JSON") or report_path_from_env(json_env, default_json)
    observed_paths = observed_timeline_kernel_paths(rows)
    payload = {
        "device": device_payload(),
        "measure": "torch_profiler_cuda_kernel_sum_us_per_iter",
        "python_overhead_included": False,
        "standard_table_columns": standard_table_columns(),
        "observed_timeline_kernel_path_columns": observed_timeline_kernel_path_columns(),
        "observed_timeline_kernel_paths": observed_paths,
        **metadata,
        "rows": rows,
    }
    write_json_report(report_path, payload)
    write_markdown_summary(os.path.splitext(report_path)[0] + ".md", rows, metadata, observed_paths)
    write_csv_summary(os.path.splitext(report_path)[0] + ".csv", rows)
    print_standard_summary(rows, metadata)
    return report_path


def observed_timeline_kernel_path_columns() -> list[str]:
    return [
        "op",
        "model_profile",
        "shape_group",
        "role",
        "shape",
        "M_values",
        "baseline_timeline_kernel_path",
        "candidate_timeline_kernel_path",
    ]


def standard_table_columns() -> list[str]:
    return [
        "op",
        "model_profile",
        "shape_group",
        "role",
        "M",
        "shape",
        "baseline_kernel_us",
        "candidate_kernel_us",
        "speedup_kernel_sum",
        "baseline_launches",
        "candidate_launches",
    ]


def _standard_shape(row: dict) -> str:
    if "K" in row:
        return f"K={row['K']}"
    if "H" in row and "D" in row:
        return f"H={row['H']} D={row['D']}"
    if "head_dim" in row:
        return f"head_dim={row['head_dim']}"
    if "D" in row:
        return f"D={row['D']}"
    return ""


def _kernel_path(measure: dict) -> list[str]:
    names = measure.get("kernel_names") or []
    return [str(name) for name in names]


def _kernel_path_key(names: Sequence[str]) -> str:
    if not names:
        return "(no CUDA kernel observed)"
    return " -> ".join(names)


def _markdown_cell(value: object) -> str:
    return str(value).replace("|", "\\|").replace("\n", "<br>")


def _markdown_kernel_path(names: Sequence[str]) -> str:
    if not names:
        return "(no CUDA kernel observed)"
    return "<br>".join(f"`{_markdown_cell(name)}`" for name in names)


def observed_timeline_kernel_paths(rows: Iterable[dict]) -> list[dict]:
    grouped: dict[tuple[str, str, str, str, str, str, str], dict] = {}
    for row in rows:
        baseline_path = _kernel_path(row.get("baseline", {}))
        candidate_path = _kernel_path(row.get("candidate", {}))
        key = (
            str(row.get("op", "")),
            str(row.get("model_profile", "")),
            str(row.get("shape_group", "")),
            str(row.get("role", row.get("case", ""))),
            _standard_shape(row),
            _kernel_path_key(baseline_path),
            _kernel_path_key(candidate_path),
        )
        entry = grouped.get(key)
        if entry is None:
            entry = {
                "op": key[0],
                "model_profile": key[1],
                "shape_group": key[2],
                "role": key[3],
                "shape": key[4],
                "M_values": [],
                "baseline_timeline_kernel_path": baseline_path,
                "candidate_timeline_kernel_path": candidate_path,
            }
            grouped[key] = entry
        entry["M_values"].append(int(row.get("M", 0)))
    result = list(grouped.values())
    for entry in result:
        entry["M_values"] = sorted(entry["M_values"])
    result.sort(
        key=lambda item: (
            item["op"],
            item["model_profile"],
            item["shape_group"],
            item["role"],
            item["shape"],
            item["M_values"][0] if item["M_values"] else -1,
        )
    )
    return result


def _standard_row(row: dict) -> dict:
    base = row.get("baseline", {})
    cand = row.get("candidate", {})
    return {
        "op": row.get("op", ""),
        "model_profile": row.get("model_profile", ""),
        "shape_group": row.get("shape_group", ""),
        "role": row.get("role", row.get("case", "")),
        "M": row.get("M", ""),
        "shape": _standard_shape(row),
        "baseline_kernel_us": float(base.get("kernel_sum_us", 0.0)),
        "candidate_kernel_us": float(cand.get("kernel_sum_us", 0.0)),
        "speedup_kernel_sum": float(row.get("speedup_kernel_sum", 0.0)),
        "baseline_launches": row.get("baseline_launches", ""),
        "candidate_launches": row.get("candidate_launches", ""),
    }


def _format_standard_row(row: dict) -> str:
    item = _standard_row(row)
    return (
        "| {op} | {profile} | {group} | {role} | {m} | {shape} | "
        "{base:.3f} | {cand:.3f} | {speedup:.3f} | {bl} | {cl} |"
    ).format(
        op=item["op"],
        profile=item["model_profile"],
        group=item["shape_group"],
        role=item["role"],
        m=item["M"],
        shape=item["shape"],
        base=item["baseline_kernel_us"],
        cand=item["candidate_kernel_us"],
        speedup=item["speedup_kernel_sum"],
        bl=item["baseline_launches"],
        cl=item["candidate_launches"],
    )


def write_markdown_summary(
    path: str,
    rows: Iterable[dict],
    metadata: dict,
    observed_paths: Iterable[dict] | None = None,
) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    row_list = list(rows)
    observed_path_rows = (
        list(observed_paths)
        if observed_paths is not None
        else observed_timeline_kernel_paths(row_list)
    )
    lines = [
        f"# {metadata.get('title', 'DSV4 GraphFX Fusion Perf')}",
        "",
        "## Observed Timeline Kernel Paths",
        "",
        "Source: Torch Profiler CUDA timeline kernel events. "
        "These are observed kernel names from the timeline, not code-inferred paths.",
        "",
        "| op | profile | group | role | shape | M values | "
        "GraphFX before timeline kernel path | GraphFX after timeline kernel path |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for item in observed_path_rows:
        lines.append(
            "| {op} | {profile} | {group} | {role} | {shape} | {m_values} | {base_path} | {cand_path} |".format(
                op=_markdown_cell(item.get("op", "")),
                profile=_markdown_cell(item.get("model_profile", "")),
                group=_markdown_cell(item.get("shape_group", "")),
                role=_markdown_cell(item.get("role", "")),
                shape=_markdown_cell(item.get("shape", "")),
                m_values=_markdown_cell(",".join(str(m) for m in item.get("M_values", []))),
                base_path=_markdown_kernel_path(item.get("baseline_timeline_kernel_path", [])),
                cand_path=_markdown_kernel_path(item.get("candidate_timeline_kernel_path", [])),
            )
        )
    lines.extend(
        [
            "",
            "## Performance Table",
            "",
            "Measure: Torch Profiler CUDA kernel duration sum per iteration.",
            "",
            "| op | profile | group | role | M | shape | baseline kernel us | "
            "candidate kernel us | speedup | baseline launches | candidate launches |",
            "| --- | --- | --- | --- | ---: | --- | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in row_list:
        lines.append(_format_standard_row(row))
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def write_csv_summary(path: str, rows: Iterable[dict]) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    columns = standard_table_columns()
    with open(path, "w", encoding="utf-8") as f:
        f.write(",".join(columns) + "\n")
        for row in rows:
            item = _standard_row(row)
            values = [str(item.get(col, "")) for col in columns]
            f.write(",".join(value.replace(",", ";") for value in values) + "\n")


def print_standard_summary(rows: Iterable[dict], metadata: dict) -> None:
    print(f"\n[{metadata.get('title', 'DSV4 GraphFX Fusion Perf')}]")
    print("Measure: Torch Profiler CUDA kernel duration sum per iteration; Python overhead is excluded.")
    print(
        "| op | profile | group | role | M | shape | baseline kernel us | "
        "candidate kernel us | speedup | baseline launches | candidate launches |"
    )
    print("| --- | --- | --- | --- | ---: | --- | ---: | ---: | ---: | ---: | ---: |")
    for row in rows:
        print(_format_standard_row(row))


def trace_dir_for_report(json_env: str, default_json: str, trace_env: str) -> str:
    report_path = os.environ.get("PERF_JSON") or report_path_from_env(json_env, default_json)
    return trace_dir_from_report(report_path, trace_env)
