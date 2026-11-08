"""Require actual projection construction and NVFP4 execution on every rank."""

import json
from collections import Counter

from rtp_llm.models.kimi_k3.nvfp4_config import selected_nvfp4_modules, uses_fp8_output

MARKER = "[K31_PROJECTION_EVENT] "
LINEAR_POLICIES = {
    "NVFP4": ("CudaFp4GEMMLinear", "torch.uint8", "dynamic"),
    "FP8": ("CudaFp8GEMMLinear", "torch.float8_e4m3fn", "native"),
    "BF16": ("CudaF16Linear", "torch.bfloat16", "native"),
}


def matches_precision(row, expected):
    return (
        tuple(row.get(field) for field in ("linear", "weight_dtype", "activation_scheme"))
        == LINEAR_POLICIES[expected]
        and row.get("weight_precision") == row.get("activation_precision") == expected
    )


def audit_projection_precision(directory, config, *, layers=93, ranks=8):
    initialized, executed, draft = {}, set(), set()
    modules = selected_nvfp4_modules(config["kda_nvfp4"], config["dsa_nvfp4"])
    for path in (directory / "logs").rglob("*.log"):
        for line in path.read_text(errors="replace").splitlines():
            if MARKER not in line:
                continue
            row = json.loads(line.split(MARKER, 1)[1])
            key = (row["rank"], row["layer"], row["module"])
            if row["model"] == "draft":
                if (
                    not matches_precision(row, "BF16")
                    or row["kda_nvfp4"]
                    or row["dsa_nvfp4"]
                ):
                    raise ValueError(f"draft precision changed: {key}")
                draft.add(row["rank"])
                continue
            if row["model"] != "target":
                raise ValueError("unknown projection model identity")
            if any(row[key] is not config[key] for key in ("kda_nvfp4", "dsa_nvfp4")):
                raise ValueError(f"projection switches differ: {key}")
            module = row["module"]
            if uses_fp8_output(module, modules):
                expected = "FP8"
            elif module in modules:
                expected = "NVFP4"
            else:
                expected = "FP8" if config.get("fp8_gemm", True) else "BF16"
            if not matches_precision(row, expected):
                raise ValueError(
                    f"projection precision differs: {key}, expected {expected}, "
                    f"got {row['linear']} W={row.get('weight_precision')} "
                    f"A={row.get('activation_precision')} ({row['weight_dtype']})"
                )
            if row["event"] == "initialized":
                if key in initialized and initialized[key] != row:
                    raise ValueError(f"conflicting projection receipts: {key}")
                initialized[key] = row
            elif row["event"] == "executed":
                expected_input = "torch.uint8" if row.get("prequantized_input") else "torch.bfloat16"
                if (
                    row["input_dtype"] != expected_input
                    or row["output_dtype"] != "torch.bfloat16"
                ):
                    raise ValueError(f"NVFP4 input/output dtype differs: {key}")
                executed.add((*key, row.get("fragment", "fused")))
            else:
                raise ValueError("unknown projection evidence event")
    for rank in range(ranks):
        for layer in range(layers):
            keys = [key for key in initialized if key[:2] == (rank, layer)]
            modules = {key[2] for key in keys}
            if modules not in (
                {"kda.input", "kda.forget", "kda.output"},
                {"dsa.input", "dsa.query", "dsa.output"},
            ):
                raise ValueError(
                    f"missing projection initialization: rank={rank} layer={layer}"
                )
    if draft != set(range(ranks)):
        raise ValueError("missing native draft precision receipts")
    selected = {key for key, row in initialized.items() if row["weight_precision"] == "NVFP4"}
    required = {
        (*key, fragment)
        for key in selected
        for fragment in initialized[key].get("execution_fragments", ["fused"])
    }
    if required - executed:
        raise ValueError(
            f"NVFP4 projections never executed: {sorted(required - executed)[:8]}"
        )
    return dict(
        passed=True,
        initialized=len(initialized),
        initialized_by_precision=dict(Counter(row["weight_precision"] for row in initialized.values())),
        nvfp4_executed=len(selected),
        executed_fragments=len(required),
        draft_ranks=sorted(draft),
    )
