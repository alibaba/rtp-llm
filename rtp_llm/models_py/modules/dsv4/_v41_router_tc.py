"""Tensor-core router GEMM for V4.1 (R5-a), default off.

Production runs the router as an FP32 SIMT sgemm (~1 ms/layer on 8192x5120
x384, ~21 ms/chunk). Under ``DSV41_ROUTER_TENSORCORE=tf32`` the same FP32
operands run with TF32 tensor-core math for the single ``F.linear`` call
(restored immediately afterwards; the module-level TF32-disabled guard in
V41Router stays authoritative). Both operands are BF16-derived — hidden
states arrive BF16 and the checkpoint weight is promoted from BF16 — so they
are exactly representable in TF32 (10-bit mantissa superset of BF16's) and
the only deviation from the SIMT path is FP32 accumulation order.

``bf16`` rounds the GEMM output to BF16 before promotion: measurably
coarser (microbench: 11% of rows flip top-6 on synthetic near-ties); kept
for comparison only.

``DSV41_ROUTER_TC_VERIFY=1`` additionally recomputes the FP32 reference and
compares the full gate epilogue's top-k ids on live rows, accumulating flip
counts and margins and logging ``[V41ROUTERVERIFY]`` lines. Verification
only: it doubles the router epilogue and adds host syncs.
"""

import os

import torch
import torch.nn.functional as F

ROUTER_TC_FLAG = "DSV41_ROUTER_TENSORCORE"
ROUTER_TC_VERIFY_FLAG = "DSV41_ROUTER_TC_VERIFY"
MODES = ("0", "tf32", "bf16")

# Process-local verification counters (reset only by process restart).
VERIFY_STATE = {
    "calls": 0,
    "live_rows": 0,
    "flip_rows": 0,
    "max_logit_absdiff": 0.0,
    "worst_flip_margin": None,
}


def router_tc_mode() -> str:
    mode = os.environ.get(ROUTER_TC_FLAG, "0")
    return mode if mode in MODES else "0"


def router_verify_enabled() -> bool:
    return os.environ.get(ROUTER_TC_VERIFY_FLAG, "0") == "1"


def router_linear(clean: torch.Tensor, weight: torch.Tensor, mode: str) -> torch.Tensor:
    """FP32 [rows,dim] x [experts,dim]^T router GEMM, optionally tensor-core."""
    if mode == "tf32" and clean.is_cuda:
        prev = torch.backends.cuda.matmul.allow_tf32
        torch.backends.cuda.matmul.allow_tf32 = True
        try:
            return F.linear(clean, weight)
        finally:
            torch.backends.cuda.matmul.allow_tf32 = prev
    if mode == "bf16" and clean.is_cuda:
        return torch.matmul(clean.bfloat16(), weight.bfloat16().transpose(0, 1)).float()
    return F.linear(clean, weight)


def verify_router_parity(
    ref_logits: torch.Tensor,
    tc_logits: torch.Tensor,
    gate_temp: float,
    bias: torch.Tensor,
    topk: int,
    valid_rows: torch.Tensor,
) -> None:
    """Compare gate top-k ids between the tensor-core and SIMT paths.

    Replicates the production epilogue exactly (softplus/temp -> sqrt ->
    +bias -> topk) on both logit sets and counts id flips among LIVE rows
    only; dead rows are masked to -1 downstream and cannot affect routing.
    ``valid_rows=None`` treats every row as live (the moe/gate.py path runs
    on restored real rows without a mask).
    """
    if valid_rows is None:
        valid_rows = torch.ones(
            ref_logits.shape[0], dtype=torch.bool, device=ref_logits.device
        )
    ref_scores = F.softplus(ref_logits / gate_temp).sqrt()
    tc_scores = F.softplus(tc_logits / gate_temp).sqrt()
    ref_rank = ref_scores + bias
    tc_rank = tc_scores + bias
    ref_ids = ref_rank.topk(topk, dim=-1).indices
    tc_ids = tc_rank.topk(topk, dim=-1).indices
    flipped = (ref_ids != tc_ids).any(dim=-1) & valid_rows
    n_flips = int(flipped.sum())
    diff = float((tc_logits - ref_logits).abs().max())
    state = VERIFY_STATE
    state["calls"] += 1
    state["live_rows"] += int(valid_rows.sum())
    state["flip_rows"] += n_flips
    state["max_logit_absdiff"] = max(state["max_logit_absdiff"], diff)
    worst = None
    if n_flips:
        ordered = ref_rank.sort(dim=-1, descending=True).values
        margins = ordered[flipped, topk - 1] - ordered[flipped, topk]
        worst = float(margins.min())
        prev_worst = state["worst_flip_margin"]
        state["worst_flip_margin"] = (
            worst if prev_worst is None else min(prev_worst, worst)
        )
    if n_flips or state["calls"] % 21 == 0:
        wm = state["worst_flip_margin"]
        print(
            "[V41ROUTERVERIFY] calls=%d live_rows=%d flip_rows=%d "
            "max_logit_absdiff=%.6e worst_flip_margin=%s last_flips=%d%s"
            % (
                state["calls"],
                state["live_rows"],
                state["flip_rows"],
                state["max_logit_absdiff"],
                "None" if wm is None else "%.6e" % wm,
                n_flips,
                "" if worst is None else " this_margin=%.6e" % worst,
            ),
            flush=True,
        )
