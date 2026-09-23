"""CPU tests of actual MoE/count methods with simulated leaf modules and collectives.

Cover per-rank decisions, record lifetime and failures; not GPU or factory selection."""

from __future__ import annotations

import ast
import hashlib
import importlib.util
import json
import os
import pathlib
import sys
import types

import torch

assert os.environ.get("CUDA_VISIBLE_DEVICES") == "", "CUDA must be hidden for CPU tests"

HERE = pathlib.Path(__file__).resolve().parent
WT = HERE.parents[5]
STRAT_REL = "rtp_llm/models_py/modules/dsv4/moe/strategies/nccl_ep_mxfp8.py"
LAYER_REL = "rtp_llm/models_py/modules/dsv4/moe/moe_layer.py"
CHUNK_REL = "rtp_llm/models_py/modules/dsv4/chunk_env.py"
CANDIDATE = WT / STRAT_REL
PKG = "rtp_llm.models_py.modules.dsv4.moe.strategies"
FUSE_FLAG = "DSV4_MOE_EXTENT_COUNT_FUSE"

checks = []
FAILS = []


def ck(name, cond, detail=""):
    checks.append({"name": name, "pass": bool(cond), "detail": str(detail)})
    if not cond:
        FAILS.append("%s: %s" % (name, detail))
    return bool(cond)


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, str(path))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


class _NullCtx:
    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


# ---------------- stubbed package chain (heavy siblings only) ----------------
def _stub_packages():
    for n in (
        "rtp_llm",
        "rtp_llm.models_py",
        "rtp_llm.models_py.modules",
        "rtp_llm.models_py.modules.dsv4",
        "rtp_llm.models_py.modules.dsv4.moe",
        PKG,
    ):
        if n not in sys.modules:
            m = types.ModuleType(n)
            m.__path__ = []
            sys.modules[n] = m
    base = _load(
        PKG + ".base", WT / "rtp_llm/models_py/modules/dsv4/moe/strategies/base.py"
    )
    setattr(sys.modules[PKG], "base", base)

    utils_pkg = types.ModuleType("rtp_llm.models_py.utils")
    utils_pkg.__path__ = []
    sys.modules[utils_pkg.__name__] = utils_pkg
    setattr(sys.modules["rtp_llm.models_py"], "utils", utils_pkg)
    arch = types.ModuleType("rtp_llm.models_py.utils.arch")
    arch.is_sm120 = lambda *_a, **_k: False
    sys.modules[arch.__name__] = arch
    setattr(utils_pkg, "arch", arch)

    prof = types.ModuleType("rtp_llm.models_py.modules.dsv4._profiler")
    prof.record_function_range = lambda *_a, **_k: _NullCtx()
    sys.modules["rtp_llm.models_py.modules.dsv4._profiler"] = prof
    setattr(sys.modules["rtp_llm.models_py.modules.dsv4"], "_profiler", prof)

    comb = types.ModuleType("rtp_llm.models_py.modules.dsv4.moe._nccl_ep_mxfp8_combine")
    comb.SCALE_BLOCK = 32
    comb.mxfp8_dequant_peer_sum = lambda *a, **k: None
    sys.modules[comb.__name__] = comb
    setattr(
        sys.modules["rtp_llm.models_py.modules.dsv4.moe"],
        "_nccl_ep_mxfp8_combine",
        comb,
    )

    ws = types.ModuleType("rtp_llm.models_py.modules.dsv4.moe.warmup_sync")
    ws.cuda_graph_warmup_forward_enabled = lambda: False
    sys.modules[ws.__name__] = ws
    setattr(sys.modules["rtp_llm.models_py.modules.dsv4.moe"], "warmup_sync", ws)

    gfp4 = types.ModuleType(PKG + ".grouped_fp4")

    class _GroupedFP4Double:
        def __init__(self, *_a, **_k):
            pass

    gfp4.GroupedFP4Strategy = _GroupedFP4Double
    gfp4._has_fp8_fp4_grouped_kernel = lambda: False
    sys.modules[gfp4.__name__] = gfp4
    setattr(sys.modules[PKG], "grouped_fp4", gfp4)

    ll = types.ModuleType(PKG + ".local_loop")

    class _LocalLoopDouble:
        def __init__(self, *_a, **_k):
            pass

    ll.LocalLoopStrategy = _LocalLoopDouble
    sys.modules[ll.__name__] = ll
    setattr(sys.modules[PKG], "local_loop", ll)

    rt = types.ModuleType("rtp_llm.models_py.modules.dsv4._record_tensor")
    rt.should_record_layer = lambda _lid: False
    rt.record_if_level = lambda *_a, **_k: None
    sys.modules[rt.__name__] = rt
    setattr(sys.modules["rtp_llm.models_py.modules.dsv4"], "_record_tensor", rt)


_stub_packages()
_load(
    "rtp_llm.models_py.modules.dsv4.moe.forward_ep_plan",
    WT / "rtp_llm/models_py/modules/dsv4/moe/forward_ep_plan.py",
)
cand = _load(PKG + "._cand_caller", CANDIDATE)
chunk_env = _load("_chunk_env_real", WT / CHUNK_REL)

# ---------------- AST-extract the ACTUAL caller surface ----------------------
LAYER_SRC = (WT / LAYER_REL).read_text()
tree = ast.parse(LAYER_SRC)
METHODS = ("forward", "_should_chunk", "_run_chunk", "_forward_chunked")
MODULE_FUNCS = (
    "chunked_moe_enabled",
    "moe_chunk_tokens_from_env",
    "cp_padded_tokens_per_rank_bound",
    "resolve_moe_max_tokens_per_rank",
    "_get_or_create_final_out",
)
method_nodes = {}
for node in tree.body:
    if isinstance(node, ast.ClassDef) and node.name == "MoE":
        for sub in node.body:
            if isinstance(sub, ast.FunctionDef) and sub.name in METHODS:
                method_nodes[sub.name] = sub
    if isinstance(node, ast.FunctionDef) and node.name in MODULE_FUNCS:
        method_nodes[node.name] = node
ck(
    "caller_surface_extracted",
    sorted(method_nodes) == sorted(set(METHODS) | set(MODULE_FUNCS)),
    sorted(method_nodes),
)

fwd_node = method_nodes["forward"]
first_stmt = [s for s in fwd_node.body if not isinstance(s, ast.Global)][0]
ck(
    "caller_forward_is_the_real_source",
    "_forward_chunked" in LAYER_SRC and "synchronized_chunk_extent" in LAYER_SRC,
    "extraction must come from the real moe_layer.py",
)

from copy import deepcopy  # noqa: E402

_env = {
    "torch": torch,
    "os": os,
    "logging": types.SimpleNamespace(info=lambda *a, **k: None),
    "record_function_range": lambda *_a, **_k: _NullCtx(),
    "DEFAULT_DSV4_CHUNK_TOKENS": chunk_env.DEFAULT_DSV4_CHUNK_TOKENS,
    "DEFAULT_MOE_CHUNK_TOKENS": chunk_env.DEFAULT_DSV4_CHUNK_TOKENS,
    "dsv4_chunk_tokens_from_env": chunk_env.dsv4_chunk_tokens_from_env,
    "dsv4_global_chunk_tokens_configured": chunk_env.dsv4_global_chunk_tokens_configured,
    "combine_routed_and_shared": None,  # installed below
    "_FINAL_OUT_CACHE": {},
    "_CHUNKED_MOE_LOGGED": False,
}
exec(
    compile(
        ast.Module(body=[deepcopy(n) for n in method_nodes.values()], type_ignores=[]),
        str(WT / LAYER_REL),
        "exec",
    ),
    _env,
)


def combine_routed_and_shared(routed, shared, dtype, out=None):
    """Caller-contract double: writes routed (+ shared if any) into `out`."""
    src = routed if shared is None else routed + shared
    if out is None:
        return src.to(dtype)
    out.copy_(src.to(dtype))
    return out


_env["combine_routed_and_shared"] = combine_routed_and_shared

# ---------------- collective doubles (stage-local) ---------------------------
CALLS = []


class SimGroup:
    """A stage roster. `global_ranks` is the world roster (for leak detection);
    `local_rank` is the STAGE-LOCAL rank (0..3) — the value the real
    EpStageContext carries; `counts[i]` is the row count of global_ranks[i]."""

    def __init__(self, global_ranks, local_rank, counts):
        self.ranks = list(global_ranks)
        self.rank = int(local_rank)
        self.counts = list(counts)

    def __repr__(self):
        return "SimGroup(ranks=%r, local=%d, counts=%r)" % (
            self.ranks,
            self.rank,
            self.counts,
        )


def sim_all_reduce(t, op, group):
    assert isinstance(group, SimGroup), "collective left the stage group: %r" % (group,)
    CALLS.append({"op": "ar", "group": group.ranks, "value": int(t.item())})
    assert op == torch.distributed.ReduceOp.MAX
    t.fill_(max(group.counts))


def sim_all_gather_into_tensor(out, inp, group):
    assert isinstance(group, SimGroup), "collective left the stage group: %r" % (group,)
    CALLS.append({"op": "ag", "group": group.ranks, "local": int(inp.item())})
    out.view(-1)[:] = torch.tensor(group.counts, dtype=out.dtype)


torch.distributed.all_reduce = sim_all_reduce
torch.distributed.all_gather_into_tensor = sim_all_gather_into_tensor
torch.distributed.is_initialized = lambda: True


class FakeCtx:
    def __init__(self, group):
        self.process_group = group
        self.group_size = len(group.ranks)
        self.group_rank = group.rank


class StratDouble(cand.NcclEpMxfp8Strategy):
    """Runs the ACTUAL count/guard methods; the tensor work is doubled."""

    def __init__(self, group, hidden=32, topk=2):
        cfg = cand.MoeCfg(
            layer_id=22,
            dim=hidden,
            moe_inter_dim=hidden,
            n_routed_experts=64,
            n_activated_experts=topk,
            swiglu_limit=10.0,
            ep_size=len(group.ranks),
            ep_rank=group.rank,
            n_local_experts=64,
            local_expert_start=0,
            local_expert_end=64,
            max_tokens_per_rank=16384,
            stage_context=FakeCtx(group),
        )
        super().__init__(cfg)
        self._group = group
        self.hidden = hidden
        self.fail_next = False
        self.forward_calls = []

    def _stage(self):
        return self._group, len(self._group.ranks), self._group.rank

    def forward(self, x, weights, indices):
        # mirrors the real entry order: ownership transfer FIRST
        pending = self._take_pending_counts()
        TRACE.append(
            {
                "ev": "forward_entry",
                "rows": int(x.size(0)),
                "took_record": pending is not None,
            }
        )
        if self.fail_next:
            self.fail_next = False
            raise RuntimeError("injected forward failure (after the ownership pop)")
        n_local = int(x.size(0))
        counts = self._counts_for_forward(
            n_local, self._group, len(self._group.ranks), x.device, pending
        )
        TRACE.append({"ev": "counts", "rows": n_local, "counts": list(counts)})
        self.forward_calls.append((n_local, tuple(counts)))
        return torch.full(
            (n_local, self.hidden), float(sum(counts)), dtype=torch.float32
        )


class GateDouble:
    def __init__(self, topk=2):
        self.topk = topk
        self.route_scale = 1.0

    def __call__(self, x, input_ids):
        n = int(x.size(0))
        return (
            torch.ones(n, self.topk, dtype=torch.float32),
            torch.zeros(n, self.topk, dtype=torch.int64),
        )


class SharedDouble:
    def start(self, shared, x):
        return None

    def finish(self):
        return None


class FakeMoE:
    """Carries the AST-extracted real methods; the wiring is a test double."""

    def __init__(self, strat, max_tokens, dim=32, topk=2, decode=False):
        self.layer_id = 22
        self.dim = dim
        self.max_tokens_per_rank = max_tokens
        self._is_decode_role = decode
        self._gate_pack_static = False
        self._routed_includes_shared = False
        self.gate = GateDouble(topk)
        self.shared_experts = None
        self._shared_executor = SharedDouble()
        self._strategy = strat
        self._dbg_positions = None
        for name in METHODS:
            setattr(self, name, types.MethodType(_env[name], self))

    def __call__(self, x, input_ids):
        return self.forward(x, input_ids)


TRACE = []


def fresh():
    CALLS.clear()
    TRACE.clear()


def ops():
    return [c["op"] for c in CALLS]


def run_caller(
    rows,
    group_counts,
    fused,
    max_tokens=4096,
    chunking=True,
    local_group=(0, 1, 2, 3),
    local_rank=0,
    fail_next=False,
):
    os.environ[FUSE_FLAG] = "1" if fused else "0"
    os.environ["DSV4_MOE_CHUNK_PREFILL"] = "1" if chunking else "0"
    os.environ.pop("DSV4_CHUNK_TOKENS", None)
    group = SimGroup(local_group, local_rank, group_counts)
    strat = StratDouble(group)
    strat.fail_next = fail_next
    moe = FakeMoE(strat, max_tokens)
    x = torch.randn(rows, moe.dim)
    ids = torch.zeros(rows, dtype=torch.int64)
    fresh()
    err = None
    try:
        y = moe(x, ids)
    except Exception as e:  # noqa: BLE001
        err = e
        y = None
    return {
        "ops": list(ops()),
        "trace": list(TRACE),
        "y": y,
        "err": err,
        "strat": strat,
        "calls": CALLS,
        "rows": rows,
    }


# ---------------- 1. uniform no-subchunk: eligible reuse ----------------------
r_on = run_caller(4096, [4096] * 4, fused=True)
r_off = run_caller(4096, [4096] * 4, fused=False)
ck("uniform_on_ops", r_on["ops"] == ["ag"], str(r_on["ops"]))
ck("uniform_off_ops", r_off["ops"] == ["ar", "ag"], str(r_off["ops"]))
ck("uniform_savings_one", len(r_off["ops"]) - len(r_on["ops"]) == 1)
ck(
    "uniform_value_agreement",
    bool(torch.equal(r_on["y"].view(torch.uint8), r_off["y"].view(torch.uint8))),
    "caller outputs must agree bitwise between arms",
)
ck("uniform_record_consumed", r_on["strat"]._pending_counts is None)

# every LOCAL rank of the same group state makes the same decision (rank-uniform)
decisions = []
for lr in range(4):
    rr = run_caller(4096, [4096] * 4, fused=True, local_rank=lr)
    decisions.append(
        (rr["ops"], rr["trace"][1]["counts"] if len(rr["trace"]) > 1 else None)
    )
ck(
    "uniform_decision_identical_across_local_ranks",
    all(d[0] == ["ag"] and d[1] == [4096] * 4 for d in decisions),
    str(decisions),
)

# ---------------- 2. uniform subchunk: fallback, zero savings -----------------
r_on = run_caller(8192, [8192] * 4, fused=True, max_tokens=4096)
r_off = run_caller(8192, [8192] * 4, fused=False, max_tokens=4096)
ck("subchunk_on_ops", r_on["ops"] == ["ag", "ag", "ag"], str(r_on["ops"]))
ck("subchunk_off_ops", r_off["ops"] == ["ar", "ag", "ag"], str(r_off["ops"]))
ck("subchunk_savings_zero", len(r_off["ops"]) - len(r_on["ops"]) == 0)
ck(
    "subchunk_counts_per_chunk",
    [t["counts"] for t in r_on["trace"] if t["ev"] == "counts"] == [[8192] * 4] * 2,
    str(r_on["trace"]),
)
ck(
    "subchunk_value_agreement",
    bool(torch.equal(r_on["y"].view(torch.uint8), r_off["y"].view(torch.uint8))),
)

# ---------------- 3. ragged: fallback, zero savings ---------------------------
r_on = run_caller(3072, [4096, 3072, 4096, 4096], fused=True)
r_off = run_caller(3072, [4096, 3072, 4096, 4096], fused=False)
ck("ragged_on_ops", r_on["ops"] == ["ag", "ag"], str(r_on["ops"]))
ck("ragged_off_ops", r_off["ops"] == ["ar", "ag"], str(r_off["ops"]))
ck("ragged_savings_zero", len(r_off["ops"]) - len(r_on["ops"]) == 0)
ck(
    "ragged_counts_authoritative",
    [t["counts"] for t in r_on["trace"] if t["ev"] == "counts"]
    == [[4096, 3072, 4096, 4096]],
    str(r_on["trace"]),
)
ck(
    "ragged_value_agreement",
    bool(torch.equal(r_on["y"].view(torch.uint8), r_off["y"].view(torch.uint8))),
)

# ---------------- 4. uniform empty: eligible reuse ----------------------------
r_on = run_caller(0, [0, 0, 0, 0], fused=True)
r_off = run_caller(0, [0, 0, 0, 0], fused=False)
ck("empty_on_ops", r_on["ops"] == ["ag"], str(r_on["ops"]))
ck("empty_off_ops", r_off["ops"] == ["ar", "ag"], str(r_off["ops"]))
ck("empty_savings_one", len(r_off["ops"]) - len(r_on["ops"]) == 1)
# mixed empty (this rank empty, peers not): non-uniform => ALL fall back
mix = []
for lr in range(4):
    rr = run_caller(0, [0, 2048, 0, 0], fused=True, local_rank=lr)
    mix.append((rr["ops"], [t["counts"] for t in rr["trace"] if t["ev"] == "counts"]))
ck(
    "mixed_empty_falls_back_everywhere",
    all(m[0] == ["ag", "ag"] and m[1] == [[0, 2048, 0, 0]] for m in mix),
    str(mix),
)

# ---------------- 5. skipped extent: no stale record, fresh gather ------------
r_on = run_caller(4096, [4096] * 4, fused=True, chunking=False)
ck("skipped_extent_single_gather", r_on["ops"] == ["ag"], str(r_on["ops"]))
ck(
    "skipped_extent_counts_current",
    [t["counts"] for t in r_on["trace"] if t["ev"] == "counts"] == [[4096] * 4],
    str(r_on["trace"]),
)

# leak sequence through the caller: round 1 arms a record and the forward fails
# right after the entry pop; round 2 skips the extent (chunking off) and the
# peers have changed -> the later forward must gather the CURRENT state.
os.environ[FUSE_FLAG] = "1"
os.environ["DSV4_MOE_CHUNK_PREFILL"] = "1"
group = SimGroup([0, 1, 2, 3], 0, [4096, 4096, 4096, 4096])
strat = StratDouble(group)
strat.fail_next = True
moe = FakeMoE(strat, 4096)
fresh()
err = None
try:
    moe(torch.randn(4096, moe.dim), torch.zeros(4096, dtype=torch.int64))
except Exception as e:  # noqa: BLE001
    err = e
ck("leak_round1_forward_failed", err is not None and "injected" in str(err), repr(err))
ck(
    "leak_round1_record_destroyed",
    strat._pending_counts is None,
    "the entry pop must destroy the record even though the forward raised",
)
group.counts = [4096, 2048, 4096, 4096]  # peers changed while no extent ran
os.environ["DSV4_MOE_CHUNK_PREFILL"] = "0"
fresh()
moe(torch.randn(4096, moe.dim), torch.zeros(4096, dtype=torch.int64))
ck(
    "leak_round2_no_stale_reuse",
    ops() == ["ag"]
    and [t["counts"] for t in TRACE if t["ev"] == "counts"]
    == [[4096, 2048, 4096, 4096]],
    "ops=%r trace=%r" % (ops(), TRACE),
)

# cross-layer: layer 2 has its own carrier and must gather for itself
os.environ["DSV4_MOE_CHUNK_PREFILL"] = "0"
strat2 = StratDouble(group)
moe2 = FakeMoE(strat2, 4096)
fresh()
moe2(torch.randn(4096, moe2.dim), torch.zeros(4096, dtype=torch.int64))
ck(
    "cross_layer_no_bleed",
    ops() == ["ag"]
    and [t["counts"] for t in TRACE if t["ev"] == "counts"]
    == [[4096, 2048, 4096, 4096]],
    "layer 2 must gather its own counts: %r" % (TRACE,),
)

# ---------------- 6. exception path leaves nothing behind ---------------------
os.environ[FUSE_FLAG] = "1"
os.environ["DSV4_MOE_CHUNK_PREFILL"] = "1"
group = SimGroup([0, 1, 2, 3], 0, [4096] * 4)
strat = StratDouble(group)
strat.fail_next = True
moe = FakeMoE(strat, 4096)
fresh()
err = None
try:
    moe(torch.randn(4096, moe.dim), torch.zeros(4096, dtype=torch.int64))
except Exception as e:  # noqa: BLE001
    err = e
ck(
    "exception_propagates",
    isinstance(err, RuntimeError) and "injected" in str(err),
    repr(err),
)
ck("exception_leaves_no_record", strat._pending_counts is None)
fresh()
y = moe(torch.randn(4096, moe.dim), torch.zeros(4096, dtype=torch.int64))
ck(
    "post_exception_forward_healthy",
    ops() == ["ag"]
    and [t["counts"] for t in TRACE if t["ev"] == "counts"] == [[4096] * 4],
    "ops=%r" % (ops(),),
)

# ---------------- 7. both stage groups, local ranks 0..3 ----------------------
for group_ranks in ([0, 1, 2, 3], [4, 5, 6, 7]):
    seen = []
    for lr in range(4):
        rr = run_caller(
            4096, [4096] * 4, fused=True, local_group=group_ranks, local_rank=lr
        )
        seen.append(
            (
                rr["ops"],
                (
                    tuple(rr["strat"].forward_calls[0][1])
                    if rr["strat"].forward_calls
                    else None
                ),
            )
        )
        ck(
            "group_%s_local_%d_no_foreign_collective" % (group_ranks[0], lr),
            all(c["group"] == group_ranks for c in rr["calls"]),
            str(rr["calls"]),
        )
    ck(
        "group_%s_uniform_agreement" % group_ranks[0],
        all(s == (["ag"], (4096, 4096, 4096, 4096)) for s in seen),
        str(seen),
    )

# ---------------- 8. opt-in default-off at the caller -------------------------
os.environ.pop(FUSE_FLAG, None)
r = run_caller(4096, [4096] * 4, fused=False)
ck("caller_default_off_uses_allreduce", r["ops"] == ["ar", "ag"], str(r["ops"]))

# ---------------- 9. actual received-MXFP8 entry selection ------------------
# Exercise the candidate's real _compute_local body with CPU tensors and only
# lightweight sibling doubles. This is deliberately not a numerical/NaN test.
from types import SimpleNamespace  # noqa: E402

flashinfer = types.ModuleType("flashinfer")
flashinfer.mxfp8_quantize = lambda *_a, **_k: (_ for _ in ()).throw(
    AssertionError(
        "_compute_local imports this symbol only; CPU test must not quantize"
    )
)
sys.modules["flashinfer"] = flashinfer


class EagerLocal(cand.GroupedFP4Strategy):
    def __init__(self):
        self.calls = []

    def forward_sm120_eager(self, x, weights, indices, input_scale=None):
        self.calls.append(
            ("eager", x.dtype, weights.clone(), indices.clone(), input_scale.dtype)
        )
        return torch.full((x.size(0), x.size(1)), 7.0, dtype=torch.bfloat16)

    def __call__(self, x, weights, indices):
        self.calls.append(("fallback", x.dtype, weights.clone(), indices.clone(), None))
        return torch.full((x.size(0), x.size(1)), 3.0, dtype=torch.bfloat16)


class FallbackLocal:
    def __init__(self):
        self.calls = []

    def __call__(self, x, weights, indices):
        self.calls.append(("fallback", x.dtype, weights.clone(), indices.clone(), None))
        return torch.full((x.size(0), x.size(1)), 3.0, dtype=torch.bfloat16)


def compute_probe(
    local, prequant, required=False, capture=False, warmup=False, queries=None
):
    os.environ["DSV4_MOE_PREQUANT_INPUT"] = "1" if prequant else "0"
    os.environ["DSV4_MOE_PREQUANT_INPUT_REQUIRED"] = "1" if required else "0"
    queries = queries if queries is not None else []
    old_capture = cand.torch.cuda.is_current_stream_capturing
    old_warmup = cand.cuda_graph_warmup_forward_enabled

    def capture_probe():
        queries.append("capture")
        return capture

    def warmup_probe():
        queries.append("warmup")
        return warmup

    cand.torch.cuda.is_current_stream_capturing = capture_probe
    cand.cuda_graph_warmup_forward_enabled = warmup_probe
    try:
        probe = object.__new__(cand.NcclEpMxfp8Strategy)
        probe.cfg = SimpleNamespace(local_expert_start=0, local_expert_end=2)
        probe._local = local
        hidden, scales, topk = 128, 4, 1
        payload_cols = hidden + scales + topk * 4 + topk * 4
        recv = torch.zeros((1, payload_cols), dtype=torch.uint8)
        recv[:, hidden + scales : hidden + scales + 4].view(torch.float32).fill_(1.0)
        recv[:, hidden + scales + 4 :].view(torch.int32).fill_(0)
        return probe._compute_local(
            recv, [1, 0, 0, 0], hidden, scales, topk, payload_cols
        )
    finally:
        cand.torch.cuda.is_current_stream_capturing = old_capture
        cand.cuda_graph_warmup_forward_enabled = old_warmup


prequant = EagerLocal()
prequant_queries = []
out_prequant = compute_probe(prequant, prequant=True, queries=prequant_queries)
ck(
    "prequant_actual_compute_selects_eager",
    [c[0] for c in prequant.calls] == ["eager"],
    prequant.calls,
)
ck(
    "prequant_requested_queries_capture_and_warmup",
    prequant_queries == ["capture", "warmup"],
    prequant_queries,
)
ck(
    "prequant_forwards_received_float8_and_scale",
    prequant.calls[0][1] == torch.float8_e4m3fn and prequant.calls[0][4] == torch.uint8,
    repr(prequant.calls[0]),
)
ck(
    "prequant_preserves_remapped_route",
    bool(
        torch.equal(prequant.calls[0][2], torch.ones((1, 1)))
        and torch.equal(prequant.calls[0][3], torch.zeros((1, 1), dtype=torch.int64))
    ),
    repr(prequant.calls[0][2:4]),
)
ck(
    "prequant_actual_compute_returns_partial",
    out_prequant.dtype == torch.float32 and bool(torch.all(out_prequant == 7.0)),
    repr(out_prequant),
)

fallback = EagerLocal()
default_queries = []
out_fallback = compute_probe(fallback, prequant=False, queries=default_queries)
ck(
    "prequant_default_off_uses_bf16_fallback",
    [c[0] for c in fallback.calls] == ["fallback"]
    and fallback.calls[0][1] == torch.bfloat16,
    fallback.calls,
)
ck(
    "prequant_default_off_output_contract",
    out_fallback.dtype == torch.float32 and bool(torch.all(out_fallback == 3.0)),
    repr(out_fallback),
)
ck(
    "prequant_default_off_makes_zero_availability_queries",
    default_queries == [],
    default_queries,
)

required_error = None
required_disabled_queries = []
try:
    compute_probe(
        FallbackLocal(),
        prequant=False,
        required=True,
        queries=required_disabled_queries,
    )
except RuntimeError as exc:
    required_error = exc
ck(
    "required_without_enable_rejects_configuration",
    required_error is not None
    and "requires DSV4_MOE_PREQUANT_INPUT=1" in str(required_error),
    repr(required_error),
)
ck(
    "required_without_enable_makes_zero_availability_queries",
    required_disabled_queries == [],
    required_disabled_queries,
)

for label, capture, warmup in (("capture", True, False), ("warmup", False, True)):
    optional = EagerLocal()
    optional_queries = []
    optional_out = compute_probe(
        optional,
        prequant=True,
        capture=capture,
        warmup=warmup,
        queries=optional_queries,
    )
    ck(
        label + "_optional_prequant_falls_back",
        [c[0] for c in optional.calls] == ["fallback"]
        and optional_out.dtype == torch.float32
        and bool(torch.all(optional_out == 3.0)),
        optional.calls,
    )
    expected_queries = ["capture"] if capture else ["capture", "warmup"]
    ck(
        label + "_optional_queries_expected_availability",
        optional_queries == expected_queries,
        optional_queries,
    )

    required_error = None
    required_queries = []
    try:
        compute_probe(
            EagerLocal(),
            prequant=True,
            required=True,
            capture=capture,
            warmup=warmup,
            queries=required_queries,
        )
    except RuntimeError as exc:
        required_error = exc
    ck(
        label + "_required_prequant_rejects_fallback",
        required_error is not None and "PREQUANT_INPUT_REQUIRED" in str(required_error),
        repr(required_error),
    )
    ck(
        label + "_required_queries_expected_availability",
        required_queries == expected_queries,
        required_queries,
    )

os.environ.pop("DSV4_MOE_PREQUANT_INPUT", None)
os.environ.pop("DSV4_MOE_PREQUANT_INPUT_REQUIRED", None)
os.environ.pop(FUSE_FLAG, None)
os.environ.pop("DSV4_MOE_CHUNK_PREFILL", None)

summary = {
    "ttft_port_cpu_checks": len(checks),
    "failed": [f for f in FAILS],
    "pass": not FAILS,
}
print(json.dumps(summary, indent=1))
sys.exit(0 if not FAILS else 1)
