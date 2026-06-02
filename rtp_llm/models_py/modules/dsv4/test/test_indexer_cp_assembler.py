"""Stage 5b-4 — indexer CP assembler UTs (CPU).

Verifies:
  * ``build_indexer_cp_chunk_plan`` — restore_indices + local lengths math
    matches the per-iteration restore contract.
  * ``build_local_cu_kv_seqlens`` — int32 cumsum, leading 0.
  * ``assemble_indexer_k`` — gather (stubbed identity for cp_size=1) +
    restore writes the right rows into the out buffers; rejects shape
    mismatches.
"""

import importlib.util
import sys
import types
from contextlib import contextmanager
from pathlib import Path

import torch

_REPO_ROOT = Path(__file__).resolve().parents[5]


def _stub_modules():
    mod_name = "rtp_llm.models_py.distributed.collective_torch"
    if mod_name not in sys.modules:
        stub = types.ModuleType(mod_name)

        class _Group:
            TP = "TP"

        def _stub_all_gather(local, group=None):
            return local

        stub.Group = _Group
        stub.all_gather = _stub_all_gather
        sys.modules[mod_name] = stub
        for p in (
            "rtp_llm",
            "rtp_llm.models_py",
            "rtp_llm.models_py.distributed",
        ):
            if p not in sys.modules:
                sys.modules[p] = types.ModuleType(p)

    cp_name = "rtp_llm.models_py.modules.dsv4.cp"
    if cp_name not in sys.modules:
        profiler_name = "rtp_llm.models_py.modules.dsv4._profiler"
        if profiler_name not in sys.modules:
            profiler = types.ModuleType(profiler_name)

            @contextmanager
            def _record_function_range(*args, **kwargs):
                yield

            profiler.record_function_range = _record_function_range
            sys.modules[profiler_name] = profiler

        spec = importlib.util.spec_from_file_location(
            cp_name,
            _REPO_ROOT / "rtp_llm/models_py/modules/dsv4/cp.py",
        )
        mod = importlib.util.module_from_spec(spec)
        sys.modules[cp_name] = mod
        spec.loader.exec_module(mod)

        class _IdHandle:
            def __init__(self, t):
                self._t = t

            def wait(self):
                return self._t

        def _identity_gather(local, cp_ctx, group=None):
            return _IdHandle(local)

        mod.cp_all_gather_full_async = _identity_gather

    for p in (
        "rtp_llm.models_py.modules",
        "rtp_llm.models_py.modules.dsv4",
        "rtp_llm.models_py.modules.dsv4.fp8",
    ):
        if p not in sys.modules:
            sys.modules[p] = types.ModuleType(p)


def _import_asm():
    _stub_modules()
    name = "_dsv4_indexer_cp_asm_for_test"
    spec = importlib.util.spec_from_file_location(
        name,
        _REPO_ROOT / "rtp_llm/models_py/modules/dsv4/fp8/_indexer_cp_assembler.py",
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


A = _import_asm()


def _ctx(cp_size, cp_rank=0):
    c = types.SimpleNamespace()
    c.cp_size = cp_size
    c.cp_rank = cp_rank
    return c


def test_build_plan_lengths_and_restore():
    plan = A.build_indexer_cp_chunk_plan(
        cp_ctx=_ctx(2, 0),
        per_req_total_kv_lens=torch.tensor([8, 12], dtype=torch.int64),
        block_size=4,
        device=torch.device("cpu"),
    )
    # vb=8. T=8 → 1 vb → local 4; T=12 → 2 vb → local 8. total_local=12.
    assert torch.equal(plan.per_req_local_kv_lens, torch.tensor([4, 8]))
    assert plan.total_local_T == 12
    # chunk_T = 8+12 = 20
    assert plan.restore_indices.numel() == 20


def test_build_plan_uses_owner_block_size_for_restore_lengths():
    plan = A.build_indexer_cp_chunk_plan(
        cp_ctx=_ctx(4, 0),
        per_req_total_kv_lens=torch.tensor([16], dtype=torch.int64),
        block_size=2,
        owner_block_size=8,
        device=torch.device("cpu"),
    )
    assert plan.block_size == 2
    assert plan.owner_block_size == 8
    assert torch.equal(plan.per_req_local_kv_lens, torch.tensor([8]))
    assert plan.total_local_T == 8
    assert plan.restore_indices.numel() == 16


def test_build_local_cu_kv_seqlens():
    plan = A.build_indexer_cp_chunk_plan(
        cp_ctx=_ctx(2, 0),
        per_req_total_kv_lens=torch.tensor([8, 12, 0], dtype=torch.int64),
        block_size=4,
        device=torch.device("cpu"),
    )
    cu = A.build_local_cu_kv_seqlens(plan)
    assert cu.dtype == torch.int32
    assert torch.equal(cu, torch.tensor([0, 4, 12, 12], dtype=torch.int32))


def test_assemble_indexer_k_cp1_passthrough():
    """cp_size=1 + identity gather → restore is identity → out == local."""
    plan = A.build_indexer_cp_chunk_plan(
        cp_ctx=_ctx(1, 0),
        per_req_total_kv_lens=torch.tensor([6, 4], dtype=torch.int64),
        block_size=2,
        device=torch.device("cpu"),
    )
    # vb=2. T=6→3vb→6 local. T=4→2vb→4 local. total_local=10. chunk_T=10.
    assert plan.total_local_T == 10
    local_q = torch.arange(10 * 4, dtype=torch.uint8).reshape(10, 4)
    local_s = torch.arange(10 * 2, dtype=torch.uint8).reshape(10, 2) + 100
    out_q = torch.zeros((10, 4), dtype=torch.uint8)
    out_s = torch.zeros((10, 2), dtype=torch.uint8)
    A.assemble_indexer_k(
        plan=plan,
        local_k_quant=local_q,
        local_k_scale=local_s,
        out_k_quant=out_q,
        out_k_scale=out_s,
    )
    assert torch.equal(out_q, local_q)
    assert torch.equal(out_s, local_s)


def test_assemble_indexer_k_shape_mismatch_raises():
    plan = A.build_indexer_cp_chunk_plan(
        cp_ctx=_ctx(1, 0),
        per_req_total_kv_lens=torch.tensor([4], dtype=torch.int64),
        block_size=2,
        device=torch.device("cpu"),
    )
    bad_local = torch.zeros((3, 4), dtype=torch.uint8)  # wrong row count
    out_q = torch.zeros((4, 4), dtype=torch.uint8)
    out_s = torch.zeros((4, 2), dtype=torch.uint8)
    try:
        A.assemble_indexer_k(
            plan=plan,
            local_k_quant=bad_local,
            local_k_scale=torch.zeros((4, 2), dtype=torch.uint8),
            out_k_quant=out_q,
            out_k_scale=out_s,
        )
    except ValueError:
        return
    raise AssertionError("expected ValueError")


def test_assemble_indexer_k_zero_chunk_no_op():
    plan = A.build_indexer_cp_chunk_plan(
        cp_ctx=_ctx(2, 0),
        per_req_total_kv_lens=torch.tensor([], dtype=torch.int64),
        block_size=4,
        device=torch.device("cpu"),
    )
    A.assemble_indexer_k(
        plan=plan,
        local_k_quant=torch.zeros((0, 4), dtype=torch.uint8),
        local_k_scale=torch.zeros((0, 2), dtype=torch.uint8),
        out_k_quant=torch.zeros((0, 4), dtype=torch.uint8),
        out_k_scale=torch.zeros((0, 2), dtype=torch.uint8),
    )


def test_build_plan_rejects_bad_cp_size():
    try:
        A.build_indexer_cp_chunk_plan(
            cp_ctx=_ctx(0),
            per_req_total_kv_lens=torch.tensor([4], dtype=torch.int64),
            block_size=4,
            device=torch.device("cpu"),
        )
    except ValueError:
        return
    raise AssertionError("expected ValueError")


def test_build_plan_rejects_bad_block_size():
    try:
        A.build_indexer_cp_chunk_plan(
            cp_ctx=_ctx(2),
            per_req_total_kv_lens=torch.tensor([4], dtype=torch.int64),
            block_size=0,
            device=torch.device("cpu"),
        )
    except ValueError:
        return
    raise AssertionError("expected ValueError")


if __name__ == "__main__":
    failures = 0
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print(f"ok  {name}")
            except Exception as exc:  # noqa: BLE001
                failures += 1
                print(f"FAIL {name}: {exc!r}")
    sys.exit(1 if failures else 0)
