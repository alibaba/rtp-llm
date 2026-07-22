"""Stage 5b-1 — per-iteration KV all_gather restore-indices builder UT.

Pure CPU. The builder produces an index tensor that reorders rank-major
``all_gather`` output (each rank contributes its OWNED + padded local K)
into request-concatenated logical token order.

Tested against a hand-rolled ``ground_truth_restore`` reference that mirrors
the formula but in Python loops, so the production vectorized path is
verified end-to-end without any kernel dependency.
"""

import importlib.util
import sys
import types
from pathlib import Path

import torch

_REPO_ROOT = Path(__file__).resolve().parents[5]


def _stub_distributed():
    mod_name = "rtp_llm.models_py.distributed.collective_torch"
    if mod_name in sys.modules:
        return
    stub = types.ModuleType(mod_name)

    class _Group:
        TP = "TP"

    def _fake_all_gather(local, group=None):
        return local

    stub.Group = _Group
    stub.all_gather = _fake_all_gather
    sys.modules[mod_name] = stub
    for p in (
        "rtp_llm",
        "rtp_llm.models_py",
        "rtp_llm.models_py.distributed",
    ):
        if p not in sys.modules:
            sys.modules[p] = types.ModuleType(p)


def _import_cp():
    _stub_distributed()
    spec = importlib.util.spec_from_file_location(
        "_dsv4_cp_for_test_5b1",
        _REPO_ROOT / "rtp_llm/models_py/modules/dsv4/cp.py",
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules["_dsv4_cp_for_test_5b1"] = mod
    spec.loader.exec_module(mod)
    return mod


CP = _import_cp()


def _ground_truth_restore(per_req_total_kv_lens, cp_size, block_size, device):
    """Reference: build the restore index by simulating the writer + gather.

    Place a unique tag at each (rank, local_pos) cell, simulate all_gather,
    and assert that ``gathered[restore]`` recovers the requests in logical
    token order.
    """
    virtual_block_size = block_size * cp_size
    local_per_req = []
    for T_r in per_req_total_kv_lens.tolist():
        n_vb = (T_r + virtual_block_size - 1) // virtual_block_size
        local_per_req.append(n_vb * block_size)
    total_local_kv = sum(local_per_req)
    # rank-major all_gather output: shape [cp_size * total_local_kv]
    # tag[r, j] encodes (r, j) so we can verify the reordering.
    gathered = torch.empty(cp_size * total_local_kv, dtype=torch.int64, device=device)
    for r in range(cp_size):
        for j in range(total_local_kv):
            gathered[r * total_local_kv + j] = r * 10_000 + j
    # Build expected per-token tag in logical order: token t in request r
    # was written by rank (block_idx % cp_size) at position
    # (req_local_offset[r] + local_block_idx * block_size + tib).
    req_local_offset = [0]
    for L_r in local_per_req:
        req_local_offset.append(req_local_offset[-1] + L_r)
    expected = []
    for ridx, T_r in enumerate(per_req_total_kv_lens.tolist()):
        for t in range(T_r):
            block_idx = t // block_size
            tib = t % block_size
            owner = block_idx % cp_size
            local_block_idx = block_idx // cp_size
            local_pos = local_block_idx * block_size + tib
            expected.append(owner * 10_000 + (req_local_offset[ridx] + local_pos))
    return gathered, torch.tensor(expected, dtype=torch.int64, device=device)


def _check(per_req, cp_size, block_size):
    device = torch.device("cpu")
    per_req_t = torch.tensor(per_req, dtype=torch.int64, device=device)
    restore = CP.build_kv_allgather_restore_indices(
        per_req_t, cp_size, block_size, device
    )
    gathered, expected_tags = _ground_truth_restore(
        per_req_t, cp_size, block_size, device
    )
    actual_tags = gathered[restore]
    assert torch.equal(actual_tags, expected_tags), (
        f"per_req={per_req} cp={cp_size} block={block_size}\n"
        f"expected={expected_tags.tolist()}\nactual  ={actual_tags.tolist()}"
    )


def test_single_request_evenly_split():
    # T=16, cp=2, block=4 → 2 virtual blocks of 8 tokens each.
    _check(per_req=[16], cp_size=2, block_size=4)


def test_single_request_uneven_tail():
    # T=17, cp=2, block=4 → owner of last token is rank 0 (block 4).
    _check(per_req=[17], cp_size=2, block_size=4)


def test_single_request_partial_last_virtual_block():
    # T=20, cp=4, block=4 → 2 virtual blocks (32 tokens padded), real 20.
    _check(per_req=[20], cp_size=4, block_size=4)


def test_multi_request_mixed():
    _check(per_req=[8, 12, 4], cp_size=2, block_size=4)


def test_multi_request_cp4():
    _check(per_req=[16, 8, 32], cp_size=4, block_size=4)


def test_cp1_passthrough():
    # cp=1 should give identity restore (gathered == logical order).
    _check(per_req=[16, 4, 8], cp_size=1, block_size=4)


def test_empty_request_list():
    device = torch.device("cpu")
    restore = CP.build_kv_allgather_restore_indices(
        torch.tensor([], dtype=torch.int64), cp_size=4, block_size=4, device=device
    )
    assert restore.numel() == 0


def test_zero_kv_request_in_batch():
    """A request with T_r=0 should contribute zero entries; others unaffected."""
    _check(per_req=[8, 0, 4], cp_size=2, block_size=4)


def test_block_size_one():
    """block_size=1 degenerate case: every token is its own block, RR by token."""
    _check(per_req=[6], cp_size=2, block_size=1)


def test_rejects_negative_cp_size():
    try:
        CP.build_kv_allgather_restore_indices(
            torch.tensor([8]), cp_size=0, block_size=4, device=torch.device("cpu")
        )
    except ValueError:
        return
    raise AssertionError("expected ValueError for cp_size=0")


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
