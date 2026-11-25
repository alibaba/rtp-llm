"""Stage 5b-1 — per-iteration KV all_gather restore-indices builder UT.

Pure CPU. The builder produces an index tensor that reorders rank-major
``all_gather`` output (each rank contributes its OWNED + padded local K)
into request-concatenated logical token order.

Tested against a hand-rolled ``ground_truth_restore`` reference that mirrors
the formula but in Python loops, so the production vectorized path is
verified end-to-end without any kernel dependency.
"""

import contextlib
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
    profiler = types.ModuleType("rtp_llm.models_py.modules.dsv4._profiler")
    profiler.record_function_range = lambda _: contextlib.nullcontext()
    sys.modules["rtp_llm.models_py.modules.dsv4._profiler"] = profiler


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


def test_single_request_known_total_avoids_tensor_length_and_repeat_helpers():
    per_req = torch.tensor([17], dtype=torch.int64)
    gathered, expected_tags = _ground_truth_restore(per_req, 2, 4, torch.device("cpu"))
    old_padded = CP.cp_padded_local_kv_lens
    old_repeat = torch.repeat_interleave
    CP.cp_padded_local_kv_lens = lambda *args, **kwargs: (_ for _ in ()).throw(
        AssertionError("tensor padded-length helper should not run")
    )
    torch.repeat_interleave = lambda *args, **kwargs: (_ for _ in ()).throw(
        AssertionError("repeat_interleave should not run")
    )
    try:
        restore = CP.build_kv_allgather_restore_indices(
            per_req,
            cp_size=2,
            block_size=4,
            device=torch.device("cpu"),
            total_kv_len=17,
            total_local_kv=12,
        )
    finally:
        CP.cp_padded_local_kv_lens = old_padded
        torch.repeat_interleave = old_repeat

    assert torch.equal(gathered[restore], expected_tags)


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


def test_scalar_length_helpers_match_tensor_formulas():
    for cp_size in (1, 2, 4):
        for block_size in (1, 2, 4, 8):
            for total in (0, 1, block_size - 1, block_size, 17, 65):
                lengths = torch.tensor([total], dtype=torch.int64)
                padded = CP.cp_padded_local_kv_lens(lengths, cp_size, block_size).item()
                assert CP.cp_padded_local_kv_len(total, cp_size, block_size) == padded
                for cp_rank in range(cp_size):
                    actual = CP.cp_actual_owned_kv_lens(
                        lengths, cp_size, block_size, cp_rank
                    ).item()
                    assert (
                        CP.cp_actual_owned_kv_len(total, cp_size, block_size, cp_rank)
                        == actual
                    )


def test_known_host_lengths_match_default_builder():
    per_req = torch.tensor([8, 12, 4], dtype=torch.int64)
    expected = CP.build_kv_allgather_restore_indices(per_req, 2, 4, torch.device("cpu"))
    actual = CP.build_kv_allgather_restore_indices(
        per_req,
        2,
        4,
        torch.device("cpu"),
        total_kv_len=24,
        total_local_kv=16,
    )
    assert torch.equal(actual, expected)


def test_rejects_negative_cp_size():
    try:
        CP.build_kv_allgather_restore_indices(
            torch.tensor([8]), cp_size=0, block_size=4, device=torch.device("cpu")
        )
    except ValueError:
        return
    raise AssertionError("expected ValueError for cp_size=0")


def test_triton_gather_cuda_parity_if_available():
    """Verify CP Triton 2D/1D gathers and complex freqs_cis against ATen on CUDA."""
    if not torch.cuda.is_available():
        return
    device = torch.device("cuda")
    torch.manual_seed(20261125)
    # 1) _cp_restore_gathered_full_2d across bf16/fp16/fp32 and unaligned widths.
    for H, dtype in (
        (512, torch.bfloat16),
        (512, torch.float16),
        (1024, torch.float32),
        (2048, torch.float32),
        (13, torch.bfloat16),  # exercises byte fallback path
    ):
        cp_size = 4
        chunk_len = 35
        seq_len_full = 123
        gathered = torch.randn(cp_size * chunk_len, H, dtype=dtype, device=device)
        restore = torch.randperm(cp_size * chunk_len, dtype=torch.long, device=device)[
            :seq_len_full
        ]
        ctx = CP.CPContext(
            cp_size=cp_size,
            cp_rank=0,
            chunk_length=chunk_len,
            padded_seq_len=cp_size * chunk_len,
            seq_len_full=seq_len_full,
            relative_positions=torch.empty(0, dtype=torch.long, device=device),
            prefix_length=0,
            global_positions=torch.empty(0, dtype=torch.long, device=device),
            local_is_real=torch.empty(0, dtype=torch.bool, device=device),
            unpad_restore=restore,
            seq_len_total=seq_len_full,
            cp_info=object(),
            unpad_restore_is_prefix=False,
        )
        ref = gathered.index_select(0, restore)
        got = CP._cp_restore_gathered_full_2d(gathered, ctx)
        assert torch.equal(got, ref), f"2D restore mismatch for H={H}, dtype={dtype}"
        out_buf = torch.empty_like(ref)
        got_out = CP._cp_restore_gathered_full_2d(gathered, ctx, out=out_buf)
        assert got_out.data_ptr() == out_buf.data_ptr()
        assert torch.equal(got_out, ref)

    # 2) cp_freqs_cis_local (complex64)
    freqs = torch.complex(
        torch.randn(128, 32, device=device), torch.randn(128, 32, device=device)
    )
    global_pos = torch.randint(0, 150, (64,), dtype=torch.int64, device=device)
    ctx_f = CP.CPContext(
        cp_size=2,
        cp_rank=0,
        chunk_length=64,
        padded_seq_len=128,
        seq_len_full=64,
        relative_positions=torch.empty(0, dtype=torch.long, device=device),
        prefix_length=0,
        global_positions=global_pos,
        local_is_real=torch.empty(0, dtype=torch.bool, device=device),
        unpad_restore=torch.arange(64, dtype=torch.long, device=device),
        seq_len_total=64,
        cp_info=object(),
        unpad_restore_is_prefix=True,
    )
    got_f = CP.cp_freqs_cis_local(freqs, ctx_f)
    ref_f = freqs.index_select(0, global_pos.clamp_max(freqs.size(0) - 1))
    assert torch.equal(got_f, ref_f), "cp_freqs_cis_local complex64 mismatch"

    # 3) _gather_1d with int32 and int64 indices
    src_1d = torch.randint(0, 1000, (17,), dtype=torch.int64, device=device)
    idx_1d = torch.randint(0, 17, (300,), dtype=torch.int32, device=device)
    assert torch.equal(CP._gather_1d(src_1d, idx_1d), src_1d.gather(0, idx_1d.long()))

    # 4) cp_gather_request_pool_blocks with strided outer dim
    raw_backing = torch.randn(20, 4, 300, dtype=torch.float32, device=device)
    strided_pool = raw_backing[:, :, :256]  # stride(0) padded vs inner [4, 256]
    bt_req = torch.tensor([5, 1, 9, 3], dtype=torch.int32, device=device)
    got_pool = CP.cp_gather_request_pool_blocks(
        strided_pool, bt_req, total_logical_blocks=4, cp_size=1, cp_rank=0
    )
    ref_pool = strided_pool.index_select(0, bt_req.long()).contiguous()
    assert torch.equal(got_pool, ref_pool), "strided pool block gather mismatch"


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
