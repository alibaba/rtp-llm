"""CPU UTs for CP-sharded compressed-attention planning helpers."""

import importlib.util
import sys
import types
from contextlib import nullcontext
from pathlib import Path

import torch

_REPO_ROOT = Path(__file__).resolve().parents[5]


def _stub_and_import():
    ct_name = "rtp_llm.models_py.distributed.collective_torch"
    if ct_name not in sys.modules:
        ct = types.ModuleType(ct_name)

        class _Group:
            TP = "TP"

        ct.Group = _Group
        ct.all_gather = lambda local, group=None: local
        sys.modules[ct_name] = ct
        for p in (
            "rtp_llm",
            "rtp_llm.models_py",
            "rtp_llm.models_py.distributed",
            "rtp_llm.models_py.modules",
            "rtp_llm.models_py.modules.dsv4",
            "rtp_llm.models_py.modules.dsv4.fp8",
        ):
            if p not in sys.modules:
                sys.modules[p] = types.ModuleType(p)
        profiler_name = "rtp_llm.models_py.modules.dsv4._profiler"
        if profiler_name not in sys.modules:
            profiler = types.ModuleType(profiler_name)
            profiler.record_function_range = lambda *args, **kwargs: nullcontext()
            sys.modules[profiler_name] = profiler

    cp_name = "rtp_llm.models_py.modules.dsv4.cp"
    if cp_name not in sys.modules:
        cp_spec = importlib.util.spec_from_file_location(
            cp_name, _REPO_ROOT / "rtp_llm/models_py/modules/dsv4/cp.py"
        )
        cp_mod = importlib.util.module_from_spec(cp_spec)
        sys.modules[cp_name] = cp_mod
        cp_spec.loader.exec_module(cp_mod)

    name = "rtp_llm.models_py.modules.dsv4.fp8._cp_attention_shard"
    spec = importlib.util.spec_from_file_location(
        name,
        _REPO_ROOT / "rtp_llm/models_py/modules/dsv4/fp8/_cp_attention_shard.py",
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


SHARD = _stub_and_import()


def test_csa_prefers_merge_only_for_long_prefix() -> None:
    # cp_size=2 break-even (output/lse NOT divided by C in formula):
    #   P/T >= 1363. Pick test points clearly straddling that line.
    short = SHARD.prefer_raw_q_merge_attention(
        prefix_len=1000 * 128,
        input_len=128,
        cp_size=2,
        compress_ratio=4,
        include_topk_gather=True,
    )
    long = SHARD.prefer_raw_q_merge_attention(
        prefix_len=1500 * 128,
        input_len=128,
        cp_size=2,
        compress_ratio=4,
        include_topk_gather=True,
    )
    assert not short
    assert long


def test_hca_threshold_is_much_higher_than_csa() -> None:
    # cp_size=2 break-evens: CSA P/T >= 1363, HCA P/T >= ~43k. Pick
    # P/T = 2500: CSA prefers raw_q, HCA doesn't.
    csa = SHARD.cp_attention_comm_bytes(
        prefix_len=2500 * 128,
        input_len=128,
        cp_size=2,
        compress_ratio=4,
        include_topk_gather=True,
    )
    hca = SHARD.cp_attention_comm_bytes(
        prefix_len=2500 * 128,
        input_len=128,
        cp_size=2,
        compress_ratio=128,
        include_topk_gather=False,
    )
    assert csa.raw_q_merge_total < csa.packed_kv_gather
    assert hca.raw_q_merge_total > hca.packed_kv_gather


def test_conservative_thresholds_match_plan() -> None:
    # CSA threshold tightened to P/T >= 1500 after the output/lse formula
    # correction. HCA threshold tightened to P/T >= 48000.
    assert SHARD.prefer_raw_q_merge_attention_conservative(
        prefix_len=1500 * 7,
        input_len=7,
        compress_ratio=4,
        include_topk_gather=True,
    )
    assert not SHARD.prefer_raw_q_merge_attention_conservative(
        prefix_len=1499 * 7,
        input_len=7,
        compress_ratio=4,
        include_topk_gather=True,
    )
    assert SHARD.prefer_raw_q_merge_attention_conservative(
        prefix_len=48000 * 3,
        input_len=3,
        compress_ratio=128,
        include_topk_gather=False,
    )
    assert not SHARD.prefer_raw_q_merge_attention_conservative(
        prefix_len=47999 * 3,
        input_len=3,
        compress_ratio=128,
        include_topk_gather=False,
    )


def test_comm_bytes_formula_pinned() -> None:
    """Pin the exact per-rank gather byte values for both paths.

    Regressions in the byte formula (e.g. accidentally dividing
    ``output_gather`` / ``lse_gather`` by ``cp_size``) flip the
    ``raw_q_merge_total`` semantics and silently move the auto-selected
    runtime path. Pin the exact values for one representative config so any
    formula tweak forces an intentional update here.

    Config: cp_size=2, DSV4-Flash/Pro dims (H=64, head_dim=512, BF16),
    topk=512, packed_kv_slot_bytes=584. With ``alpha = cp_size-1 = 1``:

      raw_q     = alpha * T * H * D * elem / C    # Q sharded T/C
      topk      = alpha * T * topk * 4   / C      # topk sharded T/C
      output    = alpha * T * H * D * elem        # partial O for full T
      lse       = alpha * T * H * 4               # partial LSE for full T
      packed_kv = alpha * (P+T)/r * slot / C

    With T=64, P=128, r=4 (CSA):
      raw_q     = 1 * 64 * 64 * 512 * 2 / 2 = 2_097_152
      topk      = 1 * 64 * 512 * 4 / 2      = 65_536
      output    = 1 * 64 * 64 * 512 * 2     = 4_194_304
      lse       = 1 * 64 * 64 * 4           = 16_384
      raw_total = 6_373_376
      packed    = 1 * (128+64)/4 * 584 / 2  = 14_016
    """
    est = SHARD.cp_attention_comm_bytes(
        prefix_len=128,
        input_len=64,
        cp_size=2,
        compress_ratio=4,
        include_topk_gather=True,
    )
    assert est.raw_q_gather == 2_097_152
    assert est.topk_gather == 65_536
    assert est.output_gather == 4_194_304
    assert est.lse_gather == 16_384
    assert est.raw_q_merge_total == 6_373_376
    assert est.packed_kv_gather == 14_016


def test_remap_topk_b1_cp2_block4() -> None:
    topk = torch.tensor(
        [
            [0, 1, 4, 5, 8, -1],
            [2, 6, 7, 10, 11, 12],
        ],
        dtype=torch.int32,
    )
    per_req = torch.tensor([13], dtype=torch.int64)
    rank0 = SHARD.remap_topk_to_cp_local(
        topk,
        per_req_total_kv_lens=per_req,
        cp_size=2,
        cp_rank=0,
        block_size=4,
    )
    rank1 = SHARD.remap_topk_to_cp_local(
        topk,
        per_req_total_kv_lens=per_req,
        cp_size=2,
        cp_rank=1,
        block_size=4,
    )
    assert torch.equal(
        rank0,
        torch.tensor(
            [
                [0, 1, -1, -1, 4, -1],
                [2, -1, -1, 6, 7, -1],
            ],
            dtype=torch.int32,
        ),
    )
    assert torch.equal(
        rank1,
        torch.tensor(
            [
                [-1, -1, 0, 1, -1, -1],
                [-1, 2, 3, -1, -1, 4],
            ],
            dtype=torch.int32,
        ),
    )


def test_remap_topk_batched_uses_per_request_local_offsets() -> None:
    topk = torch.tensor(
        [
            [0, 3, 4],
            [1, 5, -1],
            [0, 4, 7],
        ],
        dtype=torch.int32,
    )
    req_ids = torch.tensor([0, 0, 1], dtype=torch.int64)
    per_req = torch.tensor([6, 9], dtype=torch.int64)
    rank0 = SHARD.remap_topk_to_cp_local(
        topk,
        per_req_total_kv_lens=per_req,
        cp_size=2,
        cp_rank=0,
        block_size=4,
        req_id_per_token=req_ids,
    )
    rank1 = SHARD.remap_topk_to_cp_local(
        topk,
        per_req_total_kv_lens=per_req,
        cp_size=2,
        cp_rank=1,
        block_size=4,
        req_id_per_token=req_ids,
    )
    # local_lens: req0=4, req1=8, so req1 rank-local base is 4.
    assert torch.equal(
        rank0,
        torch.tensor([[0, 3, -1], [1, -1, -1], [4, -1, -1]], dtype=torch.int32),
    )
    assert torch.equal(
        rank1,
        torch.tensor([[-1, -1, 0], [-1, 1, -1], [-1, 4, 7]], dtype=torch.int32),
    )


def test_build_swa_cp_local_indices_b1_partitions_visible_window() -> None:
    gp = torch.tensor([4, 5], dtype=torch.int64)
    prefix = torch.tensor([4], dtype=torch.int64)
    rank0_idx, rank0_lens = SHARD.build_swa_cp_local_indices(
        gp,
        prefix_lengths=prefix,
        cp_size=2,
        cp_rank=0,
        window_size=4,
        M=20,
        N=10,
    )
    rank1_idx, rank1_lens = SHARD.build_swa_cp_local_indices(
        gp,
        prefix_lengths=prefix,
        cp_size=2,
        cp_rank=1,
        window_size=4,
        M=20,
        N=10,
    )
    # query pos 4 sees keys [1,2,3,4], gather_start=1.
    assert torch.equal(rank0_idx[0, :2], torch.tensor([11, 13], dtype=torch.int32))
    assert torch.equal(rank1_idx[0, :2], torch.tensor([10, 12], dtype=torch.int32))
    # query pos 5 sees keys [2,3,4,5], gather_start=1.
    assert torch.equal(rank0_idx[1, :2], torch.tensor([11, 13], dtype=torch.int32))
    assert torch.equal(rank1_idx[1, :2], torch.tensor([12, 14], dtype=torch.int32))
    assert torch.equal(rank0_lens, torch.tensor([2, 2], dtype=torch.int32))
    assert torch.equal(rank1_lens, torch.tensor([2, 2], dtype=torch.int32))


def test_build_swa_cp_local_indices_batched_varlen_offsets() -> None:
    gp = torch.tensor([5, 6, 2], dtype=torch.int64)
    req_ids = torch.tensor([0, 0, 1], dtype=torch.int64)
    prefix = torch.tensor([5, 2], dtype=torch.int64)
    rank0_idx, rank0_lens = SHARD.build_swa_cp_local_indices(
        gp,
        prefix_lengths=prefix,
        cp_size=2,
        cp_rank=0,
        window_size=4,
        M=20,
        N=7,
        req_id_per_token=req_ids,
    )
    rank1_idx, rank1_lens = SHARD.build_swa_cp_local_indices(
        gp,
        prefix_lengths=prefix,
        cp_size=2,
        cp_rank=1,
        window_size=4,
        M=20,
        N=7,
        req_id_per_token=req_ids,
    )
    # req0: prefix=5, gather_start=2. pos5 sees [2,3,4,5].
    assert torch.equal(rank0_idx[0, :2], torch.tensor([7, 9], dtype=torch.int32))
    assert torch.equal(rank1_idx[0, :2], torch.tensor([8, 10], dtype=torch.int32))
    # req1: prefix=2, gather_start=0, req base=20. pos2 sees [0,1,2].
    assert torch.equal(rank0_idx[2, :2], torch.tensor([27, 29], dtype=torch.int32))
    assert torch.equal(rank1_idx[2, :1], torch.tensor([28], dtype=torch.int32))
    assert torch.equal(rank0_lens, torch.tensor([2, 2, 2], dtype=torch.int32))
    assert torch.equal(rank1_lens, torch.tensor([2, 2, 1], dtype=torch.int32))


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
