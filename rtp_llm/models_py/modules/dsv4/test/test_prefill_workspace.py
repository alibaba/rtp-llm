"""UT: ``PrefillWorkspace`` — the per-forward prefill scratch (union buffer).

CPU-only. ``PrefillWorkspace`` is now ONE ``uint8`` union tensor time-multiplexed
between the Q projection output and the compressor CP gather/restore buffers
((main/idx) × (gather/restore)), each owned by a concurrent CP gather role within
a single layer. These tests lock the byte-offset layout and the per-role getter
contracts the prefill path relies on: eager union allocation sized to
``max(q_bytes, 2*main + 2*idx)``, stable storage across repeated gets, the
``reserve_cp=False`` metadata state, the SEPARATE main / indexer byte
sub-regions, and the dtype
reinterpretation that lets one byte region back both an fp32 and a bf16 gather.

NOTE: Q (``[0, q_bytes)``) INTENTIONALLY overlaps the front of the compressor
region — they never live simultaneously, so we do NOT assert Q ⊥ compressor
disjointness. ``align_bytes=1`` is passed throughout to avoid the production
1 GiB alignment forcing a 1 GiB CPU allocation.
"""

import importlib.util
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch

# This module only needs Torch; keep the CPU tests independent of compiled ops.
_SPEC = importlib.util.spec_from_file_location(
    "_prefill_workspace_test",
    Path(__file__).resolve().parents[1] / "prefill_workspace.py",
)
_WORKSPACE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_WORKSPACE)
PrefillWorkspace = _WORKSPACE.PrefillWorkspace


def _expect_runtime_error(fn):
    try:
        fn()
    except RuntimeError:
        return
    raise AssertionError("expected tensor view construction to reject the request")


def test_prefill_q_eager_alloc_shape_and_dtype():
    ws = PrefillWorkspace(
        torch.device("cpu"), q_rows=5, q_dim=4, reserve_cp=False, align_bytes=1
    )
    # The union is allocated eagerly in __init__ as a uint8 base. With no CP
    # region reserved, its size is just q_bytes (bf16 == 2 bytes).
    assert ws._union.dtype == torch.uint8
    assert ws._union.numel() == 5 * 4 * 2

    q = ws.prefill_q(3)
    assert tuple(q.shape) == (3, 4)
    assert q.dtype == torch.bfloat16


def test_prefill_q_capacity_boundaries():
    ws = PrefillWorkspace(
        torch.device("cpu"), q_rows=5, q_dim=4, reserve_cp=False, align_bytes=1
    )
    assert tuple(ws.prefill_q(5).shape) == (5, 4)
    assert tuple(ws.prefill_q(0).shape) == (0, 4)


def test_prefill_q_storage_is_stable_across_gets():
    ws = PrefillWorkspace(
        torch.device("cpu"), q_rows=5, q_dim=4, reserve_cp=False, align_bytes=1
    )
    # Repeated gets are views over the same backing storage (no realloc). Q sits
    # at offset 0 of the union, so it shares the union's base pointer.
    assert ws.prefill_q(3).data_ptr() == ws.prefill_q(2).data_ptr()
    assert ws.prefill_q(3).data_ptr() == ws._union.data_ptr()


def test_cp_region_not_reserved_when_reserve_cp_false():
    ws = PrefillWorkspace(
        torch.device("cpu"), q_rows=1, q_dim=1, reserve_cp=False, align_bytes=1
    )
    assert ws._has_main is False
    assert ws._has_idx is False
    for getter in (
        ws.cp_gather_main,
        ws.cp_restore_main,
        ws.cp_gather_idx,
        ws.cp_restore_idx,
    ):
        _expect_runtime_error(lambda g=getter: g(1, 1, torch.float32))


def test_cp_main_idx_are_separately_sized_and_distinct():
    # main sub-region: cp_rows*main_w*4 B (fp32);
    # idx  sub-region: cp_rows*idx_w *4 B (fp32).
    ws = PrefillWorkspace(
        torch.device("cpu"),
        q_rows=2,
        q_dim=2,
        reserve_cp=True,
        cp_rows=4,
        main_w=6,
        idx_w=3,
        align_bytes=1,
    )
    assert ws._main_bytes == 4 * 6 * 4
    assert ws._idx_bytes == 4 * 3 * 4

    gm = ws.cp_gather_main(4, 6, torch.float32)
    rm = ws.cp_restore_main(4, 6, torch.float32)
    gi = ws.cp_gather_idx(4, 3, torch.float32)
    ri = ws.cp_restore_idx(4, 3, torch.float32)
    assert tuple(gm.shape) == (4, 6) and gm.dtype == torch.float32
    assert tuple(gi.shape) == (4, 3) and gi.dtype == torch.float32
    # All four compressor role buffers occupy distinct byte offsets within the
    # union (no mutual aliasing).
    ptrs = {
        gm.data_ptr(),
        rm.data_ptr(),
        gi.data_ptr(),
        ri.data_ptr(),
    }
    assert len(ptrs) == 4
    # Repeated gets are stable views over the same storage.
    assert ws.cp_gather_main(2, 6, torch.float32).data_ptr() == gm.data_ptr()
    assert ws.cp_gather_idx(2, 3, torch.float32).data_ptr() == gi.data_ptr()


def test_cp_idx_region_skipped_when_idx_width_zero():
    # An HCA-only / no-indexer model has idx_w==0 → no idx region reserved,
    # while the main region is still present.
    ws = PrefillWorkspace(
        torch.device("cpu"),
        q_rows=1,
        q_dim=1,
        reserve_cp=True,
        cp_rows=4,
        main_w=6,
        idx_w=0,
        align_bytes=1,
    )
    assert ws._has_main is True
    assert ws._has_idx is False
    _expect_runtime_error(lambda: ws.cp_gather_idx(1, 1, torch.float32))


def test_cp_buffer_reinterprets_dtype_from_same_base():
    # The same byte region must serve both the fp32 compressor gather and a
    # bf16 gather. 4*6 fp32 == 4*12 bf16 == 96 bytes. cp_gather_main is at
    # offset 0 of the union, so it shares the union's base pointer.
    ws = PrefillWorkspace(
        torch.device("cpu"),
        q_rows=1,
        q_dim=1,
        reserve_cp=True,
        cp_rows=4,
        main_w=6,
        idx_w=3,
        align_bytes=1,
    )
    g_fp32 = ws.cp_gather_main(4, 6, torch.float32)
    g_bf16 = ws.cp_gather_main(4, 12, torch.bfloat16)
    assert g_fp32.dtype == torch.float32 and tuple(g_fp32.shape) == (4, 6)
    assert g_bf16.dtype == torch.bfloat16 and tuple(g_bf16.shape) == (4, 12)
    assert g_fp32.data_ptr() == g_bf16.data_ptr() == ws._union.data_ptr()


def test_union_rounds_up_to_align_bytes():
    # #5: the union byte size is round_up(max(q_bytes, compressor_sum),
    # align_bytes). The 1 GiB production default is DELIBERATE (clean allocator
    # reuse across forwards); here we lock the rounding math with a small align
    # so no 1 GiB host alloc is needed.
    #
    # q-only, q_bytes = 5*4*2 = 40; align 64 -> round up to 64.
    ws = PrefillWorkspace(
        torch.device("cpu"), q_rows=5, q_dim=4, reserve_cp=False, align_bytes=64
    )
    assert ws._union.numel() == 64

    # Already-aligned size is left unchanged: q_bytes = 8*4*2 = 64, align 64.
    ws_exact = PrefillWorkspace(
        torch.device("cpu"), q_rows=8, q_dim=4, reserve_cp=False, align_bytes=64
    )
    assert ws_exact._union.numel() == 64

    # cp_region_sum dominates: 2*main + 2*idx
    #   = 2*(4*6*4) + 2*(4*3*4) = 192 + 96 = 288;
    # q_bytes = 2*2*2 = 8; max = 288; align 256 -> round up to 512.
    ws_cp = PrefillWorkspace(
        torch.device("cpu"),
        q_rows=2,
        q_dim=2,
        reserve_cp=True,
        cp_rows=4,
        main_w=6,
        idx_w=3,
        align_bytes=256,
    )
    assert 2 * ws_cp._main_bytes + 2 * ws_cp._idx_bytes == 288
    assert ws_cp._union.numel() == 512


def test_default_align_bytes_is_one_gib():
    # #5: pin the production default so a regression to a different alignment is
    # caught here (the value, not the 1 GiB allocation, is what we assert — we
    # never construct with the default on CPU to avoid the 1 GiB host alloc).
    import inspect

    sig = inspect.signature(PrefillWorkspace.__init__)
    assert sig.parameters["align_bytes"].default == (1 << 30)


def test_cp_role_byte_offsets_match_documented_layout():
    # #6: lock the exact byte-offset layout the CP gather/restore path depends
    # on (PrefillWorkspace docstring): main g|r, then idx g|r.
    ws = PrefillWorkspace(
        torch.device("cpu"),
        q_rows=2,
        q_dim=2,
        reserve_cp=True,
        cp_rows=4,
        main_w=6,
        idx_w=3,
        align_bytes=1,
    )
    assert ws._off_gather_main == 0
    assert ws._off_restore_main == ws._main_bytes
    assert ws._off_gather_idx == 2 * ws._main_bytes
    assert ws._off_restore_idx == 2 * ws._main_bytes + ws._idx_bytes


def test_cp_restore_region_does_not_alias_gather_region():
    # #6: the WHOLE reason restore is a separate sub-region (not reusing the
    # gather buffer) is that within one CSA layer both main+indexer gathers can
    # be in flight. A restore that aliased its gather (or another role's buffer)
    # would clobber an un-drained gather. Assert pairwise disjointness across all
    # four compressor role byte ranges.
    ws = PrefillWorkspace(
        torch.device("cpu"),
        q_rows=2,
        q_dim=2,
        reserve_cp=True,
        cp_rows=4,
        main_w=6,
        idx_w=3,
        align_bytes=1,
    )

    def _range(off, nbytes):
        return (off, off + nbytes)

    ranges = {
        "gm": _range(ws._off_gather_main, ws._main_bytes),
        "rm": _range(ws._off_restore_main, ws._main_bytes),
        "gi": _range(ws._off_gather_idx, ws._idx_bytes),
        "ri": _range(ws._off_restore_idx, ws._idx_bytes),
    }

    def _disjoint(a, b):
        return a[1] <= b[0] or b[1] <= a[0]

    names = list(ranges.keys())
    for i, na in enumerate(names):
        for nb in names[i + 1 :]:
            assert _disjoint(ranges[na], ranges[nb]), f"{na} vs {nb} overlap"


def test_cp_views_cannot_cross_role_boundaries():
    ws = PrefillWorkspace(
        torch.device("cpu"),
        q_rows=1,
        q_dim=1,
        reserve_cp=True,
        cp_rows=4,
        main_w=6,
        idx_w=3,
        align_bytes=1,
    )
    _expect_runtime_error(lambda: ws.cp_gather_main(5, 6, torch.float32))
    _expect_runtime_error(lambda: ws.cp_restore_main(5, 6, torch.float32))
    _expect_runtime_error(lambda: ws.cp_gather_idx(5, 3, torch.float32))
    _expect_runtime_error(lambda: ws.cp_restore_idx(5, 3, torch.float32))


def test_live_padded_batch_gather_restore_and_q_aliases():
    for cp_size in (2, 4, 8):
        for lengths in ([1], [7], [8], [9], [1, 9, 6, 17]):
            local_rows = sum(
                ((length + 2 * cp_size - 1) // (2 * cp_size)) * 2 for length in lengths
            )
            padded_rows = local_rows * cp_size
            real_rows = sum(lengths)
            ws = PrefillWorkspace(
                torch.device("cpu"),
                q_rows=32,
                q_dim=16,
                reserve_cp=True,
                cp_rows=128,
                main_w=8,
                idx_w=4,
                align_bytes=1,
            )
            # V4 keeps fixed capacity; its async consumers request live rows:
            # start() gathers padded rows; wait() restores only real new tokens.
            for dtype in (torch.float32, torch.bfloat16):
                views = [
                    ws.cp_gather_main(padded_rows, 8, dtype),
                    ws.cp_restore_main(real_rows, 8, dtype),
                    ws.cp_gather_idx(padded_rows, 4, dtype),
                    ws.cp_restore_idx(real_rows, 4, dtype),
                ]
                for sentinel, view in enumerate(views, 1):
                    view.fill_(sentinel)
                for sentinel, view in enumerate(views, 1):
                    assert torch.equal(view, torch.full_like(view, sentinel))
                for gather, restore in ((views[0], views[1]), (views[2], views[3])):
                    indices = torch.arange(real_rows - 1, -1, -1)
                    torch.index_select(gather, 0, indices, out=restore)
                    assert torch.equal(restore, gather.index_select(0, indices))
            q = ws.prefill_q(local_rows)
            assert q.shape == (local_rows, 16)
            assert (
                q.data_ptr()
                == ws.cp_gather_main(padded_rows, 8, torch.float32).data_ptr()
            )
            q.fill_(9)
            assert ws.cp_gather_main(padded_rows, 16, torch.bfloat16)[0, 0] == 9


def test_production_bucket_boundaries_on_meta_device():
    gib = 1 << 30
    # Meta tensors exercise the real byte layout without allocating GiB of RAM.
    for rows, expected in ((0, 0), (1, gib), (gib // 2, gib), (gib // 2 + 1, 2 * gib)):
        ws = PrefillWorkspace(
            torch.device("meta"), q_rows=rows, q_dim=1, reserve_cp=False
        )
        assert ws._union.numel() == expected
        assert ws.prefill_q(rows).shape == (rows, 1)
    for local_rows, expected_gib in (
        (0, 0),
        (26770, 2),
        (40588, 3),
        (28800, 2),
        (50468, 4),
        (262144, 16),
    ):
        ws = PrefillWorkspace(
            torch.device("meta"),
            q_rows=local_rows,
            q_dim=64 * 512,
            reserve_cp=False,
            cp_rows=1048576,
            main_w=2048,
            idx_w=512,
        )
        assert ws._union.numel() == expected_gib * gib
        assert ws.prefill_q(local_rows).shape == (local_rows, 64 * 512)
        assert ws._main_bytes == ws._idx_bytes == 0


def test_v41_forward_uses_64_mib_buckets_and_v4_keeps_default():
    from rtp_llm.models_py.modules.dsv4 import chunk_env
    from rtp_llm.models_py.modules.dsv4.prefill import forward

    class AllocationCaptured(Exception):
        pass

    mib = 1 << 20
    for v41 in (False, True):
        for rows, v41_mib in ((32768, 2048), (32770, 2112), (32856, 2112)):
            v4 = SimpleNamespace(
                fp8_kv_cache=True,
                layers=(),
                args=SimpleNamespace(v41_config={} if v41 else None),
                _prefill_ws_q_rows=rows,
                _prefill_ws_q_dim=64 * 512,
                _prefill_ws_full_rows=0,
                _prefill_ws_main_w=0,
                _prefill_ws_idx_w=0,
            )
            allocated = []

            def allocate(*args, **kwargs):
                allocated.append((kwargs, PrefillWorkspace(*args, **kwargs)))
                raise AllocationCaptured

            with (
                patch.object(chunk_env, "FLASH_MLA_SPARSE_Q_CHUNK", 33280),
                patch.object(forward, "PrefillWorkspace", side_effect=allocate),
            ):
                try:
                    forward.forward_layers(
                        v4, None, torch.empty(rows, device="meta"), None, None, None
                    )
                except AllocationCaptured:
                    pass
                else:
                    raise AssertionError("forward did not allocate a workspace")
            kwargs, ws = allocated[0]
            assert kwargs["align_bytes"] == (64 * mib if v41 else 1024 * mib)
            expected_mib = v41_mib if v41 else (2048 if rows == 32768 else 3072)
            assert ws._union.numel() == expected_mib * mib
            assert ws._q_rows == rows
            assert ws.prefill_q(rows).shape == (rows, 64 * 512)
            assert ws._main_bytes == ws._idx_bytes == 0


if __name__ == "__main__":
    test_prefill_q_eager_alloc_shape_and_dtype()
    print("PASS test_prefill_q_eager_alloc_shape_and_dtype")
    test_prefill_q_capacity_boundaries()
    print("PASS test_prefill_q_capacity_boundaries")
    test_prefill_q_storage_is_stable_across_gets()
    print("PASS test_prefill_q_storage_is_stable_across_gets")
    test_cp_region_not_reserved_when_reserve_cp_false()
    print("PASS test_cp_region_not_reserved_when_reserve_cp_false")
    test_cp_main_idx_are_separately_sized_and_distinct()
    print("PASS test_cp_main_idx_are_separately_sized_and_distinct")
    test_cp_idx_region_skipped_when_idx_width_zero()
    print("PASS test_cp_idx_region_skipped_when_idx_width_zero")
    test_cp_buffer_reinterprets_dtype_from_same_base()
    print("PASS test_cp_buffer_reinterprets_dtype_from_same_base")
    test_union_rounds_up_to_align_bytes()
    print("PASS test_union_rounds_up_to_align_bytes")
    test_default_align_bytes_is_one_gib()
    print("PASS test_default_align_bytes_is_one_gib")
    test_cp_role_byte_offsets_match_documented_layout()
    print("PASS test_cp_role_byte_offsets_match_documented_layout")
    test_cp_restore_region_does_not_alias_gather_region()
    print("PASS test_cp_restore_region_does_not_alias_gather_region")
    test_cp_views_cannot_cross_role_boundaries()
    print("PASS test_cp_views_cannot_cross_role_boundaries")
    test_live_padded_batch_gather_restore_and_q_aliases()
    print("PASS test_live_padded_batch_gather_restore_and_q_aliases")
    test_production_bucket_boundaries_on_meta_device()
    print("PASS test_production_bucket_boundaries_on_meta_device")
    test_v41_forward_uses_64_mib_buckets_and_v4_keeps_default()
    print("PASS test_v41_forward_uses_64_mib_buckets_and_v4_keeps_default")
    print("ALL TESTS PASSED")
