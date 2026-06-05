"""UT: ``PrefillWorkspace`` — the per-forward prefill scratch (union buffer).

CPU-only. ``PrefillWorkspace`` is now ONE ``uint8`` union tensor time-multiplexed
between the Q projection output and the six CP gather/restore buffers
((main/idx/swa) × (gather/restore)), each owned by a concurrent CP gather role
within a single layer. These tests lock the byte-offset layout and the per-role
getter contracts the prefill path relies on: eager union allocation sized to
``max(q_bytes, 2*main + 2*idx + 2*swa)``, fit-asserts (no silent growth / no
fallback alloc), stable storage across repeated gets, the ``reserve_cp=False``
guard on the CP region, the SEPARATE main / indexer / swa byte sub-regions (so
all three gathers can be concurrently in flight within one layer — main+idx on
the compressor side stream, swa on its own side stream), and the dtype
reinterpretation that lets one byte region back both an fp32 and a bf16 gather.

NOTE: Q (``[0, q_bytes)``) INTENTIONALLY overlaps the front of the compressor
region — they never live simultaneously, so we do NOT assert Q ⊥ compressor
disjointness. ``align_bytes=1`` is passed throughout to avoid the production
1 GiB alignment forcing a 1 GiB CPU allocation.
"""

import torch

from rtp_llm.models_py.modules.dsv4.prefill_workspace import PrefillWorkspace


def _assert_raises(fn, exc_type, msg_substr: str):
    try:
        fn()
    except exc_type as exc:
        assert msg_substr in str(exc), str(exc)
        return
    raise AssertionError(f"expected {exc_type.__name__} containing {msg_substr!r}")


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


def test_prefill_q_full_capacity_and_overflow():
    ws = PrefillWorkspace(
        torch.device("cpu"), q_rows=5, q_dim=4, reserve_cp=False, align_bytes=1
    )
    assert tuple(ws.prefill_q(5).shape) == (5, 4)
    assert tuple(ws.prefill_q(0).shape) == (0, 4)

    _assert_raises(
        lambda: ws.prefill_q(6), AssertionError, "prefill_q overflow: num_tokens=6"
    )


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
    assert ws._has_swa is False

    for getter, name in (
        (ws.cp_gather_main, "cp_gather_main"),
        (ws.cp_restore_main, "cp_restore_main"),
        (ws.cp_gather_idx, "cp_gather_idx"),
        (ws.cp_restore_idx, "cp_restore_idx"),
        (ws.cp_gather_swa, "cp_gather_swa"),
        (ws.cp_restore_swa, "cp_restore_swa"),
    ):
        _assert_raises(
            lambda g=getter: g(1, 1, torch.float32),
            AssertionError,
            f"{name} region not reserved (reserve_cp=False)",
        )


def test_cp_main_idx_swa_are_separately_sized_and_distinct():
    # main sub-region: cp_rows*main_w*4 B (fp32);
    # idx  sub-region: cp_rows*idx_w *4 B (fp32);
    # swa  sub-region: cp_rows*swa_w *2 B (bf16 — swa's only dtype).
    ws = PrefillWorkspace(
        torch.device("cpu"),
        q_rows=2,
        q_dim=2,
        reserve_cp=True,
        cp_rows=4,
        main_w=6,
        idx_w=3,
        swa_w=5,
        align_bytes=1,
    )
    assert ws._main_bytes == 4 * 6 * 4
    assert ws._idx_bytes == 4 * 3 * 4
    assert ws._swa_bytes == 4 * 5 * 2

    gm = ws.cp_gather_main(4, 6, torch.float32)
    rm = ws.cp_restore_main(4, 6, torch.float32)
    gi = ws.cp_gather_idx(4, 3, torch.float32)
    ri = ws.cp_restore_idx(4, 3, torch.float32)
    gs = ws.cp_gather_swa(4, 5, torch.bfloat16)
    rs = ws.cp_restore_swa(4, 5, torch.bfloat16)
    assert tuple(gm.shape) == (4, 6) and gm.dtype == torch.float32
    assert tuple(gi.shape) == (4, 3) and gi.dtype == torch.float32
    assert tuple(gs.shape) == (4, 5) and gs.dtype == torch.bfloat16
    # All six CP role buffers occupy distinct byte offsets within the union (no
    # mutual aliasing — required because main+indexer (compressor side stream)
    # and swa (its own side stream) can all be concurrently live within one
    # layer).
    ptrs = {
        gm.data_ptr(),
        rm.data_ptr(),
        gi.data_ptr(),
        ri.data_ptr(),
        gs.data_ptr(),
        rs.data_ptr(),
    }
    assert len(ptrs) == 6
    # Repeated gets are stable views over the same storage.
    assert ws.cp_gather_main(2, 6, torch.float32).data_ptr() == gm.data_ptr()
    assert ws.cp_gather_idx(2, 3, torch.float32).data_ptr() == gi.data_ptr()
    assert ws.cp_gather_swa(2, 5, torch.bfloat16).data_ptr() == gs.data_ptr()


def test_cp_idx_region_skipped_when_idx_width_zero():
    # An HCA-only / no-indexer model has idx_w==0 → no idx region reserved,
    # while the main+swa regions are still present (swa runs on every CP layer).
    ws = PrefillWorkspace(
        torch.device("cpu"),
        q_rows=1,
        q_dim=1,
        reserve_cp=True,
        cp_rows=4,
        main_w=6,
        idx_w=0,
        swa_w=5,
        align_bytes=1,
    )
    assert ws._has_main is True
    assert ws._has_idx is False
    assert ws._has_swa is True
    _assert_raises(
        lambda: ws.cp_gather_idx(1, 1, torch.float32),
        AssertionError,
        "cp_gather_idx region not reserved (reserve_cp=False)",
    )


def test_cp_swa_region_skipped_when_swa_width_zero():
    # A no-SWA / no-CP-prefill config (or a CP-disabled forward) has swa_w==0
    # → no swa region reserved. main/idx unaffected.
    ws = PrefillWorkspace(
        torch.device("cpu"),
        q_rows=1,
        q_dim=1,
        reserve_cp=True,
        cp_rows=4,
        main_w=6,
        idx_w=3,
        swa_w=0,
        align_bytes=1,
    )
    assert ws._has_main is True
    assert ws._has_idx is True
    assert ws._has_swa is False
    _assert_raises(
        lambda: ws.cp_gather_swa(1, 1, torch.bfloat16),
        AssertionError,
        "cp_gather_swa region not reserved (reserve_cp=False)",
    )
    _assert_raises(
        lambda: ws.cp_restore_swa(1, 1, torch.bfloat16),
        AssertionError,
        "cp_restore_swa region not reserved (reserve_cp=False)",
    )


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

    # cp_region_sum dominates: 2*main + 2*idx + 2*swa
    #   = 2*(4*6*4) + 2*(4*3*4) + 2*(4*5*2) = 192 + 96 + 80 = 368;
    # q_bytes = 2*2*2 = 8; max = 368; align 256 -> round up to 512.
    ws_cp = PrefillWorkspace(
        torch.device("cpu"),
        q_rows=2,
        q_dim=2,
        reserve_cp=True,
        cp_rows=4,
        main_w=6,
        idx_w=3,
        swa_w=5,
        align_bytes=256,
    )
    assert 2 * ws_cp._main_bytes + 2 * ws_cp._idx_bytes + 2 * ws_cp._swa_bytes == 368
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
    # on (PrefillWorkspace docstring): main g|r, then idx g|r, then swa g|r.
    ws = PrefillWorkspace(
        torch.device("cpu"),
        q_rows=2,
        q_dim=2,
        reserve_cp=True,
        cp_rows=4,
        main_w=6,
        idx_w=3,
        swa_w=5,
        align_bytes=1,
    )
    assert ws._off_gather_main == 0
    assert ws._off_restore_main == ws._main_bytes
    assert ws._off_gather_idx == 2 * ws._main_bytes
    assert ws._off_restore_idx == 2 * ws._main_bytes + ws._idx_bytes
    assert ws._off_gather_swa == 2 * ws._main_bytes + 2 * ws._idx_bytes
    assert ws._off_restore_swa == (
        2 * ws._main_bytes + 2 * ws._idx_bytes + ws._swa_bytes
    )


def test_cp_restore_region_does_not_alias_gather_region():
    # #6: the WHOLE reason restore is a separate sub-region (not reusing the
    # gather buffer) is that within one layer THREE gathers can be in flight at
    # once — main+indexer on the compressor side stream, swa on its own side
    # stream — and a restore that aliased its gather (or another role's
    # buffer) would clobber an un-drained gather. Assert pairwise disjointness
    # across all six role byte ranges.
    ws = PrefillWorkspace(
        torch.device("cpu"),
        q_rows=2,
        q_dim=2,
        reserve_cp=True,
        cp_rows=4,
        main_w=6,
        idx_w=3,
        swa_w=5,
        align_bytes=1,
    )

    def _range(off, nbytes):
        return (off, off + nbytes)

    ranges = {
        "gm": _range(ws._off_gather_main, ws._main_bytes),
        "rm": _range(ws._off_restore_main, ws._main_bytes),
        "gi": _range(ws._off_gather_idx, ws._idx_bytes),
        "ri": _range(ws._off_restore_idx, ws._idx_bytes),
        "gs": _range(ws._off_gather_swa, ws._swa_bytes),
        "rs": _range(ws._off_restore_swa, ws._swa_bytes),
    }

    def _disjoint(a, b):
        return a[1] <= b[0] or b[1] <= a[0]

    names = list(ranges.keys())
    for i, na in enumerate(names):
        for nb in names[i + 1 :]:
            assert _disjoint(ranges[na], ranges[nb]), f"{na} vs {nb} overlap"


def test_cp_gather_restore_overflow():
    ws = PrefillWorkspace(
        torch.device("cpu"),
        q_rows=1,
        q_dim=1,
        reserve_cp=True,
        cp_rows=4,
        main_w=6,
        idx_w=3,
        swa_w=5,
        align_bytes=1,
    )
    # 5*6 fp32 == 120 B > 96 B reserved (main).
    _assert_raises(
        lambda: ws.cp_gather_main(5, 6, torch.float32),
        AssertionError,
        "cp_gather_main overflow",
    )
    _assert_raises(
        lambda: ws.cp_restore_main(5, 6, torch.float32),
        AssertionError,
        "cp_restore_main overflow",
    )
    # 5*3 fp32 == 60 B > 48 B reserved (idx).
    _assert_raises(
        lambda: ws.cp_gather_idx(5, 3, torch.float32),
        AssertionError,
        "cp_gather_idx overflow",
    )
    _assert_raises(
        lambda: ws.cp_restore_idx(5, 3, torch.float32),
        AssertionError,
        "cp_restore_idx overflow",
    )
    # 5*5 bf16 == 50 B > 40 B reserved (swa).
    _assert_raises(
        lambda: ws.cp_gather_swa(5, 5, torch.bfloat16),
        AssertionError,
        "cp_gather_swa overflow",
    )
    _assert_raises(
        lambda: ws.cp_restore_swa(5, 5, torch.bfloat16),
        AssertionError,
        "cp_restore_swa overflow",
    )


if __name__ == "__main__":
    test_prefill_q_eager_alloc_shape_and_dtype()
    print("PASS test_prefill_q_eager_alloc_shape_and_dtype")
    test_prefill_q_full_capacity_and_overflow()
    print("PASS test_prefill_q_full_capacity_and_overflow")
    test_prefill_q_storage_is_stable_across_gets()
    print("PASS test_prefill_q_storage_is_stable_across_gets")
    test_cp_region_not_reserved_when_reserve_cp_false()
    print("PASS test_cp_region_not_reserved_when_reserve_cp_false")
    test_cp_main_idx_swa_are_separately_sized_and_distinct()
    print("PASS test_cp_main_idx_swa_are_separately_sized_and_distinct")
    test_cp_idx_region_skipped_when_idx_width_zero()
    print("PASS test_cp_idx_region_skipped_when_idx_width_zero")
    test_cp_swa_region_skipped_when_swa_width_zero()
    print("PASS test_cp_swa_region_skipped_when_swa_width_zero")
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
    test_cp_gather_restore_overflow()
    print("PASS test_cp_gather_restore_overflow")
    print("ALL TESTS PASSED")
