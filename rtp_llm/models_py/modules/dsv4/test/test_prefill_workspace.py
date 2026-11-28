"""UT: ``PrefillWorkspace`` — the per-forward prefill scratch (union buffer).

CPU-only. ``PrefillWorkspace`` is now ONE ``uint8`` union tensor time-multiplexed
between the Q projection output and the compressor CP gather/restore buffers
((main/idx) × (gather/restore)), each owned by a concurrent CP gather role within
a single layer. These tests lock the byte-offset layout and the per-role getter
contracts the prefill path relies on: eager union allocation sized to
``max(q_bytes, 2*main + 2*idx)``, fit-asserts (no silent growth / no fallback
alloc), stable storage across repeated gets, the ``reserve_cp=False`` guard on
the CP region, the SEPARATE main / indexer byte sub-regions, and the dtype
reinterpretation that lets one byte region back both an fp32 and a bf16 gather.

NOTE: Q (``[0, q_bytes)``) INTENTIONALLY overlaps the front of the compressor
region — they never live simultaneously, so we do NOT assert Q ⊥ compressor
disjointness. ``align_bytes=1`` is passed throughout to avoid the production
1 GiB alignment forcing a 1 GiB CPU allocation.
"""

import torch

from rtp_llm.models_py.distributed import collective_torch
from rtp_llm.models_py.distributed.collective_torch import Group
from rtp_llm.models_py.modules.dsv4 import cp as _cp_mod
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

    for getter, name in (
        (ws.cp_gather_main, "cp_gather_main"),
        (ws.cp_restore_main, "cp_restore_main"),
        (ws.cp_gather_idx, "cp_gather_idx"),
        (ws.cp_restore_idx, "cp_restore_idx"),
    ):
        _assert_raises(
            lambda g=getter: g(1, 1, torch.float32),
            AssertionError,
            f"{name} region not reserved (reserve_cp=False)",
        )


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
    _assert_raises(
        lambda: ws.cp_gather_idx(1, 1, torch.float32),
        AssertionError,
        "cp_gather_idx region not reserved (reserve_cp=False)",
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


def test_cp_gather_restore_overflow():
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


def test_scratch_respects_enable_flag_entry_limit_budget_and_growth():
    ws = PrefillWorkspace(
        torch.device("cpu"), q_rows=1, q_dim=1, reserve_cp=False, align_bytes=1
    )
    # 1. enable_scratch_cache=False disables caching and returns ephemeral storage.
    ws.enable_scratch_cache = False
    e1 = ws.scratch("disabled", (8,), torch.int32, zero=True)
    e2 = ws.scratch("disabled", (8,), torch.int32)
    assert e1.data_ptr() != e2.data_ptr()
    assert int(e1.sum().item()) == 0
    assert not ws.owns_tensor(e1)
    assert ws._scratch == {}
    assert ws._scratch_total_bytes == 0

    # 2. Entry larger than scratch_max_entry_bytes is uncached.
    ws.enable_scratch_cache = True
    ws.scratch_max_entry_bytes = 64
    ws.scratch_budget_bytes = 128
    over_entry = ws.scratch("big", (20,), torch.float32)  # 80 B > 64 B
    assert not ws.owns_tensor(over_entry)
    assert "big" not in ws._scratch
    assert ws._scratch_total_bytes == 0

    # 3. Cached allocation grows and updates total without double counting old size.
    k1 = ws.scratch("k", (8,), torch.float32)  # 32 B
    assert ws.owns_tensor(k1)
    assert ws._scratch_total_bytes == 32
    k1_same = ws.scratch("k", (4,), torch.float32)  # 16 B fits in 32 B
    assert k1_same.data_ptr() == k1.data_ptr()
    assert ws._scratch_total_bytes == 32

    k1_grown = ws.scratch("k", (12,), torch.float32)  # 48 B <= 64 B, <= 128 B
    assert ws.owns_tensor(k1_grown)
    assert ws._scratch_total_bytes == 48

    # 4. Adding entries within budget caches; pushing over budget falls back ephemeral.
    m1 = ws.scratch("m", (16,), torch.float32)  # 64 B -> total 112 B <= 128 B
    assert ws.owns_tensor(m1)
    assert ws._scratch_total_bytes == 112

    spill = ws.scratch("spill", (8,), torch.float32)  # 112 + 32 = 144 B > 128 B
    assert not ws.owns_tensor(spill)
    assert "spill" not in ws._scratch
    assert ws._scratch_total_bytes == 112

    # Growing an existing key uses (total - cur + new) <= budget.
    # m currently holds 64 B; growing m to 64 B is cached, total=112.
    # Growing k (48 B) to 64 B: 112 - 48 + 64 = 128 B <= 128 B -> cached!
    k1_edge = ws.scratch("k", (16,), torch.float32)
    assert ws.owns_tensor(k1_edge)
    assert ws._scratch_total_bytes == 128


def test_grow_only_buffer_caps_and_clear_epoch_sync():
    orig_max_cache = collective_torch._GROW_ONLY_MAX_CACHE_BYTES
    orig_max_total = collective_torch._GROW_ONLY_MAX_TOTAL_BYTES
    try:
        collective_torch.clear_grow_only_buffers()
        collective_torch._GROW_ONLY_MAX_CACHE_BYTES = 64
        collective_torch._GROW_ONLY_MAX_TOTAL_BYTES = 96

        # Oversized single buffer (> 64 B) stays ephemeral.
        big_u8, big_holder, big_gen = collective_torch.acquire_grow_only_u8_holder(
            "cpu", "ut.big", 80, check_capture=False
        )
        assert big_u8.numel() == 80
        assert big_holder is None and big_gen == -1
        assert len(collective_torch._GROW_ONLY_BUFFERS) == 0

        # Small buffer within cache & total budget is cached.
        a_u8, a_holder, a_gen = collective_torch.acquire_grow_only_u8_holder(
            "cpu", "ut.a", 48, check_capture=False
        )
        assert a_u8.numel() == 48
        assert a_holder is not None and a_gen == 1
        assert a_holder[0].numel() == 48

        # Growing within caps bumps gen.
        a2_u8, a2_holder, a2_gen = collective_torch.acquire_grow_only_u8_holder(
            "cpu", "ut.a", 64, check_capture=False
        )
        assert a2_holder is a_holder
        assert a2_gen == 2
        assert a2_u8.numel() == 64

        # Another key that pushes per-device total > 96 B stays ephemeral.
        b_u8, b_holder, b_gen = collective_torch.acquire_grow_only_u8_holder(
            "cpu", "ut.b", 48, check_capture=False
        )
        assert b_u8.numel() == 48
        assert b_holder is None and b_gen == -1
        assert (("cpu", None), "ut.b") not in collective_torch._GROW_ONLY_BUFFERS

        # clear_grow_only_buffers invalidates holder gen, drops backing tensor,
        # and bumps _GROW_ONLY_CLEAR_EPOCH.
        epoch_before = collective_torch._GROW_ONLY_CLEAR_EPOCH
        collective_torch.clear_grow_only_buffers()
        assert collective_torch._GROW_ONLY_CLEAR_EPOCH == epoch_before + 1
        assert a_holder[1] == 3
        assert a_holder[0].numel() == 0
        assert len(collective_torch._GROW_ONLY_BUFFERS) == 0
    finally:
        collective_torch._GROW_ONLY_MAX_CACHE_BYTES = orig_max_cache
        collective_torch._GROW_ONLY_MAX_TOTAL_BYTES = orig_max_total
        collective_torch.clear_grow_only_buffers()


def test_coalesced_plan_handles_uncached_buffers_and_clear_epoch():
    from types import SimpleNamespace
    from unittest.mock import patch

    orig_max_cache = collective_torch._GROW_ONLY_MAX_CACHE_BYTES
    orig_max_total = collective_torch._GROW_ONLY_MAX_TOTAL_BYTES
    try:
        collective_torch.clear_grow_only_buffers()
        _cp_mod._COALESCED_PLAN_CACHE.clear()
        _cp_mod._COALESCED_LAST_PLAN_BY_SLOT.clear()

        fake_pg = SimpleNamespace(rank=lambda: 0, size=lambda: 1)
        with (
            patch.object(collective_torch, "_get_group", return_value=fake_pg),
            patch.object(torch.distributed, "get_world_size", return_value=1),
            patch.object(torch.distributed, "is_initialized", return_value=True),
            patch.object(torch.distributed, "all_gather_into_tensor"),
        ):
            # Case 1: oversized staging exceeds cap -> holder is None, but
            # _build_coalesced_plan still builds an uncached plan without crashing.
            collective_torch._GROW_ONLY_MAX_CACHE_BYTES = 32
            collective_torch._GROW_ONLY_MAX_TOTAL_BYTES = 64
            t_big = torch.ones((4, 16), dtype=torch.float32)  # 256 B > 32 B
            handle_big = _cp_mod.start_cp_all_gather_coalesced(
                (t_big,),
                group=Group.TP,
                async_op=False,
                slot_key="ut.plan.oversize",
            )
            assert len(handle_big.outputs) == 1
            assert tuple(handle_big.outputs[0].shape) == (4, 16)
            assert len(_cp_mod._COALESCED_PLAN_CACHE) == 0
            assert len(_cp_mod._COALESCED_LAST_PLAN_BY_SLOT) == 0

            # Case 2: small tensors fit cache, then clear_grow_only_buffers()
            # bumps epoch and purges both coalesced plan caches cleanly.
            collective_torch._GROW_ONLY_MAX_CACHE_BYTES = 4096
            collective_torch._GROW_ONLY_MAX_TOTAL_BYTES = 8192
            t_small = torch.ones((2, 8), dtype=torch.float32)  # 64 B
            plan1 = _cp_mod._get_or_build_coalesced_plan(
                (t_small,),
                device=torch.device("cpu"),
                group=Group.TP,
                allow_col2d=True,
                slot_key="ut.plan.small",
            )
            assert plan1 is not None and plan1.send_holder is not None
            assert len(_cp_mod._COALESCED_PLAN_CACHE) == 1
            assert len(_cp_mod._COALESCED_LAST_PLAN_BY_SLOT) == 1

            collective_torch.clear_grow_only_buffers()
            t_tiny = torch.ones((1, 8), dtype=torch.float32)
            plan2 = _cp_mod._get_or_build_coalesced_plan(
                (t_tiny,),
                device=torch.device("cpu"),
                group=Group.TP,
                allow_col2d=True,
                slot_key="ut.plan.other",
            )
            assert plan2 is not None
            # Old cached plan for ut.plan.small must have been evicted by epoch sync.
            assert all(k[0] != "ut.plan.small" for k in _cp_mod._COALESCED_PLAN_CACHE)
            assert all(
                k[0] != "ut.plan.small" for k in _cp_mod._COALESCED_LAST_PLAN_BY_SLOT
            )
    finally:
        collective_torch._GROW_ONLY_MAX_CACHE_BYTES = orig_max_cache
        collective_torch._GROW_ONLY_MAX_TOTAL_BYTES = orig_max_total
        collective_torch.clear_grow_only_buffers()
        _cp_mod._COALESCED_PLAN_CACHE.clear()
        _cp_mod._COALESCED_LAST_PLAN_BY_SLOT.clear()


if __name__ == "__main__":
    test_prefill_q_eager_alloc_shape_and_dtype()
    print("PASS test_prefill_q_eager_alloc_shape_and_dtype")
    test_prefill_q_full_capacity_and_overflow()
    print("PASS test_prefill_q_full_capacity_and_overflow")
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
    test_cp_gather_restore_overflow()
    print("PASS test_cp_gather_restore_overflow")
    test_scratch_respects_enable_flag_entry_limit_budget_and_growth()
    print("PASS test_scratch_respects_enable_flag_entry_limit_budget_and_growth")
    test_grow_only_buffer_caps_and_clear_epoch_sync()
    print("PASS test_grow_only_buffer_caps_and_clear_epoch_sync")
    test_coalesced_plan_handles_uncached_buffers_and_clear_epoch()
    print("PASS test_coalesced_plan_handles_uncached_buffers_and_clear_epoch")
    print("ALL TESTS PASSED")
