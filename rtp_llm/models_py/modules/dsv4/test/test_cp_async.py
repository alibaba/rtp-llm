from types import SimpleNamespace
from unittest.mock import patch

import torch

from rtp_llm.models_py.distributed import pynccl_cp
from rtp_llm.models_py.modules.dsv4.cp import (
    _CP_ROLE_MAIN,
    CPContext,
    CPSyncGatherHandle,
    CudaAsyncCPGatherImpl,
    SyncCPGatherImpl,
    _cp_gather_2d,
    _cp_restore_gathered_full_2d,
    build_cp_context,
    cp_all_gather_full,
    cp_wait_gather_full,
)
from rtp_llm.models_py.modules.dsv4.prefill_workspace import PrefillWorkspace


def _make_cp_ctx() -> CPContext:
    return CPContext(
        cp_size=2,
        cp_rank=0,
        chunk_length=2,
        padded_seq_len=4,
        seq_len_full=3,
        relative_positions=torch.tensor([0, 3], dtype=torch.long),
        prefix_length=0,
        global_positions=torch.tensor([0, 3], dtype=torch.long),
        local_is_real=torch.tensor([True, True]),
        unpad_restore=torch.tensor([0, 3, 1], dtype=torch.long),
        seq_len_total=3,
        cp_info=object(),
    )


def _assert_raises(fn, exc_type, msg_substr: str):
    try:
        fn()
    except exc_type as exc:
        assert msg_substr in str(exc), str(exc)
        return
    raise AssertionError(f"expected {exc_type.__name__} containing {msg_substr!r}")


def test_cp_all_gather_full_restores_2d_and_unpads():
    ctx = _make_cp_ctx()
    local = torch.zeros((2, 6), dtype=torch.float32)
    gathered = torch.arange(24, dtype=torch.float32).reshape(4, 6)

    with patch("rtp_llm.models_py.modules.dsv4.cp.all_gather", return_value=gathered):
        full = cp_all_gather_full(local, ctx)

    assert tuple(full.shape) == (3, 6)
    assert torch.equal(full, gathered.index_select(0, ctx.unpad_restore))


def test_sync_cp_gather_impl_restores_2d_on_cpu():
    ctx = _make_cp_ctx()
    local = torch.zeros((2, 2), dtype=torch.float32)
    gathered = torch.tensor(
        [
            [10.0, 11.0],
            [20.0, 21.0],
            [30.0, 31.0],
            [40.0, 41.0],
        ]
    )

    with patch("rtp_llm.models_py.modules.dsv4.cp.all_gather", return_value=gathered):
        handle = SyncCPGatherImpl().start(local, ctx)
        full = cp_wait_gather_full(handle)

    expected = gathered.index_select(0, ctx.unpad_restore)
    assert isinstance(handle, CPSyncGatherHandle)
    assert torch.equal(full, expected)


def test_cp_all_gather_full_rejects_non_2d_and_wrong_t_local():
    ctx = _make_cp_ctx()

    _assert_raises(
        lambda: cp_all_gather_full(torch.zeros((2,), dtype=torch.float32), ctx),
        ValueError,
        "expects 2D",
    )
    _assert_raises(
        lambda: cp_all_gather_full(torch.zeros((1, 2, 6), dtype=torch.float32), ctx),
        ValueError,
        "expects 2D",
    )
    _assert_raises(
        lambda: cp_all_gather_full(torch.zeros((3, 6), dtype=torch.float32), ctx),
        ValueError,
        "T_local",
    )


def test_cuda_async_cp_gather_impl_fails_fast_on_cpu():
    ctx = _make_cp_ctx()
    local = torch.zeros((2, 6), dtype=torch.float32)
    # The workspace assert fires first; pass a (minimal) one so the test
    # exercises the intended CUDA fail-fast rather than the workspace guard.
    ws = PrefillWorkspace(
        torch.device("cpu"), q_rows=1, q_dim=1, reserve_cp=False, align_bytes=1
    )

    _assert_raises(
        lambda: CudaAsyncCPGatherImpl().start(
            local, ctx, workspace=ws, cp_role=_CP_ROLE_MAIN
        ),
        RuntimeError,
        "requires CUDA",
    )


def test_cp_wait_gather_full_rejects_unknown_handle():
    _assert_raises(
        lambda: cp_wait_gather_full(object()),
        TypeError,
        "unsupported CP gather handle",
    )


def test_build_cp_context_single_stream_direct_slice_unpad_restore():
    cp_info = SimpleNamespace(
        prefill_qkv_padding_mask=torch.tensor(
            [1, 1, 1, 1, 1, 1, 0, 0], dtype=torch.int32
        ),
        prefill_qkv_restore_indice=torch.tensor(
            [0, 1, 6, 7, 2, 3, 4, 5], dtype=torch.int32
        ),
        prefill_cp_chunk_lengths=torch.tensor([4], dtype=torch.int32),
        prefill_actual_input_lengths_cpu=torch.tensor([6], dtype=torch.int32),
    )

    ctx = build_cp_context(
        cp_info,
        cp_size=2,
        cp_rank=0,
        chunk_length=4,
        device=torch.device("cpu"),
        position_offset=10,
    )

    assert ctx.seq_len_full == 6
    assert ctx.prefix_length == 10
    assert torch.equal(ctx.relative_positions, torch.tensor([0, 1, 6, 7]))
    assert torch.equal(ctx.global_positions, torch.tensor([10, 11, 15, 15]))
    assert torch.equal(ctx.local_is_real, torch.tensor([True, True, False, False]))
    assert torch.equal(ctx.unpad_restore, cp_info.prefill_qkv_restore_indice[:6].long())
    assert not ctx.unpad_restore_is_prefix


def test_build_cp_context_marks_identity_restore_prefix():
    cp_info = SimpleNamespace(
        prefill_qkv_padding_mask=torch.ones(8, dtype=torch.int32),
        prefill_qkv_restore_indice=torch.arange(8, dtype=torch.int32),
        prefill_cp_chunk_lengths=torch.tensor([4], dtype=torch.int32),
        prefill_actual_input_lengths_cpu=torch.tensor([6], dtype=torch.int32),
    )

    ctx = build_cp_context(
        cp_info,
        cp_size=2,
        cp_rank=0,
        chunk_length=4,
        device=torch.device("cpu"),
    )

    assert torch.equal(ctx.unpad_restore, torch.arange(6, dtype=torch.long))
    assert ctx.unpad_restore_is_prefix


def test_cp_restore_prefix_returns_view_without_index_select():
    ctx = CPContext(
        cp_size=2,
        cp_rank=0,
        chunk_length=4,
        padded_seq_len=8,
        seq_len_full=6,
        relative_positions=torch.arange(4, dtype=torch.long),
        prefix_length=0,
        global_positions=torch.arange(4, dtype=torch.long),
        local_is_real=torch.ones(4, dtype=torch.bool),
        unpad_restore=torch.arange(6, dtype=torch.long),
        seq_len_total=6,
        cp_info=object(),
        unpad_restore_is_prefix=True,
    )
    gathered = torch.arange(8 * 3, dtype=torch.float32).reshape(8, 3)
    full = _cp_restore_gathered_full_2d(gathered, ctx)

    assert torch.equal(full, gathered[:6])
    assert full.data_ptr() == gathered.data_ptr()
    assert full.storage_offset() == gathered.storage_offset()


def test_cp_gather_2d_keeps_contiguous_input_object():
    ctx = _make_cp_ctx()
    local = torch.zeros((2, 6), dtype=torch.float32)
    out = _cp_gather_2d(local, ctx)

    assert out is local


def test_build_cp_context_supports_multi_prefill_stream():
    cp_info = SimpleNamespace(
        prefill_qkv_padding_mask=torch.ones(16, dtype=torch.int32),
        prefill_qkv_restore_indice=torch.arange(16, dtype=torch.int32),
        prefill_cp_chunk_lengths=torch.tensor([4, 4], dtype=torch.int32),
        prefill_actual_input_lengths_cpu=torch.tensor([8, 8], dtype=torch.int32),
    )

    ctx = build_cp_context(
        cp_info,
        cp_size=2,
        cp_rank=0,
        chunk_length=8,
        device=torch.device("cpu"),
        position_offset=torch.tensor([0, 100], dtype=torch.int64),
    )

    assert ctx.seq_len_full == 16
    assert ctx.chunk_lengths_per_req == (4, 4)
    assert torch.equal(ctx.relative_positions, torch.tensor([0, 1, 6, 7, 8, 9, 14, 15]))
    assert torch.equal(
        ctx.global_positions, torch.tensor([0, 1, 6, 7, 100, 101, 106, 107])
    )
    assert torch.equal(
        ctx.req_id_per_token,
        torch.tensor([0, 0, 0, 0, 1, 1, 1, 1], dtype=torch.int32),
    )
    assert torch.equal(ctx.unpad_restore, torch.arange(16, dtype=torch.long))
    assert torch.equal(ctx.local_is_real, torch.ones(8, dtype=torch.bool))


def test_pynccl_symm_rounds_rows_by_default_for_prefix_cache_shapes():
    with patch.dict("os.environ", {}, clear=True):
        assert pynccl_cp._round_symm_rows(1) == 1
        assert pynccl_cp._round_symm_rows(24576) == 32768
        assert pynccl_cp._round_symm_rows(28672) == 32768
        assert pynccl_cp._round_symm_rows(32768) == 32768


def test_pynccl_symm_allows_only_safe_default_roles():
    with patch.dict("os.environ", {}, clear=True):
        assert pynccl_cp.symm_role_allowed("state_read")
        assert pynccl_cp.symm_role_allowed("full_sync")
        assert pynccl_cp.symm_role_allowed("varlen")
        assert pynccl_cp.symm_role_allowed("main")
        assert pynccl_cp.symm_role_allowed("indexer")
        assert not pynccl_cp.symm_role_allowed("gather_cmp")
        assert not pynccl_cp.symm_role_allowed("swa_prefix")
        assert not pynccl_cp.symm_role_allowed("indexer_k_packed")


def test_pynccl_symm_uses_torch_max_bytes_as_capacity_limit():
    max_nbytes = 64 * 1024 * 1024
    assert pynccl_cp._fits_torch_symm_mem_limit(
        rows=1024,
        cols=1024,
        dtype=torch.bfloat16,
        max_nbytes=max_nbytes,
    )
    assert not pynccl_cp._fits_torch_symm_mem_limit(
        rows=max_nbytes // 2 + 1,
        cols=1,
        dtype=torch.bfloat16,
        max_nbytes=max_nbytes,
    )
    assert not pynccl_cp._fits_torch_symm_mem_limit(
        rows=1,
        cols=1,
        dtype=torch.bfloat16,
        max_nbytes=None,
    )


def test_pynccl_symm_over_limit_falls_back_to_plain_pynccl():
    process_group = object()
    device = torch.device("cuda", 0)
    with patch.dict("os.environ", {"DSV4_CP_SYMM": "1"}, clear=True):
        with patch.object(pynccl_cp, "symm_available", return_value=True):
            with patch.object(pynccl_cp, "symm_persistent", return_value=None):
                with patch.object(pynccl_cp, "_comm_ready", return_value=True) as ready:
                    symm_out, symm_ok, use_pynccl = pynccl_cp._select_cp_backend(
                        "main",
                        1,
                        1,
                        torch.bfloat16,
                        device,
                        process_group,
                        False,
                        "test.profile",
                    )
    assert symm_out is None
    assert not symm_ok
    assert use_pynccl
    ready.assert_called_once_with(process_group)


def test_pynccl_symm_disallowed_roles_fall_back_to_plain_pynccl():
    process_group = object()
    device = torch.device("cuda", 0)
    with patch.dict("os.environ", {"DSV4_CP_SYMM": "1"}, clear=True):
        with patch.object(pynccl_cp, "symm_available", return_value=True):
            with patch.object(pynccl_cp, "symm_persistent") as symm_persistent:
                with patch.object(pynccl_cp, "_comm_ready", return_value=True) as ready:
                    symm_out, symm_ok, use_pynccl = pynccl_cp._select_cp_backend(
                        "gather_cmp",
                        1,
                        1,
                        torch.uint8,
                        device,
                        process_group,
                        True,
                        "test.profile",
                    )
    assert symm_out is None
    assert not symm_ok
    assert use_pynccl
    symm_persistent.assert_not_called()
    ready.assert_called_once_with(process_group)


def test_pynccl_symm_init_defaults_cover_variable_roles_and_dtypes():
    with patch.dict("os.environ", {}, clear=True):
        pairs = pynccl_cp._symm_init_pairs()

    assert ("main", torch.bfloat16) in pairs
    assert ("main", torch.float32) in pairs
    assert ("indexer", torch.bfloat16) in pairs
    assert ("indexer", torch.float32) in pairs
    assert ("state_read", torch.float32) in pairs
    assert ("gather_cmp", torch.uint8) not in pairs
    if hasattr(torch, "float8_e4m3fn"):
        assert ("indexer_k_quant", torch.float8_e4m3fn) not in pairs


def test_pynccl_symm_init_registers_windows_within_budget():
    process_group = object()
    device = torch.device("cuda", 0)
    with patch.object(pynccl_cp, "symm_enabled", return_value=True):
        with patch.object(pynccl_cp, "_symm_pool", return_value=object()):
            with patch.object(pynccl_cp, "symm_available", return_value=True):
                with patch.object(pynccl_cp, "_torch_symm_mem_max_nbytes", return_value=128):
                    with patch.object(
                        pynccl_cp,
                        "_symm_init_pairs",
                        return_value=(
                            ("main", torch.bfloat16),
                            ("gather_cmp", torch.uint8),
                            ("state_read", torch.float32),
                            ("indexer_k_scale", torch.float32),
                        ),
                    ):
                        with patch.object(pynccl_cp, "_ring_symm_base") as ring_base:
                            pynccl_cp._init_symm_windows(process_group, device)

    assert ring_base.call_count == 2
    ring_base.assert_any_call(
        "main", torch.bfloat16, device, process_group, 128, create=True
    )
    ring_base.assert_any_call(
        "state_read", torch.float32, device, process_group, 128, create=True
    )


def test_pynccl_symm_runtime_does_not_create_missing_windows():
    process_group = object()
    device = torch.device("cuda", 0)
    with patch.dict("os.environ", {"DSV4_CP_SYMM": "1"}, clear=True):
        with patch.object(pynccl_cp, "_torch_symm_mem_max_nbytes", return_value=1024):
            with patch.object(pynccl_cp, "_ring_symm_base", return_value=None) as ring_base:
                out = pynccl_cp.symm_persistent(
                    "main",
                    1,
                    1,
                    torch.bfloat16,
                    device,
                    process_group,
                    "test.profile",
                )

    assert out is None
    ring_base.assert_called_once_with(
        "main", torch.bfloat16, device, process_group, 1024
    )


def test_build_cp_context_moves_restore_indices_independently_when_cuda_available():
    if not torch.cuda.is_available():
        return
    cp_info = SimpleNamespace(
        prefill_qkv_padding_mask=torch.ones(8, dtype=torch.int32, device="cuda"),
        prefill_qkv_restore_indice=torch.arange(8, dtype=torch.int32),
        prefill_cp_chunk_lengths=torch.tensor([4], dtype=torch.int32),
        prefill_actual_input_lengths_cpu=torch.tensor([6], dtype=torch.int32),
    )

    ctx = build_cp_context(
        cp_info,
        cp_size=2,
        cp_rank=0,
        chunk_length=4,
        device=torch.device("cuda"),
    )

    assert ctx.unpad_restore.device.type == "cuda"
    assert torch.equal(ctx.unpad_restore.cpu(), torch.arange(6, dtype=torch.long))


if __name__ == "__main__":
    test_cp_all_gather_full_restores_2d_and_unpads()
    print("PASS test_cp_all_gather_full_restores_2d_and_unpads")
    test_sync_cp_gather_impl_restores_2d_on_cpu()
    print("PASS test_sync_cp_gather_impl_restores_2d_on_cpu")
    test_cp_all_gather_full_rejects_non_2d_and_wrong_t_local()
    print("PASS test_cp_all_gather_full_rejects_non_2d_and_wrong_t_local")
    test_cuda_async_cp_gather_impl_fails_fast_on_cpu()
    print("PASS test_cuda_async_cp_gather_impl_fails_fast_on_cpu")
    test_cp_wait_gather_full_rejects_unknown_handle()
    print("PASS test_cp_wait_gather_full_rejects_unknown_handle")
    test_build_cp_context_single_stream_direct_slice_unpad_restore()
    print("PASS test_build_cp_context_single_stream_direct_slice_unpad_restore")
    test_build_cp_context_marks_identity_restore_prefix()
    print("PASS test_build_cp_context_marks_identity_restore_prefix")
    test_cp_restore_prefix_returns_view_without_index_select()
    print("PASS test_cp_restore_prefix_returns_view_without_index_select")
    test_cp_gather_2d_keeps_contiguous_input_object()
    print("PASS test_cp_gather_2d_keeps_contiguous_input_object")
    test_build_cp_context_supports_multi_prefill_stream()
    print("PASS test_build_cp_context_supports_multi_prefill_stream")
    test_pynccl_symm_rounds_rows_by_default_for_prefix_cache_shapes()
    print("PASS test_pynccl_symm_rounds_rows_by_default_for_prefix_cache_shapes")
    test_pynccl_symm_allows_only_safe_default_roles()
    print("PASS test_pynccl_symm_allows_only_safe_default_roles")
    test_pynccl_symm_uses_torch_max_bytes_as_capacity_limit()
    print("PASS test_pynccl_symm_uses_torch_max_bytes_as_capacity_limit")
    test_pynccl_symm_over_limit_falls_back_to_plain_pynccl()
    print("PASS test_pynccl_symm_over_limit_falls_back_to_plain_pynccl")
    test_pynccl_symm_disallowed_roles_fall_back_to_plain_pynccl()
    print("PASS test_pynccl_symm_disallowed_roles_fall_back_to_plain_pynccl")
    test_pynccl_symm_init_defaults_cover_variable_roles_and_dtypes()
    print("PASS test_pynccl_symm_init_defaults_cover_variable_roles_and_dtypes")
    test_pynccl_symm_init_registers_windows_within_budget()
    print("PASS test_pynccl_symm_init_registers_windows_within_budget")
    test_pynccl_symm_runtime_does_not_create_missing_windows()
    print("PASS test_pynccl_symm_runtime_does_not_create_missing_windows")
    test_build_cp_context_moves_restore_indices_independently_when_cuda_available()
    print(
        "PASS test_build_cp_context_moves_restore_indices_independently_when_cuda_available"
    )
    print("ALL TESTS PASSED")
