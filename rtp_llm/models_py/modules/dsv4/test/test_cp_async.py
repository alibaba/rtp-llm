from types import SimpleNamespace
from unittest.mock import patch

import torch

from rtp_llm.models_py.modules.dsv4.cp import (
    _CP_ROLE_INDEXER,
    _CP_ROLE_MAIN,
    _CP_ROLE_SWA_KV_FULL,
    CPContext,
    CPSyncGatherHandle,
    CudaAsyncCPGatherImpl,
    SyncCPGatherImpl,
    _cp_gather_2d,
    _cp_restore_gathered_full_2d,
    _cp_workspace_gather_buffer,
    _cp_workspace_restore_buffer,
    build_cp_context,
    cp_all_gather_full,
    cp_all_gather_full_async,
    cp_wait_gather_full,
    cp_wait_gather_rank_major,
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


def test_sync_cp_gather_handle_can_defer_restore_and_expose_rank_major():
    ctx = _make_cp_ctx()
    gathered = torch.tensor(
        [
            [10.0, 11.0],
            [20.0, 21.0],
            [30.0, 31.0],
            [40.0, 41.0],
        ]
    )
    handle = CPSyncGatherHandle(
        full_2d=None,
        gathered=gathered,
        cp_ctx=ctx,
        profile_name="unit",
    )

    raw = cp_wait_gather_rank_major(handle)
    assert raw is not None
    raw_gathered, raw_ctx = raw
    assert raw_gathered is gathered
    assert raw_ctx is ctx

    full = cp_wait_gather_full(handle)
    expected = gathered.index_select(0, ctx.unpad_restore)
    assert torch.equal(full, expected)
    assert handle.full_2d is full


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


def test_cp_all_gather_full_async_env_dispatches_to_sync_workspace():
    ctx = _make_cp_ctx()
    local = torch.zeros((2, 2), dtype=torch.float32)
    ws = PrefillWorkspace(
        torch.device("cpu"), q_rows=1, q_dim=1, reserve_cp=False, align_bytes=1
    )
    expected = CPSyncGatherHandle(full_2d=torch.ones(3, 2), profile_name="test")
    calls = []

    def fake_sync(local_arg, ctx_arg, *, profile_name, workspace, cp_role):
        calls.append((local_arg, ctx_arg, profile_name, workspace, cp_role))
        return expected

    with patch.dict(
        "rtp_llm.models_py.modules.dsv4.cp.os.environ",
        {"DSV4_CP_GATHER_SYNC_WORKSPACE": "1"},
    ), patch(
        "rtp_llm.models_py.modules.dsv4.cp._cp_all_gather_full_sync_workspace",
        fake_sync,
    ):
        handle = cp_all_gather_full_async(
            local,
            ctx,
            workspace=ws,
            cp_role=_CP_ROLE_MAIN,
            profile_name="unit",
        )

    assert handle is expected
    assert len(calls) == 1
    assert calls[0] == (local, ctx, "unit", ws, _CP_ROLE_MAIN)


def test_cp_workspace_role_buffer_mapping_and_unknown_role():
    ws = PrefillWorkspace(
        torch.device("cpu"),
        q_rows=1,
        q_dim=1,
        reserve_cp=True,
        cp_rows=4,
        main_w=2,
        idx_w=3,
        swa_w=5,
        align_bytes=1,
    )

    main = _cp_workspace_gather_buffer(ws, _CP_ROLE_MAIN, 4, 2, torch.float32)
    idx = _cp_workspace_gather_buffer(ws, _CP_ROLE_INDEXER, 4, 3, torch.float32)
    swa = _cp_workspace_gather_buffer(ws, _CP_ROLE_SWA_KV_FULL, 4, 5, torch.bfloat16)
    main_restore = _cp_workspace_restore_buffer(
        ws, _CP_ROLE_MAIN, 4, 2, torch.float32
    )

    assert tuple(main.shape) == (4, 2)
    assert tuple(idx.shape) == (4, 3)
    assert tuple(swa.shape) == (4, 5)
    assert swa.dtype == torch.bfloat16
    assert tuple(main_restore.shape) == (4, 2)
    assert main.data_ptr() != idx.data_ptr()
    assert main.data_ptr() != swa.data_ptr()
    assert main_restore.data_ptr() != main.data_ptr()

    _assert_raises(
        lambda: _cp_workspace_gather_buffer(ws, "bad", 4, 2, torch.float32),
        ValueError,
        "unknown cp_role",
    )
    _assert_raises(
        lambda: _cp_workspace_restore_buffer(ws, "bad", 4, 2, torch.float32),
        ValueError,
        "unknown cp_role",
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
    test_sync_cp_gather_handle_can_defer_restore_and_expose_rank_major()
    print("PASS test_sync_cp_gather_handle_can_defer_restore_and_expose_rank_major")
    test_cp_all_gather_full_rejects_non_2d_and_wrong_t_local()
    print("PASS test_cp_all_gather_full_rejects_non_2d_and_wrong_t_local")
    test_cuda_async_cp_gather_impl_fails_fast_on_cpu()
    print("PASS test_cuda_async_cp_gather_impl_fails_fast_on_cpu")
    test_cp_wait_gather_full_rejects_unknown_handle()
    print("PASS test_cp_wait_gather_full_rejects_unknown_handle")
    test_cp_all_gather_full_async_env_dispatches_to_sync_workspace()
    print("PASS test_cp_all_gather_full_async_env_dispatches_to_sync_workspace")
    test_cp_workspace_role_buffer_mapping_and_unknown_role()
    print("PASS test_cp_workspace_role_buffer_mapping_and_unknown_role")
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
    test_build_cp_context_moves_restore_indices_independently_when_cuda_available()
    print(
        "PASS test_build_cp_context_moves_restore_indices_independently_when_cuda_available"
    )
    print("ALL TESTS PASSED")
