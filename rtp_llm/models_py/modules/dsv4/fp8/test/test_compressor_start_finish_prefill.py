"""UT: CompressorFP8.start_prefill + finish_prefill is bit-equal to forward.

The overlap orchestrator (attention.py) drives compressors via the new
``start_prefill`` / ``finish_prefill`` split so it can interleave the CP
all-gather with default-stream compute (indexer / SWA write). This UT
locks the split's externally observable behaviour against the existing
``forward`` baseline:

  * exactly one CP all-gather per pending, with the same ``local_2d``
    source and ``cp_ctx`` as ``forward``;
  * the wait + split + ``_launch`` path receives identical
    ``kv_flat`` / ``score_flat`` / ``meta`` arguments;
  * warmup (no pool bound) → ``start_prefill`` returns ``None`` and
    ``finish_prefill(None)`` is a no-op (mirrors ``forward``'s early
    return → ``None``);
  * ``cp_gather_stream`` provided by the caller is forwarded verbatim to
    ``cp_all_gather_full_async`` (shared FIFO stream contract).
"""

from __future__ import annotations

import unittest
from unittest.mock import patch

import torch

from rtp_llm.models_py.modules.dsv4.cp import CPContext
from rtp_llm.models_py.modules.dsv4.fp8._compressor_consts import (
    INDEXER_ENTRY_BYTES,
    INDEXER_HEAD_DIM,
)
from rtp_llm.models_py.modules.dsv4.fp8.compressor import (
    CompressorFP8,
    CompressorMeta,
    _CompressorPending,
)


def _build_compressor(
    *,
    dim: int,
    head_dim: int,
    compress_ratio: int,
    device: torch.device,
    bind_pool: bool = True,
) -> CompressorFP8:
    coff = 1 + (compress_ratio == 4)
    weights = {
        "ape": torch.zeros(compress_ratio, coff * head_dim, dtype=torch.float32),
        "wkv": torch.randn(coff * head_dim, dim, dtype=torch.bfloat16),
        "wgate": torch.randn(coff * head_dim, dim, dtype=torch.bfloat16),
        "norm": torch.ones(head_dim, dtype=torch.bfloat16),
    }
    cmp = CompressorFP8(
        dim=dim,
        head_dim=head_dim,
        rope_head_dim=0,
        compress_ratio=compress_ratio,
        max_batch_size=1,
        compressor_weights=weights,
    ).to(device)
    # _fuse_wkv_wgate stores the fused matrix on construction (CPU); rebind
    # to ``device`` so the fused linear runs where the input lives.
    if cmp._wkv_wgate_fused is not None:
        cmp._wkv_wgate_fused = cmp._wkv_wgate_fused.to(device)
    if not bind_pool:
        return cmp

    state_eb = 4
    state_view = torch.zeros(4, 2 * coff * head_dim, dtype=torch.float32, device=device)
    state_block_table = torch.ones(1, 1, dtype=torch.int32, device=device)
    kv_pool = torch.zeros(1, 1, INDEXER_ENTRY_BYTES, dtype=torch.uint8, device=device)
    kv_block_table = torch.ones(1, 1, dtype=torch.int32, device=device)
    cmp.set_pool_context(
        kv_pool_view=kv_pool,
        kv_block_table=kv_block_table,
        kv_eb=1,
        state_pool_view=state_view,
        state_block_table=state_block_table,
        state_eb=state_eb,
        state_tokens_per_block=state_eb,
        kv_tokens_per_block=compress_ratio,
    )
    return cmp


def _make_cp_ctx(device: torch.device) -> CPContext:
    return CPContext(
        cp_size=2,
        cp_rank=0,
        chunk_length=2,
        padded_seq_len=4,
        seq_len_full=3,
        relative_positions=torch.tensor([0, 1], dtype=torch.long, device=device),
        prefix_length=0,
        global_positions=torch.tensor([0, 1], dtype=torch.long, device=device),
        local_is_real=torch.tensor([True, True], device=device),
        unpad_restore=torch.tensor([0, 2, 3], dtype=torch.long, device=device),
        seq_len_total=3,
        cp_info=object(),
        input_lengths_global=torch.tensor([3], dtype=torch.int32, device=device),
        cu_seqlens_global=torch.tensor([0, 3], dtype=torch.int32, device=device),
        unpad_restore_is_prefix=False,
    )


def _make_meta(cp_ctx: CPContext, device: torch.device) -> CompressorMeta:
    return CompressorMeta(
        positions=torch.arange(cp_ctx.seq_len_full, dtype=torch.long, device=device),
        b_idx=torch.zeros(cp_ctx.seq_len_full, dtype=torch.long, device=device),
        state_slots=torch.zeros(cp_ctx.seq_len_full, dtype=torch.long, device=device),
        kv_slots=torch.zeros(cp_ctx.seq_len_full, dtype=torch.long, device=device),
        token_to_req=torch.zeros(cp_ctx.seq_len_full, dtype=torch.int32, device=device),
        seq_start_per_req=torch.tensor([0], dtype=torch.int32, device=device),
        cu_seq_per_req=torch.tensor(
            [0, cp_ctx.seq_len_full], dtype=torch.int32, device=device
        ),
    )


@unittest.skipUnless(torch.cuda.is_available(), "CompressorFP8 requires CUDA gemm")
class CompressorFP8StartFinishPrefillTest(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(0)
        self.device = torch.device("cuda")
        self.dim = 8
        self.head_dim = INDEXER_HEAD_DIM
        self.compress_ratio = 4
        self.coff = 1 + (self.compress_ratio == 4)
        self.out_dim = self.coff * self.head_dim
        self.cp_ctx = _make_cp_ctx(self.device)
        self.meta = _make_meta(self.cp_ctx, self.device)
        self.x = torch.randn(
            self.cp_ctx.chunk_length, self.dim, dtype=torch.bfloat16, device=self.device
        )
        self.gathered_fused = torch.arange(
            self.cp_ctx.seq_len_full * 2 * self.out_dim,
            dtype=torch.bfloat16,
            device=self.device,
        ).reshape(self.cp_ctx.seq_len_full, 2 * self.out_dim)

    def _run(self, cmp: CompressorFP8, *, use_split: bool, cp_gather_stream=None):
        """Returns (gather_inputs, launch_args, pending). pending is None when
        running through forward (sync path) — only used for split assertions."""
        gather_inputs = []
        handle = object()
        launch_args = {}

        def fake_start(
            local_2d, ctx, stream=None, restored_buf=None, profile_name=None
        ):
            del restored_buf, profile_name
            gather_inputs.append((local_2d, ctx, stream))
            return handle

        def fake_wait(actual_handle):
            assert actual_handle is handle
            return self.gathered_fused

        def fake_launch(kv_flat, score_flat, actual_meta):
            launch_args["kv_flat"] = kv_flat
            launch_args["score_flat"] = score_flat
            launch_args["meta"] = actual_meta

        with (
            patch(
                "rtp_llm.models_py.modules.dsv4.fp8.compressor.cp_all_gather_full_async",
                side_effect=fake_start,
            ),
            patch(
                "rtp_llm.models_py.modules.dsv4.fp8.compressor.cp_wait_gather_full",
                side_effect=fake_wait,
            ),
            patch.object(cmp, "_launch", side_effect=fake_launch),
        ):
            if use_split:
                pending = cmp.start_prefill(
                    self.x, 0, meta=self.meta, cp_gather_stream=cp_gather_stream
                )
                cmp.finish_prefill(pending)
            else:
                pending = None
                cmp.forward(self.x, 0, meta=self.meta)
        return gather_inputs, launch_args, pending

    def test_start_finish_bit_equal_to_forward_under_cp(self):
        cmp_sync = _build_compressor(
            dim=self.dim,
            head_dim=self.head_dim,
            compress_ratio=self.compress_ratio,
            device=self.device,
        )
        cmp_split = _build_compressor(
            dim=self.dim,
            head_dim=self.head_dim,
            compress_ratio=self.compress_ratio,
            device=self.device,
        )
        # Bind the same fused weight so fused_linear is deterministic.
        with torch.no_grad():
            cmp_split._wkv_wgate_fused.copy_(cmp_sync._wkv_wgate_fused)
        cmp_sync.set_cp_ctx(self.cp_ctx)
        cmp_split.set_cp_ctx(self.cp_ctx)

        sync_inputs, sync_launch, _ = self._run(cmp_sync, use_split=False)
        split_inputs, split_launch, pending = self._run(cmp_split, use_split=True)

        # Pending plumbed through correctly.
        self.assertIsInstance(pending, _CompressorPending)
        self.assertEqual(pending.sp, 0)
        self.assertEqual(pending.bsz, 1)
        self.assertEqual(pending.seqlen, int(self.x.size(0)))
        self.assertEqual(pending.out_dim, self.out_dim)
        self.assertIs(pending.meta, self.meta)

        # Both paths called exactly one gather with the same local source +
        # the same cp_ctx (the source values must be equal since fused
        # weights were copied).
        self.assertEqual(len(sync_inputs), 1)
        self.assertEqual(len(split_inputs), 1)
        sync_local, sync_ctx, _ = sync_inputs[0]
        split_local, split_ctx, _ = split_inputs[0]
        self.assertIs(sync_ctx, self.cp_ctx)
        self.assertIs(split_ctx, self.cp_ctx)
        self.assertEqual(tuple(sync_local.shape), tuple(split_local.shape))
        self.assertTrue(torch.equal(sync_local, split_local))

        # _launch receives identical kv_flat / score_flat slices (both are
        # views of the same gathered_fused via the split).
        self.assertTrue(torch.equal(sync_launch["kv_flat"], split_launch["kv_flat"]))
        self.assertTrue(
            torch.equal(sync_launch["score_flat"], split_launch["score_flat"])
        )
        self.assertIs(sync_launch["meta"], split_launch["meta"])

    def test_caller_supplied_cp_gather_stream_is_forwarded(self):
        cmp = _build_compressor(
            dim=self.dim,
            head_dim=self.head_dim,
            compress_ratio=self.compress_ratio,
            device=self.device,
        )
        cmp.set_cp_ctx(self.cp_ctx)
        sentinel = object()  # cpu path -> stream is opaque, fine to be non-Stream

        gather_inputs, _, _ = self._run(cmp, use_split=True, cp_gather_stream=sentinel)
        self.assertEqual(len(gather_inputs), 1)
        _, _, forwarded_stream = gather_inputs[0]
        self.assertIs(forwarded_stream, sentinel)

    def test_wait_prefill_gather_fences_without_launch_and_finish_writes_once(self):
        cmp = _build_compressor(
            dim=self.dim,
            head_dim=self.head_dim,
            compress_ratio=self.compress_ratio,
            device=self.device,
        )
        cmp.set_cp_ctx(self.cp_ctx)
        handle = object()
        wait_calls = []
        launch_calls = []

        def fake_start(
            local_2d, ctx, stream=None, restored_buf=None, profile_name=None
        ):
            del local_2d, ctx, stream, restored_buf, profile_name
            return handle

        def fake_wait(actual_handle):
            wait_calls.append(actual_handle)
            return self.gathered_fused

        def fake_launch(kv_flat, score_flat, actual_meta):
            launch_calls.append((kv_flat, score_flat, actual_meta))

        with (
            patch(
                "rtp_llm.models_py.modules.dsv4.fp8.compressor.cp_all_gather_full_async",
                side_effect=fake_start,
            ),
            patch(
                "rtp_llm.models_py.modules.dsv4.fp8.compressor.cp_wait_gather_full",
                side_effect=fake_wait,
            ),
            patch.object(cmp, "_launch", side_effect=fake_launch),
        ):
            pending = cmp.start_prefill(self.x, 0, meta=self.meta)
            self.assertIsNotNone(pending)
            cmp.wait_prefill_gather(pending)
            self.assertEqual(wait_calls, [handle])
            self.assertEqual(launch_calls, [])
            self.assertIsNone(pending.fused_gather_handle)

            cmp.finish_prefill(pending)

        # ``finish_prefill`` must not wait the already-fenced handle again, and
        # must perform the single pool write after the caller's explicit fence.
        self.assertEqual(wait_calls, [handle])
        self.assertEqual(len(launch_calls), 1)
        kv_flat, score_flat, actual_meta = launch_calls[0]
        self.assertTrue(torch.equal(kv_flat, self.gathered_fused[:, : self.out_dim]))
        self.assertTrue(torch.equal(score_flat, self.gathered_fused[:, self.out_dim :]))
        self.assertIs(actual_meta, self.meta)

    def test_warmup_start_returns_none_and_finish_is_noop(self):
        cmp = _build_compressor(
            dim=self.dim,
            head_dim=self.head_dim,
            compress_ratio=self.compress_ratio,
            device=self.device,
            bind_pool=False,  # no pool ⇒ warmup
        )
        cmp.set_cp_ctx(self.cp_ctx)

        # Even without patching, neither cp_all_gather_full_async nor _launch
        # should be touched on the warmup path.
        launched = []

        def fake_launch(*args, **kwargs):
            launched.append((args, kwargs))

        with patch.object(cmp, "_launch", side_effect=fake_launch):
            pending = cmp.start_prefill(self.x, 0, meta=self.meta)
            self.assertIsNone(pending)
            cmp.finish_prefill(pending)  # no-op
        self.assertEqual(launched, [])

    def test_finish_prefill_none_handle_is_idempotent_noop(self):
        # Belt-and-braces: even if a caller wrapped start_prefill in a try
        # / except and only got back None, double-calling finish must not
        # raise.
        cmp = _build_compressor(
            dim=self.dim,
            head_dim=self.head_dim,
            compress_ratio=self.compress_ratio,
            device=self.device,
            bind_pool=False,
        )
        cmp.finish_prefill(None)
        cmp.finish_prefill(None)


if __name__ == "__main__":
    unittest.main()
