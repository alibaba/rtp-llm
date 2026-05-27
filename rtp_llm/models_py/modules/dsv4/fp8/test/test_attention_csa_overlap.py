"""UT: AttentionFP8 CSA CP-overlap orchestrator.

CSA layers issue TWO NCCL collectives per step (nested indexer
compressor + main CSA compressor); both must share one cp_gather_stream
for rank-consistent FIFO ordering inside the ProcessGroup. This test
locks the orchestrator's externally observable call sequence:

  1. indexer.start_prefill_nested_compressor — NCCL #1 on cp_gather_stream
  2. compressor.start_prefill        — NCCL #2 on the SAME stream
  3. _prefill_write_swa_fp8_paged    — default-stream overlap zone
  4. indexer.forward_with_pending_nested — drains nested NCCL #1, then waits
                                       main NCCL #2 before gather_k/score/topk
  5. compressor.finish_prefill       — writes main CSA pool after indexer topk
  6. _forward_prefill_compressed(_skip_compressor_write=True,
                                  cmp_topk_runtime=raw)
                                     — workspace_attn

Also verifies that both compressor stream-binds use the *same* stream
sentinel (FIFO contract).
"""

from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock

import torch

from rtp_llm.models_py.modules.dsv4.fp8.attention import (
    AttentionFP8,
    PrefillMeta,
    PrefillQKV,
)
from rtp_llm.models_py.modules.dsv4.fp8.indexer import IndexerFP8


def _make_attention_stub() -> AttentionFP8:
    layer = AttentionFP8.__new__(AttentionFP8)
    torch.nn.Module.__init__(layer)
    layer.compress_ratio = 4
    # IndexerFP8 isinstance() is enforced inside the orchestrator — we
    # need a real IndexerFP8 instance. Bypass __init__ on it too.
    ind = IndexerFP8.__new__(IndexerFP8)
    torch.nn.Module.__init__(ind)
    ind.start_prefill_nested_compressor = MagicMock(
        name="indexer.start_prefill_nested_compressor"
    )
    ind.forward_with_pending_nested = MagicMock(
        name="indexer.forward_with_pending_nested"
    )
    ind.compressor = MagicMock(name="indexer.compressor")
    layer.indexer = ind
    layer.compressor = MagicMock(name="compressor")
    layer._prefill_write_swa_fp8_paged = MagicMock(name="_prefill_write_swa_fp8_paged")
    layer._forward_prefill_compressed = MagicMock(
        name="_forward_prefill_compressed",
        return_value=torch.zeros(2, 8, dtype=torch.bfloat16),
    )
    return layer


def _make_common(device: torch.device = torch.device("cpu")) -> PrefillMeta:
    cp_ctx = SimpleNamespace(cp_size=2, seq_len_full=4)
    indexer_meta = SimpleNamespace(
        sp_int=0,
        compressor_meta=SimpleNamespace(name="nested_compressor_meta"),
    )
    csa_meta = SimpleNamespace(
        indexer_meta=indexer_meta,
        compressor_meta=SimpleNamespace(name="main_csa_compressor_meta"),
        workspace_meta=SimpleNamespace(name="csa_workspace_meta"),
    )
    return PrefillMeta(
        seqlen=2,
        seqlen_full=4,
        rd=0,
        device=device,
        cp_ctx=cp_ctx,
        cp_on=True,
        freqs_cis=torch.zeros(1, dtype=torch.float32, device=device),
        topk_idxs=torch.zeros(1, dtype=torch.int32, device=device),
        sp_int=0,
        any_cont=False,
        row_seqlens_full=torch.tensor([2], dtype=torch.long, device=device),
        csa_meta=csa_meta,
    )


def _make_qkv(device: torch.device) -> PrefillQKV:
    return PrefillQKV(
        qr=torch.zeros(2, 4, dtype=torch.bfloat16, device=device),
        q=torch.zeros(2, 4, dtype=torch.bfloat16, device=device),
        kv_full=torch.zeros(2, 4, dtype=torch.bfloat16, device=device),
    )


class CSAOverlapOrchestratorTest(unittest.TestCase):
    def setUp(self) -> None:
        self.device = torch.device("cpu")

    def test_csa_orchestrator_call_sequence_and_shared_stream(self) -> None:
        layer = _make_attention_stub()
        stream_sentinel = object()
        layer._get_cp_gather_stream = MagicMock(  # type: ignore[assignment]
            return_value=stream_sentinel
        )

        main_pending = object()
        nested_pending = object()
        raw_topk = torch.zeros(2, 16, dtype=torch.int32, device=self.device)
        layer.compressor.start_prefill.return_value = main_pending
        layer.indexer.start_prefill_nested_compressor.return_value = nested_pending
        layer.indexer.forward_with_pending_nested.return_value = raw_topk

        seq: list = []
        layer.compressor.start_prefill.side_effect = (
            lambda *a, **kw: seq.append(("start_main", a, kw)) or main_pending
        )
        layer.indexer.start_prefill_nested_compressor.side_effect = (
            lambda *a, **kw: seq.append(("start_nested", a, kw)) or nested_pending
        )
        layer._prefill_write_swa_fp8_paged.side_effect = lambda *a, **kw: seq.append(
            ("swa_write", a, kw)
        )

        def fake_indexer_finish(*a, **kw):
            seq.append(("indexer_enter", a, kw))
            kw["before_gather_k"]()
            seq.append(("indexer_finish", a, kw))
            return raw_topk

        layer.indexer.forward_with_pending_nested.side_effect = fake_indexer_finish
        layer.compressor.finish_prefill.side_effect = lambda p: seq.append(
            ("finish_main", p)
        )
        layer.compressor.wait_prefill_gather.side_effect = lambda p: seq.append(
            ("wait_main", p)
        )
        layer._forward_prefill_compressed.side_effect = lambda *a, **kw: seq.append(
            ("compressed", a, kw)
        ) or torch.zeros(2, 8, dtype=torch.bfloat16)

        common = _make_common(self.device)
        qkv = _make_qkv(self.device)
        x = torch.zeros(2, 4, dtype=torch.bfloat16, device=self.device)

        out = layer._forward_prefill_csa_overlapped(x, qkv, common)

        # Exact sequence, including the main-gather wait invoked from inside
        # indexer.forward_with_pending_nested before gather_k/score/topk, while
        # the main pool write remains after indexer topk as in the baseline.
        self.assertEqual(
            [step[0] for step in seq],
            [
                "start_nested",
                "start_main",
                "swa_write",
                "indexer_enter",
                "wait_main",
                "indexer_finish",
                "finish_main",
                "compressed",
            ],
        )

        # Both compressor starts bind the SAME cp_gather_stream — FIFO
        # ordering inside the ProcessGroup requires it.
        self.assertIs(seq[0][2]["cp_gather_stream"], stream_sentinel)
        self.assertIs(seq[1][2]["cp_gather_stream"], stream_sentinel)

        # start_main args: (x, sp_int) + main compressor meta.
        sm_args, sm_kwargs = seq[1][1], seq[1][2]
        self.assertIs(sm_args[0], x)
        self.assertEqual(sm_args[1], common.sp_int)
        self.assertIs(sm_kwargs["meta"], common.csa_meta.compressor_meta)

        # start_nested args: (x, indexer_meta.sp_int) + nested compressor meta.
        sn_args, sn_kwargs = seq[0][1], seq[0][2]
        self.assertIs(sn_args[0], x)
        self.assertEqual(sn_args[1], common.csa_meta.indexer_meta.sp_int)
        self.assertIs(sn_kwargs["meta"], common.csa_meta.indexer_meta.compressor_meta)

        # forward_with_pending_nested gets the nested_pending we returned.
        if_args = seq[3][1]
        if_kwargs = seq[3][2]
        self.assertIs(if_args[0], x)
        self.assertIs(if_args[1], qkv.qr)
        self.assertIs(if_args[2], common.csa_meta.indexer_meta)
        self.assertIs(if_args[3], nested_pending)
        self.assertIn("before_gather_k", if_kwargs)
        self.assertTrue(callable(if_kwargs["before_gather_k"]))

        # wait_main fences only the main gather before indexer returns.
        self.assertIs(seq[4][1], main_pending)

        # finish_main writes the main CSA pool after indexer topk.
        self.assertIs(seq[6][1], main_pending)

        # compressed receives the indexer's topk as cmp_topk_runtime and
        # the skip flag is set so the orchestrator's finish_main is the
        # only compressor write.
        cp_args, cp_kwargs = seq[7][1], seq[7][2]
        self.assertIs(cp_kwargs["cmp_topk_runtime"], raw_topk)
        self.assertIs(cp_kwargs["compressor_meta"], common.csa_meta.compressor_meta)
        self.assertIs(cp_kwargs["workspace_meta"], common.csa_meta.workspace_meta)
        self.assertTrue(cp_kwargs["_skip_compressor_write"])

        # Output threaded through.
        self.assertEqual(tuple(out.shape), (2, 8))

    def test_csa_orchestrator_drains_pending_gathers_on_exception(self) -> None:
        layer = _make_attention_stub()
        stream_sentinel = object()
        layer._get_cp_gather_stream = MagicMock(  # type: ignore[assignment]
            return_value=stream_sentinel
        )

        main_pending = object()
        nested_pending = object()
        layer.compressor.start_prefill.return_value = main_pending
        layer.indexer.start_prefill_nested_compressor.return_value = nested_pending
        layer._prefill_write_swa_fp8_paged.side_effect = RuntimeError("boom")

        common = _make_common(self.device)
        qkv = _make_qkv(self.device)
        x = torch.zeros(2, 4, dtype=torch.bfloat16, device=self.device)

        with self.assertRaisesRegex(RuntimeError, "boom"):
            layer._forward_prefill_csa_overlapped(x, qkv, common)

        layer.compressor.wait_prefill_gather.assert_called_once_with(main_pending)
        layer.indexer.compressor.wait_prefill_gather.assert_called_once_with(
            nested_pending
        )
        layer.indexer.compressor.clear_pool_context.assert_called_once()
        layer.compressor.finish_prefill.assert_not_called()
        layer.indexer.forward_with_pending_nested.assert_not_called()
        layer._forward_prefill_compressed.assert_not_called()


if __name__ == "__main__":
    unittest.main()
