"""UT: AttentionFP8 CP-prefill execution path selection.

These tests are intentionally CPU/mocked. They lock the Python-side
orchestrator contracts that decide which CUDA-heavy path would run:

* SWA-only layers never enter compressor overlap, even when the CP overlap
  flag is enabled.
* CSA/HCA layers use the overlapped orchestrators only when the feature gate
  says CP+CUDA+env are active; otherwise they follow the baseline path after
  the SWA pool write.
* Deferred SWA ``kv_full`` gathers are always waited before use, and failed
  Q-side compute drains the already-launched gather.
"""

from __future__ import annotations

import os
import sys
import unittest
from contextlib import ExitStack
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch

import rtp_llm.models_py.modules.dsv4.fp8.attention as attention_mod
from rtp_llm.models_py.modules.dsv4.fp8.attention import (
    AttentionFP8,
    PrefillMeta,
    PrefillQKV,
    WorkspaceMeta,
)
from rtp_llm.models_py.modules.dsv4.prefill_workspace import PrefillWorkspace


def _make_common(
    *,
    cp_on: bool,
    device: torch.device,
    cp_size: int = 2,
) -> PrefillMeta:
    cp_ctx = None
    if cp_on:
        cp_ctx = SimpleNamespace(
            cp_size=cp_size,
            cp_rank=0,
            chunk_length=3,
            seq_len_full=6,
            unpad_restore=torch.arange(6, dtype=torch.long),
            unpad_restore_is_prefix=True,
            kv_cache_sharded=False,
        )
    tensor_device = torch.device("cpu")
    # Per-forward prefill scratch is now threaded via ``PrefillMeta.workspace``;
    # the real ``_prefill_compute_qkv`` reads ``common.workspace.prefill_q``.
    # ``_make_qkv_layer`` uses n_heads=1, head_dim=4 → q_dim=4; size for it.
    workspace = PrefillWorkspace(
        tensor_device, q_rows=8, q_dim=4, reserve_cp=False, align_bytes=1
    )
    return PrefillMeta(
        seqlen=3,
        seqlen_full=6 if cp_on else 3,
        rd=0,
        device=device,
        cp_ctx=cp_ctx,
        cp_on=cp_on,
        freqs_cis=torch.zeros(1, dtype=torch.float32, device=tensor_device),
        topk_idxs=torch.zeros(1, dtype=torch.int32, device=tensor_device),
        sp_int=0,
        any_cont=False,
        row_seqlens_full=torch.tensor([3], dtype=torch.long, device=tensor_device),
        workspace=workspace,
    )


def _make_qkv(*, with_pending: bool = False) -> PrefillQKV:
    kv_full = None if with_pending else torch.zeros(3, 2, dtype=torch.bfloat16)
    return PrefillQKV(
        qr=torch.zeros(3, 4, dtype=torch.bfloat16),
        q=torch.zeros(3, 1, 4, dtype=torch.bfloat16),
        kv_full=kv_full,
        kv_full_gather_handle=object() if with_pending else None,
        kv_full_trailing_shape=(2,) if with_pending else None,
        swa_k_local=torch.zeros(3, 4, dtype=torch.bfloat16),
    )


def _make_dispatch_layer(compress_ratio: int, seq: list) -> AttentionFP8:
    layer = AttentionFP8.__new__(AttentionFP8)
    torch.nn.Module.__init__(layer)
    layer.compress_ratio = compress_ratio
    layer.layer_id = 7
    layer._prefill_common_setup = MagicMock(  # type: ignore[assignment]
        side_effect=lambda x, p: seq.append("common") or None
    )
    layer._prefill_compute_qkv = MagicMock(  # type: ignore[assignment]
        side_effect=lambda x, c: seq.append("qkv") or _make_qkv(with_pending=True)
    )
    layer._ensure_prefill_kv_full = MagicMock(  # type: ignore[assignment]
        side_effect=lambda qkv, c: seq.append("ensure")
        or qkv._replace(kv_full=torch.zeros(3, 2, dtype=torch.bfloat16))
    )
    layer._prefill_write_swa_fp8_paged = MagicMock(  # type: ignore[assignment]
        side_effect=lambda c, kv: seq.append("swa_write")
    )
    layer._forward_prefill_swa_only = MagicMock(  # type: ignore[assignment]
        side_effect=lambda qkv, c: seq.append("swa_path")
        or torch.zeros(3, 8, dtype=torch.bfloat16)
    )
    layer._forward_prefill_csa = MagicMock(  # type: ignore[assignment]
        side_effect=lambda x, qkv, c: seq.append("csa_path")
        or torch.zeros(3, 8, dtype=torch.bfloat16)
    )
    layer._forward_prefill_hca = MagicMock(  # type: ignore[assignment]
        side_effect=lambda x, qkv, c: seq.append("hca_path")
        or torch.zeros(3, 8, dtype=torch.bfloat16)
    )
    layer._forward_prefill_csa_overlapped = MagicMock(  # type: ignore[assignment]
        side_effect=lambda x, qkv, c: seq.append("csa_overlap")
        or torch.zeros(3, 8, dtype=torch.bfloat16)
    )
    layer._forward_prefill_hca_overlapped = MagicMock(  # type: ignore[assignment]
        side_effect=lambda x, qkv, c: seq.append("hca_overlap")
        or torch.zeros(3, 8, dtype=torch.bfloat16)
    )
    return layer


class AttentionCPPrefillDispatchTest(unittest.TestCase):
    def setUp(self) -> None:
        self.x = torch.zeros(3, 4, dtype=torch.bfloat16)
        self.positions = torch.arange(3, dtype=torch.long)

    def _run_dispatch(
        self,
        *,
        compress_ratio: int,
        common: PrefillMeta,
        env_value: str,
    ) -> list:
        seq: list = []
        layer = _make_dispatch_layer(compress_ratio, seq)
        layer._prefill_common_setup.side_effect = (  # type: ignore[attr-defined]
            lambda x, p: seq.append("common") or common
        )
        with patch.dict(os.environ, {"DSV4_PREFILL_CP_OVERLAP": env_value}):
            out = layer._forward_prefill(self.x, self.positions)
        self.assertEqual(tuple(out.shape), (3, 8))
        return seq

    def test_swa_only_env_on_stays_on_swa_path(self) -> None:
        common = _make_common(cp_on=True, device=torch.device("cuda"))

        seq = self._run_dispatch(
            compress_ratio=0,
            common=common,
            env_value="1",
        )

        self.assertEqual(seq, ["common", "qkv", "ensure", "swa_write", "swa_path"])

    def test_csa_env_off_uses_baseline_after_swa_write(self) -> None:
        common = _make_common(cp_on=True, device=torch.device("cuda"))

        seq = self._run_dispatch(
            compress_ratio=4,
            common=common,
            env_value="0",
        )

        self.assertEqual(seq, ["common", "qkv", "ensure", "swa_write", "csa_path"])

    def test_csa_env_on_cuda_uses_overlap_orchestrator(self) -> None:
        common = _make_common(cp_on=True, device=torch.device("cuda"))

        seq = self._run_dispatch(
            compress_ratio=4,
            common=common,
            env_value="1",
        )

        self.assertEqual(seq, ["common", "qkv", "csa_overlap"])

    def test_hca_env_on_cuda_uses_overlap_orchestrator(self) -> None:
        common = _make_common(cp_on=True, device=torch.device("cuda"))

        seq = self._run_dispatch(
            compress_ratio=128,
            common=common,
            env_value="1",
        )

        self.assertEqual(seq, ["common", "qkv", "hca_overlap"])

    def test_hca_env_on_cpu_falls_back_to_baseline(self) -> None:
        common = _make_common(cp_on=True, device=torch.device("cpu"))

        seq = self._run_dispatch(
            compress_ratio=128,
            common=common,
            env_value="1",
        )

        self.assertEqual(seq, ["common", "qkv", "ensure", "swa_write", "hca_path"])


class AttentionCPDistributedPrefillGateTest(unittest.TestCase):
    def _make_layer(self, compress_ratio: int = 128) -> AttentionFP8:
        layer = AttentionFP8.__new__(AttentionFP8)
        torch.nn.Module.__init__(layer)
        layer.compress_ratio = compress_ratio
        layer.layer_id = 9
        layer.head_dim = 4
        return layer

    def test_gate_accepts_cp8_cuda_page_rr_off_when_op_available(self) -> None:
        layer = self._make_layer()
        common = _make_common(cp_on=True, cp_size=8, device=torch.device("cuda"))

        with (
            patch.dict(os.environ, {"DSV4_CP_DISTRIBUTED_PREFILL_ATTN": "1"}),
            patch.object(
                attention_mod,
                "_cp_distributed_attention_op_available",
                return_value=True,
            ),
        ):
            self.assertTrue(
                layer._should_use_cp_distributed_prefill_attention(common, object())
            )

    def test_gate_rejects_page_rr_and_non_cp8(self) -> None:
        layer = self._make_layer()
        common_rr = _make_common(cp_on=True, cp_size=8, device=torch.device("cuda"))
        common_rr.cp_ctx.kv_cache_sharded = True
        common_cp4 = _make_common(cp_on=True, cp_size=4, device=torch.device("cuda"))

        with (
            patch.dict(os.environ, {"DSV4_CP_DISTRIBUTED_PREFILL_ATTN": "1"}),
            patch.object(
                attention_mod,
                "_cp_distributed_attention_op_available",
                return_value=True,
            ),
        ):
            self.assertFalse(
                layer._should_use_cp_distributed_prefill_attention(common_rr, object())
            )
            self.assertFalse(
                layer._should_use_cp_distributed_prefill_attention(common_cp4, object())
            )

    def test_compressed_epilogue_enters_distributed_hook_before_workspace(self) -> None:
        layer = self._make_layer(compress_ratio=128)
        common = _make_common(cp_on=True, cp_size=8, device=torch.device("cuda"))
        qkv = _make_qkv()
        expected = torch.ones(3, 8, dtype=torch.bfloat16)
        layer.compressor = MagicMock()
        layer._materialize_prefill_q = MagicMock(return_value=qkv)  # type: ignore[assignment]
        layer._should_use_cp_distributed_prefill_attention = MagicMock(  # type: ignore[assignment]
            return_value=True
        )
        layer._forward_prefill_cp_distributed_attention = MagicMock(  # type: ignore[assignment]
            return_value=expected
        )
        layer._attn_via_workspace = MagicMock(  # type: ignore[assignment]
            side_effect=AssertionError("workspace path should not run")
        )

        out = layer._forward_prefill_compressed(
            torch.zeros(3, 4, dtype=torch.bfloat16),
            qkv,
            common,
            cmp_topk_runtime=None,
            compressor_meta=object(),
            workspace_meta=object(),
            _skip_compressor_write=True,
        )

        self.assertIs(out, expected)
        layer._forward_prefill_cp_distributed_attention.assert_called_once()
        layer._attn_via_workspace.assert_not_called()

    def test_compressed_epilogue_distributed_path_uses_project_only_payload(
        self,
    ) -> None:
        layer = self._make_layer(compress_ratio=128)
        common = _make_common(cp_on=True, cp_size=8, device=torch.device("cuda"))
        qkv = _make_qkv()
        expected = torch.ones(3, 8, dtype=torch.bfloat16)
        payload = (torch.zeros(3, 4), torch.zeros(3, 4))
        compressor = MagicMock()
        compressor.project_prefill_local.return_value = payload
        layer.compressor = compressor
        layer._materialize_prefill_q = MagicMock(return_value=qkv)  # type: ignore[assignment]
        layer._should_use_cp_distributed_prefill_attention = MagicMock(  # type: ignore[assignment]
            return_value=True
        )
        layer._forward_prefill_cp_distributed_attention = MagicMock(  # type: ignore[assignment]
            return_value=expected
        )
        layer._attn_via_workspace = MagicMock(  # type: ignore[assignment]
            side_effect=AssertionError("workspace path should not run")
        )

        x = torch.zeros(3, 4, dtype=torch.bfloat16)
        meta = object()
        wm = object()
        out = layer._forward_prefill_compressed(
            x,
            qkv,
            common,
            cmp_topk_runtime=None,
            compressor_meta=meta,
            workspace_meta=wm,
            _skip_compressor_write=False,
        )

        self.assertIs(out, expected)
        compressor.assert_not_called()
        compressor.project_prefill_local.assert_called_once_with(x)
        layer._forward_prefill_cp_distributed_attention.assert_called_once_with(
            qkv,
            common,
            wm,
            None,
            meta,
            payload,
            None,
        )
        layer._attn_via_workspace.assert_not_called()

    def test_csa_distributed_path_uses_indexer_project_only_payload(self) -> None:
        layer = self._make_layer(compress_ratio=4)
        common = _make_common(cp_on=True, cp_size=8, device=torch.device("cuda"))
        indexer_meta = object()
        compressor_meta = object()
        wm = object()
        common = common._replace(
            csa_meta=SimpleNamespace(
                indexer_meta=indexer_meta,
                compressor_meta=compressor_meta,
                workspace_meta=wm,
            )
        )
        qkv = _make_qkv()
        expected = torch.ones(3, 8, dtype=torch.bfloat16)
        payload = object()
        from rtp_llm.models_py.modules.dsv4.fp8.indexer import IndexerFP8

        indexer = IndexerFP8.__new__(IndexerFP8)
        torch.nn.Module.__init__(indexer)
        indexer.project_prefill_local = MagicMock(return_value=payload)  # type: ignore[method-assign]
        indexer.forward = MagicMock(  # type: ignore[method-assign]
            side_effect=AssertionError("standalone indexer forward should not run")
        )
        layer.indexer = indexer
        layer._should_use_cp_distributed_prefill_attention = MagicMock(  # type: ignore[assignment]
            return_value=True
        )
        layer._forward_prefill_compressed = MagicMock(return_value=expected)  # type: ignore[assignment]

        x = torch.zeros(3, 4, dtype=torch.bfloat16)
        out = layer._forward_prefill_csa(x, qkv, common)

        self.assertIs(out, expected)
        indexer.forward.assert_not_called()
        indexer.project_prefill_local.assert_called_once_with(x, qkv.qr, indexer_meta)
        layer._forward_prefill_compressed.assert_called_once_with(
            x,
            qkv,
            common,
            cmp_topk_runtime=None,
            compressor_meta=compressor_meta,
            workspace_meta=wm,
            indexer_payload=payload,
        )

    def test_forward_prefill_distributed_path_skips_standalone_swa_write(self) -> None:
        common = _make_common(cp_on=True, cp_size=8, device=torch.device("cuda"))
        common = common._replace(hca_meta=SimpleNamespace(workspace_meta=object()))

        seq: list = []
        layer = _make_dispatch_layer(128, seq)
        layer._prefill_common_setup.side_effect = (  # type: ignore[attr-defined]
            lambda x, p: seq.append("common") or common
        )
        with (
            patch.dict(
                os.environ,
                {
                    "DSV4_CP_DISTRIBUTED_PREFILL_ATTN": "1",
                    "DSV4_PREFILL_CP_OVERLAP": "1",
                },
            ),
            patch.object(
                attention_mod,
                "_cp_distributed_attention_op_available",
                return_value=True,
            ),
        ):
            out = layer._forward_prefill(
                torch.zeros(3, 4, dtype=torch.bfloat16),
                torch.arange(3, dtype=torch.long),
            )

        self.assertEqual(tuple(out.shape), (3, 8))
        self.assertEqual(seq, ["common", "qkv", "hca_path"])
        layer._prefill_write_swa_fp8_paged.assert_not_called()
        layer._forward_prefill_hca_overlapped.assert_not_called()

    def test_forward_prefill_csa_distributed_path_skips_standalone_swa_write(
        self,
    ) -> None:
        common = _make_common(cp_on=True, cp_size=8, device=torch.device("cuda"))
        common = common._replace(csa_meta=SimpleNamespace(workspace_meta=object()))

        seq: list = []
        layer = _make_dispatch_layer(4, seq)
        layer._prefill_common_setup.side_effect = (  # type: ignore[attr-defined]
            lambda x, p: seq.append("common") or common
        )
        with (
            patch.dict(
                os.environ,
                {
                    "DSV4_CP_DISTRIBUTED_PREFILL_ATTN": "1",
                    "DSV4_PREFILL_CP_OVERLAP": "1",
                },
            ),
            patch.object(
                attention_mod,
                "_cp_distributed_attention_op_available",
                return_value=True,
            ),
        ):
            out = layer._forward_prefill(
                torch.zeros(3, 4, dtype=torch.bfloat16),
                torch.arange(3, dtype=torch.long),
            )

        self.assertEqual(tuple(out.shape), (3, 8))
        self.assertEqual(seq, ["common", "qkv", "csa_path"])
        layer._prefill_write_swa_fp8_paged.assert_not_called()
        layer._forward_prefill_csa_overlapped.assert_not_called()

    def test_distributed_swa_slot_mapping_uses_rank_major_padded_order(self) -> None:
        layer = self._make_layer(compress_ratio=128)
        common = _make_common(cp_on=True, cp_size=2, device=torch.device("cpu"))
        common.cp_ctx.chunk_length = 3
        common.cp_ctx.unpad_restore = torch.tensor([0, 2, 3, 5], dtype=torch.long)
        common = common._replace(
            swa_meta=SimpleNamespace(
                slot_mapping=torch.tensor([10, 11, 12, 13], dtype=torch.long)
            )
        )

        got = layer._build_cp_distributed_swa_slot_mapping(common)

        self.assertEqual(got.tolist(), [10, -1, 11, 12, -1, 13])

    def test_distributed_swa_kwargs_use_local_k_pool_and_padded_slots(self) -> None:
        layer = self._make_layer(compress_ratio=128)
        common = _make_common(cp_on=True, cp_size=2, device=torch.device("cpu"))
        common.cp_ctx.chunk_length = 3
        common.cp_ctx.unpad_restore = torch.tensor([0, 2, 3, 5], dtype=torch.long)
        common = common._replace(
            swa_meta=SimpleNamespace(
                slot_mapping=torch.tensor([20, 21, 22, 23], dtype=torch.long)
            )
        )
        pool = torch.empty(2, 8, 584, dtype=torch.uint8)
        layer._pool_view_3d_fp8 = MagicMock(return_value=pool)  # type: ignore[assignment]
        qkv = PrefillQKV(
            qr=torch.zeros(3, 4, dtype=torch.bfloat16),
            q=torch.zeros(3, 1, 4, dtype=torch.bfloat16),
            kv_full=torch.zeros(4, 4, dtype=torch.bfloat16),
            swa_k_local=torch.arange(12, dtype=torch.float32).view(3, 4),
        )

        kwargs = layer._build_cp_distributed_swa_kwargs(qkv, common)

        self.assertIs(kwargs["swa_k_cache"], pool)
        self.assertEqual(tuple(kwargs["swa_k"].shape), (3, 4))
        self.assertEqual(kwargs["swa_k"].dtype, torch.bfloat16)
        self.assertEqual(kwargs["swa_slot_mapping"].tolist(), [20, -1, 21, 22, -1, 23])

    def test_distributed_buffer_preflight_rejects_uninitialized_dist(self) -> None:
        layer = self._make_layer(compress_ratio=128)
        common = _make_common(cp_on=True, cp_size=8, device=torch.device("cuda"))
        qkv = PrefillQKV(
            qr=torch.zeros(3, 4, dtype=torch.bfloat16),
            q=torch.zeros(3, 1, 4, dtype=torch.bfloat16),
            kv_full=torch.zeros(6, 2, 4, dtype=torch.bfloat16),
            swa_k_local=torch.zeros(3, 8, dtype=torch.bfloat16),
        )

        with (
            patch.object(torch.distributed, "is_available", return_value=True),
            patch.object(torch.distributed, "is_initialized", return_value=False),
        ):
            with self.assertRaisesRegex(RuntimeError, "initialized torch.distributed"):
                layer._ensure_cp_distributed_attention_buffer(qkv, common)

    def test_distributed_buffer_preflight_builds_attention_buffer(self) -> None:
        layer = self._make_layer(compress_ratio=128)
        common = _make_common(cp_on=True, cp_size=8, device=torch.device("cuda"))
        common = common._replace(batch_size=2)
        qkv = PrefillQKV(
            qr=torch.zeros(3, 4, dtype=torch.bfloat16),
            q=torch.zeros(3, 1, 4, dtype=torch.bfloat16),
            kv_full=torch.zeros(6, 2, 4, dtype=torch.bfloat16),
            swa_k_local=torch.zeros(3, 8, dtype=torch.bfloat16),
        )
        captured = {}
        expected = object()

        def fake_spec(**kwargs):
            captured["spec_kwargs"] = kwargs
            return "spec"

        def fake_get(**kwargs):
            captured["buffer_kwargs"] = kwargs
            return expected

        with (
            patch.object(torch.distributed, "is_available", return_value=True),
            patch.object(torch.distributed, "is_initialized", return_value=True),
            patch.object(torch.distributed, "group", SimpleNamespace(WORLD=object())),
            patch.object(
                attention_mod, "build_dsv4_cp_attention_buffer_spec", fake_spec
            ),
            patch.object(
                attention_mod, "get_or_create_dsv4_cp_attention_buffer", fake_get
            ),
        ):
            got = layer._ensure_cp_distributed_attention_buffer(qkv, common)

        self.assertIs(got, expected)
        self.assertEqual(
            captured["spec_kwargs"],
            {
                "cp_size": 8,
                "actual_tokens_per_rank": 3,
                "batch_size": 2,
                "swa_bytes_per_token": 16,
                "page_rr": False,
            },
        )
        self.assertEqual(captured["buffer_kwargs"]["cp_rank"], 0)
        self.assertEqual(captured["buffer_kwargs"]["spec"], "spec")

    def test_distributed_buffer_uses_cp_chunk_length_for_capacity_key(self) -> None:
        layer = self._make_layer(compress_ratio=128)
        common = _make_common(cp_on=True, cp_size=8, device=torch.device("cuda"))
        common.cp_ctx.chunk_length = 8
        qkv = PrefillQKV(
            qr=torch.zeros(3, 4, dtype=torch.bfloat16),
            q=torch.zeros(3, 1, 4, dtype=torch.bfloat16),
            kv_full=torch.zeros(6, 2, 4, dtype=torch.bfloat16),
            swa_k_local=torch.zeros(8, 8, dtype=torch.bfloat16),
        )
        captured = {}

        def fake_spec(**kwargs):
            captured.update(kwargs)
            return "spec"

        with (
            patch.object(torch.distributed, "is_available", return_value=True),
            patch.object(torch.distributed, "is_initialized", return_value=True),
            patch.object(torch.distributed, "group", SimpleNamespace(WORLD=object())),
            patch.object(
                attention_mod, "build_dsv4_cp_attention_buffer_spec", fake_spec
            ),
            patch.object(
                attention_mod,
                "get_or_create_dsv4_cp_attention_buffer",
                return_value=object(),
            ),
        ):
            layer._ensure_cp_distributed_attention_buffer(qkv, common)

        self.assertEqual(captured["actual_tokens_per_rank"], 8)

    def test_distributed_buffer_preflight_rejects_page_rr_before_allocation(
        self,
    ) -> None:
        layer = self._make_layer(compress_ratio=128)
        common = _make_common(cp_on=True, cp_size=8, device=torch.device("cuda"))
        common.cp_ctx.kv_cache_sharded = True
        qkv = PrefillQKV(
            qr=torch.zeros(3, 4, dtype=torch.bfloat16),
            q=torch.zeros(3, 1, 4, dtype=torch.bfloat16),
            kv_full=torch.zeros(6, 2, 4, dtype=torch.bfloat16),
            swa_k_local=torch.zeros(3, 8, dtype=torch.bfloat16),
        )

        with (
            patch.object(torch.distributed, "is_available", return_value=True),
            patch.object(torch.distributed, "is_initialized", return_value=True),
            patch.object(torch.distributed, "group", SimpleNamespace(WORLD=object())),
            patch.object(
                attention_mod,
                "get_or_create_dsv4_cp_attention_buffer",
                side_effect=AssertionError("buffer allocation should not happen"),
            ),
        ):
            with self.assertRaisesRegex(ValueError, "page/RR"):
                layer._ensure_cp_distributed_attention_buffer(qkv, common)


class AttentionSwaAsyncGatherTest(unittest.TestCase):
    def test_ensure_prefill_kv_full_waits_pending_handle_once(self) -> None:
        layer = AttentionFP8.__new__(AttentionFP8)
        torch.nn.Module.__init__(layer)
        common = _make_common(cp_on=True, device=torch.device("cuda"))
        handle = object()
        qkv = PrefillQKV(
            qr=torch.zeros(3, 4, dtype=torch.bfloat16),
            q=torch.zeros(3, 1, 4, dtype=torch.bfloat16),
            kv_full=None,
            kv_full_gather_handle=handle,
            kv_full_trailing_shape=(2, 3),
            swa_k_local=torch.zeros(3, 6, dtype=torch.bfloat16),
        )
        gathered = torch.arange(36, dtype=torch.float32).view(6, 6)

        with patch.object(
            attention_mod,
            "cp_wait_gather_full",
            return_value=gathered,
        ) as wait:
            out = layer._ensure_prefill_kv_full(qkv, common)

        wait.assert_called_once_with(handle)
        self.assertEqual(tuple(out.kv_full.shape), (6, 2, 3))
        self.assertIsNone(out.kv_full_gather_handle)
        self.assertIsNone(out.kv_full_trailing_shape)

    def test_prefill_compute_qkv_starts_swa_gather_before_q_compute(self) -> None:
        seq: list = []
        layer = self._make_qkv_layer(seq)
        common = _make_common(cp_on=True, device=torch.device("cuda"))
        handle = object()
        stream = object()
        layer._get_swa_cp_gather_stream = MagicMock(  # type: ignore[assignment]
            return_value=stream
        )

        def fake_gather(local_2d, cp_ctx, stream=None, **kwargs):
            seq.append(("gather_start", tuple(local_2d.shape), stream))
            return handle

        with (
            patch.dict(os.environ, {"DSV4_PREFILL_CP_OVERLAP": "1"}),
            patch.object(attention_mod, "cp_all_gather_full_async", fake_gather),
            patch.object(attention_mod, "fused_rmsnorm_rope", lambda t, *a, **k: t),
        ):
            qkv = layer._prefill_compute_qkv(
                torch.zeros(3, 4, dtype=torch.bfloat16),
                common,
            )

        self.assertEqual(seq[0], "lin_wkv")
        self.assertEqual(seq[1], ("gather_start", (3, 6), stream))
        self.assertEqual(seq[2], "lin_wq_a")
        # q_lora_b + RoPE are deferred to _materialize_prefill_q so the big Q
        # buffer can reuse the union workspace storage; not run here.
        self.assertNotIn("lin_wq_b", seq)
        self.assertIsNone(qkv.q)
        self.assertIs(qkv.kv_full_gather_handle, handle)
        self.assertIsNone(qkv.kv_full)
        self.assertEqual(qkv.kv_full_trailing_shape, (2, 3))
        self.assertIsNotNone(qkv.swa_k_local)
        self.assertEqual(tuple(qkv.swa_k_local.shape), (3, 6))

    def test_prefill_compute_qkv_waits_swa_gather_if_q_compute_raises(self) -> None:
        seq: list = []
        layer = self._make_qkv_layer(seq, fail_q=True)
        common = _make_common(cp_on=True, device=torch.device("cuda"))
        handle = object()
        layer._get_swa_cp_gather_stream = MagicMock(  # type: ignore[assignment]
            return_value=object()
        )

        with (
            patch.dict(os.environ, {"DSV4_PREFILL_CP_OVERLAP": "1"}),
            patch.object(
                attention_mod,
                "cp_all_gather_full_async",
                lambda *a, **k: seq.append("gather_start") or handle,
            ),
            patch.object(
                attention_mod,
                "cp_wait_gather_full",
                lambda h: seq.append(("wait", h))
                or torch.zeros(6, 6, dtype=torch.bfloat16),
            ),
            patch.object(attention_mod, "fused_rmsnorm_rope", lambda t, *a, **k: t),
        ):
            with self.assertRaisesRegex(RuntimeError, "q failed"):
                layer._prefill_compute_qkv(
                    torch.zeros(3, 4, dtype=torch.bfloat16),
                    common,
                )

        self.assertIn("gather_start", seq)
        self.assertIn(("wait", handle), seq)

    def test_materialize_prefill_q_reuses_wq_b_output_for_rope(self) -> None:
        # The deferred q_lora_b + RoPE now live in _materialize_prefill_q, which
        # writes wq_b straight into the workspace Q slice and RoPEs it in-place.
        seq: list = []
        layer = self._make_qkv_layer(seq)
        common = _make_common(cp_on=False, device=torch.device("cuda"))
        seen: dict[str, torch.Tensor] = {}

        def fake_fused(t, weight, *args, **kwargs):
            if weight is None:
                seen["q_in"] = t
                seen["q_out"] = kwargs["out"]
                self.assertIs(kwargs["out"], t)
                return kwargs["out"]
            self.assertNotIn("out", kwargs)
            return t

        with patch.object(attention_mod, "fused_rmsnorm_rope", fake_fused):
            qkv = layer._prefill_compute_qkv(
                torch.zeros(3, 4, dtype=torch.bfloat16),
                common,
            )
            # _prefill_compute_qkv runs q_lora_a but defers q_lora_b + RoPE.
            self.assertIsNone(qkv.q)
            self.assertNotIn("lin_wq_b", seq)
            self.assertNotIn("q_out", seen)
            self.assertIsNotNone(qkv.swa_k_local)
            self.assertEqual(tuple(qkv.swa_k_local.shape), (3, 6))
            qkv = layer._materialize_prefill_q(qkv, common)

        self.assertEqual(seq[0], "lin_wq_a")
        self.assertEqual(seq[-1], "lin_wq_b")
        self.assertIn("q_out", seen)
        self.assertEqual(qkv.q.data_ptr(), seen["q_out"].data_ptr())

    def test_prefill_workspace_q_rejects_overflow(self) -> None:
        ws = PrefillWorkspace(
            torch.device("cpu"), q_rows=2, q_dim=4, reserve_cp=False, align_bytes=1
        )

        with self.assertRaisesRegex(AssertionError, "prefill_q overflow: num_tokens=3"):
            ws.prefill_q(3)

    def _make_qkv_layer(self, seq: list, fail_q: bool = False) -> AttentionFP8:
        layer = AttentionFP8.__new__(AttentionFP8)
        torch.nn.Module.__init__(layer)
        layer.compress_ratio = 0
        layer.layer_id = 3
        layer.rd = 0
        layer.n_heads = 1
        layer.head_dim = 4
        layer.eps = 1e-6
        layer.q_norm = object()
        layer.kv_norm = object()
        layer.wq_a = object()
        layer.wq_b = object()
        layer.wkv = object()
        layer._rmsnorm_weighted = lambda t, w: t  # type: ignore[assignment]

        def fake_lin(op, x, out=None):
            T = int(x.size(1))
            if op is layer.wkv:
                seq.append("lin_wkv")
                self.assertIsNone(out)
                return torch.zeros(1, T, 2, 3, dtype=torch.bfloat16)
            if op is layer.wq_a:
                seq.append("lin_wq_a")
                self.assertIsNone(out)
                if fail_q:
                    raise RuntimeError("q failed")
                return torch.zeros(1, T, 6, dtype=torch.bfloat16)
            if op is layer.wq_b:
                seq.append("lin_wq_b")
                self.assertIsNotNone(out)
                out.zero_()
                return out
            raise AssertionError("unexpected linear op")

        layer._lin = fake_lin  # type: ignore[assignment]
        return layer


class AttentionRawQMergeWorkspaceTest(unittest.TestCase):
    def test_raw_q_merge_cold_prefill_skips_swa_cache_read(self) -> None:
        layer = AttentionFP8.__new__(AttentionFP8)
        torch.nn.Module.__init__(layer)
        layer.layer_id = 2
        layer.compress_ratio = 4
        layer.head_dim = 2
        layer.n_heads = 1
        layer.window_size = 3
        layer.softmax_scale = 1.0
        layer.attn_sink = None

        device = torch.device("cpu")
        cp_ctx = SimpleNamespace(
            cp_size=1,
            cp_rank=0,
            seq_len_full=3,
            prefix_length=0,
            prefix_lengths=torch.tensor([0], dtype=torch.long),
            input_lengths_global=torch.tensor([3], dtype=torch.int32),
            req_id_per_token=torch.zeros(3, dtype=torch.long),
            cu_seqlens_global=torch.tensor([0, 3], dtype=torch.long),
            global_positions=torch.arange(3, dtype=torch.long),
            kv_cache_sharded=True,
        )
        common = PrefillMeta(
            seqlen=3,
            seqlen_full=3,
            rd=0,
            device=device,
            cp_ctx=cp_ctx,
            cp_on=True,
            freqs_cis=torch.zeros(1),
            topk_idxs=torch.zeros(1, dtype=torch.int32),
            sp_int=0,
            any_cont=False,
            row_seqlens_full=torch.tensor([3], dtype=torch.long),
            batch_size=1,
            input_lengths=torch.tensor([3], dtype=torch.int32),
            prefix_lengths=torch.tensor([0], dtype=torch.long),
        )
        qkv = PrefillQKV(
            qr=torch.zeros(3, 2, dtype=torch.bfloat16),
            q=torch.zeros(3, 1, 2, dtype=torch.bfloat16),
            kv_full=torch.arange(6, dtype=torch.float32).view(3, 2),
            swa_k_local=torch.zeros(3, 2, dtype=torch.bfloat16),
        )
        wm = WorkspaceMeta(
            M=3,
            N=0,
            swa_eb=1,
            cmp_eb=1,
            swa_bt_int32=torch.zeros(1, 1, dtype=torch.int32),
            cmp_bt_int32=torch.zeros(1, 1, dtype=torch.int32),
            swa_seq_lens=torch.tensor([3], dtype=torch.int32),
            cmp_seq_lens=torch.tensor([0], dtype=torch.int32),
            swa_gather_lens=torch.tensor([3], dtype=torch.int32),
            swa_cache_seq_lens=torch.tensor([0], dtype=torch.int32),
            swa_cache_gather_lens=torch.tensor([0], dtype=torch.int32),
            qsl=torch.tensor([0, 3], dtype=torch.int32),
            dense_cmp_topk=torch.empty(3, 0, dtype=torch.int32),
            new_k_slot_in_flat=torch.arange(3, dtype=torch.long),
            cmp_reader=None,
            use_cp_raw_q_merge=True,
        )

        dequant_calls = []

        def fake_flash_mla_sparse_fwd(q, kv, indices, sm_scale, attn_sink, topk_length):
            return (
                q.to(torch.bfloat16),
                None,
                torch.zeros(q.shape[0], q.shape[1], dtype=torch.float32),
            )

        with ExitStack() as stack:
            stack.enter_context(
                patch.dict(
                    sys.modules,
                    {
                        "flash_mla": SimpleNamespace(
                            flash_mla_sparse_fwd=fake_flash_mla_sparse_fwd
                        )
                    },
                )
            )
            from rtp_llm.models_py.distributed import collective_torch
            from rtp_llm.models_py.modules.dsv4.fp8 import _swa_dequant_triton

            stack.enter_context(
                patch.object(
                    _swa_dequant_triton,
                    "dequantize_and_gather_k_cache",
                    lambda **kwargs: dequant_calls.append(kwargs),
                )
            )
            stack.enter_context(
                patch.object(attention_mod, "cp_all_gather_full_varlen", lambda t, c: t)
            )
            stack.enter_context(
                patch.object(collective_torch, "all_gather", lambda t, group=None: t)
            )

            out = layer._attn_via_workspace_cp_raw_q_merge(
                qkv=qkv,
                common=common,
                workspace_meta=wm,
                cmp_topk_runtime=None,
                cmp_pool_3d=torch.empty(1, 1, 1, dtype=torch.uint8),
                swa_pool_3d=torch.empty(1, 1, 1, dtype=torch.uint8),
            )

        self.assertEqual(tuple(out.shape), (1, 3, 1, 2))
        self.assertEqual(dequant_calls, [])


if __name__ == "__main__":
    unittest.main()
