"""UT: AttentionFP8 CP-prefill execution path selection.

These tests are intentionally CPU/mocked. They lock the Python-side
orchestrator contracts that decide which CUDA-heavy path would run:

* SWA-only layers never enter compressor overlap, even when the CP overlap
  flag is enabled.
* CSA/HCA layers use the overlapped orchestrators only when the feature gate
  says CP+CUDA+env are active; otherwise they follow the baseline path after
  the SWA pool write.
* Current-layer SWA ``kv_full`` is not part of the overlap feature; it stays on
  the synchronous gather path so the returned tensor cannot alias deferred Q.
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
            chunk_length=3,
            seq_len_full=6,
            unpad_restore_is_prefix=True,
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


def _make_qkv() -> PrefillQKV:
    return PrefillQKV(
        qr=torch.zeros(3, 4, dtype=torch.bfloat16),
        q=torch.zeros(3, 1, 4, dtype=torch.bfloat16),
        kv_full=torch.zeros(3, 2, dtype=torch.bfloat16),
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
        side_effect=lambda x, c: seq.append("qkv") or _make_qkv()
    )
    layer._ensure_prefill_kv_full = MagicMock(  # type: ignore[assignment]
        side_effect=lambda qkv, c: seq.append("ensure") or qkv
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


class AttentionSwaAsyncGatherTest(unittest.TestCase):
    def test_prefill_compute_qkv_uses_sync_current_swa_kv_full(self) -> None:
        seq: list = []
        layer = self._make_qkv_layer(seq)
        common = _make_common(cp_on=True, device=torch.device("cuda"))

        with (
            patch.dict(os.environ, {"DSV4_PREFILL_CP_OVERLAP": "1"}),
            patch.object(
                attention_mod, "cp_all_gather_full_varlen", lambda t, *a, **k: t
            ),
            patch.object(attention_mod, "fused_rmsnorm_rope", lambda t, *a, **k: t),
        ):
            qkv = layer._prefill_compute_qkv(
                torch.zeros(3, 4, dtype=torch.bfloat16),
                common,
            )

        self.assertEqual(seq, ["lin_wq_a", "lin_wkv"])
        # q_lora_b + RoPE are deferred to _materialize_prefill_q; not run here.
        self.assertNotIn("lin_wq_b", seq)
        self.assertIsNone(qkv.q)
        self.assertIsNotNone(qkv.kv_full)

    def test_prefill_compute_qkv_reuses_shared_input_quant(self) -> None:
        seq: list = []
        layer = self._make_qkv_layer(seq)
        common = _make_common(cp_on=False, device=torch.device("cuda"))

        class FakeQuantLinear:
            def __init__(self, name: str, out_features: int):
                self.name = name
                self.K = 4
                self.scale_ue8m0 = True
                self.out_features = out_features

            def quantize_input(self, x):
                seq.append(f"quant_{self.name}")
                return ("fp8", "scale")

            def forward_quantized(self, q, s):
                seq.append(f"gemm_{self.name}_{q}_{s}")
                return torch.zeros(3, self.out_features, dtype=torch.bfloat16)

        layer.wq_a = FakeQuantLinear("wq_a", 6)
        layer.wkv = FakeQuantLinear("wkv", 6)

        with patch.object(attention_mod, "fused_rmsnorm_rope", lambda t, *a, **k: t):
            qkv = layer._prefill_compute_qkv(
                torch.zeros(3, 4, dtype=torch.bfloat16),
                common,
            )

        self.assertEqual(
            seq,
            [
                "quant_wq_a",
                "gemm_wq_a_fp8_scale",
                "gemm_wkv_fp8_scale",
            ],
        )
        self.assertIsNone(qkv.q)
        self.assertEqual(tuple(qkv.qr.shape), (3, 6))
        self.assertEqual(tuple(qkv.kv_full.shape), (3, 6))

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
