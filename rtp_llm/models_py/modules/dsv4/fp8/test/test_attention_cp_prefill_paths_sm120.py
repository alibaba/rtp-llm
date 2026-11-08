"""SM120 cases split out of test_attention_cp_prefill_paths.py.
These cases live in this sibling file, leaving the original test file
untouched."""

from __future__ import annotations

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
import sys
import unittest
from contextlib import ExitStack
from types import SimpleNamespace
from unittest.mock import MagicMock

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
        side_effect=lambda x, c, *, shared_input_quant=None: seq.append("qkv")
        or _make_qkv()
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


class AttentionCPPrefillDispatchTestSm120(unittest.TestCase):
    """Sibling of ``AttentionCPPrefillDispatchTest`` carrying the SM120 cases."""

    def setUp(self) -> None:
        self.x = torch.zeros(3, 4, dtype=torch.bfloat16)
        self.positions = torch.arange(3, dtype=torch.long)

    def _run_dispatch(
        self,
        *,
        compress_ratio: int,
        common: PrefillMeta,
        shared_input_quant=None,
    ) -> list:
        seq: list = []
        layer = _make_dispatch_layer(compress_ratio, seq)
        layer._prefill_common_setup.side_effect = (  # type: ignore[attr-defined]
            lambda x, p: seq.append("common") or common
        )
        out = layer._forward_prefill(
            self.x, self.positions, shared_input_quant=shared_input_quant
        )
        layer._prefill_compute_qkv.assert_called_once()  # type: ignore[attr-defined]
        call = layer._prefill_compute_qkv.call_args  # type: ignore[attr-defined]
        self.assertIs(call.args[0], self.x)
        self.assertIs(call.args[1], common)
        self.assertEqual(set(call.kwargs), {"shared_input_quant"})
        self.assertIs(call.kwargs["shared_input_quant"], shared_input_quant)
        self.assertEqual(tuple(out.shape), (3, 8))
        return seq

    def test_dispatch_forwards_shared_input_quant_without_repacking(self) -> None:
        shared = (torch.zeros(3, 4, dtype=torch.uint8), torch.ones(3, 1))
        common = _make_common(cp_on=True, device=torch.device("cuda"))
        for ratio in (0, 4, 128):
            with self.subTest(compress_ratio=ratio):
                self._run_dispatch(
                    compress_ratio=ratio,
                    common=common,
                    shared_input_quant=shared,
                )


if __name__ == "__main__":
    import unittest

    unittest.main()
