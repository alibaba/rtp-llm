"""Contract and SM103 GPU parity tests for Block, Mega adapters, and decode_transition."""

from __future__ import annotations

import os
import sys
import types
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch
import torch.nn as nn


def _install_cpu_native_stubs_if_needed() -> None:
    """Allow CPU-pool contract tests to import GPU Python modules when ``libcuda.so.1`` is absent."""
    if os.environ.get("GPU_COUNT", "0") != "0":
        return
    try:
        import librtp_compute_ops  # noqa: F401
    except ImportError:
        if "librtp_compute_ops" not in sys.modules:
            pkg = types.ModuleType("librtp_compute_ops")
            sub = types.ModuleType("librtp_compute_ops.rtp_llm_ops")
            pkg.rtp_llm_ops = sub
            sys.modules["librtp_compute_ops"] = pkg
            sys.modules["librtp_compute_ops.rtp_llm_ops"] = sub
    try:
        import deep_gemm  # noqa: F401
    except ImportError:
        if "deep_gemm" not in sys.modules:
            dg = types.ModuleType("deep_gemm")
            dg_utils = types.ModuleType("deep_gemm.utils")
            dg_layout = types.ModuleType("deep_gemm.utils.layout")
            dg_layout.get_mn_major_tma_aligned_packed_ue8m0_tensor = (
                lambda *a, **k: None
            )
            dg.utils = dg_utils
            dg_utils.layout = dg_layout
            sys.modules["deep_gemm"] = dg
            sys.modules["deep_gemm.utils"] = dg_utils
            sys.modules["deep_gemm.utils.layout"] = dg_layout


_install_cpu_native_stubs_if_needed()

from rtp_llm.models_py.modules.base.cuda.norm import RMSNorm  # noqa: E402
from rtp_llm.models_py.modules.dsv4.block import Block  # noqa: E402
from rtp_llm.models_py.modules.dsv4.decode.forward import forward_layers  # noqa: E402
from rtp_llm.models_py.modules.dsv4.fp8.decode.mega_csa_adapter import (  # noqa: E402
    MegaCSAAdapter,
)
from rtp_llm.models_py.modules.dsv4.fp8.decode.mega_csa_weights import (  # noqa: E402
    GEOMETRY_BY_DIM,
    HC,
)
from rtp_llm.models_py.modules.dsv4.fp8.decode.mega_hca_adapter import (  # noqa: E402
    MegaHCAAdapter,
)
from rtp_llm.models_py.modules.dsv4.hc import mega_mhc as _mega_mhc  # noqa: E402
from rtp_llm.models_py.modules.dsv4.hc.decode_transition import (  # noqa: E402
    PendingDecodePost,
    can_defer_decode_post,
    forward_decode_layer,
)
from rtp_llm.models_py.modules.dsv4.hc.tilelang_impl import (  # noqa: E402
    TileLangHCUnit,
)


def _make_tilelang_unit(
    *,
    dim: int = 4096,
    device: str | torch.device = "cpu",
    seed: int = 0,
    layer_id: int = 0,
    name: str = "attn_hc",
) -> TileLangHCUnit:
    gen = torch.Generator(device="cpu").manual_seed(seed)
    fn = (torch.randn((24, 4 * dim), generator=gen, dtype=torch.float32) * 0.02).to(
        device
    )
    base = (torch.randn((24,), generator=gen, dtype=torch.float32) * 0.05).to(device)
    scale = (torch.rand((3,), generator=gen, dtype=torch.float32) * 0.5 + 0.5).to(
        device
    )
    return TileLangHCUnit(
        fn,
        base,
        scale,
        dim=dim,
        hc_mult=4,
        hc_sinkhorn_iters=20,
        norm_eps=1.0e-6,
        hc_eps=1.0e-6,
        layer_id=layer_id,
        name=name,
    )


def _make_block(
    *,
    dim: int = 4096,
    device: str | torch.device = "cpu",
    seed: int = 0,
    layer_id: int = 0,
) -> Block:
    block = nn.Module.__new__(Block)
    nn.Module.__init__(block)
    block.layer_id = layer_id
    block.dim = dim
    block.hc_mult = 4
    block.attn_hc = _make_tilelang_unit(
        dim=dim, device=device, seed=seed, layer_id=layer_id, name="attn_hc"
    )
    block.ffn_hc = _make_tilelang_unit(
        dim=dim, device=device, seed=seed + 10, layer_id=layer_id, name="ffn_hc"
    )
    gen = torch.Generator(device="cpu").manual_seed(seed + 20)
    block.attn_norm = RMSNorm(
        (
            torch.ones((dim,), dtype=torch.bfloat16)
            + torch.randn((dim,), generator=gen, dtype=torch.bfloat16) * 0.1
        ).to(device),
        eps=1.0e-6,
    )
    block.ffn_norm = RMSNorm(
        (
            torch.ones((dim,), dtype=torch.bfloat16)
            + torch.randn((dim,), generator=gen, dtype=torch.bfloat16) * 0.1
        ).to(device),
        eps=1.0e-6,
    )
    block._mega_csa_adapter = None
    block._mega_hca_adapter = None
    block._cp_sync_after_attn_done = False

    # Deterministic lightweight attn & ffn that work cleanly inside CUDA graphs and CPU tests.
    attn_mod = nn.Module()
    attn_mod.forward_decode = lambda x_pre, meta, kv_cache=None: (x_pre * 0.75 + 0.05)
    attn_mod.forward = (
        lambda x_pre, positions, kv_cache=None, block_tables_by_type=None: (
            x_pre * 0.75 + 0.05
        )
    )
    block.attn = attn_mod

    ffn_mod = nn.Module()
    ffn_mod.strategy_name = "fused"
    ffn_mod.forward = lambda x_pre, input_ids, is_decode_forward=False: (
        x_pre * 0.5 - 0.025
    )
    block.ffn = ffn_mod
    return block


class _StubMegaRuntime:
    def run_csa(self, *args):
        return None

    def run_hca(self, *args):
        return None


class _FakeMegaAdapter:
    def __init__(self, *, supports: bool = True) -> None:
        self._supports = supports
        self.calls: list[str] = []

    def supports_decode_shape(self, hidden, metadata) -> bool:
        return self._supports

    def forward_attention_sublayer(self, block, hidden, metadata, *, kv_cache=None):
        self.calls.append("full")
        projected = hidden[:, :, 0, :] * 0.5
        post = torch.ones(
            *hidden.shape[:2], 4, 1, dtype=torch.float32, device=hidden.device
        )
        comb = (
            torch.eye(4, dtype=torch.float32, device=hidden.device)
            .view(1, 1, 4, 4)
            .expand(*hidden.shape[:2], 4, 4)
            .contiguous()
        )
        return block.attn_hc.post(projected, hidden, post, comb)

    def forward_attention_sublayer_outputs(
        self, block, hidden, metadata, *, kv_cache=None
    ):
        self.calls.append("outputs")
        projected = hidden[:, :, 0, :] * 0.5
        post = torch.ones(
            *hidden.shape[:2], 4, 1, dtype=torch.float32, device=hidden.device
        )
        comb = (
            torch.eye(4, dtype=torch.float32, device=hidden.device)
            .view(1, 1, 4, 4)
            .expand(*hidden.shape[:2], 4, 4)
            .contiguous()
        )
        return projected, hidden, post, comb


class TestMegaAdapterOutputsContract(unittest.TestCase):
    def test_csa_and_hca_forward_bound_outputs_and_post_wrap(self) -> None:
        geometry = GEOMETRY_BY_DIM[4096]
        dim = geometry.dim
        for adapter_cls in (MegaCSAAdapter, MegaHCAAdapter):
            adapter = adapter_cls.__new__(adapter_cls)
            adapter._geometry = geometry

            orig_kv = object()
            orig_tables = {"old": 1}
            block = SimpleNamespace(
                layer_id=0,
                attn_hc=MagicMock(),
                attn=SimpleNamespace(
                    _ensure_freqs_cis_bound=MagicMock(),
                    _kv_cache=orig_kv,
                    _block_tables_by_type=orig_tables,
                ),
            )
            block.attn_hc.post.return_value = "POST_DONE"

            # Simulate contiguous CUDA bfloat16 hidden tensor cleanly without requiring GPU memory.
            hidden = MagicMock(spec=torch.Tensor)
            hidden.dim.return_value = 4
            hidden.shape = (2, 1, HC, dim)
            hidden.dtype = torch.bfloat16
            hidden.is_cuda = True
            hidden.is_contiguous.return_value = True
            hidden.device = torch.device("cuda:0")

            proj = torch.ones(2, 1, dim, dtype=torch.bfloat16)
            post_view = torch.ones(2, 1, HC, 1, dtype=torch.float32)
            comb_view = torch.ones(2, 1, HC, HC, dtype=torch.float32)

            new_kv = object()
            new_tables = {"new": 2}
            metadata = SimpleNamespace(
                batch_size=2,
                q_len_per_req=1,
                pool_block_tables=new_tables,
                position_ids=torch.zeros(2, 1, dtype=torch.int32),
                position_ids_long=torch.zeros(2, 1, dtype=torch.int64),
            )
            fake_pools = object()
            fake_runtime = object()

            with (
                patch.object(adapter, "_require_runtime", return_value=fake_runtime),
                patch.object(adapter, "_bind_pools", return_value=fake_pools),
                patch.object(
                    adapter,
                    "_forward_bound_outputs",
                    return_value=(proj, hidden, post_view, comb_view),
                ) as out_mock,
            ):
                got_outputs = adapter.forward_attention_sublayer_outputs(
                    block, hidden, metadata, kv_cache=new_kv
                )
                self.assertEqual(got_outputs, (proj, hidden, post_view, comb_view))
                # KV cache and block tables must be restored on exit.
                self.assertIs(block.attn._kv_cache, orig_kv)
                self.assertIs(block.attn._block_tables_by_type, orig_tables)
                block.attn_hc.post.assert_not_called()

                got_full = adapter.forward_attention_sublayer(
                    block, hidden, metadata, kv_cache=new_kv
                )
                self.assertEqual(got_full, "POST_DONE")
                self.assertEqual(out_mock.call_count, 2)
                block.attn_hc.post.assert_called_once_with(
                    proj, hidden, post_view, comb_view
                )
                self.assertIs(block.attn._kv_cache, orig_kv)
                self.assertIs(block.attn._block_tables_by_type, orig_tables)


class TestDSV4BlockAndDecodeTransitionContract(unittest.TestCase):
    def setUp(self) -> None:
        self._saved_warmed = set(_mega_mhc._WARMED_STREAMS)
        self.addCleanup(
            lambda: (
                _mega_mhc._WARMED_STREAMS.clear(),
                _mega_mhc._WARMED_STREAMS.update(self._saved_warmed),
            )
        )

    def test_block_preserves_magicmock_mega_adapter_contract(self) -> None:
        block = _make_block(dim=64)
        mock_adapter = MagicMock()
        mock_adapter.supports_decode_shape.return_value = True
        x = torch.randn(2, 1, 4, 64, dtype=torch.bfloat16)
        mock_adapter.forward_attention_sublayer.return_value = x
        block._mega_csa_adapter = mock_adapter
        block.ffn_hc.pre = MagicMock(
            return_value=(
                torch.randn(2, 1, 64, dtype=torch.bfloat16),
                torch.ones(2, 1, 4, 1, dtype=torch.float32),
                torch.ones(2, 1, 4, 4, dtype=torch.float32),
            )
        )
        block.ffn_hc.post = MagicMock(return_value=x)
        block.ffn_norm.forward = lambda y: y
        block.ffn.forward = lambda x_pre, ids, is_decode_forward=True: x_pre

        out = block.forward_decode(
            x, SimpleNamespace(), torch.zeros(2, 1, dtype=torch.long)
        )
        self.assertIs(out, x)
        mock_adapter.forward_attention_sublayer.assert_called_once()
        self.assertFalse(mock_adapter.forward_attention_sublayer_outputs.called)

    def test_block_mega_adapter_outputs_uses_fused_post_pre_and_falls_back(
        self,
    ) -> None:
        block = _make_block(dim=4096)
        adapter = _FakeMegaAdapter(supports=True)
        block._mega_csa_adapter = adapter

        x = torch.randn(2, 1, 4, 4096, dtype=torch.bfloat16)
        ids = torch.zeros(2, 1, dtype=torch.long)
        meta = SimpleNamespace()

        ffn_post_seen = []
        block.ffn_hc = _make_tilelang_unit(dim=4096, layer_id=0, name="ffn_hc")
        block.ffn_hc.post = lambda o, r, p, c: (ffn_post_seen.append((o, r, p, c)), r)[
            1
        ]
        block.attn_hc = _make_tilelang_unit(dim=4096, layer_id=0, name="attn_hc")

        with (
            patch.object(_mega_mhc, "is_supported", return_value=True),
            patch.object(
                block,
                "_intra_post_pre_norm",
                return_value=(
                    x * 2.0,
                    x[:, :, 0, :] * 3.0,
                    torch.ones(2, 1, 4, 1, dtype=torch.float32),
                    torch.ones(2, 1, 4, 4, dtype=torch.float32),
                ),
            ) as fused_mock,
        ):
            out = block.forward_decode(x, meta, ids)
            self.assertEqual(adapter.calls, ["outputs"])
            fused_mock.assert_called_once()
            self.assertEqual(len(ffn_post_seen), 1)
            torch.testing.assert_close(out, x * 2.0)

        # Fallback path if _intra_post_pre_norm returns None after adapter outputs were returned.
        adapter.calls.clear()
        attn_post_called = []
        block.attn_hc.post = lambda o, r, p, c: (attn_post_called.append(True), r)[1]
        with (
            patch.object(_mega_mhc, "is_supported", return_value=True),
            patch.object(block, "_intra_post_pre_norm", return_value=None),
            patch.object(
                block,
                "_hc_pre_norm_profiled",
                return_value=(
                    x[:, :, 0, :],
                    torch.ones(2, 1, 4, 1, dtype=torch.float32),
                    torch.ones(2, 1, 4, 4, dtype=torch.float32),
                ),
            ) as pre_mock,
        ):
            block.forward_decode(x, meta, ids)
            self.assertEqual(len(attn_post_called), 1)
            pre_mock.assert_called_once()

    def test_can_defer_decode_post_gates_and_forward_layers_flow(self) -> None:
        layer0 = _make_block(dim=4096, layer_id=0)
        layer1 = _make_block(dim=4096, layer_id=1)
        layer2 = _make_block(dim=4096, layer_id=2)
        layer2._mega_csa_adapter = _FakeMegaAdapter(supports=True)
        layer3 = _make_block(dim=4096, layer_id=3)

        dev = torch.device("cuda:0")
        with patch.object(_mega_mhc, "is_supported", return_value=True):
            # Non-Mega next layer -> allowed.
            self.assertTrue(
                can_defer_decode_post(layer0, layer1, device=dev, preserve_output=False)
            )
            # Even if current layer uses Mega adapter, next non-Mega layer is allowed.
            self.assertTrue(
                can_defer_decode_post(layer2, layer3, device=dev, preserve_output=False)
            )
            # Never defer INTO a layer with Mega CSA or Mega HCA adapter.
            self.assertFalse(
                can_defer_decode_post(layer1, layer2, device=dev, preserve_output=False)
            )
            layer2._mega_csa_adapter = None
            layer2._mega_hca_adapter = _FakeMegaAdapter(supports=True)
            self.assertFalse(
                can_defer_decode_post(layer1, layer2, device=dev, preserve_output=False)
            )
            layer2._mega_hca_adapter = None
            layer2._mega_csa_adapter = _FakeMegaAdapter(supports=True)
            # Never defer if preserved for aux capture or final layer or non-Block.
            self.assertFalse(
                can_defer_decode_post(layer0, layer1, device=dev, preserve_output=True)
            )
            self.assertFalse(
                can_defer_decode_post(layer3, None, device=dev, preserve_output=False)
            )
            self.assertFalse(
                can_defer_decode_post(layer0, SimpleNamespace(), device=dev)
            )

        # Test forward_decode_layer consumes PendingDecodePost via try_fused_post_pre, and materializes on None.
        pending = PendingDecodePost(
            sublayer_out=torch.randn(2, 1, 4096, dtype=torch.bfloat16),
            residual=torch.randn(2, 1, 4, 4096, dtype=torch.bfloat16),
            post=torch.ones(2, 1, 4, 1, dtype=torch.float32),
            comb=torch.ones(2, 1, 4, 4, dtype=torch.float32),
            hc_unit=layer0.ffn_hc,
        )
        seen_prepared = []
        orig_fd = layer1.forward_decode

        def spy_forward_decode(*args, **kwargs):
            seen_prepared.append(kwargs.get("_prepared_attn"))
            return args[0]

        layer1.forward_decode = spy_forward_decode
        fused_tuple = (
            torch.randn(2, 1, 4, 4096, dtype=torch.bfloat16),
            torch.randn(2, 1, 4096, dtype=torch.bfloat16),
            torch.ones(2, 1, 4, 1, dtype=torch.float32),
            torch.ones(2, 1, 4, 4, dtype=torch.float32),
        )
        with patch.object(_mega_mhc, "try_fused_post_pre", return_value=fused_tuple):
            forward_decode_layer(
                layer1,
                pending,
                SimpleNamespace(),
                torch.zeros(2, 1, dtype=torch.long),
                defer_post=False,
            )
        self.assertIsNotNone(seen_prepared[0])

        layer1.forward_decode = orig_fd
        pending_fallback = PendingDecodePost(
            sublayer_out=pending.sublayer_out,
            residual=pending.residual,
            post=pending.post,
            comb=pending.comb,
            hc_unit=MagicMock(
                post=MagicMock(
                    return_value=torch.zeros(2, 1, 4, 4096, dtype=torch.bfloat16)
                )
            ),
        )
        with (
            patch.object(_mega_mhc, "try_fused_post_pre", return_value=None),
            patch.object(layer1, "forward_decode", return_value="MAT") as fd_mock,
        ):
            res = forward_decode_layer(
                layer1,
                pending_fallback,
                SimpleNamespace(),
                torch.zeros(2, 1, dtype=torch.long),
                defer_post=False,
            )
            self.assertEqual(res, "MAT")
            pending_fallback.hc_unit.post.assert_called_once()
            fd_mock.assert_called_once()

    def test_decode_forward_layers_respects_capture_and_mtp_boundaries(self) -> None:
        l0 = _make_block(dim=64, layer_id=0)
        l1 = _make_block(dim=64, layer_id=1)
        l2 = _make_block(dim=64, layer_id=2)

        events = []

        def make_spy_fd(idx):
            def fd(
                x,
                meta,
                ids,
                kv_cache=None,
                *,
                _prepared_attn=None,
                _defer_ffn_post=False,
            ):
                events.append(
                    (
                        idx,
                        _prepared_attn is not None,
                        _defer_ffn_post,
                        isinstance(x, PendingDecodePost),
                    )
                )
                if _defer_ffn_post:
                    return PendingDecodePost(
                        sublayer_out=torch.ones(2, 1, 64, dtype=torch.bfloat16),
                        residual=x,
                        post=torch.ones(2, 1, 4, 1, dtype=torch.float32),
                        comb=torch.ones(2, 1, 4, 4, dtype=torch.float32),
                        hc_unit=MagicMock(post=MagicMock(return_value=x)),
                    )
                return x

            return fd

        for idx, layer in enumerate((l0, l1, l2)):
            layer.forward_decode = make_spy_fd(idx)

        fake_v4 = SimpleNamespace(
            embed=lambda ids: torch.ones(2, 1, 64, dtype=torch.bfloat16),
            hc_mult=4,
            layers=[l0, l1, l2],
            capture_aux_hidden_layer_ids=[1],
            capture_aux_hidden=MagicMock(),
            _mtp_hidden_buffer=object(),
            _note_aux_hidden_rows=MagicMock(),
            _write_mtp_hidden_buffer=MagicMock(),
            _hc_head_reduce=lambda h: h[:, :, 0, :],
            norm=lambda x: x,
        )
        fake_fused = (
            torch.ones(2, 1, 4, 64, dtype=torch.bfloat16),
            torch.ones(2, 1, 64, dtype=torch.bfloat16),
            torch.ones(2, 1, 4, 1, dtype=torch.float32),
            torch.ones(2, 1, 4, 4, dtype=torch.float32),
        )
        with (
            patch.object(_mega_mhc, "is_supported", return_value=True),
            patch.object(_mega_mhc, "try_fused_post_pre", return_value=fake_fused),
        ):
            out = forward_layers(
                fake_v4,
                None,
                torch.zeros(2, 1, dtype=torch.long),
                SimpleNamespace(batch_size=2, q_len_per_req=1, is_cuda_graph=False),
            )
        self.assertEqual(tuple(out.shape), (2, 1, 64))
        # Layer 0 deferred -> consumed prepared_attn by Layer 1; Layer 1 preserved (no defer); Layer 2 final (no defer).
        self.assertEqual(
            events,
            [
                (0, False, True, False),
                (1, True, False, False),
                (2, False, False, False),
            ],
        )
        fake_v4.capture_aux_hidden.assert_called_once()
        # Capture argument is a materialized tensor, never a PendingDecodePost.
        self.assertIsInstance(fake_v4.capture_aux_hidden.call_args[0][1], torch.Tensor)
        fake_v4._note_aux_hidden_rows.assert_called_once()


def _gpu_run_required() -> bool:
    return os.environ.get("GPU_COUNT", "0") != "0"


@unittest.skipUnless(_gpu_run_required(), "GPU parity tests skipped when GPU_COUNT=0")
class TestDSV4BlockMegaMHCSm103Gpu(unittest.TestCase):
    def setUp(self) -> None:
        self.assertTrue(torch.cuda.is_available(), "CUDA required when GPU_COUNT != 0")
        major, minor = torch.cuda.get_device_capability(0)
        self.assertEqual(
            major,
            10,
            f"deep_gemm.mega_mhc requires SM100/SM103; got sm_{major}{minor}",
        )
        self.assertIsNotNone(_mega_mhc._get_mega_mhc())
        self.device = torch.device("cuda:0")

    def test_single_block_decode_and_prefill_parity(self) -> None:
        dim = 4096
        block = _make_block(dim=dim, device=self.device, seed=42)
        block_mega = _make_block(dim=dim, device=self.device, seed=42)
        block_mega._mega_csa_adapter = _FakeMegaAdapter(supports=True)

        x_dec = (torch.randn(4, 1, 4, dim, dtype=torch.float32) * 0.25).to(
            device=self.device, dtype=torch.bfloat16
        )
        ids_dec = torch.zeros(4, 1, dtype=torch.long, device=self.device)
        meta = SimpleNamespace()

        # Ordinary attention seam parity.
        with patch.object(_mega_mhc, "try_fused_post_pre", return_value=None):
            ref_dec = block.forward_decode(x_dec.clone(), meta, ids_dec)
        fused_dec = block.forward_decode(x_dec.clone(), meta, ids_dec)
        torch.testing.assert_close(fused_dec, ref_dec, atol=4.0e-2, rtol=3.0e-2)

        # Mega adapter seam parity.
        with patch.object(_mega_mhc, "is_supported", return_value=False):
            ref_mega = block_mega.forward_decode(x_dec.clone(), meta, ids_dec)
        fused_mega = block_mega.forward_decode(x_dec.clone(), meta, ids_dec)
        torch.testing.assert_close(fused_mega, ref_mega, atol=4.0e-2, rtol=3.0e-2)

        # Prefill forward seam parity.
        x_pref = (torch.randn(130, 4, dim, dtype=torch.float32) * 0.25).to(
            device=self.device, dtype=torch.bfloat16
        )
        pos = torch.arange(130, dtype=torch.long, device=self.device)
        cu = torch.tensor([0, 130], dtype=torch.long, device=self.device)
        # Monkeypatch AttentionFP8 check by calling Block.forward with non-AttentionFP8 attn using dense branch?
        # Block.forward checks self.attn.__class__ against AttentionFP8; stub forward to accept [B, S, D] or [T, D].
        block.attn.forward = (
            lambda x_pad, start_pos, sequence_lengths=None, **kw: x_pad * 0.75 + 0.05
        )
        block.attn.__call__ = block.attn.forward
        with patch.object(_mega_mhc, "try_fused_post_pre", return_value=None):
            ref_pref = block.forward(x_pref.clone(), None, pos, cu)
        fused_pref = block.forward(x_pref.clone(), None, pos, cu)
        torch.testing.assert_close(fused_pref, ref_pref, atol=4.0e-2, rtol=3.0e-2)

    def test_multilayer_decode_transition_parity_and_cuda_graph(self) -> None:
        dim = 4096
        layers_fused = [
            _make_block(dim=dim, device=self.device, seed=100 + i * 30, layer_id=i)
            for i in range(4)
        ]
        # Layer 2 has Mega adapter (cannot defer INTO layer 2, CAN fuse layer 2->3 and 0->1).
        layers_fused[2]._mega_csa_adapter = _FakeMegaAdapter(supports=True)

        layers_ref = [
            _make_block(dim=dim, device=self.device, seed=100 + i * 30, layer_id=i)
            for i in range(4)
        ]
        layers_ref[2]._mega_csa_adapter = _FakeMegaAdapter(supports=True)

        embed_w = (torch.randn(16, dim, dtype=torch.float32) * 0.25).to(
            device=self.device, dtype=torch.bfloat16
        )
        final_norm = RMSNorm(torch.ones(dim, dtype=torch.bfloat16, device=self.device))

        def make_v4(layers):
            return SimpleNamespace(
                embed=lambda ids: embed_w[ids],
                hc_mult=4,
                layers=layers,
                capture_aux_hidden_layer_ids=[],
                capture_aux_hidden=MagicMock(),
                _mtp_hidden_buffer=None,
                _hc_head_reduce=lambda h: h.mean(dim=-2),
                norm=final_norm,
                begin_decode=None,
            )

        v4_fused = make_v4(layers_fused)
        v4_ref = make_v4(layers_ref)

        input_ids = torch.tensor(
            [[1], [3], [5], [7]], dtype=torch.long, device=self.device
        )
        meta = SimpleNamespace(batch_size=4, q_len_per_req=1, is_cuda_graph=True)

        with (
            patch.object(_mega_mhc, "is_supported", return_value=False),
            patch.object(_mega_mhc, "try_fused_post_pre", return_value=None),
        ):
            ref_out = forward_layers(v4_ref, None, input_ids, meta)

        eager_fused_out = forward_layers(v4_fused, None, input_ids, meta)
        torch.testing.assert_close(eager_fused_out, ref_out, atol=6.0e-2, rtol=5.0e-2)

        # Capture and replay multi-layer decode transition inside a CUDA graph on warmed stream.
        stream = torch.cuda.Stream(device=self.device)
        with torch.cuda.stream(stream):
            _ = forward_layers(v4_fused, None, input_ids, meta)
        stream.synchronize()

        static_ids = input_ids.clone()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            graph_out = forward_layers(v4_fused, None, static_ids, meta)

        next_ids = torch.tensor(
            [[2], [4], [6], [8]], dtype=torch.long, device=self.device
        )
        static_ids.copy_(next_ids)
        graph.replay()
        torch.cuda.synchronize(self.device)

        with (
            patch.object(_mega_mhc, "is_supported", return_value=False),
            patch.object(_mega_mhc, "try_fused_post_pre", return_value=None),
        ):
            ref_next = forward_layers(v4_ref, None, next_ids, meta)
        torch.testing.assert_close(graph_out, ref_next, atol=6.0e-2, rtol=5.0e-2)


if __name__ == "__main__":
    unittest.main()
