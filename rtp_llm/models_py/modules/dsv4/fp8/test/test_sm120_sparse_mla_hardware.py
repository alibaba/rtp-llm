"""SM120 hardware gate for the long-context sparse-MLA path."""

from __future__ import annotations

import unittest
from contextlib import contextmanager
from types import SimpleNamespace
from typing import Iterator
from unittest.mock import Mock, patch

import torch

from deep_gemm.utils.layout import get_mn_major_tma_aligned_packed_ue8m0_tensor
from rtp_llm.models_py.modules.dsv4.fp8 import sm120_sparse_mla
from rtp_llm.models_py.modules.dsv4.fp8._swa_dequant_triton import (
    dequantize_and_gather_k_cache_slots,
)
from rtp_llm.models_py.modules.dsv4.fp8._swa_kv_insert_triton import (
    quantize_and_insert_k_cache,
)
from rtp_llm.models_py.modules.dsv4.fp8.attention import (
    AttentionFP8,
    _map_active_sparse_slots,
)
from rtp_llm.models_py.modules.dsv4.fp8.attention_backend import (
    SparseAttentionWorkspace,
    SparsePrefillInputs,
    get_sparse_attention_backend,
)
from rtp_llm.models_py.modules.dsv4.fp8.decode.fp8_sparse_attn_decode_op import (
    SparseAttnV4DecodeFp8Op,
)
from rtp_llm.models_py.modules.dsv4.prefill_workspace import PrefillWorkspace
from rtp_llm.models_py.utils.arch import is_sm120

# FlashInfer 0.7.5's SM120 DSv4 sparse-MLA kernel quantizes BF16 queries to
# FP8 E4M3 internally. Under standard-normal N(0, 1) synthetic inputs, enforce
# global relative L2 error along with a loose elementwise safety bound (and
# verify halved-scale mutations fail) instead of attenuating Q/KV variance.


class Sm120SparseMlaHardwareTest(unittest.TestCase):
    def setUp(self) -> None:
        if not torch.cuda.is_available():
            self.skipTest("SM120 hardware gate was scheduled without a CUDA device")
        if not is_sm120():
            self.skipTest(
                "SM120 hardware gate was scheduled on "
                f"compute capability {torch.cuda.get_device_capability()}"
            )
        self.device = torch.device("cuda", torch.cuda.current_device())
        torch.manual_seed(7)

    @contextmanager
    def _spy_sparse_mla_provider(self) -> Iterator[Mock]:
        """Record the public FlashInfer calls while still launching them.

        FlashInfer picks the decode or prefill kernel itself, so these tests
        assert only the inputs RTP prepares and the exact outputs.
        """
        provider = sm120_sparse_mla._load_sm120_sparse_mla()
        self.assertIsNotNone(provider)
        spy = Mock(wraps=provider)
        with patch.object(sm120_sparse_mla, "_load_sm120_sparse_mla", return_value=spy):
            yield spy

    def test_two_logical_compressed_indices_use_padded_native_prefill(self):
        q = torch.zeros((1, 16, 512), dtype=torch.bfloat16, device=self.device)
        kv = torch.zeros((1, 66, 512), dtype=torch.bfloat16, device=self.device)
        kv[:, :, 448:] = 2
        indices = torch.full((1, 1, 130), -1, dtype=torch.int32, device=self.device)
        indices[0, 0, :3] = torch.arange(3, device=self.device, dtype=torch.int32)
        lengths = torch.full((1,), 3, dtype=torch.int32, device=self.device)
        backend = get_sparse_attention_backend(self.device)
        inputs = backend.workspace_inputs(
            workspace=kv,
            indices=indices,
            lengths=lengths,
            M=66,
            N=2,
            window_size=128,
            extra_width=2,
            ratio=128,
            device=self.device,
            prefix_cache=None,
            prefix_slots=None,
            prefix_lens=None,
        )
        self.assertEqual(inputs.extra_indices.shape[-1], 64)
        with patch.object(sm120_sparse_mla, "run", wraps=sm120_sparse_mla.run) as run:
            prepared = backend.prepare_prefill(
                n_heads=16,
                q=q,
                kv=kv,
                indices=indices,
                topk_length=lengths,
                head_dim=512,
                softmax_scale=1.0,
                attn_sink=torch.zeros(16, device=self.device),
                prefill_workspace=PrefillWorkspace(
                    self.device,
                    q_rows=1,
                    q_dim=16 * 512,
                    reserve_cp=False,
                    align_bytes=1,
                ),
                backend_inputs=inputs,
            )
            output = prepared.forward(0, 1)
        run.assert_called_once()
        self.assertTrue(run.call_args.kwargs["is_prefill"])
        self.assertEqual(run.call_args.kwargs["extra_indices"].shape[-1], 128)
        expected = torch.zeros_like(q)
        expected[..., 448:] = 1.5
        torch.testing.assert_close(output, expected, atol=0, rtol=0)

    def _packed_cache(self, source: torch.Tensor, *, block_size: int) -> torch.Tensor:
        num_blocks = (source.shape[0] + block_size - 1) // block_size
        cache = torch.zeros(
            (num_blocks, block_size, 584), dtype=torch.uint8, device=self.device
        )
        quantize_and_insert_k_cache(
            source,
            cache,
            torch.arange(source.shape[0], dtype=torch.int64, device=self.device),
        )
        return cache

    def _packed_dual_cache_with_wide_swa(self) -> tuple[torch.Tensor, torch.Tensor]:
        swa = torch.zeros((129, 512), dtype=torch.bfloat16, device=self.device)
        swa[:, 448:] = 1
        swa[-1, 448:] = 128
        extra = torch.zeros((1, 512), dtype=torch.bfloat16, device=self.device)
        extra[:, 448:] = 4
        return self._packed_cache(swa, block_size=64), self._packed_cache(
            extra, block_size=2
        )

    def _packed_dual_cache_with_two_extra_slots(
        self,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        swa = torch.zeros((1, 512), dtype=torch.bfloat16, device=self.device)
        swa[:, 448:] = 2
        extra = torch.zeros((2, 512), dtype=torch.bfloat16, device=self.device)
        extra[0, 448:] = 4
        extra[1, 448:] = 10
        return self._packed_cache(swa, block_size=64), self._packed_cache(
            extra, block_size=2
        )

    def _assert_narrow_extra_decode(
        self, rows: int, raw_extra_width: int, *, uses_prefill: bool
    ) -> None:
        swa_cache, extra_cache = self._packed_dual_cache_with_two_extra_slots()
        query = torch.zeros(
            (rows, 1, 16, 512), dtype=torch.bfloat16, device=self.device
        )
        swa_indices = torch.full(
            (rows, 1, 128), -1, dtype=torch.int32, device=self.device
        )
        swa_indices[:, :, 0] = 0
        swa_lens = torch.ones(rows, dtype=torch.int32, device=self.device)
        extra_indices = torch.empty(
            (rows, 1, raw_extra_width), dtype=torch.int32, device=self.device
        )
        extra_lens = torch.full_like(swa_lens, raw_extra_width)
        expected_extra_lens = torch.zeros_like(extra_lens)
        expected = torch.zeros_like(query)
        expected[..., 448:] = 1
        if raw_extra_width == 2:
            extra_indices[::2, 0] = torch.tensor([-1, 0], device=self.device)
            extra_indices[1::2, 0] = torch.tensor([0, 1], device=self.device)
            expected_extra_lens[::2] = 1
            expected_extra_lens[1::2] = 2
            # With zero query and sink logits, every live value and the sink
            # gets equal weight. Compaction must preserve the entry after -1.
            expected[::2, ..., 448:] = 2
            expected[1::2, ..., 448:] = 4

        with self._spy_sparse_mla_provider() as provider:
            output = sm120_sparse_mla.forward_decode(
                query,
                swa_cache,
                torch.zeros(16, device=self.device),
                swa_indices,
                swa_lens,
                extra_cache,
                extra_indices,
                extra_lens,
                softmax_scale=1.0,
            )

        provider.assert_called_once()
        launch = provider.call_args.kwargs
        self.assertEqual(launch["sparse_indices"].shape, (rows, 128))
        torch.testing.assert_close(launch["swa_topk_lens"], swa_lens)
        # Rows above the decode limit always run FlashInfer prefill, which
        # reads full 64-entry tiles, so RTP pads the extra table for them.
        native_width = 128 if uses_prefill else 2
        self.assertIsNotNone(launch["compressed_kv_cache"])
        self.assertEqual(launch["extra_sparse_indices"].shape, (rows, native_width))
        torch.testing.assert_close(
            launch["extra_sparse_topk_lens"], expected_extra_lens
        )
        torch.testing.assert_close(output, expected, atol=0, rtol=0)

    def test_narrow_dual_decode_above_batch_limit_uses_padded_native_prefill(self):
        for raw_extra_width in (0, 2):
            with self.subTest(raw_extra_width=raw_extra_width):
                self._assert_narrow_extra_decode(65, raw_extra_width, uses_prefill=True)

    def test_narrow_dual_decode_at_batch_limit_keeps_width_two(self):
        self._assert_narrow_extra_decode(64, 2, uses_prefill=False)

    def test_two_paged_compressed_indices_use_padded_native_prefill(self):
        swa_cache, extra_cache = self._packed_dual_cache_with_two_extra_slots()
        query = torch.zeros((1, 16, 512), dtype=torch.bfloat16, device=self.device)
        swa_indices = torch.zeros((1, 128), dtype=torch.int32, device=self.device)
        swa_lens = torch.ones(1, dtype=torch.int32, device=self.device)
        extra_indices = torch.tensor([[1, 0]], dtype=torch.int32, device=self.device)
        extra_lens = torch.full_like(swa_lens, 2)
        backend = get_sparse_attention_backend(self.device)
        prepared = backend.prepare_prefill(
            n_heads=16,
            q=query,
            kv=None,
            indices=swa_indices,
            topk_length=swa_lens,
            head_dim=512,
            softmax_scale=1.0,
            attn_sink=torch.zeros(16, device=self.device),
            prefill_workspace=None,
            backend_inputs=SparsePrefillInputs(
                swa_cache=swa_cache,
                extra_cache=extra_cache,
                swa_indices=swa_indices,
                swa_lens=swa_lens,
                extra_indices=extra_indices,
                extra_lens=extra_lens,
            ),
        )
        with self._spy_sparse_mla_provider() as provider:
            output = prepared.forward(0, 1)

        provider.assert_called_once()
        launch = provider.call_args.kwargs
        self.assertEqual(launch["extra_sparse_indices"].shape, (1, 128))
        torch.testing.assert_close(launch["extra_sparse_indices"][:, :2], extra_indices)
        torch.testing.assert_close(
            launch["extra_sparse_indices"][:, 2:], extra_indices[:, :1].expand(1, 126)
        )
        torch.testing.assert_close(launch["extra_sparse_topk_lens"], extra_lens)
        expected = torch.zeros_like(query)
        expected[..., 448:] = 4
        torch.testing.assert_close(output, expected, atol=0, rtol=0)

    def test_dual_prefill_swa_width_boundary(self) -> None:
        backend = get_sparse_attention_backend(self.device)
        query = torch.zeros((1, 16, 512), dtype=torch.bfloat16, device=self.device)
        sink = torch.zeros(16, device=self.device)
        swa_cache, extra_cache = self._packed_dual_cache_with_wide_swa()
        extra_indices = torch.zeros((1, 128), dtype=torch.int32, device=self.device)
        extra_lens = torch.ones((1,), dtype=torch.int32, device=self.device)
        for width in (128, 129):
            with self.subTest(width=width):
                swa_indices = torch.arange(
                    width, dtype=torch.int32, device=self.device
                ).view(1, width)
                swa_lens = torch.full(
                    (1,), width, dtype=torch.int32, device=self.device
                )
                with (
                    patch.object(
                        sm120_sparse_mla, "run", wraps=sm120_sparse_mla.run
                    ) as native,
                    patch.object(
                        sm120_sparse_mla,
                        "run_chunked_reference",
                        wraps=sm120_sparse_mla.run_chunked_reference,
                    ) as fallback,
                ):
                    prepared = backend.prepare_prefill(
                        n_heads=16,
                        q=query,
                        kv=None,
                        indices=swa_indices,
                        topk_length=swa_lens,
                        head_dim=512,
                        softmax_scale=1.0,
                        attn_sink=sink,
                        prefill_workspace=None,
                        backend_inputs=SparsePrefillInputs(
                            swa_cache=swa_cache,
                            extra_cache=extra_cache,
                            swa_indices=swa_indices,
                            swa_lens=swa_lens,
                            extra_indices=extra_indices,
                            extra_lens=extra_lens,
                        ),
                    )
                    output = prepared.forward(0, 1)
                if width == 128:
                    native.assert_called_once()
                    self.assertTrue(native.call_args.kwargs["is_prefill"])
                    fallback.assert_not_called()
                else:
                    fallback.assert_called_once()
                    native.assert_not_called()
                    self.assertEqual(
                        fallback.call_args.kwargs["swa_indices"].shape[-1], width
                    )
                # With zero query and sink logits, all active values and the
                # sink have equal weight. Slot 128 contributes only at k=129.
                expected = torch.zeros_like(query)
                expected[..., 448:] = (132 + (128 if width == 129 else 0)) / (width + 2)
                torch.testing.assert_close(output, expected, rtol=1e-2, atol=1e-2)

    def test_wide_dual_decode_crosses_native_batch_limit(self) -> None:
        swa_cache, extra_cache = self._packed_dual_cache_with_wide_swa()
        sink = torch.zeros(16, device=self.device)
        for rows in (64, 65):
            with self.subTest(rows=rows):
                query = torch.zeros(
                    (rows, 1, 16, 512), dtype=torch.bfloat16, device=self.device
                )
                swa_indices = torch.arange(
                    129, dtype=torch.int32, device=self.device
                ).expand(rows, 1, -1)
                extra_indices = torch.zeros(
                    (rows, 1, 128), dtype=torch.int32, device=self.device
                )
                swa_lens = torch.full(
                    (rows,), 129, dtype=torch.int32, device=self.device
                )
                extra_lens = torch.ones((rows,), dtype=torch.int32, device=self.device)
                with patch.object(
                    sm120_sparse_mla, "run", wraps=sm120_sparse_mla.run
                ) as native:
                    output = sm120_sparse_mla.forward_decode(
                        query,
                        swa_cache,
                        sink,
                        swa_indices,
                        swa_lens,
                        extra_cache,
                        extra_indices,
                        extra_lens,
                        softmax_scale=1.0,
                    )
                if rows == 64:
                    native.assert_called_once()
                    self.assertEqual(
                        native.call_args.kwargs["swa_indices"].shape[-1], 512
                    )
                else:
                    native.assert_not_called()
                expected = torch.zeros_like(query)
                expected[..., 448:] = 260 / 131
                torch.testing.assert_close(output, expected, rtol=1e-2, atol=1e-2)

    def _gather_dequantized(
        self, cache: torch.Tensor, rows: list[list[int]]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        width = max(len(row) for row in rows)
        slots = torch.full(
            (len(rows), width), -1, dtype=torch.int64, device=self.device
        )
        lengths = torch.tensor(
            [len(row) for row in rows], dtype=torch.int32, device=self.device
        )
        for row_idx, row in enumerate(rows):
            slots[row_idx, : len(row)] = torch.tensor(
                row, dtype=torch.int64, device=self.device
            )
        output = torch.empty(
            (len(rows), width, 512), dtype=torch.bfloat16, device=self.device
        )
        dequantize_and_gather_k_cache_slots(output, cache, slots, lengths, 0)
        return output, lengths

    @staticmethod
    def _reference_sparse_prefill(
        query: torch.Tensor,
        sink: torch.Tensor,
        swa_values: torch.Tensor,
        swa_indices: torch.Tensor,
        swa_lengths: torch.Tensor,
        scale: float,
        extra_values: torch.Tensor | None = None,
        extra_indices: torch.Tensor | None = None,
        extra_lengths: torch.Tensor | None = None,
    ) -> torch.Tensor:
        outputs = []
        for row in range(query.shape[0]):
            swa_bound = int(swa_lengths[row].item())
            swa_slots = swa_indices[row, :swa_bound].long()
            swa_slots = swa_slots[swa_slots >= 0]
            values = swa_values.index_select(0, swa_slots)
            if extra_values is not None:
                assert extra_indices is not None and extra_lengths is not None
                extra_bound = int(extra_lengths[row].item())
                extra_slots = extra_indices[row, :extra_bound].long()
                extra_slots = extra_slots[extra_slots >= 0]
                values = torch.cat(
                    (values, extra_values.index_select(0, extra_slots)), dim=0
                )

            q = query[row].float()
            logits = torch.einsum("hd,td->ht", q, values.float()) * scale
            scores_max = torch.maximum(logits.amax(dim=-1), sink).unsqueeze(-1)
            exp_logits = torch.exp(logits - scores_max)
            exp_sink = torch.exp(sink.unsqueeze(-1) - scores_max)
            probs = exp_logits / (exp_logits.sum(dim=-1, keepdim=True) + exp_sink)
            outputs.append(torch.einsum("ht,td->hd", probs, values.float()))
        return torch.stack(outputs).to(query.dtype)

    @staticmethod
    def _assert_sparse_output_close(
        actual: torch.Tensor,
        expected: torch.Tensor,
        *,
        rtol: float = 1.5e-1,
        atol: float = 1.5e-1,
        max_rel_l2: float = 5e-2,
    ) -> float:
        assert torch.isfinite(
            actual
        ).all(), "sparse MLA output contains non-finite values"
        rel_l2 = (
            (actual.float() - expected.float()).norm()
            / expected.float().norm().clamp_min(1e-10)
        ).item()
        assert rel_l2 < max_rel_l2, f"relative L2 error {rel_l2:.5f} >= {max_rel_l2}"
        torch.testing.assert_close(actual, expected, rtol=rtol, atol=atol)
        return rel_l2

    def _assert_prefill_variant(
        self, variant: str, *, empty_extra: bool = False, tokens: int = 3
    ) -> None:
        heads, dim = 8, 512
        scale = dim**-0.5
        workspace = PrefillWorkspace(
            self.device,
            q_rows=tokens,
            q_dim=heads * dim,
            reserve_cp=False,
            align_bytes=1,
        )
        query = workspace.prefill_q(tokens).view(tokens, heads, dim)
        query.copy_(torch.randn_like(query))
        sink = torch.linspace(-0.3, 0.4, heads, dtype=torch.float32, device=self.device)
        swa_values = torch.randn(11, dim, dtype=torch.bfloat16, device=self.device)
        swa_indices = torch.full(
            (tokens, 128), -1, dtype=torch.int32, device=self.device
        )
        swa_indices[0, :3] = torch.tensor([0, 3, 5], device=self.device)
        swa_indices[1, :4] = torch.tensor([2, 4, 7, 9], device=self.device)
        swa_indices[2, :2] = torch.tensor([1, 10], device=self.device)
        swa_lengths = torch.tensor([3, 4, 2], dtype=torch.int32, device=self.device)
        if tokens > 3:
            swa_indices[3:, :3] = torch.tensor([1, 4, 8], device=self.device)
            swa_lengths = torch.cat(
                [
                    swa_lengths,
                    torch.full((tokens - 3,), 3, dtype=torch.int32, device=self.device),
                ]
            )
        swa_cache = self._packed_cache(swa_values, block_size=64)
        swa_reference_values, _ = self._gather_dequantized(
            swa_cache, [list(range(swa_values.shape[0]))]
        )
        swa_reference_values = swa_reference_values.squeeze(0)

        extra_values = extra_indices = extra_lengths = None
        extra_reference_values = None
        extra_cache = None
        if variant != "swa":
            extra_values = torch.randn(9, dim, dtype=torch.bfloat16, device=self.device)
            extra_indices = torch.full(
                (tokens, 8), -1, dtype=torch.int32, device=self.device
            )
            extra_indices[0, :2] = torch.tensor([1, 6], device=self.device)
            extra_indices[1, :3] = torch.tensor([0, 4, 8], device=self.device)
            extra_indices[2, :1] = torch.tensor([7], device=self.device)
            extra_lengths = torch.tensor(
                [2, 3, 1], dtype=torch.int32, device=self.device
            )
            if tokens > 3:
                extra_indices[3:, :2] = torch.tensor([2, 5], device=self.device)
                extra_lengths = torch.cat(
                    [
                        extra_lengths,
                        torch.full(
                            (tokens - 3,), 2, dtype=torch.int32, device=self.device
                        ),
                    ]
                )
            extra_cache = self._packed_cache(
                extra_values, block_size=64 if variant == "csa" else 2
            )
            extra_reference_values, _ = self._gather_dequantized(
                extra_cache, [list(range(extra_values.shape[0]))]
            )
            extra_reference_values = extra_reference_values.squeeze(0)

        if empty_extra:
            extra_values = torch.empty(0, dim, dtype=torch.bfloat16, device=self.device)
            extra_cache = None
            extra_indices.fill_(-1)
            extra_lengths.zero_()
            extra_reference_values = None

            def poisoned_cache(*args, **kwargs):
                # A zero-token allocation must never reach the sparse provider.
                return torch.full(
                    (1, kwargs["page_size"], 1, kwargs["entry_bytes"]),
                    255,
                    dtype=torch.uint8,
                    device=self.device,
                )

            workspace.sm120_packed_cache = poisoned_cache

        layer = AttentionFP8.__new__(AttentionFP8)
        torch.nn.Module.__init__(layer)
        attn_ws = SparseAttentionWorkspace()
        layer._sparse_mla_workspace = attn_ws
        layer.n_heads = heads
        layer.head_dim = dim
        layer.dim = heads * dim
        layer.softmax_scale = scale
        layer.attn_sink = sink
        layer._prefill_output_proj_into = lambda attention, _freqs, *, out: out.copy_(
            attention.reshape(tokens, -1)
        )
        layer._prefill_output_all_reduce = lambda _out: None
        freqs = torch.zeros(tokens, 1, dtype=torch.complex64, device=self.device)
        generic_indices = swa_indices.unsqueeze(1)
        generic_kv = swa_values.unsqueeze(1)

        def forward(out: torch.Tensor) -> torch.Tensor:
            return layer._flash_mla_sparse_fwd_chunked_projected(
                q=query,
                kv=generic_kv,
                indices=generic_indices,
                topk_length=swa_lengths,
                freqs_cis=freqs,
                prefill_workspace=workspace,
                profile_name=f"sm120.prefill.{variant}",
                out=out,
                backend_inputs=SparsePrefillInputs(
                    swa_cache=swa_cache,
                    extra_cache=extra_cache,
                    extra_kv=extra_values if empty_extra else None,
                    extra_page_size=64 if variant == "csa" else 2,
                    swa_indices=swa_indices,
                    swa_lens=swa_lengths,
                    extra_indices=extra_indices,
                    extra_lens=extra_lengths,
                ),
            )

        def reference(ref_scale: float = scale) -> torch.Tensor:
            return self._reference_sparse_prefill(
                query,
                sink,
                swa_reference_values,
                swa_indices,
                swa_lengths,
                ref_scale,
                extra_reference_values,
                extra_indices,
                extra_lengths,
            ).reshape(tokens, -1)

        eager_out = torch.empty(
            tokens, heads * dim, dtype=torch.bfloat16, device=self.device
        )
        with patch(
            "rtp_llm.models_py.modules.dsv4.fp8._swa_kv_insert_triton."
            "quantize_and_insert_k_cache",
            side_effect=AssertionError("prefill must reuse the paged FP8 cache"),
        ):
            forward(eager_out)
            self._assert_sparse_output_close(eager_out, reference())
            if not empty_extra or variant == "swa":
                with self.assertRaises(AssertionError):
                    self._assert_sparse_output_close(eager_out, reference(scale * 0.5))

            sparse_ws = layer._sparse_mla_workspace.get_or_create(
                sm120_sparse_mla.Sm120SparseMlaWorkspace
            )
            sm120_sparse_mla.warmup(self.device, owner=sparse_ws)
            graph_out = torch.empty_like(eager_out)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                forward(graph_out)
            query.copy_(torch.randn_like(query))
            graph.replay()
            torch.cuda.synchronize(self.device)
            self._assert_sparse_output_close(graph_out, reference())
            if not empty_extra or variant == "swa":
                with self.assertRaises(AssertionError):
                    self._assert_sparse_output_close(graph_out, reference(scale * 0.5))

    def test_swa_prefill_eager_and_cuda_graph(self) -> None:
        self._assert_prefill_variant("swa")

    def test_paged_lookup_masks_only_inactive_indices(self) -> None:
        for device in ("cpu", self.device):
            mapping = torch.tensor([5, 8, 11], device=device)
            logical = torch.tensor([[2, -1, 99]], device=device)
            active = torch.tensor([[True, False, False]], device=device)
            torch.testing.assert_close(
                _map_active_sparse_slots(mapping, logical, active),
                torch.tensor([[11, -1, -1]], dtype=torch.int32, device=device),
            )
        # CPU exercises the same index_select bounds contract without poisoning
        # the shared CUDA context with a deliberately invalid device lookup.
        with self.assertRaises(IndexError):
            _map_active_sparse_slots(
                torch.tensor([5, 8, 11]), torch.tensor([[3]]), torch.tensor([[True]])
            )

    def test_csa_prefill_eager_and_cuda_graph(self) -> None:
        self._assert_prefill_variant("csa")

    def test_empty_compressed_prefill_ignores_poisoned_storage(self) -> None:
        for variant in ("csa", "hca"):
            with self.subTest(variant=variant):
                self._assert_prefill_variant(variant, empty_extra=True)

    def test_cp_batch_preserves_standalone_sparse_dispatch(self) -> None:
        heads, dim = 64, 512
        sink = torch.linspace(-0.3, 0.4, heads, device=self.device)
        for lengths in ((10, 10), (10,) * 7, (16, 54, 66), (0, 10)):
            with self.subTest(lengths=lengths):
                tokens = sum(lengths)
                ws = PrefillWorkspace(
                    self.device,
                    q_rows=tokens,
                    q_dim=heads * dim,
                    reserve_cp=False,
                    align_bytes=1,
                )
                query = ws.prefill_q(tokens).view(tokens, heads, dim)
                query.copy_(torch.randn_like(query))
                values = torch.randn(
                    35 * len(lengths), dim, device=self.device, dtype=torch.bfloat16
                )
                cache = self._packed_cache(values, block_size=64)
                indices = torch.zeros(
                    tokens, 128, device=self.device, dtype=torch.int32
                )
                lens = torch.zeros(tokens, device=self.device, dtype=torch.int32)
                offset = 0
                for request, rows in enumerate(lengths):
                    indices[offset : offset + rows, :35] = (
                        torch.arange(35, device=self.device) + request * 35
                    )
                    lens[offset : offset + rows] = (
                        torch.arange(rows, device=self.device) % 35 + 1
                    )
                    offset += rows
                layer = AttentionFP8.__new__(AttentionFP8)
                torch.nn.Module.__init__(layer)
                layer.n_heads, layer.head_dim, layer.dim = heads, dim, heads * dim
                layer.softmax_scale, layer.attn_sink = dim**-0.5, sink
                layer._prefill_output_proj_into = (
                    lambda attention, _freqs, *, out: out.copy_(
                        attention.reshape(-1, heads * dim)
                    )
                )
                layer._prefill_output_all_reduce = lambda _out: None
                freqs = torch.zeros(
                    tokens, 1, dtype=torch.complex64, device=self.device
                )

                def forward(start, end):
                    return layer._flash_mla_sparse_fwd_chunked_projected(
                        q=query[start:end],
                        kv=values.unsqueeze(1),
                        indices=indices[start:end].unsqueeze(1),
                        topk_length=lens[start:end],
                        freqs_cis=freqs[start:end],
                        prefill_workspace=ws,
                        profile_name="sm120.prefill.cp_batch_regression",
                        backend_inputs=SparsePrefillInputs(
                            swa_cache=cache,
                            swa_indices=indices[start:end],
                            swa_lens=lens[start:end],
                        ),
                    )

                expected = []
                offset = 0
                for rows in lengths:
                    if rows:
                        # The production workspace contract requires Q at its base.
                        original = query.clone()
                        query[:rows].copy_(original[offset : offset + rows])
                        saved_indices, saved_lens = indices.clone(), lens.clone()
                        indices[:rows].copy_(saved_indices[offset : offset + rows])
                        lens[:rows].copy_(saved_lens[offset : offset + rows])
                        expected.append(forward(0, rows))
                        query.copy_(original)
                        indices.copy_(saved_indices)
                        lens.copy_(saved_lens)
                    offset += rows
                layer._cp_ctx = SimpleNamespace(chunk_lengths_per_req=lengths)
                actual = forward(0, tokens)
                self.assertTrue(torch.equal(actual, torch.cat(expected)))

    def test_hca_prefill_eager_and_cuda_graph(self) -> None:
        self._assert_prefill_variant("hca")

    def test_prefill_over_64_rows_non_zero_query_matches_reference(self) -> None:
        self._assert_prefill_variant("hca", tokens=72)

    def test_packed_hca_mixed_empty_rows_ignore_uninitialized_storage(self) -> None:
        batch, queries, heads, dim = 2, 4, 64, 512
        query = torch.randn(
            batch, queries, heads, dim, dtype=torch.bfloat16, device=self.device
        )
        sink = torch.linspace(-0.3, 0.4, heads, device=self.device)
        swa_cache = self._packed_cache(
            torch.randn(128, dim, dtype=torch.bfloat16, device=self.device),
            block_size=64,
        )
        hca_cache = self._packed_cache(
            torch.randn(1, dim, dtype=torch.bfloat16, device=self.device),
            block_size=2,
        )
        swa_indices = torch.arange(128, dtype=torch.int32, device=self.device)
        swa_indices = swa_indices.expand(batch, queries, 128).contiguous()
        hca_indices = torch.full(
            (batch, queries, 256), -1, dtype=torch.int32, device=self.device
        )
        # A gamma-3 verification crosses the first HCA compression boundary.
        hca_indices[0, 1:, 0] = 0
        swa_lens = torch.tensor([128, 128], dtype=torch.int32, device=self.device)
        hca_lens = torch.tensor([1, 0], dtype=torch.int32, device=self.device)
        sparse_ws = sm120_sparse_mla.Sm120SparseMlaWorkspace()
        op = SparseAttnV4DecodeFp8Op(heads, dim, dim**-0.5, attn_workspace=sparse_ws)
        swa_values, _ = self._gather_dequantized(swa_cache, [list(range(128))])
        hca_values, _ = self._gather_dequantized(hca_cache, [[0]])
        original_empty = torch.empty

        def forward():
            return op.forward(
                query,
                swa_cache,
                sink,
                swa_indices,
                None,
                topk_length=swa_lens,
                extra_k_cache=hca_cache,
                extra_topk_idxs=hca_indices,
                extra_topk_length=hca_lens,
            )

        def reference():
            return self._reference_sparse_prefill(
                query.flatten(0, 1),
                sink,
                swa_values[0],
                swa_indices.flatten(0, 1),
                swa_lens.repeat_interleave(queries),
                dim**-0.5,
                hca_values[0],
                hca_indices.flatten(0, 1),
                hca_lens.repeat_interleave(queries),
            ).reshape_as(query)

        def evaluate(fill, capture):
            def allocation(*args, **kwargs):
                result = original_empty(*args, **kwargs)
                if (
                    result.dtype == torch.uint8
                    and result.ndim == 3
                    and result.shape[-1] == 584
                ):
                    result.fill_(fill)
                return result

            with patch("torch.empty", allocation):
                for _ in range(3):
                    forward()
                if capture:
                    sm120_sparse_mla.warmup(self.device, owner=sparse_ws)
                    graph = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(graph):
                        out = forward()
                    query.neg_()
                    graph.replay()
                    torch.testing.assert_close(out, reference(), rtol=3e-2, atol=3e-2)
                    query.neg_()
                    graph.replay()
                else:
                    out = forward()
            self.assertTrue(bool(torch.isfinite(out).all()))
            torch.testing.assert_close(out, reference(), rtol=3e-2, atol=3e-2)
            return out.clone()

        for capture in (False, True):
            with self.subTest(capture=capture):
                clean = evaluate(0, capture)
                poisoned = evaluate(0x7F, capture)
                torch.testing.assert_close(clean, poisoned, rtol=0, atol=0)

    def test_large_decode_window_uses_graph_safe_generic_fallback(self) -> None:
        width, heads, dim = 1152, 8, 512
        scale = dim**-0.5
        source = torch.randn(width + 7, dim, dtype=torch.bfloat16, device=self.device)
        cache = self._packed_cache(source, block_size=64)
        reference_source, _ = self._gather_dequantized(
            cache, [list(range(source.shape[0]))]
        )
        reference_source = reference_source.squeeze(0)
        indices = torch.arange(width, dtype=torch.int32, device=self.device).view(
            1, 1, width
        )
        lengths = torch.tensor([width], dtype=torch.int32, device=self.device)
        query = torch.randn(1, 1, heads, dim, dtype=torch.bfloat16, device=self.device)
        sink = torch.linspace(-0.2, 0.3, heads, dtype=torch.float32, device=self.device)
        op = SparseAttnV4DecodeFp8Op(heads, dim, scale)

        def forward() -> torch.Tensor:
            return op.forward(
                query,
                cache,
                sink,
                indices,
                sched_meta=None,
                topk_length=lengths,
            )

        def reference() -> torch.Tensor:
            expected = self._reference_sparse_prefill(
                query.reshape(1, heads, dim),
                sink,
                reference_source,
                indices.reshape(1, width),
                lengths,
                scale,
            )
            return expected.view_as(query)

        eager = forward()
        torch.testing.assert_close(eager, reference(), rtol=4e-2, atol=4e-2)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            graph_output = forward()
        query.copy_(torch.randn_like(query))
        graph.replay()
        torch.cuda.synchronize(self.device)
        torch.testing.assert_close(graph_output, reference(), rtol=4e-2, atol=4e-2)

    @staticmethod
    def _reference_dual_pool_attention(
        query: torch.Tensor,
        sink: torch.Tensor,
        swa_values: torch.Tensor,
        swa_lengths: torch.Tensor,
        extra_values: torch.Tensor,
        extra_lengths: torch.Tensor,
        scale: float,
    ) -> torch.Tensor:
        outputs = []
        for row in range(query.shape[0]):
            swa_len = int(swa_lengths[row].item())
            extra_len = int(extra_lengths[row].item())
            values = torch.cat(
                (swa_values[row, :swa_len], extra_values[row, :extra_len]), dim=0
            ).float()
            q = query[row, 0].float()
            logits = torch.einsum("hd,td->ht", q, values) * scale
            scores_max = torch.maximum(logits.amax(dim=-1), sink).unsqueeze(-1)
            exp_logits = torch.exp(logits - scores_max)
            exp_sink = torch.exp(sink.unsqueeze(-1) - scores_max)
            probs = exp_logits / (exp_logits.sum(dim=-1, keepdim=True) + exp_sink)
            outputs.append(torch.einsum("ht,td->hd", probs, values))
        return torch.stack(outputs, dim=0).unsqueeze(1).to(query.dtype)

    def test_output_projection_eager_and_cuda_graph_against_dequantized_reference(
        self,
    ) -> None:
        groups, rank = 2, 128
        for width in (512, 1024):
            with self.subTest(width=width):
                layer = AttentionFP8.__new__(AttentionFP8)
                torch.nn.Module.__init__(layer)
                layer.o_lora_rank = rank

                weight = (
                    torch.randn(groups * rank, width, device=self.device) * 0.2
                ).to(torch.float8_e4m3fn)
                idx = torch.arange(
                    groups * (rank // 128) * (width // 128),
                    dtype=torch.float32,
                    device=self.device,
                )
                weight_exponents = ((idx * 3 + 1) % 7 - 3).view(
                    groups, rank // 128, width // 128
                )
                weight_scale = torch.exp2(weight_exponents).to(torch.float8_e8m0fnu)
                layer.wo_a_s = weight_scale.view(
                    groups * (rank // 128), width // 128
                )
                layer._wo_a_stk_w, layer._wo_a_stk_s = get_sparse_attention_backend(
                    self.device
                ).prepare_grouped_output(weight, layer.wo_a_s, groups, rank, width)

                dequantized_weight = (
                    weight.view(groups, rank, width).float()
                    * weight_scale.float()
                    .repeat_interleave(128, dim=1)
                    .repeat_interleave(128, dim=2)
                )

                def make_input(rows: int, offset: int):
                    fp8 = (torch.randn(rows, groups, width, device=self.device) * 0.25).to(
                        torch.float8_e4m3fn
                    )
                    exponents = (
                        torch.arange(
                            rows * groups * (width // 128),
                            dtype=torch.float32,
                            device=self.device,
                        ).view(rows, groups, width // 128)
                        + offset
                    ) % 5 - 2
                    scale = torch.exp2(exponents).to(torch.float8_e8m0fnu)
                    packed_scale = get_mn_major_tma_aligned_packed_ue8m0_tensor(
                        scale.float().transpose(0, 1)
                    ).transpose(0, 1)
                    return fp8, scale, packed_scale

                def reference(fp8: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
                    dequantized_input = fp8.float() * scale.float().repeat_interleave(
                        128, dim=-1
                    )
                    return (
                        torch.einsum(
                            "mgk,grk->mgr", dequantized_input, dequantized_weight
                        )
                        .to(torch.bfloat16)
                        .view(1, fp8.shape[0], groups, rank)
                    )

                for rows in (1, 3, 5):
                    with self.subTest(rows=rows):
                        fp8, scale, packed_scale = make_input(rows, 0)
                        eager = layer._wo_a_einsum_from_fp8(
                            fp8, packed_scale, 1, rows
                        )
                        torch.testing.assert_close(
                            eager.float(),
                            reference(fp8, scale).float(),
                            rtol=3e-2,
                            atol=2e-1,
                        )

                        static_fp8 = fp8.clone()
                        static_scale = get_mn_major_tma_aligned_packed_ue8m0_tensor(
                            scale.float().transpose(0, 1)
                        ).transpose(0, 1)
                        graph = torch.cuda.CUDAGraph()
                        with torch.cuda.graph(graph):
                            graph_output = layer._wo_a_einsum_from_fp8(
                                static_fp8, static_scale, 1, rows
                            )

                        replay_fp8, replay_scale, replay_packed_scale = make_input(
                            rows, 1
                        )
                        static_fp8.copy_(replay_fp8)
                        static_scale.copy_(replay_packed_scale)
                        graph.replay()
                        torch.cuda.synchronize(self.device)
                        torch.testing.assert_close(
                            graph_output.float(),
                            reference(replay_fp8, replay_scale).float(),
                            rtol=3e-2,
                            atol=2e-1,
                        )

    def test_hca_8192_eager_and_cuda_graph_replay(self) -> None:
        swa_source = torch.randn((12, 512), dtype=torch.bfloat16, device=self.device)
        extra_source = (
            torch.randn((12, 512), dtype=torch.bfloat16, device=self.device) * 0.7 + 0.2
        )
        cache = self._packed_cache(swa_source, block_size=64)
        extra_cache = self._packed_cache(extra_source, block_size=2)
        query = torch.randn((2, 1, 16, 512), dtype=torch.bfloat16, device=self.device)
        sink = torch.linspace(-0.4, 0.6, 16, dtype=torch.float32, device=self.device)
        swa_indices = torch.full((2, 1, 128), -1, dtype=torch.int32, device=self.device)
        swa_indices[0, 0, :3] = torch.tensor(
            [0, -1, 5], dtype=torch.int32, device=self.device
        )
        swa_indices[1, 0, :3] = torch.tensor(
            [7, 9, -1], dtype=torch.int32, device=self.device
        )
        swa_length = torch.full((2,), 3, dtype=torch.int32, device=self.device)
        # HCA emits one compressed entry per 128 source tokens.  Width 8192 is
        # therefore the static FlashInfer instance needed by a 1M context.
        extra_indices = torch.full(
            (2, 1, 8192), -1, dtype=torch.int32, device=self.device
        )
        extra_indices[0, 0, :4] = torch.tensor(
            [0, -1, 3, 5], dtype=torch.int32, device=self.device
        )
        extra_indices[1, 0, :3] = torch.tensor(
            [2, 7, -1], dtype=torch.int32, device=self.device
        )
        extra_length = torch.tensor([4, 3], dtype=torch.int32, device=self.device)
        scale = 512**-0.5
        sparse_ws = sm120_sparse_mla.Sm120SparseMlaWorkspace()
        op = SparseAttnV4DecodeFp8Op(16, 512, scale, attn_workspace=sparse_ws)

        swa_values, swa_value_lengths = self._gather_dequantized(
            cache, [[0, 5], [7, 9]]
        )
        extra_values, extra_value_lengths = self._gather_dequantized(
            extra_cache, [[0, 3, 5], [2, 7]]
        )

        def reference() -> torch.Tensor:
            return self._reference_dual_pool_attention(
                query,
                sink,
                swa_values,
                swa_value_lengths,
                extra_values,
                extra_value_lengths,
                scale,
            )

        def forward() -> torch.Tensor:
            return sm120_sparse_mla.forward_decode(
                query,
                cache,
                sink,
                swa_indices,
                swa_length,
                extra_cache,
                extra_indices,
                extra_length,
                softmax_scale=op.softmax_scale,
                workspace_owner=sparse_ws,
            )

        # First validate the production kernel against independent PyTorch
        # attention math, then prove graph replay consumes an updated query.
        eager = forward()
        self._assert_sparse_output_close(eager, reference())
        sm120_sparse_mla.warmup(self.device, owner=sparse_ws)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            graph_output = forward()

        query.copy_(torch.randn_like(query))
        graph.replay()
        torch.cuda.synchronize(self.device)
        replay_output = graph_output.clone()
        eager_output = forward()
        reference_output = reference()

        self.assertTrue(torch.isfinite(replay_output).all())
        self.assertGreater(replay_output.float().abs().max().item(), 0.0)
        torch.testing.assert_close(
            replay_output,
            eager_output,
            rtol=3e-2,
            atol=3e-2,
        )
        self._assert_sparse_output_close(replay_output, reference_output)


if __name__ == "__main__":
    unittest.main()
