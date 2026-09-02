"""RTX PRO 5000 gate for the long-context SM120 sparse-MLA path."""

from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from rtp_llm.config.quant_config import Fp8BlockWiseQuantConfig
from rtp_llm.device.device_type import DeviceType
from rtp_llm.model_loader.per_block_fp8_quant_weight import PerBlockFp8Weight
from rtp_llm.model_loader.weight_module import CompositeWeight
from rtp_llm.models_py.modules.dsv4.fp8._swa_dequant_triton import (
    dequantize_and_gather_k_cache_slots,
)
from rtp_llm.models_py.modules.dsv4.fp8._swa_kv_insert_triton import (
    quantize_and_insert_k_cache,
)
from rtp_llm.models_py.modules.dsv4.fp8.attention import AttentionFP8
from rtp_llm.models_py.modules.dsv4.fp8.decode.fp8_sparse_attn_decode_op import (
    SparseAttnV4DecodeFp8Op,
)
from rtp_llm.models_py.modules.dsv4.prefill_workspace import PrefillWorkspace
from rtp_llm.models_py.modules.factory.linear import LinearFactory
from rtp_llm.models_py.modules.factory.linear.impl.cuda.fp8_vllm_blockwise_sm120_linear import (
    CudaFp8VllmBlockwiseLinear,
)
from rtp_llm.models_py.utils.arch import is_sm120


class Sm120SparseMlaHardwareTest(unittest.TestCase):
    def setUp(self) -> None:
        if not torch.cuda.is_available():
            self.fail("SM120 hardware gate was scheduled without a CUDA device")
        if not is_sm120():
            self.fail(
                "SM120 hardware gate was scheduled on "
                f"compute capability {torch.cuda.get_device_capability()}"
            )
        self.device = torch.device("cuda", torch.cuda.current_device())
        torch.manual_seed(7)

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

    def _assert_prefill_variant(self, variant: str) -> None:
        tokens, heads, dim = 3, 8, 512
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
        swa_cache = self._packed_cache(swa_values, block_size=64)

        extra_values = extra_indices = extra_lengths = None
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
            extra_cache = self._packed_cache(
                extra_values, block_size=64 if variant == "csa" else 2
            )

        layer = AttentionFP8.__new__(AttentionFP8)
        torch.nn.Module.__init__(layer)
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
                sm120_swa_cache=swa_cache,
                sm120_extra_cache=extra_cache,
                sm120_swa_indices=swa_indices,
                sm120_swa_lens=swa_lengths,
                sm120_extra_indices=extra_indices,
                sm120_extra_lens=extra_lengths,
            )

        def reference() -> torch.Tensor:
            return self._reference_sparse_prefill(
                query,
                sink,
                swa_values,
                swa_indices,
                swa_lengths,
                scale,
                extra_values,
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
            torch.testing.assert_close(eager_out, reference(), rtol=3e-2, atol=3e-2)

            graph_out = torch.empty_like(eager_out)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                forward(graph_out)
            query.copy_(torch.randn_like(query))
            graph.replay()
            torch.cuda.synchronize(self.device)
            torch.testing.assert_close(graph_out, reference(), rtol=3e-2, atol=3e-2)

    def test_swa_prefill_eager_and_cuda_graph(self) -> None:
        self._assert_prefill_variant("swa")

    def test_csa_prefill_eager_and_cuda_graph(self) -> None:
        self._assert_prefill_variant("csa")

    def test_hca_prefill_eager_and_cuda_graph(self) -> None:
        self._assert_prefill_variant("hca")

    def test_large_decode_window_uses_graph_safe_generic_fallback(self) -> None:
        width, heads, dim = 1152, 8, 512
        scale = dim**-0.5
        source = torch.randn(width + 7, dim, dtype=torch.bfloat16, device=self.device)
        cache = self._packed_cache(source, block_size=64)
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
                source,
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

    def test_cutlass_blockwise_linear_eager_and_cuda_graph(self) -> None:
        # Deliberately rectangular and non-uniform across output blocks: a
        # loader-side (N,K)->(K,N) reshape cannot hide behind square geometry
        # or all-one scales.
        # Simulate force_cpu_load_weights: post-processing sees a CPU tensor,
        # while exported_device identifies the final SM120 inference target.
        weight = (
            (torch.randn((384, 256), dtype=torch.float32, device=self.device) * 0.05)
            .to(torch.float8_e4m3fn)
            .cpu()
        )
        weight_scale = torch.tensor(
            [[0.50, 0.75], [1.00, 1.25], [1.50, 1.75]], dtype=torch.float32
        )

        loader = PerBlockFp8Weight.__new__(PerBlockFp8Weight)
        loader.kernel = SimpleNamespace(name="kernel")
        loader.scale = SimpleNamespace(name="scale")
        exported_device = SimpleNamespace(
            get_device_type=lambda: DeviceType.Cuda,
            get_device_id=lambda: self.device.index,
            maybe_rewrite_weight_by_key=lambda _key, value: value,
        )
        with patch.object(
            CompositeWeight,
            "_postprocess",
            return_value={"kernel": weight, "scale": weight_scale},
        ):
            processed = loader._postprocess(
                {"kernel": weight, "scale": weight_scale},
                "cpu",
                SimpleNamespace(exported_device=exported_device),
            )

        self.assertEqual(processed["kernel"].device.type, "cpu")
        self.assertEqual(tuple(processed["kernel"].shape), (384, 256))
        self.assertEqual(tuple(processed["scale"].shape), (3, 2))
        processed = {key: value.to(self.device) for key, value in processed.items()}
        linear = LinearFactory.create_linear(
            processed["kernel"],
            None,
            processed["scale"],
            Fp8BlockWiseQuantConfig(),
        )
        self.assertIsInstance(linear, CudaFp8VllmBlockwiseLinear)
        dequant_weight = processed["kernel"].float() * processed[
            "scale"
        ].repeat_interleave(128, 0).repeat_interleave(128, 1)

        # 260 and 384 exercise the generic CUTLASS dispatch beyond the tuned
        # 64/65 and 256/257 boundary cases.
        for rows in (64, 65, 256, 257, 260, 384):
            with self.subTest(rows=rows):
                activation = torch.randn(
                    (rows, 256), dtype=torch.bfloat16, device=self.device
                )
                eager = linear(activation)
                reference = activation.float() @ dequant_weight.transpose(0, 1)
                torch.testing.assert_close(
                    eager.float(),
                    reference,
                    rtol=2e-1,
                    atol=2e-1,
                )

                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    graph_output = linear(activation)
                activation.copy_(torch.randn_like(activation))
                graph.replay()
                torch.cuda.synchronize(self.device)
                replay_output = graph_output.clone()
                eager_after_update = linear(activation)
                torch.testing.assert_close(replay_output, eager_after_update)

    def test_output_projection_eager_and_cuda_graph_against_dequantized_reference(
        self,
    ) -> None:
        groups, rank, width = 2, 128, 512
        layer = AttentionFP8.__new__(AttentionFP8)
        torch.nn.Module.__init__(layer)
        layer.o_lora_rank = rank

        weight = (torch.randn(groups, rank, width, device=self.device) * 0.2).to(
            torch.float8_e4m3fn
        )
        weight_exponents = (
            torch.arange(
                groups * (rank // 128) * (width // 128),
                dtype=torch.float32,
                device=self.device,
            ).view(groups, rank // 128, width // 128)
            % 4
            - 2
        )
        weight_scale = torch.exp2(weight_exponents).to(torch.float8_e8m0fnu)
        layer._wo_a_stk_w = weight
        layer.wo_a_s = weight_scale.view(groups * (rank // 128), width // 128)

        dequantized_weight = weight.float() * weight_scale.float().repeat_interleave(
            128, dim=1
        ).repeat_interleave(128, dim=2)

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
            # Production fused_inv_rope_fp8_quant packs four UE8M0 bytes into
            # each int32 along K. Preserve that exact ABI here.
            packed_scale = scale.contiguous().view(torch.int32)
            return fp8, scale, packed_scale

        def reference(fp8: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
            dequantized_input = fp8.float() * scale.float().repeat_interleave(
                128, dim=-1
            )
            return (
                torch.einsum("mgk,grk->mgr", dequantized_input, dequantized_weight)
                .to(torch.bfloat16)
                .view(1, fp8.shape[0], groups, rank)
            )

        # M=1 covers the decode boundary; 3 and 5 exercise both sides of the
        # internal four-row padding boundary. G=2 verifies grouped projection.
        for rows in (1, 3, 5):
            with self.subTest(rows=rows):
                fp8, scale, packed_scale = make_input(rows, 0)
                eager = layer._wo_a_einsum_from_fp8(fp8, packed_scale, 1, rows)
                torch.testing.assert_close(
                    eager.float(),
                    reference(fp8, scale).float(),
                    rtol=3e-2,
                    atol=2e-1,
                )

                static_fp8 = fp8.clone()
                static_scale = packed_scale.clone()
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    graph_output = layer._wo_a_einsum_from_fp8(
                        static_fp8, static_scale, 1, rows
                    )

                replay_fp8, replay_scale, replay_packed_scale = make_input(rows, 1)
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
        op = SparseAttnV4DecodeFp8Op(16, 512, scale)

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
            return op._forward_sm120_flashinfer(
                query,
                cache,
                sink,
                swa_indices,
                swa_length,
                extra_cache,
                extra_indices,
                extra_length,
            )

        # First validate the production kernel against independent PyTorch
        # attention math, then prove graph replay consumes an updated query.
        eager = forward()
        torch.testing.assert_close(eager, reference(), rtol=3e-2, atol=3e-2)
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
            rtol=2e-2,
            atol=2e-2,
        )
        torch.testing.assert_close(
            replay_output,
            reference_output,
            rtol=3e-2,
            atol=3e-2,
        )


if __name__ == "__main__":
    unittest.main()
