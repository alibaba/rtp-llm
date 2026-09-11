"""Single-GPU tests of TP8's rank-local CMP query and output projections.

This executes the production Glm5Cmp bridge and locked RTP-kernel operators,
with actual head-sharded weights and non-unit packed UE8M0 scales. It is NOT an
eight-rank test: all_reduce is explicitly replaced with identity. The projection
tests exclude Indexer, KV production, sparse attention, residual/norm, and MoE.
The separate prologue integration test executes real QKV producers, reuse and
three-stream Indexer producer branches, and prepared TRT attention. Only its
paged Indexer scoring/TopK is deterministic; it is not a full model test.

The ordinary reference uses CudaFp8DeepGEMMLinear, the existing Triton Q-RoPE
kernel restricted to its Q tiles (no KV work), and SparseMlaImpl's cuBLAS BMMs.
Set GLM5_CMP_TP8_BENCHMARK=1 to emit 31 alternating paired graph samples. Both
paths use the same input/weights, but separate outputs; graph timings include
Q-B/RoPE/Wkc plus Wvc/quant/O, never Indexer/TRT/all-reduce or a full CMP layer.
Run only on an explicitly reserved SM10x GPU with the locked RTP-kernel wheel
on PYTHONPATH; the old user-site package does not contain rtp_kernel.glm5.
"""

import importlib
import json
import math
import os
import statistics
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch


def _repo_root():
    for root in Path(__file__).resolve().parents:
        if (root / "rtp_llm/models_py/modules/hybrid/glm5_cmp.py").is_file():
            return root
    raise RuntimeError("cannot locate repository root")


def _pack_ue8m0(exponents):
    rows, groups = exponents.shape
    assert groups % 4 == 0
    aligned_rows = (rows + 3) // 4 * 4
    packed = torch.empty(
        aligned_rows * (groups // 4), device=exponents.device, dtype=torch.int32
    ).as_strided((rows, groups // 4), (1, aligned_rows))
    packed.copy_(exponents.contiguous().view(torch.int32))
    return packed


def _quantize_block_weight(values):
    """128x128 block scaling, expanded to the deployed native packed layout."""
    rows, width = values.shape
    if rows % 128:
        padded = torch.nn.functional.pad(values, (0, 0, 0, 128 - rows % 128))
        quantized, packed = _quantize_block_weight(padded)
        exponent = packed[:rows].contiguous().view(torch.uint8)
        return quantized[:rows].contiguous(), _pack_ue8m0(exponent)
    blocks = values.float().view(rows // 128, 128, width // 128, 128)
    initial = blocks.abs().amax((1, 3)).clamp_min(1.0e-4) / 448.0
    bits = initial.contiguous().view(torch.int32)
    exponent = (((bits >> 23) & 255) + ((bits & 0x7FFFFF) != 0)).clamp(1, 254)
    scale = torch.exp2(exponent.float() - 127)
    quantized = (blocks / scale[:, None, :, None]).clamp(-448, 448)
    packed = _pack_ue8m0(exponent.repeat_interleave(128, 0).to(torch.uint8))
    return quantized.to(torch.float8_e4m3fn).reshape_as(values), packed


def _dequantize(values, packed):
    exponent = packed.contiguous().view(torch.uint8).int()
    return values.float() * torch.exp2(exponent.float() - 127).repeat_interleave(128, 1)


def _errors(actual, expected):
    difference = (actual.float() - expected.float()).abs()
    return {
        "max_abs": difference.max().item(),
        "relative_l2": (
            difference.norm() / expected.float().norm().clamp_min(1.0e-20)
        ).item(),
        "equal_fraction": (actual == expected).float().mean().item(),
    }


class LocalProjectionFixture:
    def __init__(self, modules, rows, neox=False):
        self.modules, self.rows, self.neox = modules, rows, neox
        self.device = torch.device("cuda:0")
        rng = torch.Generator(device=self.device).manual_seed(913 + rows)

        def random(shape, scale=1.0):
            return (
                torch.randn(shape, device=self.device, generator=rng) * scale
            ).bfloat16()

        self.q_source = random((rows, 2048))
        self.q_fp8, self.q_scale = self.quantize(self.q_source)
        self.q_weight, self.q_weight_scale = _quantize_block_weight(
            random((2048, 2048), 1 / math.sqrt(2048))
        )
        self.o_weight, self.o_weight_scale = _quantize_block_weight(
            random((6144, 2048), 1 / math.sqrt(2048))
        )
        self.wkc = random((8, 192, 512), 1 / math.sqrt(192))
        self.wvc = random((8, 512, 256), 1 / math.sqrt(512))
        self.mla_output = random((rows, 8, 512))
        self.residual = random((rows, 6144))
        angles = torch.randn((1024, 32), device=self.device, generator=rng)
        self.cos_sin = torch.cat((angles.cos(), angles.sin()), dim=1)
        self.positions = torch.randint(
            0, 1024, (rows,), device=self.device, dtype=torch.int64, generator=rng
        )

        self.q_linear = modules.Linear(self.q_weight, self.q_weight_scale)
        self.o_linear = modules.Linear(self.o_weight, self.o_weight_scale)
        self.impl = object.__new__(modules.Sparse)
        self.impl.num_heads = 8
        self.impl.nope_head_dim, self.impl.rope_head_dim = 192, 64
        self.impl.kv_lora_rank = 512
        self.impl.weights = [{modules.W.mla_kc: self.wkc, modules.W.mla_vc: self.wvc}]
        self.impl._cos_sin_cache = self.cos_sin
        self.impl._is_neox_style = neox
        self.impl.fmha_params = SimpleNamespace(positions_d=self.positions)

        self.bridge = object.__new__(modules.cmp.Glm5Cmp)
        self.bridge.layer_idx = 0
        self.bridge.ops = modules.ops
        self.bridge._q_b_proj = self.q_weight, self.q_weight_scale
        self.bridge._output_projection = self.o_weight, self.o_weight_scale
        self.bridge.self_attn = SimpleNamespace(
            num_heads=8, q_b_proj=self.q_linear, o_proj=self.o_linear
        )
        self.bridge.parallelism_config = SimpleNamespace(get_attn_tp_size=lambda: 8)
        self.bridge._is_moe_layer = False
        self.bridge.disable_attention_post_moe_pre = False
        self.projected = torch.empty(
            (rows, 2048), device=self.device, dtype=torch.bfloat16
        )
        self.nope = torch.empty(
            (rows, 8, 192), device=self.device, dtype=torch.bfloat16
        )
        self.query = torch.empty(
            (rows, 8, 576), device=self.device, dtype=torch.bfloat16
        )
        self.reference_projected = torch.empty_like(self.projected)
        self.reference_query = torch.empty_like(self.query)
        # The existing kernel receives valid dummy operands, but the Q-only
        # launch grid never executes its latent/K-RoPE/KV-cache programs.
        self.dummy_k = torch.zeros((rows, 64), device=self.device, dtype=torch.bfloat16)
        self.dummy_kv = torch.zeros(
            (rows, 512), device=self.device, dtype=torch.bfloat16
        )
        self.dummy_cache = torch.zeros(
            (1, 64, 656), device=self.device, dtype=torch.uint8
        )
        self.dummy_slots = torch.full(
            (rows,), -1, device=self.device, dtype=torch.int64
        )
        self.fp8_scale_min = modules.rope._get_fp8_scale_min(self.device)

    def quantize(self, values):
        return self.modules.quantize(
            values,
            group_size=128,
            eps=1.0e-4,
            column_major_scales=True,
            scale_tma_aligned=True,
            scale_ue8m0=True,
        )

    def reference_rope(self, q):
        module = self.modules.rope
        tiles = (8 + module._BLOCK_H_TILE - 1) // module._BLOCK_H_TILE
        fp8 = self.dummy_cache.view(torch.float8_e4m3fn)
        fp32 = self.dummy_cache.view(torch.float32)
        bf16 = self.dummy_cache.view(torch.bfloat16)
        module._fused_qk_rope_cat_cache_mla_fp8_kernel[(self.rows, tiles)](
            q,
            self.dummy_k,
            fp8,
            fp32,
            bf16,
            self.dummy_kv,
            self.dummy_slots,
            self.positions,
            self.cos_sin,
            self.fp8_scale_min,
            q.stride(0),
            q.stride(1),
            self.dummy_k.stride(0),
            self.dummy_kv.stride(0),
            fp8.stride(0),
            fp8.stride(1),
            fp32.stride(0),
            fp32.stride(1),
            bf16.stride(0),
            bf16.stride(1),
            BLOCK_SIZE=64,
            H=8,
            Q_ROPE_OFFSET=192,
            KV_LORA=512,
            ROPE=64,
            HALF_ROPE=32,
            QUANT_BLOCK=128,
            IS_NEOX=self.neox,
            BLOCK_H_TILE=module._BLOCK_H_TILE,
            H_TILES=tiles,
            NUM_K_BLOCKS=module._NUM_FP8_K_BLOCKS,
            num_warps=2,
            num_stages=2,
        )

    def candidate_query(self):
        nope, query = self.bridge._project_query(
            self.q_fp8,
            self.q_scale,
            self.impl,
            out=(self.nope, self.query),
        )
        self.modules.ops.absorbed_q_nope_bmm(nope, self.wkc, out=query)
        return query

    def reference_query_forward(self):
        projected = self.q_linear(
            self.q_fp8, input_scales=self.q_scale, out=self.reference_projected
        ).view(self.rows, 8, 256)
        self.reference_rope(projected)
        return self.impl._apply_input_bmm(projected, 0, out=self.reference_query)

    def candidate_post(self):
        return self.bridge.mla_post_moe_pre(self.mla_output, self.residual, self.impl)[
            0
        ]

    def reference_post(self):
        return self.bridge._standard_mla_post(self.mla_output, self.impl)

    def candidate(self):
        return self.candidate_query(), self.candidate_post()

    def reference(self):
        return self.reference_query_forward(), self.reference_post()

    def change_inputs(self, step):
        self.q_source.mul_(-0.8)
        quantized, scale = self.quantize(self.q_source)
        self.q_fp8.copy_(quantized)
        self.q_scale.copy_(scale)
        self.positions.copy_((self.positions + 97 + step).remainder(1024))
        self.mla_output.mul_(-0.875)


class PrologueFixture(LocalProjectionFixture):
    """Actual CMP prologue; only paged Indexer scoring/TopK is substituted.

    The sequential oracle reuses the replicated RTP QKV-A/norm/cache producers,
    but uses ordinary DeepGEMM/Q-RoPE/cuBLAS for the head-sharded query. It is an
    integration/ordering oracle, not independent model-accuracy validation of
    those unchanged QKV/Indexer kernels.
    """

    def __init__(self, modules, with_indexer):
        super().__init__(modules, rows=6, neox=False)
        attention_tests = _repo_root() / (
            "rtp_llm/models_py/modules/factory/attention/cuda_mla_impl/test"
        )
        sys.path.insert(0, str(attention_tests))
        from trtllm_sparse_decode_test import PackedFixture

        self.with_indexer = with_indexer
        self.packed = PackedFixture(batch=1, queries=6, heads=8, length=128, seed=429)
        self.packed.set_counts([65] * self.rows)
        self.live_topk = self.packed.topk.clone()
        self.positions = self.positions.int()
        self.impl.fmha_params = self.packed.params
        self.impl.fmha_params.positions_d = self.positions
        self.impl.fmha_params.slot_mapping = self.packed.table[
            0, 0
        ].long() * 64 + torch.arange(self.rows, device=self.device)
        self.impl.attn_inputs = self.packed.attn_inputs
        self.impl.attn_inputs.kv_cache_block_id_device = self.packed.table
        self.impl.pinned_mla_groups = {}
        self.impl.fmha_impl = self.packed.new_op(cuda_graph=True)
        self.reference_op = self.packed.new_op(cuda_graph=True)
        self.layer_cache = SimpleNamespace(
            kv_cache_base=self.packed.cache,
            kv_scale_base=torch.zeros(
                (self.packed.pages, 64, 1, 132), dtype=torch.uint8, device=self.device
            ),
        )
        self.reference_cache = self.packed.cache.clone()
        self.reference_indexer_cache = self.layer_cache.kv_scale_base.clone()
        generator = torch.Generator(device=self.device).manual_seed(321)

        def random(shape, scale=1.0):
            return (
                torch.randn(shape, device=self.device, generator=generator) * scale
            ).bfloat16()

        self.hidden = random((self.rows, 6144))
        self.bridge.input_layernorm = SimpleNamespace(
            weight=torch.ones(6144, device=self.device, dtype=torch.bfloat16),
            variance_epsilon=1.0e-5,
        )
        self.bridge.self_attn.q_a_layernorm = SimpleNamespace(
            weight=torch.ones(2048, device=self.device, dtype=torch.bfloat16),
            variance_epsilon=1.0e-5,
        )
        self.bridge.self_attn.kv_a_layernorm = SimpleNamespace(
            weight=torch.ones(512, device=self.device, dtype=torch.bfloat16),
            variance_epsilon=1.0e-5,
        )
        self.bridge._qkv_projection = _quantize_block_weight(
            random((2624, 6144), 1 / math.sqrt(6144))
        )
        self.bridge.self_attn.has_indexer = with_indexer
        self.bridge._events = None
        self.bridge._packed_head_gate_weight = None
        self.indexer_calls = 0
        if with_indexer:
            self.bridge._indexer_k_projection = _quantize_block_weight(
                random((128, 6144), 1 / math.sqrt(6144))
            )
            self.bridge._indexer_q_projection = _quantize_block_weight(
                random((4096, 2048), 1 / math.sqrt(2048))
            )
            self.bridge.self_attn.indexer = SimpleNamespace(
                weights_proj=SimpleNamespace(
                    weight=random((32, 6144), 1 / math.sqrt(6144))
                ),
                k_norm=SimpleNamespace(
                    weight=torch.ones(128, device=self.device, dtype=torch.bfloat16),
                    beta=torch.zeros(128, device=self.device, dtype=torch.bfloat16),
                    variance_epsilon=1.0e-5,
                ),
                indexer_op=SimpleNamespace(_get_topk_paged=self.deterministic_topk),
            )

    def deterministic_topk(self, query, head_weights, cache, params, attn_inputs):
        """Device-read actual Indexer producers, then apply a live fixed selection."""
        self.indexer_calls += 1
        ready = torch.isfinite(query.float()).all(dim=(1, 2)) & torch.isfinite(
            head_weights
        ).all(dim=1)
        return torch.where(ready[:, None], self.packed.topk, -1)

    def candidate_prologue(self):
        residual, prepared, topk = self.bridge.mla_prologue(
            self.hidden,
            self.residual,
            self.impl,
            self.layer_cache,
            prev_topk_indices=self.packed.topk if not self.with_indexer else None,
            reuse_topk_indices=not self.with_indexer,
        )
        if isinstance(prepared, torch.Tensor):
            raise AssertionError("real TRT prepare path was not exercised")
        query = prepared.inputs[0]
        result = self.bridge.sparse_mla(prepared, topk, self.impl, self.layer_cache)
        return residual, query, result, topk

    def reference_prologue(self):
        ops = self.modules.ops
        attention = self.bridge.self_attn
        residual, _, hidden_fp8, hidden_scale = ops.add_norm_quant(
            self.hidden,
            self.residual,
            self.bridge.input_layernorm.weight,
            epsilon=1.0e-5,
        )
        projected = ops.qkv_a_proj(
            hidden_fp8, hidden_scale, *self.bridge._qkv_projection
        )
        q_fp8, q_scale, _ = ops.qkv_rmsnorm_quant_rope_cached(
            projected,
            attention.q_a_layernorm.weight,
            attention.kv_a_layernorm.weight,
            self.cos_sin,
            self.positions,
            self.impl.fmha_params.slot_mapping,
            cache=self.reference_cache,
        )
        q = self.q_linear(q_fp8, input_scales=q_scale).view(self.rows, 8, 256)
        self.reference_rope(q)
        query = self.impl._apply_input_bmm(q, 0)
        if self.with_indexer:
            norm = attention.indexer.k_norm
            ops.indexer_k_cache(
                hidden_fp8,
                hidden_scale,
                *self.bridge._indexer_k_projection,
                norm.weight,
                norm.beta,
                self.cos_sin,
                self.positions,
                self.impl.fmha_params.slot_mapping,
                self.reference_indexer_cache,
                epsilon=1.0e-5,
            )
        result = self.reference_op.forward(
            query, self.reference_cache, self.packed.topk, layer_id=0
        )
        return residual, query, result, self.packed.topk

    def change_prologue_inputs(self, step):
        self.hidden.mul_(-0.75)
        self.residual.mul_(0.9)
        self.positions.copy_((self.positions + 43 + step).remainder(1024))
        self.impl.fmha_params.slot_mapping.copy_(
            self.packed.table[0, 0].long() * 64
            + torch.arange(self.rows, device=self.device)
            + 12 * step
        )
        if step == 1:
            self.packed.topk.fill_(-1)
        else:
            self.packed.topk.copy_(self.live_topk)


class Glm5CmpTp8LocalGpuTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if (
            not torch.cuda.is_available()
            or torch.cuda.get_device_capability(0)[0] != 10
        ):
            raise unittest.SkipTest("requires an explicitly reserved SM10x GPU")
        sys.path.insert(0, str(_repo_root()))
        torch.cuda.set_device(0)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
        from rtp_kernel import glm5

        from rtp_llm.models_py.kernels.cuda.fp8_kernel import (
            sgl_per_token_group_quant_fp8,
        )
        from rtp_llm.models_py.modules.factory.attention.cuda_mla_impl.flashmla_sparse_impl import (
            SparseMlaImpl,
        )
        from rtp_llm.models_py.modules.factory.linear.impl.cuda.fp8_deepgemm_linear import (
            CudaFp8DeepGEMMLinear,
        )
        from rtp_llm.models_py.modules.hybrid import glm5_cmp
        from rtp_llm.utils.model_weight import W

        cls.modules = SimpleNamespace(
            ops=glm5,
            cmp=glm5_cmp,
            Sparse=SparseMlaImpl,
            W=W,
            Linear=CudaFp8DeepGEMMLinear,
            quantize=sgl_per_token_group_quant_fp8,
            rope=importlib.import_module(
                "rtp_llm.models_py.triton_kernels.sparse_mla.fused_qk_rope_cat_cache_mla"
            ),
        )
        print(
            json.dumps(
                {
                    "scope": "single_gpu_tp8_local_query_and_post_no_collectives",
                    "rtp_kernel_glm5": glm5.__file__,
                    "torch": torch.__version__,
                    "gpu": torch.cuda.get_device_name(0),
                }
            ),
            flush=True,
        )

    def setUp(self):
        self.reduce_patch = patch.object(
            self.modules.cmp, "all_reduce", side_effect=lambda tensor, **_: tensor
        )
        self.reduce = self.reduce_patch.start()
        self.addCleanup(self.reduce_patch.stop)

    def tearDown(self):
        torch.cuda.synchronize()

    def assert_numerics(self, actual, expected, label):
        self.assertTrue(bool(torch.isfinite(actual).all()), label)
        error = _errors(actual, expected)
        # BF16 GEMM reduction order may differ; this is a numerical comparison,
        # not a bitwise guarantee or an end-to-end model precision waiver.
        torch.testing.assert_close(actual, expected, atol=0.02, rtol=0.02, msg=label)
        self.assertLess(error["relative_l2"], 0.01, label)
        return error

    def test_real_projection_intermediates_and_post_all_requested_rows(self):
        for rows in (1, 4, 6, 64, 256):
            with self.subTest(rows=rows):
                fixture = LocalProjectionFixture(self.modules, rows)
                saved = [
                    x.clone()
                    for x in (
                        fixture.q_fp8,
                        fixture.q_scale,
                        fixture.q_weight,
                        fixture.q_weight_scale,
                        fixture.o_weight,
                        fixture.o_weight_scale,
                        fixture.positions,
                        fixture.mla_output,
                    )
                ]
                ordinary_projected = fixture.q_linear(
                    fixture.q_fp8, input_scales=fixture.q_scale
                )
                generic = self.modules.ops.indexer_q_proj(
                    fixture.q_fp8,
                    fixture.q_scale,
                    fixture.q_weight,
                    fixture.q_weight_scale,
                )
                gemm_error = self.assert_numerics(generic, ordinary_projected, "Q-B")
                independent = (
                    _dequantize(fixture.q_fp8, fixture.q_scale)
                    @ _dequantize(fixture.q_weight, fixture.q_weight_scale).T
                ).bfloat16()
                self.assert_numerics(generic, independent, "Q-B dequant FP32 reference")
                actual_query, actual_post = fixture.candidate()
                reference_query, reference_post = fixture.reference()
                query_error = self.assert_numerics(
                    actual_query, reference_query, "Q/Wkc"
                )
                post_error = self.assert_numerics(actual_post, reference_post, "Wvc/O")
                quantized, scales = self.modules.ops.mla_absorbed_output_bmm_quant(
                    fixture.mla_output, fixture.wvc
                )
                bmm_reference = fixture.impl._apply_output_bmm(fixture.mla_output, 0)
                expected_quantized, expected_scales = fixture.quantize(
                    bmm_reference.reshape(rows, 2048)
                )
                self.assertTrue(
                    torch.equal(scales, expected_scales), "Wvc UE8M0 scales"
                )
                self.assertTrue(
                    torch.equal(
                        quantized.view(torch.uint8),
                        expected_quantized.view(torch.uint8),
                    ),
                    "Wvc FP8 payload",
                )
                # The optional-output form used by MTP reuse must also support H8.
                nope, query = fixture.bridge._project_query(
                    fixture.q_fp8, fixture.q_scale, fixture.impl
                )
                self.assertEqual(tuple(nope.shape), (rows, 8, 192))
                self.assertEqual(tuple(query.shape), (rows, 8, 576))
                self.assertTrue(torch.equal(nope, fixture.nope))
                self.assertTrue(torch.equal(query[..., 512:], fixture.query[..., 512:]))
                for actual, before in zip(
                    (
                        fixture.q_fp8,
                        fixture.q_scale,
                        fixture.q_weight,
                        fixture.q_weight_scale,
                        fixture.o_weight,
                        fixture.o_weight_scale,
                        fixture.positions,
                        fixture.mla_output,
                    ),
                    saved,
                ):
                    self.assertTrue(
                        torch.equal(
                            actual.contiguous().view(torch.uint8),
                            before.contiguous().view(torch.uint8),
                        ),
                        "projection mutated an input",
                    )
                self.assertTrue(bool((fixture.dummy_cache == 0).all()))
                print(
                    json.dumps(
                        {
                            "rows": rows,
                            "Q-B": gemm_error,
                            "query": query_error,
                            "post": post_error,
                        }
                    ),
                    flush=True,
                )

    def test_graph_replay_changed_inputs_and_positions_both_rope_styles(self):
        for neox in (False, True):
            with self.subTest(neox=neox):
                fixture = LocalProjectionFixture(self.modules, 6, neox)
                for _ in range(3):
                    fixture.candidate()
                    fixture.reference()
                graphs, outputs = [], []
                for function in (fixture.candidate, fixture.reference):
                    graph = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(graph):
                        output = function()
                    graphs.append(graph)
                    outputs.append(output)
                graphs[0].replay()
                previous = outputs[0][0].clone()
                for step in range(3):
                    fixture.change_inputs(step)
                    for graph in graphs:
                        graph.replay()
                    captured = [tuple(x.clone() for x in pair) for pair in outputs]
                    for pair, function in zip(
                        captured, (fixture.candidate, fixture.reference)
                    ):
                        for actual, expected in zip(pair, function()):
                            torch.testing.assert_close(actual, expected, atol=0, rtol=0)
                    for actual, expected in zip(captured[0], captured[1]):
                        self.assert_numerics(actual, expected, "changed-input graph")
                    self.assertFalse(torch.equal(previous, captured[0][0]))
                    previous = captured[0][0]

    def test_actual_prologue_reuse_and_multistream_prepared_trt_graph(self):
        for with_indexer in (False, True):
            with self.subTest(with_indexer=with_indexer):
                fixture = PrologueFixture(self.modules, with_indexer)

                def compare(actual, expected):
                    for name, result, reference in zip(
                        ("residual", "query", "TRT output", "TopK"), actual, expected
                    ):
                        torch.testing.assert_close(
                            result, reference, atol=0, rtol=0, msg=name
                        )
                    self.assertTrue(
                        torch.equal(
                            fixture.layer_cache.kv_cache_base, fixture.reference_cache
                        ),
                        "MLA cache bytes differ",
                    )
                    self.assertTrue(
                        torch.equal(
                            fixture.layer_cache.kv_scale_base,
                            fixture.reference_indexer_cache,
                        ),
                        "Indexer cache bytes differ",
                    )

                # Warm actual main/index/indexer-Q streams and events outside
                # capture, and validate the initial eager branch independently.
                for _ in range(3):
                    compare(fixture.candidate_prologue(), fixture.reference_prologue())
                graphs, outputs = [], []
                for function in (
                    fixture.candidate_prologue,
                    fixture.reference_prologue,
                ):
                    graph = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(graph):
                        output = function()
                    graphs.append(graph)
                    outputs.append(output)
                for step in range(3):
                    fixture.change_prologue_inputs(step)
                    for graph in graphs:
                        graph.replay()
                    compare(outputs[0], outputs[1])
                    if step == 1:
                        self.assertTrue(bool((outputs[0][2] == 0).all()))
                    else:
                        self.assertTrue(bool((outputs[0][2] != 0).any()))
                    # Do not compare a graph only against another graph: each
                    # changed-input replay must match a fresh eager invocation.
                    captured = [tuple(x.clone() for x in pair) for pair in outputs]
                    compare(captured[0], fixture.candidate_prologue())
                    compare(captured[1], fixture.reference_prologue())
                if with_indexer:
                    self.assertGreater(fixture.indexer_calls, 0)
                    self.assertIsNotNone(fixture.bridge._events)
                else:
                    self.assertIsNone(fixture.bridge._events)
                print(
                    json.dumps(
                        {
                            "scope": "actual_cmp_prologue_prepared_trt_branch_integration",
                            "with_indexer_producers": with_indexer,
                            "fake_topk_scoring_only": with_indexer,
                            "rows": fixture.rows,
                            "heads": 8,
                            "exact_cache_query_and_attention": True,
                            "full_model_accuracy_test": False,
                        }
                    ),
                    flush=True,
                )

    @unittest.skipUnless(
        os.environ.get("GLM5_CMP_TP8_BENCHMARK") == "1", "opt-in local benchmark"
    )
    def test_local_projection_paired_graph_benchmark(self):
        for rows in (1, 4, 6, 64, 256):
            fixture = LocalProjectionFixture(self.modules, rows)
            paths = {}
            for name, function in (
                ("ordinary", fixture.reference),
                ("cmp", fixture.candidate),
            ):
                for _ in range(4):
                    function()
                start = torch.cuda.Event(enable_timing=True, external=True)
                end = torch.cuda.Event(enable_timing=True, external=True)
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    start.record()
                    output = function()
                    end.record()
                paths[name] = graph, start, end, output
            for _ in range(5):
                for graph, _, end, _ in paths.values():
                    graph.replay()
                    end.synchronize()
            samples = {name: [] for name in paths}
            for iteration in range(31):
                for name in (
                    ("ordinary", "cmp") if iteration % 2 == 0 else ("cmp", "ordinary")
                ):
                    graph, start, end, _ = paths[name]
                    graph.replay()
                    end.synchronize()
                    samples[name].append(start.elapsed_time(end) * 1000)
            for actual, expected in zip(paths["cmp"][3], paths["ordinary"][3]):
                self.assert_numerics(actual, expected, "timed graph result")
            medians = {
                name: statistics.median(values) for name, values in samples.items()
            }
            print(
                json.dumps(
                    {
                        "scope": "local_QB_rope_Wkc_plus_Wvc_quant_O_only",
                        "rows": rows,
                        "heads": 8,
                        "samples_us": samples,
                        "median_us": medians,
                        "speedup": medians["ordinary"] / medians["cmp"],
                    }
                ),
                flush=True,
            )


if __name__ == "__main__":
    unittest.main()
