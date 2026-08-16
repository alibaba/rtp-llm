import inspect
import os
import subprocess
import sys
import tempfile
import types
import unittest
from types import SimpleNamespace
from unittest import mock

import torch
import torch.nn as nn

_THIS = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.abspath(os.path.join(_THIS, "..", "..", "..", "..", ".."))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)


def _stub_package(name: str, path: str) -> None:
    module = types.ModuleType(name)
    module.__path__ = [path]
    sys.modules.setdefault(name, module)


_stub_package("rtp_llm", os.path.join(_REPO, "rtp_llm"))
_stub_package("rtp_llm.models_py", os.path.join(_REPO, "rtp_llm", "models_py"))
_stub_package(
    "rtp_llm.models_py.modules",
    os.path.join(_REPO, "rtp_llm", "models_py", "modules"),
)
_stub_package(
    "rtp_llm.models_py.modules.dsv4",
    os.path.join(_REPO, "rtp_llm", "models_py", "modules", "dsv4"),
)

from rtp_llm.models_py.modules.dsv4 import mhc_prenorm_backend
from rtp_llm.models_py.modules.dsv4.dsv4_kernel_jit_warmup import (
    _collect_dsv4_branch_kernel_configs,
    _collect_dsv4_batched_fp8_einsum_shapes,
    _collect_dsv4_dense_gemm_shapes,
    _collect_dsv4_fp8_mqa_logits_shapes,
    _collect_dsv4_mhc_head_fused_shapes,
    _collect_dsv4_mhc_prenorm_shapes,
    _cp_padded_tokens_per_rank_bound,
    _dense_gemm_m_grid,
    _dist_rank,
    _generate_dense_gemm_warmup_m_grid,
    _align_up,
    _collect_dsv4_grouped_fp8_moe_shapes,
    _collect_dsv4_wo_a_grouped_shapes,
    _generate_mhc_prenorm_warmup_specs,
    _grouped_fp8_masked_m_values,
    _mhc_prenorm_backend,
    _run_deepgemm_warmup_launch_with_retry,
    _run_tilelang_warmup_launch_with_retry,
    _run_triton_warmup_launch_with_retry,
    _sm100_dense_layout_signature,
    _state_ring_entries_warmup_values,
    _warmup_fused_kv_compress_norm_rope_insert,
    warmup_batched_fp8_einsum_jit,
    warmup_compressor_combine_branch_kernels,
    warmup_dense_gemm_jit,
    warmup_dsv4_fp8_swa_slot_dequant_jit,
    warmup_fp8_mqa_logits_jit,
    warmup_grouped_fp8_moe_jit,
    warmup_wo_a_grouped_jit,
    warmup_mhc_head_fused_jit,
    warmup_mhc_prenorm_gemm_jit,
    resolve_dense_gemm_warmup_max_m,
)
import rtp_llm.models_py.modules.dsv4.dsv4_kernel_jit_warmup as warmup_module


def _module_type(name, attrs):
    def __init__(self):
        nn.Module.__init__(self)
        for key, value in attrs.items():
            setattr(self, key, value)

    return type(name, (nn.Module,), {"__init__": __init__})


class Dsv4KernelJitWarmupTest(unittest.TestCase):
    def test_public_jit_warmup_entrypoints_skip_when_model_warmup_disabled(self):
        with mock.patch.object(
            warmup_module, "model_warm_up_enabled", return_value=False
        ):
            device = torch.device("cpu")
            warmup_compressor_combine_branch_kernels(
                device=device,
                gen_num_per_cycle=1,
            )
            warmup_dense_gemm_jit({}, max_m=1, device=device)
            warmup_batched_fp8_einsum_jit({}, max_m=1, device=device)
            warmup_mhc_prenorm_gemm_jit({}, max_m=1, device=device)
            warmup_mhc_head_fused_jit({}, device=device)
            warmup_fp8_mqa_logits_jit({}, device=device)
            warmup_dsv4_fp8_swa_slot_dequant_jit(
                kv_cache=None,
                cp_size=1,
                device=device,
            )

    def test_tilelang_prewarm_skips_when_model_warmup_disabled(self):
        from rtp_llm.models_py.modules.dsv4 import tilelang_kernels

        with mock.patch.object(
            tilelang_kernels, "model_warm_up_enabled", return_value=False
        ), mock.patch.object(tilelang_kernels, "_TILELANG_AVAILABLE", True), mock.patch(
            "torch.zeros"
        ) as zeros:
            tilelang_kernels.prewarm(64, 512, 0.125)

        zeros.assert_not_called()

    def test_dense_warmup_prefill_uses_sequence_bound_without_rank_cap(self):
        self.assertEqual(
            resolve_dense_gemm_warmup_max_m(
                max_seq_len=1048576,
                max_batch_size=1024,
                role_type_name="PREFILL",
                is_speculative=True,
                gen_num_per_cycle=4,
            ),
            1048576,
        )

    def test_dense_warmup_prefill_uses_chunk_size(self):
        self.assertEqual(
            resolve_dense_gemm_warmup_max_m(
                max_seq_len=1048576,
                max_batch_size=1024,
                role_type_name="PREFILL",
                prefill_chunk_size=16384,
                max_tokens_per_rank=65536,
                is_speculative=True,
                gen_num_per_cycle=4,
            ),
            16384,
        )

    def test_dense_warmup_prefill_falls_back_to_rank_token_cap(self):
        self.assertEqual(
            resolve_dense_gemm_warmup_max_m(
                max_seq_len=1048576,
                max_batch_size=1024,
                role_type_name="PREFILL",
                max_tokens_per_rank=65536,
            ),
            65536,
        )

    def test_dense_warmup_m_grid_includes_exact_chunk_size(self):
        self.assertIn(20000, _dense_gemm_m_grid(20000))
        self.assertEqual(_dense_gemm_m_grid(5120)[-1], 5120)

    def test_dense_warmup_prefill_cp_uses_rank_local_sequence_bound(self):
        self.assertEqual(
            resolve_dense_gemm_warmup_max_m(
                max_seq_len=1048576,
                max_batch_size=1024,
                role_type_name="PREFILL",
                cp_size=4,
                cp_enabled=True,
            ),
            262144,
        )
        self.assertEqual(
            resolve_dense_gemm_warmup_max_m(
                max_seq_len=1048577,
                max_batch_size=1024,
                role_type_name="PREFILL",
                cp_size=4,
                cp_enabled=True,
            ),
            262146,
        )
        self.assertEqual(_cp_padded_tokens_per_rank_bound(200002, 4), 50002)

    def test_dense_warmup_decode_uses_model_token_capacity(self):
        self.assertEqual(
            resolve_dense_gemm_warmup_max_m(
                max_seq_len=1048576,
                max_batch_size=1024,
                role_type_name="DECODE",
                is_speculative=True,
                gen_num_per_cycle=4,
            ),
            5120,
        )

    def test_dense_warmup_decode_fallback_accounts_for_speculative_width(self):
        self.assertEqual(
            resolve_dense_gemm_warmup_max_m(
                max_seq_len=1048576,
                max_batch_size=1024,
                role_type_name="RoleType.DECODE",
                is_speculative=True,
                gen_num_per_cycle=4,
            ),
            5120,
        )

    def test_dense_warmup_decode_fallback_defaults_to_batch_size(self):
        self.assertEqual(
            resolve_dense_gemm_warmup_max_m(
                max_seq_len=1048576,
                max_batch_size=1024,
                role_type_name="DECODE",
            ),
            1024,
        )

    def test_dist_rank_falls_back_to_world_rank_env(self):
        old_world_rank = os.environ.get("WORLD_RANK")
        try:
            os.environ["WORLD_RANK"] = "3"
            self.assertEqual(_dist_rank(), 3)
        finally:
            if old_world_rank is None:
                os.environ.pop("WORLD_RANK", None)
            else:
                os.environ["WORLD_RANK"] = old_world_rank

    def test_collect_branch_configs_covers_all_ratios(self):
        args = SimpleNamespace(
            compress_ratios=[0, 4, 128, 4],
            window_size=4096,
            rope_head_dim=64,
            index_topk=512,
        )
        configs = _collect_dsv4_branch_kernel_configs(v4=None, v4_args=args)
        self.assertEqual(
            configs["combine"],
            (
                (4096, 1, 0),
                (4096, 4, 512),
                (4096, 128, 512),
            ),
        )
        self.assertEqual(
            configs["compressor"],
            (
                (128, 64, 4, True),
                (512, 64, 4, True),
                (512, 64, 128, False),
            ),
        )

    def test_cp_sliced_compressor_warmup_includes_full_read_ring(self):
        self.assertEqual(
            _state_ring_entries_warmup_values(
                compress_ratio=4,
                overlap=True,
                gen_num_per_cycle=4,
                cp_size=8,
                prefill_sliced=True,
            ),
            (2, 16),
        )
        self.assertEqual(
            _state_ring_entries_warmup_values(
                compress_ratio=4,
                overlap=True,
                gen_num_per_cycle=4,
                cp_size=8,
                prefill_sliced=False,
            ),
            (16,),
        )

    def test_compressor_warmup_launches_local_and_full_cp_ring_keys(self):
        from rtp_llm.models_py.modules.dsv4.fp8 import _compressor_vllm_triton

        calls = []

        def fake_run_fused(**kwargs):
            calls.append(
                (
                    int(kwargs["state_cache"].shape[1]),
                    bool(kwargs.get("seq_start_per_req") is not None),
                    int(kwargs["head_dim"]),
                    int(kwargs["compress_ratio"]),
                )
            )

        old_run_fused = _compressor_vllm_triton.run_fused_compress_kv_write
        try:
            _compressor_vllm_triton.run_fused_compress_kv_write = fake_run_fused
            _warmup_fused_kv_compress_norm_rope_insert(
                head_dim=512,
                rope_head_dim=64,
                compress_ratio=4,
                overlap=True,
                device=torch.device("cpu"),
                gen_num_per_cycle=4,
                fixed_region_cp_size=8,
                fixed_region_prefill_sliced=True,
            )
        finally:
            _compressor_vllm_triton.run_fused_compress_kv_write = old_run_fused

        self.assertEqual(
            calls,
            [
                (2, False, 512, 4),
                (2, True, 512, 4),
                (16, False, 512, 4),
                (16, True, 512, 4),
            ],
        )

    def test_collect_dense_shapes_dedupes_representatives(self):
        root = nn.Module()

        Fp8Linear = _module_type(
            "CudaFp8DeepGEMMLinear",
            {
                "N": 16,
                "K": 32,
                "weight": torch.empty((16, 32), dtype=torch.bfloat16),
                "weight_scales": torch.empty((16, 1), dtype=torch.int32),
                "scale_ue8m0": True,
            },
        )
        root.add_module("fp8_a", Fp8Linear())
        root.add_module("fp8_b_same_shape", Fp8Linear())

        Fp4Linear = _module_type(
            "QuantizedLinear",
            {
                "storage": "fp4",
                "out_features": 8,
                "in_features": 64,
                "weight": torch.empty((8, 32), dtype=torch.int8),
                "scale_gemm": torch.empty((8, 1), dtype=torch.int32),
            },
        )
        root.add_module("fp4", Fp4Linear())

        Grouped = _module_type(
            "GroupedFP4Strategy",
            {
                "_w13": torch.empty((2, 10, 32), dtype=torch.int8),
                "_s13_dense_t": torch.empty((2, 2, 10), dtype=torch.int32),
                "_w2": torch.empty((2, 64, 5), dtype=torch.int8),
                "_s2_dense_t": torch.empty((2, 1, 64), dtype=torch.int32),
            },
        )
        root.add_module("grouped", Grouped())

        shapes = _collect_dsv4_dense_gemm_shapes(root)
        self.assertEqual(
            sorted(shapes.keys()),
            [
                ("fp8", 16, 32),
                ("fp8_fp4", 8, 64),
                ("fp8_fp4", 10, 64),
                ("fp8_fp4", 64, 10),
            ],
        )
        self.assertEqual(shapes[("fp8", 16, 32)]["name"], "fp8_a")

    def test_collect_dense_shapes_includes_model_level_mtp_projections(self):
        root = nn.Module()
        root.v4 = nn.Module()

        MainFp8Linear = _module_type(
            "CudaFp8DeepGEMMLinear",
            {
                "N": 512,
                "K": 7168,
                "weight": torch.empty((512, 7168), dtype=torch.bfloat16),
                "weight_scales": torch.empty((512, 14), dtype=torch.int32),
                "scale_ue8m0": True,
            },
        )
        root.v4.add_module("main_dense", MainFp8Linear())

        MtpFp8Linear = _module_type(
            "CudaFp8DeepGEMMLinear",
            {
                "N": 7168,
                "K": 7168,
                "weight": torch.empty((7168, 7168), dtype=torch.bfloat16),
                "weight_scales": torch.empty((56, 56), dtype=torch.int32),
                "scale_ue8m0": True,
            },
        )
        root.e_proj = MtpFp8Linear()
        root.h_proj = MtpFp8Linear()

        shapes = _collect_dsv4_dense_gemm_shapes(root)
        self.assertIn(("fp8", 512, 7168), shapes)
        self.assertIn(("fp8", 7168, 7168), shapes)
        self.assertEqual(shapes[("fp8", 7168, 7168)]["name"], "e_proj")

    def test_collect_mhc_prenorm_shapes_uses_tilelang_units_only(self):
        root = nn.Module()
        TileLangHCUnit = _module_type(
            "TileLangHCUnit",
            {
                "fn": torch.empty((24, 16384), dtype=torch.float32),
                "base": torch.empty((24,), dtype=torch.float32),
                "scale": torch.empty((3,), dtype=torch.float32),
                "dim": 4096,
                "hc_mult": 4,
                "norm_eps": 1.0e-5,
                "hc_eps": 1.0e-6,
                "hc_sinkhorn_iters": 20,
            },
        )
        TileLangHCHead = _module_type(
            "TileLangHCHead",
            {
                "fn": torch.empty((4, 16384), dtype=torch.float32),
            },
        )
        FallbackHCUnit = _module_type(
            "FallbackHCUnit",
            {
                "fn": torch.empty((24, 16384), dtype=torch.float32),
            },
        )
        root.add_module("attn_hc", TileLangHCUnit())
        root.add_module("ffn_hc_same_shape", TileLangHCUnit())
        root.add_module("head_hc", TileLangHCHead())
        root.add_module("fallback_hc", FallbackHCUnit())

        shapes = _collect_dsv4_mhc_prenorm_shapes(root)
        self.assertEqual(sorted(shapes.keys()), [(24, 16384)])
        self.assertEqual(shapes[(24, 16384)]["name"], "attn_hc")
        self.assertEqual(shapes[(24, 16384)]["dim"], 4096)
        self.assertEqual(shapes[(24, 16384)]["hc_mult"], 4)
        self.assertEqual(shapes[(24, 16384)]["hc_sinkhorn_iters"], 20)

    def test_collect_mhc_head_fused_shapes_uses_tilelang_heads_only(self):
        root = nn.Module()
        TileLangHCHead = _module_type(
            "TileLangHCHead",
            {
                "fn": torch.empty((4, 16384), dtype=torch.float32),
                "base": torch.empty((4,), dtype=torch.float32),
                "scale": torch.empty((1,), dtype=torch.float32),
                "dim": 4096,
                "hc_mult": 4,
                "norm_eps": 1.0e-5,
                "hc_eps": 1.0e-6,
            },
        )
        FallbackHCHead = _module_type(
            "FallbackHCHead",
            {
                "fn": torch.empty((4, 16384), dtype=torch.float32),
                "base": torch.empty((4,), dtype=torch.float32),
                "scale": torch.empty((1,), dtype=torch.float32),
                "dim": 4096,
                "hc_mult": 4,
            },
        )
        root.add_module("head_hc", TileLangHCHead())
        root.add_module("head_hc_same_shape", TileLangHCHead())
        root.add_module("fallback_head", FallbackHCHead())

        shapes = _collect_dsv4_mhc_head_fused_shapes(root)
        self.assertEqual(sorted(shapes.keys()), [(4, 4096)])
        self.assertEqual(shapes[(4, 4096)]["name"], "head_hc")
        self.assertEqual(tuple(shapes[(4, 4096)]["base"].shape), (4,))
        self.assertEqual(tuple(shapes[(4, 4096)]["scale"].shape), (1,))

    def test_collect_batched_fp8_einsum_shapes_uses_wo_a_buffers(self):
        root = nn.Module()
        Attention = _module_type(
            "Attention",
            {
                "_wo_a_stk_w": torch.empty((8, 1024, 4096), dtype=torch.float32),
                "_wo_a_stk_s": torch.empty((8, 1024, 8), dtype=torch.int32),
            },
        )
        root.add_module("attn", Attention())
        root.add_module("attn_same_shape", Attention())
        root.add_module("no_wo_a", nn.Linear(1, 1))

        shapes = _collect_dsv4_batched_fp8_einsum_shapes(root)
        self.assertEqual(sorted(shapes.keys()), [(8, 1024, 4096)])
        self.assertEqual(shapes[(8, 1024, 4096)]["name"], "attn")

    def test_collect_fp8_mqa_logits_shapes_uses_indexer_modules(self):
        root = nn.Module()
        IndexerFP8 = _module_type(
            "IndexerFP8",
            {
                "n_heads": 64,
                "head_dim": 128,
            },
        )
        root.add_module("indexer", IndexerFP8())
        root.add_module("indexer_same_shape", IndexerFP8())
        root.add_module("not_indexer", nn.Linear(1, 1))

        shapes = _collect_dsv4_fp8_mqa_logits_shapes(root)
        self.assertEqual(sorted(shapes.keys()), [(64, 128)])
        self.assertEqual(shapes[(64, 128)]["name"], "indexer")

    def test_dense_warmup_m_grid_covers_sm100_non_power_of_two_layouts(self):
        grid = _generate_dense_gemm_warmup_m_grid(
            max_m=500000,
            n_value=512,
            k_value=4096,
            kind="fp8_fp4",
            num_sms=148,
        )
        self.assertLess(max(grid), 32768)

        block_ms = {
            _sm100_dense_layout_signature(
                m_value=m,
                n_value=512,
                k_value=4096,
                kind="fp8_fp4",
                num_sms=148,
            )[1]
            for m in grid
        }
        self.assertTrue({144, 176, 192, 208}.issubset(block_ms))

    def test_dense_warmup_m_grid_scales_per_shape(self):
        small_n_grid = _generate_dense_gemm_warmup_m_grid(
            max_m=1048576,
            n_value=512,
            k_value=4096,
            kind="fp8_fp4",
            num_sms=148,
        )
        large_n_grid = _generate_dense_gemm_warmup_m_grid(
            max_m=1048576,
            n_value=32768,
            k_value=1024,
            kind="fp8_fp4",
            num_sms=148,
        )

        self.assertGreater(len(small_n_grid), len(large_n_grid))
        self.assertLessEqual(max(large_n_grid), 4096)

    def test_batched_fp8_einsum_m_grid_covers_observed_layout(self):
        grid = _generate_dense_gemm_warmup_m_grid(
            max_m=200000,
            n_value=1024,
            k_value=4096,
            kind="fp8_batched",
            num_sms=148,
            num_groups=8,
        )
        block_ms = {
            _sm100_dense_layout_signature(
                m_value=m,
                n_value=1024,
                k_value=4096,
                kind="fp8_batched",
                num_sms=148,
                num_groups=8,
            )[1]
            for m in grid
        }
        self.assertIn(96, block_ms)

    def test_mhc_prenorm_warmup_specs_cover_reachable_num_splits(self):
        specs = _generate_mhc_prenorm_warmup_specs(
            max_m=200000,
            k_value=16384,
            num_sms=148,
            backend=mhc_prenorm_backend.DEEPGEMM,
        )
        splits = {split for split, _ in specs}
        self.assertTrue({1, 8, 49, 64}.issubset(splits))

        # Each representative M must resolve back to the split it stands for, via
        # the same function the runtime calls -- that agreement is the point of
        # the spec list, and re-deriving the heuristic here would not test it.
        for split, m_value in specs:
            self.assertEqual(
                mhc_prenorm_backend.resolve_n_splits(
                    mhc_hidden_size=16384,
                    num_tokens=m_value,
                    num_sms=148,
                    backend=mhc_prenorm_backend.DEEPGEMM,
                ),
                split,
                f"m={m_value} no longer resolves to {split}",
            )

    def test_mhc_prenorm_warmup_specs_follow_the_tilelang_divisor_clamp(self):
        """The TileLang backend launches fewer distinct splits than DeepGEMM.

        Its count has to divide the kernel's K block count (16384/256 = 64), so
        DeepGEMM's 49 and 39 are unreachable there. Warming DeepGEMM's list would
        compile variants no request runs and miss the ones they do.
        """
        deepgemm = {
            split
            for split, _ in _generate_mhc_prenorm_warmup_specs(
                max_m=200000,
                k_value=16384,
                num_sms=148,
                backend=mhc_prenorm_backend.DEEPGEMM,
            )
        }
        splitk_specs = _generate_mhc_prenorm_warmup_specs(
            max_m=200000,
            k_value=16384,
            num_sms=148,
            backend=mhc_prenorm_backend.TILELANG_SPLITK,
        )
        splitk = {split for split, _ in splitk_specs}
        self.assertTrue(all(64 % split == 0 for split in splitk), splitk)
        self.assertIn(49, deepgemm)
        self.assertNotIn(49, splitk)
        for split, m_value in splitk_specs:
            self.assertEqual(
                mhc_prenorm_backend.resolve_n_splits(
                    mhc_hidden_size=16384,
                    num_tokens=m_value,
                    num_sms=148,
                    backend=mhc_prenorm_backend.TILELANG_SPLITK,
                ),
                split,
            )

    def test_mhc_prenorm_warmup_specs_single_backend_has_one_variant(self):
        """tilelang_single is one kernel for every M, so one spec is enough."""
        specs = _generate_mhc_prenorm_warmup_specs(
            max_m=200000,
            k_value=16384,
            num_sms=148,
            backend=mhc_prenorm_backend.TILELANG_SINGLE,
        )
        self.assertEqual({split for split, _ in specs}, {1})

    def test_mhc_warmup_launches_prenorm_gemm_and_pre_big_fuse(self):
        calls = []
        shapes = {
            (24, 16384): {
                "name": "attn_hc",
                "fn": torch.empty((24, 16384), dtype=torch.float32),
                "base": torch.empty((24,), dtype=torch.float32),
                "scale": torch.empty((3,), dtype=torch.float32),
                "dim": 4096,
                "hc_mult": 4,
                "norm_eps": 1.0e-6,
                "hc_eps": 1.0e-6,
                "hc_sinkhorn_iters": 20,
            }
        }

        def with_patch(name, value):
            old = getattr(warmup_module, name)
            setattr(warmup_module, name, value)
            return old

        old_values = {}
        try:
            old_values["_is_cuda_device"] = with_patch(
                "_is_cuda_device", lambda device: True
            )
            old_values["_assert_not_capturing"] = with_patch(
                "_assert_not_capturing", lambda: None
            )
            old_values["_get_deep_gemm_num_sms"] = with_patch(
                "_get_deep_gemm_num_sms", lambda device: 148
            )
            old_values["_mhc_prenorm_backend"] = with_patch(
                "_mhc_prenorm_backend", lambda: mhc_prenorm_backend.DEEPGEMM
            )
            old_values["_generate_mhc_prenorm_warmup_specs"] = with_patch(
                "_generate_mhc_prenorm_warmup_specs",
                lambda **kwargs: ((8, 65), (1, 129)),
            )
            old_values["_dist_rank"] = with_patch("_dist_rank", lambda: 0)
            old_values["_sync_cuda"] = with_patch("_sync_cuda", lambda device: None)
            old_values["_release_cuda_cache"] = with_patch(
                "_release_cuda_cache", lambda device: None
            )
            old_values["_run_deepgemm_warmup_launches_serialized"] = with_patch(
                "_run_deepgemm_warmup_launches_serialized",
                lambda label, fn: fn(),
            )
            old_values["_run_deepgemm_warmup_launch_with_retry"] = with_patch(
                "_run_deepgemm_warmup_launch_with_retry",
                lambda label, desc, launch_fn, device: (
                    calls.append(("gemm", desc)),
                    launch_fn(),
                ),
            )
            old_values["_launch_dummy_mhc_prenorm_gemm"] = with_patch(
                "_launch_dummy_mhc_prenorm_gemm",
                lambda **kwargs: calls.append(("gemm_launch", kwargs["num_splits"])),
            )
            old_values["_launch_dummy_mhc_pre_big_fuse"] = with_patch(
                "_launch_dummy_mhc_pre_big_fuse",
                lambda **kwargs: calls.append(("fuse_launch", kwargs["num_splits"])),
            )
            warmup_module._MHC_PRENORM_GEMM_JIT_WARMED_KEYS.clear()

            warmup_module.warmup_mhc_prenorm_gemm_jit(
                shapes,
                max_m=1024,
                device=torch.device("cuda"),
            )
        finally:
            for name, value in old_values.items():
                setattr(warmup_module, name, value)
            warmup_module._MHC_PRENORM_GEMM_JIT_WARMED_KEYS.clear()

        self.assertEqual(
            [c for c in calls if c[0] == "gemm_launch"],
            [("gemm_launch", 8), ("gemm_launch", 1)],
        )
        self.assertEqual(
            [c for c in calls if c[0] == "fuse_launch"],
            [("fuse_launch", 8), ("fuse_launch", 1)],
        )

    def test_mhc_tilelang_single_warmup_uses_runtime_wrapper(self):
        calls = []
        shapes = {
            (24, 16384): {
                "name": "attn_hc",
                "fn": torch.empty((24, 16384), dtype=torch.float32),
                "base": torch.empty((24,), dtype=torch.float32),
                "scale": torch.empty((3,), dtype=torch.float32),
                "dim": 4096,
                "hc_mult": 4,
                "norm_eps": 1.0e-6,
                "hc_eps": 1.0e-6,
                "hc_sinkhorn_iters": 20,
            }
        }

        def with_patch(name, value):
            old = getattr(warmup_module, name)
            setattr(warmup_module, name, value)
            return old

        old_values = {}
        try:
            old_values["_is_cuda_device"] = with_patch(
                "_is_cuda_device", lambda device: True
            )
            old_values["_assert_not_capturing"] = with_patch(
                "_assert_not_capturing", lambda: None
            )
            old_values["_get_deep_gemm_num_sms"] = with_patch(
                "_get_deep_gemm_num_sms", lambda device: 148
            )
            old_values["_mhc_prenorm_backend"] = with_patch(
                "_mhc_prenorm_backend", lambda: mhc_prenorm_backend.TILELANG_SINGLE
            )
            old_values["_dist_rank"] = with_patch("_dist_rank", lambda: 0)
            old_values["_sync_cuda"] = with_patch("_sync_cuda", lambda device: None)
            old_values["_release_cuda_cache"] = with_patch(
                "_release_cuda_cache", lambda device: None
            )
            old_values["_run_deepgemm_warmup_launches_serialized"] = with_patch(
                "_run_deepgemm_warmup_launches_serialized",
                lambda label, fn: fn(),
            )
            old_values["_launch_dummy_mhc_prenorm_gemm"] = with_patch(
                "_launch_dummy_mhc_prenorm_gemm",
                lambda **kwargs: calls.append(("deepgemm", kwargs)),
            )
            old_values["_launch_dummy_mhc_pre_big_fuse"] = with_patch(
                "_launch_dummy_mhc_pre_big_fuse",
                lambda **kwargs: calls.append(("raw_fuse", kwargs)),
            )
            old_values["_launch_dummy_mhc_pre_wrapper"] = with_patch(
                "_launch_dummy_mhc_pre_wrapper",
                lambda **kwargs: calls.append(("wrapper", kwargs["m_value"])),
            )
            warmup_module._MHC_PRENORM_GEMM_JIT_WARMED_KEYS.clear()

            warmup_module.warmup_mhc_prenorm_gemm_jit(
                shapes,
                max_m=1024,
                device=torch.device("cuda"),
            )
        finally:
            for name, value in old_values.items():
                setattr(warmup_module, name, value)
            warmup_module._MHC_PRENORM_GEMM_JIT_WARMED_KEYS.clear()

        self.assertEqual(calls, [("wrapper", 1)])

    def test_mhc_head_fused_warmup_uses_batched_two_token_shape(self):
        calls = []
        shapes = {
            (4, 4096): {
                "name": "head_hc",
                "fn": torch.empty((4, 16384), dtype=torch.float32),
                "base": torch.empty((4,), dtype=torch.float32),
                "scale": torch.empty((1,), dtype=torch.float32),
                "norm_eps": 1.0e-6,
                "hc_eps": 1.0e-6,
            }
        }

        def with_patch(name, value):
            old = getattr(warmup_module, name)
            setattr(warmup_module, name, value)
            return old

        old_values = {}
        try:
            old_values["_is_cuda_device"] = with_patch(
                "_is_cuda_device", lambda device: True
            )
            old_values["_assert_not_capturing"] = with_patch(
                "_assert_not_capturing", lambda: None
            )
            old_values["_dist_rank"] = with_patch("_dist_rank", lambda: 0)
            old_values["_sync_cuda"] = with_patch("_sync_cuda", lambda device: None)
            old_values["_release_cuda_cache"] = with_patch(
                "_release_cuda_cache", lambda device: None
            )
            old_values["_run_deepgemm_warmup_launches_serialized"] = with_patch(
                "_run_deepgemm_warmup_launches_serialized",
                lambda label, fn: fn(),
            )
            old_values["_launch_dummy_mhc_head_fused"] = with_patch(
                "_launch_dummy_mhc_head_fused",
                lambda **kwargs: calls.append(
                    (
                        kwargs["key"],
                        tuple(kwargs["token_values"]),
                        kwargs["info"]["name"],
                    )
                ),
            )

            import rtp_llm.models_py.modules.dsv4.hc.mhc_tilelang as mhc_tilelang

            old_enabled = mhc_tilelang.tk_mhc_head_fused_enabled
            mhc_tilelang.tk_mhc_head_fused_enabled = lambda: True
            self.addCleanup(
                lambda: setattr(
                    mhc_tilelang, "tk_mhc_head_fused_enabled", old_enabled
                )
            )

            warmup_module._MHC_HEAD_FUSED_JIT_WARMED_KEYS.clear()
            warmup_module.warmup_mhc_head_fused_jit(
                shapes,
                device=torch.device("cuda"),
            )
        finally:
            for name, value in old_values.items():
                setattr(warmup_module, name, value)
            warmup_module._MHC_HEAD_FUSED_JIT_WARMED_KEYS.clear()

        self.assertEqual(calls, [((4, 4096), (2,), "head_hc")])

    def test_mhc_head_fused_warmup_skips_when_disabled(self):
        calls = []
        shapes = {
            (4, 4096): {
                "name": "head_hc",
                "fn": torch.empty((4, 16384), dtype=torch.float32),
            }
        }

        old_disabled = os.environ.get("DSV4_MHC_HEAD_FUSED")
        old_launch = warmup_module._launch_dummy_mhc_head_fused
        try:
            os.environ["DSV4_MHC_HEAD_FUSED"] = "0"
            warmup_module._launch_dummy_mhc_head_fused = lambda **kwargs: calls.append(
                kwargs
            )
            warmup_module._MHC_HEAD_FUSED_JIT_WARMED_KEYS.clear()
            warmup_module.warmup_mhc_head_fused_jit(
                shapes,
                device=torch.device("cuda"),
            )
        finally:
            if old_disabled is None:
                os.environ.pop("DSV4_MHC_HEAD_FUSED", None)
            else:
                os.environ["DSV4_MHC_HEAD_FUSED"] = old_disabled
            warmup_module._launch_dummy_mhc_head_fused = old_launch
            warmup_module._MHC_HEAD_FUSED_JIT_WARMED_KEYS.clear()

        self.assertEqual(calls, [])

    def test_collect_mhc_head_fused_shapes_skips_fallback_head(self):
        root = nn.Module()
        FallbackHCHead = _module_type(
            "FallbackHCHead",
            {
                "fn": torch.empty((4, 16384), dtype=torch.float32),
                "dim": 4096,
                "hc_mult": 4,
            },
        )
        root.add_module("fallback_head", FallbackHCHead())

        self.assertEqual(_collect_dsv4_mhc_head_fused_shapes(root), {})

    def test_slot_dequant_warmup_uses_padded_cp_full_stride(self):
        from rtp_llm.models_py.modules.dsv4.fp8 import _swa_dequant_triton

        calls = []
        local_slice_bytes = 74880
        cp_size = 2
        expected_full_stride = local_slice_bytes * cp_size
        expected_entries = expected_full_stride // _swa_dequant_triton.ENTRY_BYTES

        def fake_dequantize(pool_3d, slot_indices):
            calls.append(
                (
                    tuple(pool_3d.shape),
                    tuple(pool_3d.stride()),
                    slot_indices.tolist(),
                )
            )
            return torch.empty(
                (int(slot_indices.numel()), _swa_dequant_triton.HEAD_DIM),
                dtype=torch.bfloat16,
                device=pool_3d.device,
            )

        def with_patch(obj, name, value):
            old = getattr(obj, name)
            setattr(obj, name, value)
            return old

        old_values = []
        try:
            old_values.append(
                (
                    warmup_module,
                    "_is_cuda_device",
                    with_patch(warmup_module, "_is_cuda_device", lambda device: True),
                )
            )
            old_values.append(
                (
                    warmup_module,
                    "_assert_not_capturing",
                    with_patch(warmup_module, "_assert_not_capturing", lambda: None),
                )
            )
            old_values.append(
                (
                    warmup_module,
                    "_swa_kv_local_slice_bytes",
                    with_patch(
                        warmup_module,
                        "_swa_kv_local_slice_bytes",
                        lambda kv_cache: local_slice_bytes,
                    ),
                )
            )
            old_values.append(
                (
                    warmup_module,
                    "_dist_rank",
                    with_patch(warmup_module, "_dist_rank", lambda: 0),
                )
            )
            old_values.append(
                (
                    warmup_module,
                    "_sync_cuda",
                    with_patch(warmup_module, "_sync_cuda", lambda device: None),
                )
            )
            old_values.append(
                (
                    _swa_dequant_triton,
                    "dequantize_slots_to_bf16",
                    with_patch(
                        _swa_dequant_triton,
                        "dequantize_slots_to_bf16",
                        fake_dequantize,
                    ),
                )
            )
            warmup_module._SWA_SLOT_DEQUANT_JIT_WARMED_KEYS.clear()

            warmup_module.warmup_dsv4_fp8_swa_slot_dequant_jit(
                kv_cache=object(),
                cp_size=cp_size,
                device=torch.device("cpu"),
            )
        finally:
            for obj, name, value in old_values:
                setattr(obj, name, value)
            warmup_module._SWA_SLOT_DEQUANT_JIT_WARMED_KEYS.clear()

        self.assertEqual(
            calls,
            [
                (
                    (1, expected_entries, _swa_dequant_triton.ENTRY_BYTES),
                    (
                        expected_full_stride,
                        _swa_dequant_triton.ENTRY_BYTES,
                        1,
                    ),
                    [0, -1],
                )
            ],
        )

    def test_mhc_pre_big_fuse_warmup_initializes_tilelang_env_first(self):
        source = inspect.getsource(warmup_module._launch_dummy_mhc_pre_big_fuse)
        self.assertIn("tilelang_kernels", source)
        self.assertIn("pre_big_fuse_kernel", source)
        self.assertLess(
            source.find("tilelang_kernels"),
            source.find("pre_big_fuse_kernel"),
        )

    def test_jit_kernel_specialization_contracts(self):
        from rtp_llm.models_py.modules.dsv4.fp8 import _compressor_vllm_triton
        from rtp_llm.models_py.modules.dsv4.fp8 import _swa_dequant_triton
        from rtp_llm.models_py.modules.dsv4.fp8 import _swa_kv_insert_triton

        compress_src = inspect.getsource(
            _compressor_vllm_triton._fused_kv_compress_norm_rope_insert_sparse_attn.fn
        )
        self.assertIn('"KV_BLOCK_STRIDE"', compress_src)
        self.assertIn("KV_BLOCK_STRIDE,", compress_src)
        self.assertNotIn("KV_BLOCK_STRIDE: tl.constexpr", compress_src)
        self.assertIn('"NUM_STATE_BLOCKS"', compress_src)
        self.assertIn('"NUM_KV_BLOCKS"', compress_src)
        self.assertNotIn("NUM_STATE_BLOCKS: tl.constexpr", compress_src)
        self.assertNotIn("NUM_KV_BLOCKS: tl.constexpr", compress_src)
        indexer_src = inspect.getsource(
            _compressor_vllm_triton._fused_kv_compress_norm_rope_insert_indexer_attn.fn
        )
        self.assertIn('"KV_BLOCK_STRIDE"', indexer_src)
        self.assertIn('"NUM_STATE_BLOCKS"', indexer_src)
        self.assertIn('"NUM_KV_BLOCKS"', indexer_src)
        self.assertNotIn("KV_BLOCK_STRIDE: tl.constexpr", indexer_src)
        self.assertNotIn("NUM_STATE_BLOCKS: tl.constexpr", indexer_src)
        self.assertNotIn("NUM_KV_BLOCKS: tl.constexpr", indexer_src)

        quant_src = inspect.getsource(
            _swa_kv_insert_triton._quantize_and_insert_k_kernel.fn
        )
        self.assertIn('"block_stride"', quant_src)
        self.assertIn('"num_cache_blocks"', quant_src)
        self.assertNotIn("block_stride: tl.constexpr", quant_src)
        self.assertNotIn("num_cache_blocks: tl.constexpr", quant_src)

        dequant_src = inspect.getsource(_swa_dequant_triton._dequantize_slots_kernel.fn)
        self.assertIn('"pool_block_stride"', dequant_src)
        self.assertIn('"num_cache_blocks"', dequant_src)
        self.assertNotIn("pool_block_stride: tl.constexpr", dequant_src)
        self.assertNotIn("num_cache_blocks: tl.constexpr", dequant_src)

        gather_src = inspect.getsource(
            _swa_dequant_triton._gather_k_cache_packed_kernel.fn
        )
        self.assertIn('"out_stride0"', gather_src)
        self.assertIn('"out_stride1"', gather_src)
        self.assertIn('"offset"', gather_src)
        self.assertIn('"max_blocks_per_seq"', gather_src)
        self.assertIn('"block_stride"', gather_src)
        self.assertNotIn("out_stride0: tl.constexpr", gather_src)
        self.assertNotIn("out_stride1: tl.constexpr", gather_src)
        self.assertNotIn("offset: tl.constexpr", gather_src)
        self.assertNotIn("max_blocks_per_seq: tl.constexpr", gather_src)
        self.assertNotIn("block_stride: tl.constexpr", gather_src)

    def test_deepgemm_warmup_retry_handles_nvcc_compile_failure(self):
        calls = []

        def launch():
            calls.append(None)
            if len(calls) == 1:
                raise RuntimeError("NVCC compilation failed")

        _run_deepgemm_warmup_launch_with_retry(
            "test",
            "shape=(24, 16384) num_splits=5",
            launch,
            device=torch.device("cpu"),
        )

        self.assertEqual(len(calls), 2)

    def test_deepgemm_warmup_retry_does_not_swallow_other_errors(self):
        def launch():
            raise RuntimeError("bad shape")

        with self.assertRaisesRegex(RuntimeError, "bad shape"):
            _run_deepgemm_warmup_launch_with_retry(
                "test",
                "shape=(24, 16384)",
                launch,
                device=torch.device("cpu"),
            )

    def test_tilelang_warmup_retry_handles_tmpxft_source_race(self):
        calls = []

        def launch():
            calls.append(None)
            if len(calls) == 1:
                raise RuntimeError(
                    'Catastrophic error: cannot open source file '
                    '"/tmp/tmpxft_000011ba_00000000-7_tvm_kernels.cpp1.ii"'
                )

        _run_tilelang_warmup_launch_with_retry(
            "test",
            "shape=(24, 28672) num_splits=18",
            launch,
            device=torch.device("cpu"),
        )

        self.assertEqual(len(calls), 2)

    def test_tilelang_warmup_retry_handles_chained_tmpxft_source_race(self):
        calls = []

        def launch():
            calls.append(None)
            if len(calls) == 1:
                cause = RuntimeError(
                    'Catastrophic error: cannot open source file '
                    '"/tmp/tmpxft_000011ba_00000000-7_tvm_kernels.cpp1.ii"'
                )
                raise RuntimeError("TileLang mhc_pre failed: shape=(1, 2)") from cause

        _run_tilelang_warmup_launch_with_retry(
            "test",
            "shape=(24, 28672) num_splits=18",
            launch,
            device=torch.device("cpu"),
        )

        self.assertEqual(len(calls), 2)

    def test_tilelang_warmup_retry_does_not_swallow_other_errors(self):
        def launch():
            raise RuntimeError("TileLang semantic compile error")

        with self.assertRaisesRegex(RuntimeError, "semantic compile"):
            _run_tilelang_warmup_launch_with_retry(
                "test",
                "shape=(24, 28672)",
                launch,
                device=torch.device("cpu"),
            )

    def test_triton_warmup_retry_handles_ptxas_tmp_log_race(self):
        calls = []

        def launch():
            calls.append(None)
            if len(calls) == 1:
                cause = RuntimeError(
                    "Command '['/usr/local/cuda-13.2/bin/ptxas', '-lineinfo']' "
                    "returned non-zero exit status 255."
                )
                raise FileNotFoundError(
                    "[Errno 2] No such file or directory: "
                    "'/jit_cache/rtp_llm_dsv4_triton_warmup_tmp/rank_3/tmpabc123.log'"
                ) from cause

        _run_triton_warmup_launch_with_retry(
            "test",
            "shape S=1 T=262144 H=128 D=512",
            launch,
            device=torch.device("cpu"),
        )

        self.assertEqual(len(calls), 2)

    def test_triton_warmup_retry_handles_called_process_context(self):
        calls = []

        def launch():
            calls.append(None)
            if len(calls) == 1:
                try:
                    raise subprocess.CalledProcessError(255, ["ptxas", "-lineinfo"])
                except subprocess.CalledProcessError:
                    raise FileNotFoundError(
                        "[Errno 2] No such file or directory: "
                        "'/jit_cache/rtp_llm_dsv4_triton_warmup_tmp/rank_3/tmpabc123.log'"
                    )

        _run_triton_warmup_launch_with_retry(
            "test",
            "shape S=1 T=262144 H=128 D=512",
            launch,
            device=torch.device("cpu"),
        )

        self.assertEqual(len(calls), 2)

    def test_triton_warmup_retry_uses_rank_local_tmpdir_and_restores(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            with mock.patch.dict(
                os.environ,
                {
                    "TRITON_CACHE_DIR": tmpdir,
                    "WORLD_RANK": "3",
                    "TMPDIR": "/old/tmp",
                },
                clear=True,
            ):
                _run_triton_warmup_launch_with_retry(
                    "test",
                    "shape S=1 T=16",
                    lambda: None,
                    device=torch.device("cpu"),
                )

                self.assertEqual(os.environ["TMPDIR"], "/old/tmp")
                self.assertTrue(
                    os.path.isdir(
                        os.path.join(
                            tmpdir,
                            "rtp_llm_dsv4_triton_warmup_tmp",
                            "rank_3",
                        )
                    )
                )

    def test_triton_warmup_retry_does_not_swallow_other_errors(self):
        def launch():
            raise FileNotFoundError("/tmp/not-a-triton-log.txt")

        with self.assertRaisesRegex(FileNotFoundError, "not-a-triton"):
            _run_triton_warmup_launch_with_retry(
                "test",
                "shape S=1 T=262144",
                launch,
                device=torch.device("cpu"),
            )


class MhcPrenormBackendResolutionTest(unittest.TestCase):
    """The warmup and the runtime must not disagree about the backend.

    They did, silently: this module resolved an unset or ``auto``
    ``DSV4_MHC_PRE_GEMM_BACKEND`` to ``deepgemm`` unconditionally, while the
    TileLang wrapper probes for ``deep_gemm.tf32_hc_prenorm_gemm`` and falls back
    to ``tilelang_splitk``. On a build without that symbol -- the H20 build this
    PR targets -- warmup drove the DeepGEMM path and raised during startup, and
    the split-K kernel requests actually run was never compiled ahead of time.

    Pure host: no CUDA, no ``deep_gemm``, no ``tilelang``.
    """

    def setUp(self):
        self._saved = os.environ.get(mhc_prenorm_backend.BACKEND_ENV)
        self.addCleanup(self._restore_env)
        # Both are @functools.cache'd; clear on entry *and* exit so neither a
        # probe result nor a log-once marker leaks between cases or out of them.
        self.addCleanup(mhc_prenorm_backend.has_deepgemm_prenorm.cache_clear)
        self.addCleanup(mhc_prenorm_backend._log_backend_choice.cache_clear)
        mhc_prenorm_backend.has_deepgemm_prenorm.cache_clear()
        mhc_prenorm_backend._log_backend_choice.cache_clear()

    def _restore_env(self):
        if self._saved is None:
            os.environ.pop(mhc_prenorm_backend.BACKEND_ENV, None)
        else:
            os.environ[mhc_prenorm_backend.BACKEND_ENV] = self._saved

    def _with_probe(self, available: bool):
        """Replace the DeepGEMM symbol probe, restoring it on cleanup."""
        original = mhc_prenorm_backend.has_deepgemm_prenorm
        mhc_prenorm_backend.has_deepgemm_prenorm = lambda: available
        self.addCleanup(setattr, mhc_prenorm_backend, "has_deepgemm_prenorm", original)

    # (env value, backend when the DeepGEMM symbol is present, and when it is not).
    # Spelled out rather than compared against resolve_backend(): asserting the two
    # agree would hold no matter what either did, and would not pin the alias table
    # or the case/whitespace normalisation at all.
    _RESOLUTION_TABLE = (
        (None, mhc_prenorm_backend.DEEPGEMM, mhc_prenorm_backend.TILELANG_SPLITK),
        ("", mhc_prenorm_backend.DEEPGEMM, mhc_prenorm_backend.TILELANG_SPLITK),
        ("   ", mhc_prenorm_backend.DEEPGEMM, mhc_prenorm_backend.TILELANG_SPLITK),
        ("auto", mhc_prenorm_backend.DEEPGEMM, mhc_prenorm_backend.TILELANG_SPLITK),
        ("AUTO", mhc_prenorm_backend.DEEPGEMM, mhc_prenorm_backend.TILELANG_SPLITK),
        ("  auto  ", mhc_prenorm_backend.DEEPGEMM, mhc_prenorm_backend.TILELANG_SPLITK),
        ("deepgemm", mhc_prenorm_backend.DEEPGEMM, mhc_prenorm_backend.DEEPGEMM),
        ("DeepGEMM", mhc_prenorm_backend.DEEPGEMM, mhc_prenorm_backend.DEEPGEMM),
        ("dg", mhc_prenorm_backend.DEEPGEMM, mhc_prenorm_backend.DEEPGEMM),
        ("tilelang", mhc_prenorm_backend.TILELANG_SINGLE, mhc_prenorm_backend.TILELANG_SINGLE),
        ("single", mhc_prenorm_backend.TILELANG_SINGLE, mhc_prenorm_backend.TILELANG_SINGLE),
        (" TileLang_Single ", mhc_prenorm_backend.TILELANG_SINGLE, mhc_prenorm_backend.TILELANG_SINGLE),
        ("splitk", mhc_prenorm_backend.TILELANG_SPLITK, mhc_prenorm_backend.TILELANG_SPLITK),
        ("tilelang_splitk", mhc_prenorm_backend.TILELANG_SPLITK, mhc_prenorm_backend.TILELANG_SPLITK),
        ("bogus", "bogus", "bogus"),
    )

    def test_resolution_table(self):
        for value, with_symbol, without_symbol in self._RESOLUTION_TABLE:
            for probe, expected in ((True, with_symbol), (False, without_symbol)):
                with self.subTest(value=value, probe=probe):
                    self._with_probe(probe)
                    mhc_prenorm_backend._log_backend_choice.cache_clear()
                    if value is None:
                        os.environ.pop(mhc_prenorm_backend.BACKEND_ENV, None)
                    else:
                        os.environ[mhc_prenorm_backend.BACKEND_ENV] = value
                    self.assertEqual(_mhc_prenorm_backend(), expected)

    def test_warmup_does_not_resolve_the_env_itself(self):
        """Structural: the warmup entry point must be the shared resolver.

        The table above already fails if this module re-implements resolution and
        drifts; this pins the intent directly, so a re-implementation that happens
        to agree today still shows up as a change here.
        """
        sentinel = "sentinel_backend"
        original = mhc_prenorm_backend.resolve_backend
        mhc_prenorm_backend.resolve_backend = lambda: sentinel
        self.addCleanup(
            setattr, mhc_prenorm_backend, "resolve_backend", original
        )
        os.environ[mhc_prenorm_backend.BACKEND_ENV] = "tilelang_single"
        self.assertEqual(_mhc_prenorm_backend(), sentinel)

    def test_auto_follows_the_symbol_probe(self):
        for value in (None, "", "auto"):
            with self.subTest(value=value):
                if value is None:
                    os.environ.pop(mhc_prenorm_backend.BACKEND_ENV, None)
                else:
                    os.environ[mhc_prenorm_backend.BACKEND_ENV] = value

                self._with_probe(True)
                mhc_prenorm_backend._log_backend_choice.cache_clear()
                self.assertEqual(_mhc_prenorm_backend(), mhc_prenorm_backend.DEEPGEMM)

                self._with_probe(False)
                mhc_prenorm_backend._log_backend_choice.cache_clear()
                self.assertEqual(
                    _mhc_prenorm_backend(), mhc_prenorm_backend.TILELANG_SPLITK
                )

    def test_explicit_deepgemm_is_not_downgraded_by_the_probe(self):
        """An explicit pin must reach the runtime error, not a silent fallback."""
        os.environ[mhc_prenorm_backend.BACKEND_ENV] = "deepgemm"
        self._with_probe(False)
        self.assertEqual(_mhc_prenorm_backend(), mhc_prenorm_backend.DEEPGEMM)

    def test_unknown_value_is_passed_through(self):
        os.environ[mhc_prenorm_backend.BACKEND_ENV] = "tilelang_splitq"
        self.assertEqual(_mhc_prenorm_backend(), "tilelang_splitq")

    def test_num_sms_has_one_source(self):
        """The warmup must read the SM count from the shared resolver.

        The split count is a function of it, so two sources meant the warmup could
        compile a count the runtime never launches -- the same class of drift as the
        backend name. Patching the shared function has to move what the warmup sees.
        """
        original = mhc_prenorm_backend.device_num_sms
        mhc_prenorm_backend.device_num_sms = lambda device=None: 137
        self.addCleanup(
            setattr, mhc_prenorm_backend, "device_num_sms", original
        )
        seen = {}

        def spy(**kwargs):
            seen.update(kwargs)
            return 1

        with mock.patch.object(
            warmup_module, "model_warm_up_enabled", return_value=True
        ), mock.patch.object(
            warmup_module, "_is_cuda_device", return_value=True
        ), mock.patch.object(
            warmup_module, "_assert_not_capturing"
        ), mock.patch.object(
            mhc_prenorm_backend, "resolve_n_splits", side_effect=spy
        ), mock.patch.object(
            warmup_module, "_run_deepgemm_warmup_launches_serialized"
        ):
            warmup_module._generate_mhc_prenorm_warmup_specs.cache_clear()
            warmup_module.warmup_mhc_prenorm_gemm_jit(
                {(24, 16384): {"name": "probe"}},
                max_m=1,
                device=torch.device("cpu"),
            )
        self.assertEqual(seen.get("num_sms"), 137)

    def test_probe_never_raises_without_deep_gemm(self):
        """The real probe answers False on a build/box without the symbol."""
        mhc_prenorm_backend.has_deepgemm_prenorm.cache_clear()
        self.assertIsInstance(mhc_prenorm_backend.has_deepgemm_prenorm(), bool)


class GroupedFp8AndWoAWarmupCollectorTest(unittest.TestCase):
    """The two SM90 collectors and the masked ``expected_m`` plan.

    Pure host: the collectors only read shapes off registered tensors, and the plan
    is arithmetic. What they feed -- the actual DeepGEMM launches -- needs a Hopper
    GPU and is exercised by bringing a role up with WARM_UP=1.
    """

    def _grouped_fp8_module(self, *, experts=8, inter=256, dim=512, ep_size=4):
        cls = _module_type(
            "GroupedFP8Strategy",
            {
                "_w13": torch.empty(
                    (experts, 2 * inter, dim), dtype=torch.float8_e4m3fn
                ),
                "_s13": torch.empty((experts, 2 * inter // 128, dim // 128)),
                "_w2": torch.empty((experts, dim, inter), dtype=torch.float8_e4m3fn),
                "_s2": torch.empty((experts, dim // 128, inter // 128)),
                "cfg": SimpleNamespace(ep_size=ep_size),
            },
        )
        return cls()

    def test_align_up(self):
        self.assertEqual(_align_up(1, 128), 128)
        self.assertEqual(_align_up(128, 128), 128)
        self.assertEqual(_align_up(129, 128), 256)
        self.assertEqual(_align_up(0, 128), 0)
        self.assertEqual(_align_up(65, 64), 128)

    def test_masked_m_values_are_the_two_the_strategy_computes(self):
        # N = tokens_per_rank * ep_size; max_m = align(N, 128) and
        # t_rows = min(max_m, align(N, 64)).
        # conc 8 over ep 4: N=32, max_m=align(32,128)=128, t_rows=align(32,64)=64
        self.assertEqual(
            _grouped_fp8_masked_m_values(tokens_per_rank=8, ep_size=4), (64, 128)
        )
        # N=128: both land on 128, so one launch covers the pair
        self.assertEqual(
            _grouped_fp8_masked_m_values(tokens_per_rank=32, ep_size=4), (128,)
        )
        # N=64: max_m still rounds to 128 while t_rows stays 64
        self.assertEqual(
            _grouped_fp8_masked_m_values(tokens_per_rank=8, ep_size=8), (64, 128)
        )

    def test_masked_m_values_empty_past_the_refusal_cap(self):
        """Prefill never masks, so there is nothing to warm for it."""
        self.assertEqual(
            _grouped_fp8_masked_m_values(tokens_per_rank=16384, ep_size=4), ()
        )
        self.assertEqual(
            _grouped_fp8_masked_m_values(tokens_per_rank=1024, ep_size=1), (1024,)
        )
        self.assertEqual(
            _grouped_fp8_masked_m_values(tokens_per_rank=1025, ep_size=1), ()
        )

    def test_collect_grouped_fp8_shapes(self):
        root = nn.Module()
        root.add_module("layer0", self._grouped_fp8_module())
        root.add_module("layer1", self._grouped_fp8_module())
        shapes = _collect_dsv4_grouped_fp8_moe_shapes(root)
        # Both layers share a shape, so one entry with all four operands.
        self.assertEqual(len(shapes), 1)
        (key,) = list(shapes)
        self.assertEqual(key, (8, 512, 512, 512, 256, 4))
        info = shapes[key]
        self.assertEqual(info["ep_size"], 4)
        for attr in ("_w13", "_s13", "_w2", "_s2"):
            self.assertIn(attr, info)

    def test_collect_grouped_fp8_shapes_skips_incomplete_modules(self):
        """A strategy whose weights are not set up yet must not be collected."""
        root = nn.Module()
        partial = _module_type(
            "GroupedFP8Strategy",
            {"_w13": torch.empty((8, 512, 512), dtype=torch.float8_e4m3fn)},
        )()
        root.add_module("partial", partial)
        self.assertEqual(_collect_dsv4_grouped_fp8_moe_shapes(root), {})

    def test_collect_grouped_fp8_shapes_ignores_other_strategies(self):
        root = nn.Module()
        root.add_module(
            "fp4",
            _module_type(
                "GroupedFP4Strategy",
                {
                    "_w13": torch.empty((8, 512, 256), dtype=torch.float8_e4m3fn),
                    "_s13": torch.empty((8, 4, 2)),
                    "_w2": torch.empty((8, 512, 128), dtype=torch.float8_e4m3fn),
                    "_s2": torch.empty((8, 4, 1)),
                    "cfg": SimpleNamespace(ep_size=4),
                },
            )(),
        )
        self.assertEqual(_collect_dsv4_grouped_fp8_moe_shapes(root), {})

    def test_collect_wo_a_grouped_shapes(self):
        root = nn.Module()
        attn = nn.Module()
        attn.register_buffer(
            "_wo_a_grp_w", torch.empty((9, 256, 512), dtype=torch.float8_e4m3fn)
        )
        attn.register_buffer("_wo_a_grp_s", torch.empty((9, 2, 4)))
        root.add_module("attn", attn)
        shapes = _collect_dsv4_wo_a_grouped_shapes(root)
        self.assertEqual(list(shapes), [(9, 256, 512)])
        self.assertEqual(shapes[(9, 256, 512)]["name"], "attn")

    def test_collect_wo_a_grouped_shapes_ignores_the_sm100_buffers(self):
        """SM100 registers ``_wo_a_stk_*``; this collector must not match them."""
        root = nn.Module()
        attn = nn.Module()
        attn.register_buffer(
            "_wo_a_stk_w", torch.empty((9, 256, 512), dtype=torch.float8_e4m3fn)
        )
        attn.register_buffer("_wo_a_stk_s", torch.empty((9, 256, 4)))
        root.add_module("attn", attn)
        self.assertEqual(_collect_dsv4_wo_a_grouped_shapes(root), {})

    def test_grouped_fp8_plan_reaches_the_launcher(self):
        """The (layout, m) sequence actually launched, not just the computed plan."""
        root = nn.Module()
        root.add_module("layer0", self._grouped_fp8_module(ep_size=4))
        shapes = _collect_dsv4_grouped_fp8_moe_shapes(root)
        seen = []

        def record(_label, desc, _fn, device=None):  # noqa: ARG001
            seen.append(desc)

        with mock.patch.object(
            warmup_module, "model_warm_up_enabled", return_value=True
        ), mock.patch.object(
            warmup_module, "_is_cuda_device", return_value=True
        ), mock.patch.object(
            warmup_module, "_assert_not_capturing"
        ), mock.patch.object(
            warmup_module, "_sync_cuda"
        ), mock.patch.object(
            warmup_module, "_release_cuda_cache"
        ), mock.patch.object(
            warmup_module,
            "_run_deepgemm_warmup_launches_serialized",
            side_effect=lambda _label, fn: fn(),
        ), mock.patch.object(
            warmup_module,
            "_run_deepgemm_warmup_launch_with_retry",
            side_effect=record,
        ):
            warmup_grouped_fp8_moe_jit(
                shapes, tokens_per_rank=8, device=torch.device("cpu")
            )

        # conc 8 over ep 4: one contiguous at the 128 alignment, then the two
        # masked expected_m values (t_rows 64 and max_m 128).
        self.assertEqual(len(seen), 3, seen)
        self.assertIn("contiguous m=128", seen[0])
        self.assertIn("masked m=64", seen[1])
        self.assertIn("masked m=128", seen[2])

    def test_grouped_fp8_plan_is_contiguous_only_for_prefill_shapes(self):
        """Past the masked cap only the contiguous launch remains."""
        root = nn.Module()
        root.add_module("layer0", self._grouped_fp8_module(ep_size=4))
        shapes = _collect_dsv4_grouped_fp8_moe_shapes(root)
        seen = []
        with mock.patch.object(
            warmup_module, "model_warm_up_enabled", return_value=True
        ), mock.patch.object(
            warmup_module, "_is_cuda_device", return_value=True
        ), mock.patch.object(
            warmup_module, "_assert_not_capturing"
        ), mock.patch.object(
            warmup_module, "_sync_cuda"
        ), mock.patch.object(
            warmup_module, "_release_cuda_cache"
        ), mock.patch.object(
            warmup_module,
            "_run_deepgemm_warmup_launches_serialized",
            side_effect=lambda _label, fn: fn(),
        ), mock.patch.object(
            warmup_module,
            "_run_deepgemm_warmup_launch_with_retry",
            side_effect=lambda _l, desc, _f, device=None: seen.append(desc),
        ):
            warmup_grouped_fp8_moe_jit(
                shapes, tokens_per_rank=16384, device=torch.device("cpu")
            )
        self.assertEqual(len(seen), 1, seen)
        self.assertIn("contiguous", seen[0])

    def test_warmups_are_noops_when_model_warmup_is_disabled(self):
        with mock.patch.object(
            warmup_module, "model_warm_up_enabled", return_value=False
        ):
            warmup_grouped_fp8_moe_jit(
                {(8, 512, 512, 512, 256, 4): {"name": "x", "ep_size": 4}},
                tokens_per_rank=8,
                device=torch.device("cpu"),
            )
            warmup_wo_a_grouped_jit(
                {(9, 256, 512): {"name": "x"}},
                max_m=8,
                device=torch.device("cpu"),
            )

    def test_warmups_are_noops_on_a_cpu_device(self):
        """Guards the SM100 case: the collectors return {} and this must not run."""
        warmup_grouped_fp8_moe_jit({}, tokens_per_rank=8, device=torch.device("cpu"))
        warmup_wo_a_grouped_jit({}, max_m=8, device=torch.device("cpu"))


if __name__ == "__main__":
    unittest.main()
