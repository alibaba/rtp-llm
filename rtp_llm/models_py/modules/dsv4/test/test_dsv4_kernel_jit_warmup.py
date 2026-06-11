import os
import sys
import types
import unittest
from types import SimpleNamespace

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

from rtp_llm.models_py.modules.dsv4.dsv4_kernel_jit_warmup import (
    _collect_dsv4_branch_kernel_configs,
    _collect_dsv4_batched_fp8_einsum_shapes,
    _collect_dsv4_dense_gemm_shapes,
    _collect_dsv4_fp8_mqa_logits_shapes,
    _collect_dsv4_mhc_prenorm_shapes,
    _compute_mhc_prenorm_num_split,
    _cp_padded_tokens_per_rank_bound,
    _dense_gemm_m_grid,
    _dist_rank,
    _generate_dense_gemm_warmup_m_grid,
    _generate_mhc_prenorm_warmup_specs,
    _run_deepgemm_warmup_launch_with_retry,
    _sm100_dense_layout_signature,
    resolve_dense_gemm_warmup_max_m,
)


def _module_type(name, attrs):
    def __init__(self):
        nn.Module.__init__(self)
        for key, value in attrs.items():
            setattr(self, key, value)

    return type(name, (nn.Module,), {"__init__": __init__})


class Dsv4KernelJitWarmupTest(unittest.TestCase):
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
                role_type_name="DECODE",
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

    def test_collect_mhc_prenorm_shapes_uses_tilelang_units_only(self):
        root = nn.Module()
        TileLangHCUnit = _module_type(
            "TileLangHCUnit",
            {
                "fn": torch.empty((24, 16384), dtype=torch.float32),
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
        )
        splits = {split for split, _ in specs}
        self.assertTrue({1, 8, 49, 64}.issubset(splits))

        reps_by_split = dict(specs)
        self.assertEqual(
            _compute_mhc_prenorm_num_split(
                m_value=reps_by_split[49],
                k_value=16384,
                num_sms=148,
            ),
            49,
        )
        self.assertEqual(
            _compute_mhc_prenorm_num_split(
                m_value=reps_by_split[8],
                k_value=16384,
                num_sms=148,
            ),
            8,
        )

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


if __name__ == "__main__":
    unittest.main()
