"""GLM checkpoint selection and its input layout on the common MegaMoE path."""

import sys
import tempfile
import unittest
from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from rtp_llm.models.glm53_prefill_parallel import (
    glm_prefill_moe_chunk_plan,
    resolve_moe_policy,
)
from rtp_llm.models_py.modules.factory.fused_moe.utils.fp8_fp4 import chunked_layer
from rtp_llm.models_py.modules.factory.fused_moe.utils.mega_moe import (
    buffer,
    fp8_capacity,
)
from rtp_llm.ops import RoleType


def _model():
    return SimpleNamespace(
        model_type="glm5_3_flash",
        moe_layer_index=[1],
        moe_k=8,
        n_shared_experts=1,
        expert_num=24,
        swiglu_limit=0.0,
        eplb_config=SimpleNamespace(phy_exp_num=lambda n: n),
        quant_config=SimpleNamespace(
            get_method=lambda: "FP8_PER_BLOCK",
            is_quanted=lambda: True,
            group_size=lambda: 128,
        ),
    )


def _parallelism(role=RoleType.PREFILL, rank=0, size=8):
    tp = size if role == RoleType.PREFILL else 1
    return SimpleNamespace(
        ep_size=size,
        world_size=size,
        ep_rank=rank,
        world_rank=rank,
        role_type=role,
        local_rank=rank,
        get_attn_tp_size=lambda: tp,
        get_ffn_tp_size=lambda: tp,
        prefill_cp_config=SimpleNamespace(is_enabled=lambda: False),
    )


def _backend():
    return SimpleNamespace(
        transform_weights_for_mega_moe_fp8=mock.Mock(),
        get_block_m_for_mega_moe_fp8=mock.Mock(),
    )


def _layers(tp=1):
    executor = SimpleNamespace(_prepared_policy=object(), _is_decode_role=False)
    return [
        SimpleNamespace(
            mlp=SimpleNamespace(
                requires_ep_sync=True,
                routed_tp_size=tp,
                route_local_tokens=tp > 1,
                fused_moe=SimpleNamespace(fused_experts=executor),
            )
        )
        for _ in range(2)
    ]


def _distributed_contract(rank, url):
    dist.init_process_group(
        "gloo", init_method=url, rank=rank, world_size=3, timeout=timedelta(seconds=30)
    )
    try:
        par = _parallelism(rank=rank, size=3)
        with mock.patch.dict(sys.modules, {"deep_gemm": _backend()}), mock.patch.object(
            buffer,
            "_mega_moe_unavailable_reason",
            return_value="missing symbol" if rank == 1 else None,
        ):
            selected = SimpleNamespace(moe_strategy="auto")
            assert resolve_moe_policy(_model(), par, selected) is None
            assert selected.moe_strategy == "auto"

        for lengths in ((0, 3, 9), (0, 0, 0), (3, 4, 1)):
            tokens = lengths[rank]
            x = torch.arange(tokens * 2, dtype=torch.float32).view(tokens, 2)
            calls = []

            def kernel(start, end, out):
                participants = torch.ones((), dtype=torch.int64)
                dist.all_reduce(participants)
                assert participants.item() == 3
                calls.append(end - start)
                return x[start:end] * 3

            layers = _layers()
            with glm_prefill_moe_chunk_plan(layers, tokens, x.device):
                for _ in layers:
                    torch.testing.assert_close(
                        chunked_layer.run_moe_chunks(x, 4, kernel), x * 3
                    )
            count = max((max(lengths) + 3) // 4, 1)
            assert calls == [
                max(min(tokens - i * 4, 4), 0) for i in range(count)
            ] * len(layers)
    finally:
        dist.destroy_process_group()


class FP8CapacityTest(unittest.TestCase):
    def test_unknown_and_unsafe_backends_keep_legacy_limit(self):
        backends = [
            SimpleNamespace(),
            SimpleNamespace(get_block_m_candidates_for_mega_moe_fp8=lambda: [192]),
        ]
        backends.extend(
            SimpleNamespace(
                get_block_m_candidates_for_mega_moe_fp8=lambda values=values: values,
                _C=SimpleNamespace(
                    get_token_alignment_for_mega_moe_fp8=lambda alignment=alignment: alignment
                ),
            )
            for values, alignment in (
                ([16, 192, 224], 384),
                ([], 384),
                ([0, 192], 384),
                ([-192, 192], 384),
                ([192], 0),
                ([192], -384),
            )
        )
        for backend in backends:
            with mock.patch.dict(sys.modules, {"deep_gemm": backend}):
                self.assertEqual(fp8_capacity.fp8_capacity(131072, 288, 8, 8), 2880)
                self.assertEqual(fp8_capacity.fp8_capacity(48, 288, 8, 8), 48)

    def test_ring_aligned_backend_preserves_configured_budget(self):
        backend = SimpleNamespace(
            get_block_m_candidates_for_mega_moe_fp8=lambda: [
                8,
                16,
                32,
                64,
                96,
                128,
                192,
            ],
            _C=SimpleNamespace(get_token_alignment_for_mega_moe_fp8=lambda: 384),
        )
        with mock.patch.dict(sys.modules, {"deep_gemm": backend}):
            for tokens in (1, 48, 2880, 2883, 16384, 65536, 131072):
                self.assertEqual(fp8_capacity.fp8_capacity(tokens, 288, 8, 8), tokens)

    def test_public_wheel_requires_its_exact_version_and_compatible_alignment(self):
        backend = SimpleNamespace(
            _C=SimpleNamespace(get_token_alignment_for_mega_moe_fp8=lambda: 1920),
        )
        with mock.patch.dict(sys.modules, {"deep_gemm": backend}):
            with mock.patch.object(
                fp8_capacity, "version", return_value="2.8.2+522942a.cu132"
            ):
                self.assertEqual(fp8_capacity.fp8_capacity(16384, 288, 8, 8), 16384)
                backend._C.get_token_alignment_for_mega_moe_fp8 = lambda: 384
                self.assertEqual(fp8_capacity.fp8_capacity(16384, 288, 8, 8), 2880)
            backend._C.get_token_alignment_for_mega_moe_fp8 = lambda: 1920
            with mock.patch.object(
                fp8_capacity, "version", return_value="2.8.2+other.cu132"
            ):
                self.assertEqual(fp8_capacity.fp8_capacity(16384, 288, 8, 8), 2880)


class GlmMegaMoeContractTest(unittest.TestCase):
    def test_auto_uses_fp8_and_replicates_only_prefill_shared_weights(self):
        for role in (RoleType.PREFILL, RoleType.DECODE, RoleType.PDFUSION):
            selected = SimpleNamespace(moe_strategy="auto")
            with self.subTest(role=role), mock.patch.dict(
                sys.modules, {"deep_gemm": _backend()}
            ), mock.patch.object(
                buffer, "_mega_moe_unavailable_reason", return_value=None
            ):
                policy = resolve_moe_policy(_model(), _parallelism(role), selected)
                self.assertEqual(selected.moe_strategy, "mega_moe_fp8_se")
                self.assertEqual(
                    policy.replicate_shared_weights, role == RoleType.PREFILL
                )
                self.assertTrue(policy.preserve_fp32_scales)
                self.assertIsNone(policy.activation_clamp)

    def test_shared_api_miss_keeps_routed_fp8_with_separate_shared_weights(self):
        backend = _backend()
        del backend.get_block_m_for_mega_moe_fp8
        with mock.patch.dict(sys.modules, {"deep_gemm": backend}), mock.patch.object(
            buffer, "_mega_moe_unavailable_reason", return_value=None
        ):
            selected = SimpleNamespace(moe_strategy="auto")
            policy = resolve_moe_policy(_model(), _parallelism(), selected)
            self.assertEqual(selected.moe_strategy, "mega_moe_fp8")
            self.assertTrue(policy.replicate_shared_weights)

    def test_unsupported_backend_declines_auto_and_explains_forced_strategy(self):
        disabled = _parallelism()
        disabled.ep_size = 4
        with mock.patch.object(
            buffer, "_mega_moe_unavailable_reason", return_value="missing FP8 allocator"
        ):
            for parallelism, reason in (
                (disabled, "EP == WORLD"),
                (_parallelism(), "missing FP8 allocator"),
            ):
                auto = SimpleNamespace(moe_strategy="auto")
                self.assertIsNone(resolve_moe_policy(_model(), parallelism, auto))
                self.assertEqual(auto.moe_strategy, "auto")
                with self.assertRaisesRegex(ValueError, reason):
                    resolve_moe_policy(
                        _model(),
                        parallelism,
                        SimpleNamespace(moe_strategy="mega_moe_fp8_se"),
                    )

    def test_stack_plan_uses_routed_tokens_and_skips_heterogeneous_stacks(self):
        for sharded, expected in ((False, 3), (True, 17)):
            layers = _layers(8)
            with mock.patch.object(
                chunked_layer, "synchronize_moe_chunk_tokens", return_value=expected
            ) as synchronize:
                with glm_prefill_moe_chunk_plan(
                    layers, 17, torch.device("cpu"), token_sharded=sharded
                ):
                    self.assertEqual(chunked_layer.planned_moe_tokens(), expected)
                synchronize.assert_called_once_with(expected, torch.device("cpu"))
            layers[1].mlp.routed_tp_size = 1
            with glm_prefill_moe_chunk_plan(layers, 17, torch.device("cpu")):
                self.assertIsNone(chunked_layer.planned_moe_tokens())

    def test_ep_agrees_before_loading_and_runs_empty_chunks(self):
        with tempfile.TemporaryDirectory() as directory:
            url = (Path(directory) / "rendezvous").as_uri()
            mp.start_processes(
                _distributed_contract,
                args=(url,),
                nprocs=3,
                join=True,
                start_method="fork",
            )


if __name__ == "__main__":
    unittest.main()
