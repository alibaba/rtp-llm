"""Tests for CP MoE allgather+reduce_scatter communication pattern.

Tests:
- is_cp_prefill property on FMHA base classes
- skip_tp_allreduce in PureTpRouter and DenseMLP
- allgather+reduce_scatter wrapping in GenericMoeDecoderLayer
- allgather+reduce_scatter roundtrip with real NCCL (multiprocessing)
"""

import logging
import multiprocessing as mp
import os
import unittest
from unittest.mock import MagicMock, patch

logging.basicConfig(level=logging.INFO)

import torch

from rtp_llm.models_py.distributed.collective_torch import (
    Group,
    all_gather,
    destroy_distributed_environment,
    init_distributed_environment,
    reduce_scatter,
)
from rtp_llm.models_py.modules.factory.attention.fmha_impl_base import (
    FMHAImplBase,
    MlaImplBase,
)
from rtp_llm.models_py.modules.factory.fused_moe.defs.fused_moe import (
    CombineForwardPayload,
)
from rtp_llm.ops import NcclCommConfig, ParallelismConfig
from rtp_llm.test.utils.port_util import PortManager


class TestIsCpPrefillProperty(unittest.TestCase):
    """Test is_cp_prefill property on FMHA/MLA base classes."""

    def test_fmha_impl_base_default_false(self):
        class ConcreteFMHA(FMHAImplBase):
            def forward(self, qkv, kv_cache):
                pass

            @staticmethod
            def support(attn_configs, attn_inputs):
                return True

        impl = ConcreteFMHA.__new__(ConcreteFMHA)
        self.assertFalse(impl.is_cp_prefill)

    def test_mla_impl_base_default_false(self):
        class ConcreteMLA(MlaImplBase):
            def forward(self, q, compressed_kv, k_pe, kv_cache, layer_id, topk_indices=None):
                pass

            @staticmethod
            def support(attn_configs, attn_inputs):
                return True

        impl = ConcreteMLA.__new__(ConcreteMLA)
        self.assertFalse(impl.is_cp_prefill)

    def test_cp_override_returns_true(self):
        class CPFmhaImpl(FMHAImplBase):
            def __init__(self):
                self._is_cp_prefill = True

            @property
            def is_cp_prefill(self):
                return self._is_cp_prefill

            def forward(self, qkv, kv_cache):
                pass

            @staticmethod
            def support(attn_configs, attn_inputs):
                return True

        self.assertTrue(CPFmhaImpl().is_cp_prefill)

    def test_getattr_fallback_for_plain_object(self):
        self.assertFalse(getattr(object(), "is_cp_prefill", False))


class TestSkipTpAllreduce(unittest.TestCase):
    """Test skip_tp_allreduce in PureTpRouter and DenseMLP."""

    def test_pure_tp_router_skip_allreduce(self):
        from rtp_llm.models_py.modules.factory.fused_moe.impl.cuda.routers.pure_tp_router import (
            PureTpRouterBase,
        )

        router = PureTpRouterBase.__new__(PureTpRouterBase)
        router.tp_size = 4
        payload = CombineForwardPayload(fused_expert_output=torch.randn(4, 8))

        with patch(
            "rtp_llm.models_py.modules.factory.fused_moe.impl.cuda.routers.pure_tp_router.all_reduce"
        ) as mock_ar:
            router.finalize(payload, torch.randn(4, 2), torch.randint(0, 8, (4, 2)),
                            False, {"skip_tp_allreduce": True})
            mock_ar.assert_not_called()

    def test_pure_tp_router_normal_allreduce(self):
        from rtp_llm.models_py.modules.factory.fused_moe.impl.cuda.routers.pure_tp_router import (
            PureTpRouterBase,
        )

        router = PureTpRouterBase.__new__(PureTpRouterBase)
        router.tp_size = 4
        payload = CombineForwardPayload(fused_expert_output=torch.randn(4, 8))

        with patch(
            "rtp_llm.models_py.modules.factory.fused_moe.impl.cuda.routers.pure_tp_router.all_reduce"
        ) as mock_ar:
            mock_ar.return_value = torch.randn(4, 8)
            router.finalize(payload, torch.randn(4, 2), torch.randint(0, 8, (4, 2)),
                            False, None)
            mock_ar.assert_called_once()

    def test_dense_mlp_skip_allreduce(self):
        from rtp_llm.models_py.modules.hybrid.dense_mlp import DenseMLP

        mlp = DenseMLP.__new__(DenseMLP)
        mlp.skip_tp_allreduce = True
        mlp.parallelism_config = MagicMock()
        mlp.parallelism_config.get_ffn_tp_size.return_value = 4
        mlp.is_gated = True
        mlp.up_proj = MagicMock(return_value=torch.randn(2, 8))
        mlp.act_fn = MagicMock(return_value=torch.randn(2, 4))
        mlp.down_proj = MagicMock(return_value=torch.randn(2, 4))

        with patch("rtp_llm.models_py.modules.hybrid.dense_mlp.all_reduce") as mock_ar:
            mlp.forward(torch.randn(2, 4))
            mock_ar.assert_not_called()

    def test_dense_mlp_normal_allreduce(self):
        from rtp_llm.models_py.modules.hybrid.dense_mlp import DenseMLP

        mlp = DenseMLP.__new__(DenseMLP)
        mlp.skip_tp_allreduce = False
        mlp.parallelism_config = MagicMock()
        mlp.parallelism_config.get_ffn_tp_size.return_value = 4
        mlp.is_gated = True
        mlp.up_proj = MagicMock(return_value=torch.randn(2, 8))
        mlp.act_fn = MagicMock(return_value=torch.randn(2, 4))
        mlp.down_proj = MagicMock(return_value=torch.randn(2, 4))

        with patch("rtp_llm.models_py.modules.hybrid.dense_mlp.all_reduce") as mock_ar:
            mock_ar.return_value = torch.randn(2, 4)
            mlp.forward(torch.randn(2, 4))
            mock_ar.assert_called_once()


class TestDecoderLayerCpMoeWrapping(unittest.TestCase):
    """Test allgather+reduce_scatter wrapping in GenericMoeDecoderLayer."""

    def _make_layer(self, use_cp_moe_allgather_rs=True):
        from rtp_llm.models_py.model_desc.generic_moe import GenericMoeDecoderLayer

        layer = GenericMoeDecoderLayer.__new__(GenericMoeDecoderLayer)
        layer.layer_idx = 1
        layer.use_cp_moe_allgather_rs = use_cp_moe_allgather_rs
        layer.input_layernorm = MagicMock(side_effect=lambda h, r: h)
        layer.self_attn = MagicMock(
            side_effect=lambda hidden_states, fmha_impl, kv_cache: hidden_states
        )
        layer.post_attention_layernorm = MagicMock(side_effect=lambda h, r: h)
        layer.mlp = MagicMock(side_effect=lambda x: x * 2)
        return layer

    def _make_cp_fmha(self):
        fmha = MagicMock()
        fmha.is_cp_prefill = True
        return fmha

    def _make_non_cp_fmha(self):
        fmha = MagicMock()
        fmha.is_cp_prefill = False
        return fmha

    def test_cp_active_calls_allgather_and_reduce_scatter(self):
        layer = self._make_layer(use_cp_moe_allgather_rs=True)

        with patch("rtp_llm.models_py.model_desc.generic_moe.all_gather") as mock_ag, \
             patch("rtp_llm.models_py.model_desc.generic_moe.reduce_scatter") as mock_rs:
            mock_ag.side_effect = lambda t, group: torch.cat([t, t], dim=0)
            mock_rs.side_effect = lambda t, group: t[: t.shape[0] // 2]

            layer.forward(torch.randn(4, 8), torch.randn(4, 8), self._make_cp_fmha())

            self.assertEqual(mock_ag.call_count, 2)
            self.assertEqual(mock_rs.call_count, 2)

    def test_decode_no_allgather_reduce_scatter(self):
        layer = self._make_layer(use_cp_moe_allgather_rs=True)

        with patch("rtp_llm.models_py.model_desc.generic_moe.all_gather") as mock_ag, \
             patch("rtp_llm.models_py.model_desc.generic_moe.reduce_scatter") as mock_rs:
            layer.forward(torch.randn(4, 8), torch.randn(4, 8), self._make_non_cp_fmha())
            mock_ag.assert_not_called()
            mock_rs.assert_not_called()

    def test_flag_disabled_no_wrapping(self):
        layer = self._make_layer(use_cp_moe_allgather_rs=False)

        with patch("rtp_llm.models_py.model_desc.generic_moe.all_gather") as mock_ag, \
             patch("rtp_llm.models_py.model_desc.generic_moe.reduce_scatter") as mock_rs:
            layer.forward(torch.randn(4, 8), torch.randn(4, 8), self._make_cp_fmha())
            mock_ag.assert_not_called()
            mock_rs.assert_not_called()

    def test_mlp_receives_full_tokens_during_cp(self):
        layer = self._make_layer(use_cp_moe_allgather_rs=True)
        scattered_size = 4
        tp_size = 2

        with patch("rtp_llm.models_py.model_desc.generic_moe.all_gather") as mock_ag, \
             patch("rtp_llm.models_py.model_desc.generic_moe.reduce_scatter") as mock_rs:
            mock_ag.side_effect = lambda t, group: torch.cat([t] * tp_size, dim=0)
            mock_rs.side_effect = lambda t, group: t[: t.shape[0] // tp_size]

            layer.forward(
                torch.randn(scattered_size, 8),
                torch.randn(scattered_size, 8),
                self._make_cp_fmha(),
            )

            mlp_input = layer.mlp.call_args[0][0]
            self.assertEqual(mlp_input.shape[0], scattered_size * tp_size)


def _test_cp_moe_roundtrip_worker(
    rank: int, world_size: int, tp_size: int, dp_size: int, nccl_port: int
):
    """Worker: verify allgather -> partial_compute -> reduce_scatter recovers data.

    Simulates the CP MoE pattern:
    1. Each rank starts with scattered tokens (chunk_size tokens)
    2. allgather: each rank gets full tokens (world_size * chunk_size)
    3. Each rank computes partial result (1/world_size of final answer)
    4. reduce_scatter: sum partials and scatter back
    Expected: output == original scattered chunk (within fp tolerance)
    """
    try:
        parallelism_config = ParallelismConfig()
        base_port = nccl_port + 11
        nccl_comm_config = NcclCommConfig(
            nccl_ip="127.0.0.1",
            tp_nccl_port=base_port - 2,
            dp_tp_nccl_port=base_port - 10,
            ffn_tp_nccl_port=base_port - 5,
        )
        parallelism_config.world_rank = rank
        parallelism_config.world_size = world_size
        parallelism_config.local_rank = (
            rank % torch.cuda.device_count() if torch.cuda.is_available() else 0
        )
        parallelism_config.tp_size = tp_size
        parallelism_config.dp_size = dp_size

        torch.cuda.set_device(parallelism_config.local_rank)
        torch.set_default_device(f"cuda:{parallelism_config.local_rank}")
        init_distributed_environment(
            parallelism_config,
            nccl_comm_config=nccl_comm_config,
            nccl_init_port=nccl_port,
            backend="nccl",
            timeout=60,
        )

        chunk_size = 8
        hidden_dim = 16
        device = f"cuda:{parallelism_config.local_rank}"

        # Simulate scattered hidden_states (each rank has its own chunk)
        torch.manual_seed(42 + rank)
        hidden_scattered = torch.randn(chunk_size, hidden_dim, device=device)
        residual_scattered = torch.randn(chunk_size, hidden_dim, device=device)

        # Step 1: allgather
        hidden_full = all_gather(hidden_scattered, group=Group.TP)
        residual_full = all_gather(residual_scattered, group=Group.TP)
        assert hidden_full.shape[0] == tp_size * chunk_size

        # Step 2: simulate MoE partial compute (each rank does 1/tp_size)
        hidden_partial = hidden_full / tp_size
        residual_partial = residual_full / tp_size

        # Step 3: reduce_scatter
        hidden_out = reduce_scatter(hidden_partial, group=Group.TP)
        residual_out = reduce_scatter(residual_partial, group=Group.TP)
        torch.cuda.synchronize()

        # Verify: output should match original scattered chunk
        assert hidden_out.shape == hidden_scattered.shape, (
            f"Rank {rank}: shape mismatch {hidden_out.shape} vs {hidden_scattered.shape}"
        )
        assert torch.allclose(hidden_out, hidden_scattered, atol=1e-5), (
            f"Rank {rank}: hidden max diff = {(hidden_out - hidden_scattered).abs().max().item()}"
        )
        assert torch.allclose(residual_out, residual_scattered, atol=1e-5), (
            f"Rank {rank}: residual max diff = {(residual_out - residual_scattered).abs().max().item()}"
        )

        torch.distributed.barrier()
        torch.cuda.synchronize()
        destroy_distributed_environment()
    except Exception as e:
        print(f"Rank {rank} error: {e}")
        import traceback
        traceback.print_exc()
        raise


class TestCpMoeRoundtrip(unittest.TestCase):
    """Test allgather+reduce_scatter roundtrip with real NCCL (multi-GPU)."""

    def setUp(self):
        if not torch.cuda.is_available():
            self.skipTest("CUDA not available")
        try:
            mp.set_start_method("spawn", force=True)
        except RuntimeError:
            pass
        self.port_manager = PortManager()

    def _run_test(self, worker_func, world_size, tp_size, dp_size):
        ports, locks = self.port_manager.get_consecutive_ports(1)
        nccl_port = ports[0]
        try:
            processes = []
            for rank in range(world_size):
                p = mp.Process(
                    target=worker_func,
                    args=(rank, world_size, tp_size, dp_size, nccl_port),
                    name=f"rank-{rank}",
                )
                p.start()
                processes.append(p)
            for p in processes:
                p.join(timeout=120)
                if p.exitcode != 0:
                    raise RuntimeError(f"Process {p.name} exited with code {p.exitcode}")
        finally:
            for lock in locks:
                lock.__exit__(None, None, None)

    def test_roundtrip_tp4(self):
        """allgather+reduce_scatter roundtrip with tp_size=4"""
        self._run_test(_test_cp_moe_roundtrip_worker, world_size=4, tp_size=4, dp_size=1)

    def test_roundtrip_tp2_dp2(self):
        """allgather+reduce_scatter roundtrip with tp_size=2, dp_size=2"""
        self._run_test(_test_cp_moe_roundtrip_worker, world_size=4, tp_size=2, dp_size=2)


if __name__ == "__main__":
    os.environ["NCCL_DEBUG"] = "INFO"
    unittest.main()
