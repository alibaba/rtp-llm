import sys
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch
from torch import nn

from rtp_llm.models_py.model_desc.kimi_k3 import resolve_kimi_k3_moe_strategy
from rtp_llm.models_py.modules.kimi_k3.input_packer_se import (
    FusedKimiK3MegaMoeSeInputPacker,
    TorchKimiK3MegaMoeSeInputPacker,
    get_kimi_k3_mega_moe_se_input_packer,
)
from rtp_llm.models_py.modules.kimi_k3.moe import KimiK3LatentMoE
from rtp_llm.models_py.modules.kimi_k3.moe_se import KimiK3LatentMoESE


class KimiK3MegaMoeSeUnitTest(unittest.TestCase):
    @staticmethod
    def _regular_module() -> KimiK3LatentMoE:
        module = KimiK3LatentMoE.__new__(KimiK3LatentMoE)
        nn.Module.__init__(module)
        module._mega_buf = SimpleNamespace(num_max_tokens_per_rank=8)
        module._mega_input_packer = SimpleNamespace(pack=MagicMock())
        module._mega_y = torch.empty((8, 4), dtype=torch.bfloat16)
        module._mega_l1_w = object()
        module._mega_l1_sf = object()
        module._mega_l2_w = object()
        module._mega_l2_sf = object()
        module.beta = 4.0
        module.linear_beta = 25.0
        return module

    @staticmethod
    def _se_module() -> KimiK3LatentMoESE:
        module = KimiK3LatentMoESE.__new__(KimiK3LatentMoESE)
        nn.Module.__init__(module)
        module._mega_buf = SimpleNamespace(num_max_tokens_per_rank=8)
        module._mega_input_packer = SimpleNamespace(pack=MagicMock())
        module._mega_y = torch.empty((8, 4), dtype=torch.bfloat16)
        module._mega_shared_x = torch.empty((8, 6), dtype=torch.bfloat16)
        module._mega_shared_y = torch.empty((8, 6), dtype=torch.bfloat16)
        module._mega_l1_w = object()
        module._mega_l1_sf = object()
        module._mega_l2_w = object()
        module._mega_l2_sf = object()
        module._mega_shared_l1_w = object()
        module._mega_shared_l2_w = object()
        module._mega_shared_hidden = 6
        module.beta = 4.0
        module.linear_beta = 25.0
        return module

    def test_strategy_names_are_explicit_and_auto_preserves_regular_mega(self) -> None:
        for configured, expected in (
            ("auto", "mega_moe"),
            ("mega_moe", "mega_moe"),
            ("mega_moe_se", "mega_moe_se"),
        ):
            self.assertEqual(
                resolve_kimi_k3_moe_strategy(SimpleNamespace(moe_strategy=configured)),
                expected,
            )
        with self.assertRaisesRegex(ValueError, "supports only"):
            resolve_kimi_k3_moe_strategy(SimpleNamespace(moe_strategy="fp4_no_dp"))

    def test_regular_mega_call_does_not_pass_shared_arguments(self) -> None:
        module = self._regular_module()
        peer_kernel = MagicMock()
        routed_input = torch.empty((3, 4), dtype=torch.bfloat16)
        expert_ids = torch.zeros((3, 2), dtype=torch.int64)
        routing_weights = torch.ones((3, 2), dtype=torch.float32)
        with patch.dict(
            sys.modules,
            {"deep_gemm": SimpleNamespace(fp8_fp4_mega_moe=peer_kernel)},
        ):
            output = module._deep_gemm_mega_expert_sum(
                routed_input,
                expert_ids,
                routing_weights,
            )

        self.assertEqual(tuple(output.shape), (3, 4))
        kwargs = peer_kernel.call_args.kwargs
        self.assertFalse(any(name.startswith("shared_") for name in kwargs))

    def test_se_call_passes_capacity_storages_and_returns_two_outputs(self) -> None:
        module = self._se_module()
        peer_kernel = MagicMock()
        routed_input = torch.empty((3, 4), dtype=torch.bfloat16)
        shared_input = torch.randn((3, 6), dtype=torch.bfloat16)
        expert_ids = torch.zeros((3, 2), dtype=torch.int64)
        routing_weights = torch.ones((3, 2), dtype=torch.float32)
        with patch.dict(
            sys.modules,
            {"deep_gemm": SimpleNamespace(fp8_fp4_mega_moe=peer_kernel)},
        ):
            routed_y, shared_y = module._deep_gemm_mega_expert_sum_with_shared(
                routed_input,
                shared_input,
                expert_ids,
                routing_weights,
            )

        self.assertEqual(tuple(routed_y.shape), (3, 4))
        self.assertEqual(tuple(shared_y.shape), (3, 6))
        torch.testing.assert_close(
            module._mega_shared_x[:3], shared_input, rtol=0, atol=0
        )
        kwargs = peer_kernel.call_args.kwargs
        self.assertIs(kwargs["shared_x"], module._mega_shared_x)
        self.assertIs(kwargs["shared_y"], module._mega_shared_y)
        self.assertIs(kwargs["shared_l1_weights"], module._mega_shared_l1_w)
        self.assertIs(kwargs["shared_l2_weights"], module._mega_shared_l2_w)

    def test_se_packer_has_independent_environment_switch(self) -> None:
        with patch.dict(
            "os.environ",
            {"KIMI_K3_MEGA_MOE_SE_INPUT_PACKER": "torch"},
            clear=False,
        ):
            self.assertIsInstance(
                get_kimi_k3_mega_moe_se_input_packer(),
                TorchKimiK3MegaMoeSeInputPacker,
            )
        with patch.dict(
            "os.environ",
            {"KIMI_K3_MEGA_MOE_SE_INPUT_PACKER": "fused"},
            clear=False,
        ):
            self.assertIsInstance(
                get_kimi_k3_mega_moe_se_input_packer(),
                FusedKimiK3MegaMoeSeInputPacker,
            )

    def test_multi_host_se_setup_delegates_to_nccl_ep_base(self) -> None:
        module = KimiK3LatentMoESE.__new__(KimiK3LatentMoESE)
        nn.Module.__init__(module)
        module.shared_expert_weight_shard = False
        module.parallelism_config = SimpleNamespace(local_world_size=8)
        with (
            patch.object(module, "_validate_mega_preconditions"),
            patch("torch.distributed.get_world_size", return_value=16),
            patch.object(
                KimiK3LatentMoE,
                "_setup_deep_gemm_mega",
                autospec=True,
            ) as base_setup,
        ):
            module._setup_deep_gemm_mega()
        base_setup.assert_called_once_with(module)

    def test_multi_host_se_forward_uses_base_nccl_ep_path(self) -> None:
        module = KimiK3LatentMoESE.__new__(KimiK3LatentMoESE)
        nn.Module.__init__(module)
        module._use_nccl_ep = True
        hidden_states = torch.randn((2, 4), dtype=torch.bfloat16)
        expected = torch.randn_like(hidden_states)
        valid_token_mask = torch.tensor([True, False])
        with patch.object(
            KimiK3LatentMoE,
            "forward",
            autospec=True,
            return_value=expected,
        ) as base_forward:
            actual = module.forward(
                hidden_states,
                sequence_parallel=False,
                valid_token_count=1,
                valid_token_mask=valid_token_mask,
            )
        self.assertIs(actual, expected)
        base_forward.assert_called_once_with(
            module,
            hidden_states,
            sequence_parallel=False,
            valid_token_count=1,
            valid_token_mask=valid_token_mask,
        )


if __name__ == "__main__":
    unittest.main()
