from types import SimpleNamespace
from unittest import TestCase, main
from unittest.mock import MagicMock, patch

import torch

from rtp_llm.models_py.model_desc.generic_moe import GenericMoeLayer
from rtp_llm.models_py.modules import FusedMoeFactory, LinearFactory
from rtp_llm.ops import HWKernelConfig
from rtp_llm.utils.model_weight import W


class GenericMoeHWKernelConfigTest(TestCase):
    def test_shared_expert_gate_uses_no_swizzle(self) -> None:
        quant_config = object()
        hw_kernel_config = HWKernelConfig()
        hw_kernel_config.use_swizzleA = True
        config = SimpleNamespace(
            hidden_size=32,
            inter_size=64,
            expert_num=2,
            moe_k=1,
            eplb_config=SimpleNamespace(phy_exp_num=lambda count: count),
            quant_config=quant_config,
            moe_style=0,
        )
        parallelism_config = SimpleNamespace(
            dp_rank=0,
            dp_size=1,
            ep_size=1,
            get_ffn_tp_size=lambda: 1,
        )
        moe_config = SimpleNamespace(fake_balance_expert=False)
        weights = {
            W.moe_w1: torch.empty(2, 32, 64),
            W.moe_w2: torch.empty(2, 64, 32),
            W.shared_expert_gate: torch.empty(32, 1),
        }

        with (
            patch("rtp_llm.models_py.model_desc.generic_moe.MoEConfigAdapter"),
            patch.object(
                FusedMoeFactory,
                "create_fused_moe",
                return_value=MagicMock(),
            ),
            patch.object(
                LinearFactory,
                "create_linear_from_weights",
                side_effect=lambda *args, **kwargs: MagicMock(),
            ) as create_linear,
        ):
            GenericMoeLayer(
                config,
                parallelism_config,
                weights,
                moe_config,
                hw_kernel_config=hw_kernel_config,
            )

        gate_calls = {
            call.args[1]: call
            for call in create_linear.call_args_list
            if len(call.args) > 1
        }
        self.assertIs(
            gate_calls[W.moe_gate].args[5],
            hw_kernel_config,
            "the routed gate must retain the configured swizzle path",
        )
        shared_gate_call = gate_calls[W.shared_expert_gate]
        self.assertIs(shared_gate_call.kwargs["quant_config"], quant_config)
        self.assertIsNone(
            shared_gate_call.kwargs["hw_kernel_config"],
            "the unshuffled scalar gate must use NoSwizzle",
        )


if __name__ == "__main__":
    main()
