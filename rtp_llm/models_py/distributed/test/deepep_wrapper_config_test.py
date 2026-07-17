from types import SimpleNamespace
from unittest import TestCase, main
from unittest.mock import patch

from rtp_llm.config.model_config import ModelConfig
from rtp_llm.models_py.distributed.deepep_wrapper import (
    DeepEPWrapper,
    init_deepep_wrapper,
)
from rtp_llm.ops import CPRotateMethod, MoeConfig, ParallelismConfig


class DeepEPWrapperConfigTest(TestCase):
    def test_cp_low_latency_uses_router_token_capacity(self) -> None:
        model_config = ModelConfig()
        model_config.hidden_size = 2048
        model_config.expert_num = 256
        model_config.moe_k = 8

        parallelism_config = ParallelismConfig()
        parallelism_config.dp_size = 1
        parallelism_config.dp_rank = 0
        parallelism_config.tp_size = 2
        parallelism_config.tp_rank = 0
        parallelism_config.ep_size = 2
        parallelism_config.ep_rank = 0
        parallelism_config.world_size = 2
        parallelism_config.local_rank = 0
        parallelism_config.prefill_cp_config.method = CPRotateMethod.ALL_GATHER
        self.assertEqual(parallelism_config.get_attn_tp_size(), 1)

        moe_config = MoeConfig()
        moe_config.use_deepep_low_latency = True
        engine_config = SimpleNamespace(
            hw_kernel_config=None,
            parallelism_config=parallelism_config,
            moe_config=moe_config,
        )

        cases = [
            ("base", 1757, 1792),
            ("mtp_eagle_gen4", 1757 * (4 + 1), 8832),
        ]
        with patch.object(DeepEPWrapper, "supported", return_value=True), patch.object(
            DeepEPWrapper, "create"
        ) as mock_create, patch(
            "rtp_llm.models_py.distributed.deepep_wrapper.allow_mnnvl",
            return_value=False,
        ):
            for case_name, ll_num_max_token, expected_per_rank in cases:
                with self.subTest(case=case_name):
                    moe_config.ll_num_max_token = ll_num_max_token
                    mock_create.reset_mock()

                    init_deepep_wrapper(engine_config, model_config)

                    created_config = mock_create.call_args.args[0]
                    self.assertEqual(created_config.tp_size, 2)
                    self.assertEqual(
                        created_config.ll_num_max_token_per_rank,
                        expected_per_rank,
                    )


if __name__ == "__main__":
    main()
