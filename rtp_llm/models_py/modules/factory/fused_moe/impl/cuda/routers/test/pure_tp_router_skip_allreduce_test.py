"""CUDA Pure TP router contract tests for deferred TP all-reduce."""

from unittest import TestCase, main
from unittest.mock import patch

import torch

from rtp_llm.config.model_config import ModelConfig
from rtp_llm.models_py.distributed.collective_torch import Group
from rtp_llm.models_py.modules.factory.fused_moe.defs.config_adapter import (
    MoEConfigAdapter,
)
from rtp_llm.models_py.modules.factory.fused_moe.defs.fused_moe import (
    SKIP_TP_ALLREDUCE_ARG,
    CombineForwardPayload,
)
from rtp_llm.models_py.modules.factory.fused_moe.defs.quant_config import (
    FusedMoEQuantConfig,
)
from rtp_llm.models_py.modules.factory.fused_moe.impl.cuda.routers.pure_tp_router import (
    PureTpRouterNoQuant,
)
from rtp_llm.ops import MoeConfig, ParallelismConfig


def _make_router(tp_size=2):
    model_config = ModelConfig()
    model_config.expert_num = 8
    model_config.moe_k = 2
    parallelism_config = ParallelismConfig()
    parallelism_config.tp_size = tp_size
    parallelism_config.tp_rank = 0
    parallelism_config.ep_size = 1
    parallelism_config.ep_rank = 0
    parallelism_config.dp_size = 1
    parallelism_config.dp_rank = 0
    parallelism_config.world_size = tp_size
    parallelism_config.world_rank = 0
    parallelism_config.local_rank = 0
    parallelism_config.local_world_size = tp_size
    moe_config = MoeConfig()
    moe_config.use_all_gather = True
    config = MoEConfigAdapter(
        model_config=model_config,
        parallelism_config=parallelism_config,
        moe_config=moe_config,
    )
    return PureTpRouterNoQuant(config, FusedMoEQuantConfig())


class PureTpRouterSkipAllreduceTest(TestCase):
    def test_pure_tp_router_declares_deferred_reduce_support(self):
        self.assertTrue(_make_router().supports_skip_tp_allreduce)

    @patch(
        "rtp_llm.models_py.modules.factory.fused_moe.impl.cuda.routers.pure_tp_router.all_reduce"
    )
    def test_finalize_reduction_matrix(self, mock_all_reduce):
        cases = (
            (1, None, False),
            (2, None, True),
            (2, {SKIP_TP_ALLREDUCE_ARG: False}, True),
            (2, {SKIP_TP_ALLREDUCE_ARG: True}, False),
            (1, {SKIP_TP_ALLREDUCE_ARG: True}, False),
        )
        for tp_size, extra_finalize_args, expect_all_reduce in cases:
            with self.subTest(
                tp_size=tp_size,
                extra_finalize_args=extra_finalize_args,
                expect_all_reduce=expect_all_reduce,
            ):
                mock_all_reduce.reset_mock()
                router = _make_router(tp_size=tp_size)
                expert_output = torch.randn(4, 8)
                reduced_output = torch.full_like(expert_output, 3.0)
                mock_all_reduce.return_value = reduced_output
                if extra_finalize_args is None:
                    extra_finalize_args = {
                        "a1_shape": expert_output.shape,
                        "original_num_tokens": expert_output.shape[0],
                    }
                else:
                    extra_finalize_args = {
                        "a1_shape": expert_output.shape,
                        "original_num_tokens": expert_output.shape[0],
                        **extra_finalize_args,
                    }
                result = router.finalize(
                    payload=CombineForwardPayload(fused_expert_output=expert_output),
                    topk_weights=torch.ones(4, 2),
                    topk_ids=torch.zeros(4, 2, dtype=torch.int32),
                    apply_router_weight_on_input=False,
                    extra_finalize_args=extra_finalize_args,
                )

                if expect_all_reduce:
                    mock_all_reduce.assert_called_once()
                    self.assertIs(mock_all_reduce.call_args.kwargs["group"], Group.TP)
                    self.assertIs(mock_all_reduce.call_args.args[0], expert_output)
                    self.assertIs(result, reduced_output)
                else:
                    mock_all_reduce.assert_not_called()
                    self.assertIs(result, expert_output)


if __name__ == "__main__":
    main()
