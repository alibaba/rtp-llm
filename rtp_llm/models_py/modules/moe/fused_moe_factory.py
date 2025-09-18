from typing import Dict

import torch

import rtp_llm.models_py.modules.utils as utils
from rtp_llm.config.gpt_init_model_parameters import GptInitModelParameters
from rtp_llm.models_py.modules.moe import FusedMoe
from rtp_llm.models_py.modules.moe.fused_batched_moe import (
    BatchedDataRouter,
    BatchedTritonExperts,
)
from rtp_llm.utils.model_weight import W

# TODO@miji move it to model init process?
initialized = False


def init_deepep_env_once(config: GptInitModelParameters):
    global initialized
    if initialized:
        return
    initialized = True
    from rtp_llm.distribute.deep_ep import init_deepep_wrapper
    from rtp_llm.distribute.process_group_state import (
        get_ep_group,
        init_distributed_environment,
    )

    init_distributed_environment(params=config, backend="nccl", timeout=None)
    ep_group = get_ep_group()
    assert ep_group.device_group is not None, "ep group device group is not initialized"
    init_deepep_wrapper(group=ep_group.device_group, params=config)


class FusedMoeFactory(object):
    @staticmethod
    def _create_fp8_per_block_fused_moe(
        config: GptInitModelParameters, weights: Dict[str, torch.Tensor]
    ):
        assert utils.is_cuda(), "FP8_PER_BLOCK only supports cuda"
        # single gpu
        if config.ep_size == 1 or config.tp_size == config.ep_size:
            from rtp_llm.models_py.modules.moe.executors.deepep_normal_executor import (
                DeepGemmContinousExecutor,
            )
            from rtp_llm.models_py.modules.moe.routers.deepgeemm_coutinous_router import (
                DeepGemmCountinousRouter,
            )

            router = DeepGemmCountinousRouter(config)
            executor = DeepGemmContinousExecutor(config, weights)
            return FusedMoe(router, executor, config.expert_num)
        # ep moe
        else:
            init_deepep_env_once(config)
            if config.moe_config.use_deepep_low_latency:
                raise ValueError("deep ep low latency mode not supported yet")
            else:
                from rtp_llm.models_py.modules.moe.executors.deepep_normal_executor import (
                    DeepGemmContinousExecutor,
                )
                from rtp_llm.models_py.modules.moe.routers.deepep_normal_router import (
                    DeepepNormalRouter,
                )

                router = DeepepNormalRouter(config)
                executor = DeepGemmContinousExecutor(config, weights)
                return FusedMoe(router, executor, config.expert_num)

    @staticmethod
    def create_fused_moe(
        config: GptInitModelParameters, weights: Dict[str, torch.Tensor]
    ) -> FusedMoe:
        # TODO get_method should return enu class other than string
        if config.quant_config is None:
            max_num_tokens = (
                config.max_generate_batch_size + config.tp_size - 1
            ) // config.tp_size

            router = BatchedDataRouter(
                max_num_tokens=max_num_tokens,
                num_local_experts=config.expert_num,
                num_dispatchers=1,
                rank=0,
            )

            experts = BatchedTritonExperts(
                max_num_tokens=max_num_tokens,
                num_dispatchers=1,
                w1=weights[W.moe_w1],
                w2=weights[W.moe_w2],
            )

            return FusedMoe(router, experts)
        elif config.quant_config.get_method() == "FP8_PER_BLOCK":
            return FusedMoeFactory._create_fp8_per_block_fused_moe(config, weights)
        else:
            raise ValueError(
                f"Unsupported quantization method: {config.quant_config.get_method()}"
            )
