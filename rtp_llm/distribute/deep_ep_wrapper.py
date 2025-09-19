from enum import Enum

import deep_ep
import torch
import torch.distributed as dist

from rtp_llm.config.gpt_init_model_parameters import GptInitModelParameters
from rtp_llm.distribute.worker_info import g_master_info, g_parallel_info


class DeepBufferWrapper(object):
    deep_ep_buffer: deep_ep.Buffer

    def __init__(self, config: GptInitModelParameters):
        self.config = config
        # TODO@miji not suitable for binding dist group init with deepep buffer
        if not dist.is_initialized():
            print(
                f"-----------------rank {g_parallel_info.world_rank} init dist: tcp://{g_master_info.ip}:{g_master_info.th_nccl_port}"
            )
            dist.init_process_group(
                backend="nccl",
                init_method=f"tcp://{g_master_info.ip}:{g_master_info.th_nccl_port}",
                world_size=g_parallel_info.world_size,
                rank=g_parallel_info.world_rank,
            )
        if config.moe_config.use_deepep_low_latency:
            raise NotImplementedError("Low latency mode is not supported now")
        # intranode mode
        if g_parallel_info.world_size == g_parallel_info.local_world_size:
            num_rdma_bytes = 0
            num_qps_per_rank = 1
            num_nvl_bytes = int(2e9)
            self.deep_ep_buffer = deep_ep.Buffer(
                dist.GroupMember.WORLD,
                num_nvl_bytes=num_nvl_bytes,
                num_rdma_bytes=num_rdma_bytes,
                low_latency_mode=False,
                num_qps_per_rank=num_qps_per_rank,
            )
        else:
            raise NotImplementedError("Only support intranode mode now")

    def __del__(self):
        if dist.is_initialized():
            dist.destroy_process_group()

    @property
    def buffer(self):
        return self.deep_ep_buffer
