"""PureDP router communication with real NCCL, with and without PP."""

from unittest import TestCase, main
from unittest.mock import patch

import torch
import torch.multiprocessing as mp
import torch.nn.functional as F

from rtp_llm.config.model_config import ModelConfig
from rtp_llm.models_py.distributed.collective_torch import (
    destroy_distributed_environment,
    init_distributed_environment,
)
from rtp_llm.models_py.modules.factory.fused_moe.defs.config_adapter import (
    MoEConfigAdapter,
)
from rtp_llm.models_py.modules.factory.fused_moe.defs.fused_moe import (
    CombineForwardPayload,
)
from rtp_llm.models_py.modules.factory.fused_moe.defs.quant_config import (
    FusedMoEQuantConfig,
)
from rtp_llm.models_py.modules.factory.fused_moe.impl.cuda.routers.pure_dp_router import (
    PureDpRouterFp8PerBlock,
)
from rtp_llm.ops import MoeConfig, NcclCommConfig, ParallelismConfig
from rtp_llm.test.utils.port_util import PortsContext


def _inputs(pp_rank, dp_rank, device):
    # Different stages have different data and maxima for the token-count gather.
    num_tokens = 2 + 2 * pp_rank + dp_rank
    x = torch.arange(num_tokens * 4, dtype=torch.float32, device=device).reshape(-1, 4)
    x = x + pp_rank * 100 + dp_rank * 10
    weights = torch.full(
        (num_tokens, 1), 1 + pp_rank * 2 + dp_rank, dtype=torch.float32, device=device
    )
    ids = torch.full(
        (num_tokens, 1), dp_rank * 2 + pp_rank, dtype=torch.int32, device=device
    )
    return x, weights, ids


def _worker(rank, pp_size, nccl_port):
    torch.cuda.set_device(rank)
    device = torch.device("cuda", rank)
    parallel = ParallelismConfig()
    parallel.pp_size = pp_size
    parallel.pp_rank = rank // 2
    parallel.tp_size = 1
    parallel.dp_size = 2
    parallel.dp_rank = rank % 2
    parallel.ep_size = 2
    parallel.ep_rank = rank % 2
    parallel.world_size = pp_size * 2
    parallel.world_rank = rank
    parallel.local_world_size = parallel.world_size
    parallel.local_rank = rank
    nccl_config = NcclCommConfig(
        nccl_ip="127.0.0.1",
        tp_nccl_port=nccl_port + 9,
        dp_tp_nccl_port=nccl_port + 1,
        ffn_tp_nccl_port=nccl_port + 6,
    )
    init_distributed_environment(
        parallel,
        nccl_comm_config=nccl_config,
        nccl_init_port=nccl_port,
        backend="nccl",
        timeout=60,
    )
    try:
        _check_router(parallel, device)
    finally:
        destroy_distributed_environment()


def _check_router(parallel, device):
    model = ModelConfig()
    model.hidden_size = 4
    model.expert_num = 4
    model.moe_k = 1
    moe = MoeConfig()
    moe.use_all_gather = True
    router = PureDpRouterFp8PerBlock(
        MoEConfigAdapter(model, parallel, moe), FusedMoEQuantConfig()
    )
    x, weights, ids = _inputs(parallel.pp_rank, parallel.dp_rank, device)
    # Quantization is tested separately; collectives and expert-id remapping are real.
    with patch.object(router, "_do_quant", side_effect=lambda tensor: (tensor, None)):
        payload = router.prepare(x, None, None, weights, ids)

    max_tokens = 3 + 2 * parallel.pp_rank
    stage_inputs = [_inputs(parallel.pp_rank, dp_rank, device) for dp_rank in range(2)]
    expected_x, expected_weights, expected_ids = (
        torch.cat(
            [
                F.pad(
                    inputs[field],
                    (0, 0, 0, max_tokens - inputs[field].shape[0]),
                    value=pad,
                )
                for inputs in stage_inputs
            ]
        )
        for field, pad in ((0, 0), (1, 0), (2, -1))
    )
    # Check stage membership, order and padding, not just the final roundtrip.
    torch.testing.assert_close(payload.expert_x, expected_x)
    torch.testing.assert_close(payload.expert_topk_weights, expected_weights)
    local_ids = expected_ids - parallel.ep_rank * 2
    local_ids = torch.where((local_ids >= 0) & (local_ids < 2), local_ids, -1)
    torch.testing.assert_close(payload.expert_topk_ids, local_ids)

    # Identity experts contribute only on the rank owning each routed expert.
    partial = (
        payload.expert_x
        * payload.expert_topk_weights
        * (payload.expert_topk_ids >= 0)
    )
    output = router.finalize(
        CombineForwardPayload(fused_expert_output=partial),
        weights,
        ids,
        False,
        {"original_num_tokens": x.shape[0]},
    )
    torch.testing.assert_close(output, x * weights)


class PureDPStageTest(TestCase):
    def test_stage_communication_with_and_without_pp(self):
        for pp_size in (1, 2):
            with self.subTest(pp_size=pp_size), PortsContext() as ports:
                mp.spawn(
                    _worker, args=(pp_size, ports[0]), nprocs=pp_size * 2, join=True
                )


if __name__ == "__main__":
    main()
