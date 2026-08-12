"""ROCm tests for GenericMoe unified TP all-reduce wiring."""

import multiprocessing as mp
import os
from unittest import TestCase, main, skipUnless
from unittest.mock import patch

import torch
import torch.distributed as dist
from torch import nn

from rtp_llm.config.model_config import ModelConfig
from rtp_llm.models_py.distributed import collective_torch
from rtp_llm.models_py.distributed.collective_torch import (
    Group,
    destroy_distributed_environment,
    init_distributed_environment,
)
from rtp_llm.models_py.model_desc import generic_moe as generic_moe_module
from rtp_llm.models_py.model_desc.generic_moe import GenericMoeLayer
from rtp_llm.models_py.modules.factory.fused_moe.impl.rocm.routers import (
    pure_tp_router as pure_tp_router_module,
)
from rtp_llm.models_py.modules.hybrid import dense_mlp as dense_mlp_module
from rtp_llm.ops import ActivationType, MoeConfig, NcclCommConfig, ParallelismConfig
from rtp_llm.test.utils.port_util import PortManager
from rtp_llm.utils.model_weight import W


class _FixedSelectTopk(nn.Module):
    def __init__(self, expert_num):
        super().__init__()
        self.expert_num = expert_num

    def forward(self, logits, topk_ids, topk_weights):
        token_ids = torch.arange(topk_ids.size(0), device=topk_ids.device)
        for topk_idx in range(topk_ids.size(1)):
            topk_ids[:, topk_idx] = (token_ids + topk_idx * 7) % self.expert_num
        topk_weights.fill_(1.0 / topk_ids.size(1))


class _ZeroRouterGate(nn.Module):
    def __init__(self, expert_num):
        super().__init__()
        self.expert_num = expert_num

    def forward(self, hidden_states):
        return torch.zeros(
            hidden_states.size(0),
            self.expert_num,
            dtype=torch.float32,
            device=hidden_states.device,
        )


def _make_real_parallelism(rank, world_size):
    parallelism_config = ParallelismConfig()
    parallelism_config.tp_size = world_size
    parallelism_config.tp_rank = rank
    parallelism_config.ffn_tp_size = world_size
    parallelism_config.ffn_tp_rank = rank
    parallelism_config.ep_size = 1
    parallelism_config.ep_rank = 0
    parallelism_config.dp_size = 1
    parallelism_config.dp_rank = 0
    parallelism_config.world_size = world_size
    parallelism_config.world_rank = rank
    parallelism_config.local_rank = rank
    parallelism_config.local_world_size = world_size
    return parallelism_config


def _build_real_layer(rank, parallelism_config, with_gate):
    device = torch.device(f"cuda:{rank}")
    hidden_size = 512
    inter_size = 1024
    expert_num = 32

    config = ModelConfig()
    config.hidden_size = hidden_size
    config.inter_size = inter_size
    config.expert_num = expert_num
    config.moe_k = 4
    config.activation_type = ActivationType.Swiglu
    config.moe_style = 2
    config.quant_config = None

    # Give each rank a different routed-expert contribution so the test checks
    # an actual cross-rank sum instead of two identical local tensors.
    torch.manual_seed(1000 + rank)
    weights = {
        W.moe_gate: torch.empty(1, device=device),
        W.moe_w1: torch.randn(
            expert_num, 2 * inter_size, hidden_size, device=device, dtype=torch.bfloat16
        )
        * 0.02,
        W.moe_w2: torch.randn(
            expert_num, hidden_size, inter_size, device=device, dtype=torch.bfloat16
        )
        * 0.02,
        W.ffn_w13: torch.empty(hidden_size, 2 * inter_size, device=device),
        W.ffn_w2: torch.empty(inter_size, hidden_size, device=device),
    }
    if with_gate:
        weights[W.shared_expert_gate] = torch.empty(1, device=device)

    moe_config = MoeConfig()
    moe_config.use_all_gather = True
    moe_config.fake_balance_expert = False

    def make_linear(_weights, weight_key, *args, **kwargs):
        if weight_key == W.moe_gate:
            return _ZeroRouterGate(expert_num)
        if weight_key == W.ffn_w13:
            torch.manual_seed(3000 + rank)
            return nn.Linear(
                hidden_size,
                2 * inter_size,
                bias=False,
                device=device,
                dtype=torch.bfloat16,
            )
        if weight_key == W.ffn_w2:
            torch.manual_seed(4000 + rank)
            return nn.Linear(
                inter_size,
                hidden_size,
                bias=False,
                device=device,
                dtype=torch.bfloat16,
            )
        if weight_key == W.shared_expert_gate:
            torch.manual_seed(5000)
            return nn.Linear(
                hidden_size,
                1,
                bias=False,
                device=device,
                dtype=torch.bfloat16,
            )
        raise AssertionError(f"unexpected test linear weight key: {weight_key}")

    with patch(
        "rtp_llm.models_py.model_desc.generic_moe.LinearFactory.create_linear_from_weights",
        side_effect=make_linear,
    ):
        layer = GenericMoeLayer(
            config,
            parallelism_config,
            weights,
            moe_config,
        )

    if type(layer.fused_moe.router) is not pure_tp_router_module.PureTpRouterNoQuant:
        raise AssertionError(
            "real TP integration test selected "
            f"{type(layer.fused_moe.router).__name__}, expected "
            "the ROCm PureTpRouterNoQuant class"
        )
    if not layer.use_unified_tp_allreduce:
        raise AssertionError("real pure-TP layer did not enable unified all-reduce")
    layer.select_topk = _FixedSelectTopk(expert_num)
    return layer


def _run_real_two_gpu_case(rank, world_size, ports, with_gate):
    parallelism_config = _make_real_parallelism(rank, world_size)
    initialized = False
    try:
        torch.cuda.set_device(rank)
        init_distributed_environment(
            parallelism_config,
            NcclCommConfig(
                nccl_ip="127.0.0.1",
                tp_nccl_port=ports[1],
                dp_tp_nccl_port=ports[2],
                ffn_tp_nccl_port=ports[3],
            ),
            nccl_init_port=ports[0],
            backend="nccl",
            timeout=300,
        )
        initialized = True
        layer = _build_real_layer(rank, parallelism_config, with_gate)
        # TP ranks consume the same replicated hidden states.  Keep the routed
        # expert weights rank-specific so the collective still sums distinct
        # partial outputs, while the shared-expert gate remains rank-consistent.
        torch.manual_seed(2024)
        hidden_states = torch.randn(
            32, 512, device=torch.device(f"cuda:{rank}"), dtype=torch.bfloat16
        )
        dist.broadcast(hidden_states, src=0)
        if with_gate:
            gate_local = layer.shared_expert_gate(hidden_states)
            gate_rank0 = gate_local.clone()
            dist.broadcast(gate_rank0, src=0)
            torch.testing.assert_close(gate_local, gate_rank0, rtol=0.0, atol=0.0)
            dist.barrier()
        counts = {"all_reduce": 0}

        def counted_all_reduce(tensor, group):
            if group is not Group.TP:
                raise AssertionError(f"unexpected collective group: {group}")
            counts["all_reduce"] += 1
            return collective_torch.all_reduce(tensor, group)

        with (
            patch.object(
                generic_moe_module, "all_reduce", side_effect=counted_all_reduce
            ),
            patch.object(
                dense_mlp_module, "all_reduce", side_effect=counted_all_reduce
            ),
            patch.object(
                pure_tp_router_module, "all_reduce", side_effect=counted_all_reduce
            ),
        ):
            layer.use_unified_tp_allreduce = True
            unified_output = layer(hidden_states.clone()).detach().clone()
            unified_calls = counts["all_reduce"]
            dist.barrier()

            layer.use_unified_tp_allreduce = False
            legacy_output = layer(hidden_states.clone()).detach().clone()
            legacy_calls = counts["all_reduce"] - unified_calls
            dist.barrier()

        if unified_calls != 1 or legacy_calls != 2:
            raise AssertionError(
                f"unexpected TP collective counts: unified={unified_calls}, "
                f"legacy={legacy_calls}"
            )
        torch.testing.assert_close(
            unified_output,
            legacy_output,
            rtol=2e-2,
            atol=2e-3,
        )
        if unified_output is legacy_output:
            raise AssertionError("unified and legacy outputs unexpectedly alias")
        if rank == 0:
            print(
                f"[real_tp_unified] gate={with_gate} unified_calls={unified_calls} "
                f"legacy_calls={legacy_calls} ✓"
            )
    finally:
        if initialized:
            destroy_distributed_environment()


def _launch_real_two_gpu_case(with_gate):
    world_size = 2
    ports, port_locks = PortManager().get_consecutive_ports(4)
    context = mp.get_context("spawn")
    processes = [
        context.Process(
            target=_run_real_two_gpu_case,
            args=(rank, world_size, ports, with_gate),
            name=f"generic-moe-rank-{rank}",
        )
        for rank in range(world_size)
    ]
    try:
        for process in processes:
            process.start()
        for process in processes:
            process.join(timeout=300)
        failed = [process for process in processes if process.exitcode != 0]
        if failed:
            for process in processes:
                if process.is_alive():
                    process.terminate()
            raise RuntimeError(
                "real GenericMoe two-GPU worker failed: "
                + ", ".join(
                    f"{process.name} exitcode={process.exitcode}" for process in failed
                )
            )
    finally:
        for lock in port_locks:
            lock.__exit__(None, None, None)


def _configured_gpu_count():
    """Read CI GPU allocation without initializing CUDA/HIP in the parent."""

    if (gpu_count := os.environ.get("GPU_COUNT")) is not None:
        return int(gpu_count)
    visible_devices = os.environ.get("HIP_VISIBLE_DEVICES")
    if visible_devices is None:
        return 0
    return len([device for device in visible_devices.split(",") if device.strip()])


@skipUnless(
    torch.version.hip is not None,
    "requires a ROCm build of PyTorch",
)
class GenericMoeRealAllreduceTest(TestCase):
    def test_two_gpu_real_layer_matches_legacy_for_gated_and_ungated(self):
        if _configured_gpu_count() < 2:
            self.fail(
                "test target must allocate at least two GPUs; "
                f"GPU_COUNT={os.environ.get('GPU_COUNT')!r}"
            )
        for with_gate in (False, True):
            with self.subTest(with_gate=with_gate):
                _launch_real_two_gpu_case(with_gate)


if __name__ == "__main__":
    main()
