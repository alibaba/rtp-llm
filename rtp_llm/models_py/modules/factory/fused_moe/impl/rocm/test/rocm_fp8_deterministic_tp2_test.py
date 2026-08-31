"""Real two-GPU coverage for deterministic ROCm FP8-per-channel MoE."""

import multiprocessing as mp
import os
import socket
import unittest
from datetime import timedelta
from unittest.mock import patch

import torch
import torch.distributed as dist

from rtp_llm.config.model_config import ModelConfig
from rtp_llm.device.device_impl import RocmImpl
from rtp_llm.models_py.kernel_tuning import ROCM_FP8_MOE_DETERMINISTIC_REDUCE_ENV
from rtp_llm.models_py.modules.factory.fused_moe.defs.config_adapter import (
    MoEConfigAdapter,
)
from rtp_llm.models_py.modules.factory.fused_moe.defs.fused_moe import (
    ExpertForwardPayload,
    ExpertTokensMetadata,
)
from rtp_llm.models_py.modules.factory.fused_moe.defs.quant_config import (
    FusedMoEQuantConfig,
)
from rtp_llm.models_py.modules.factory.fused_moe.impl.rocm.executors import (
    deterministic_fp8_moe,
)
from rtp_llm.models_py.modules.factory.fused_moe.impl.rocm.executors import (
    rocm_moe as rocm_moe_module,
)
from rtp_llm.models_py.modules.factory.fused_moe.impl.rocm.executors.rocm_moe import (
    RocmExpertsFp8PerChannel,
)
from rtp_llm.ops import MoeConfig, ParallelismConfig
from rtp_llm.utils.model_weight import W

_WORLD_SIZE = 2
_HIDDEN_SIZE = 2048
_LOCAL_INTER_SIZE = 256
_EXPERTS = 256
_TOPK = 8
_TOKENS = 4
_REPEATS = 10
_FP8_MAX = 240.0


def _configured_gpu_count() -> int:
    if (gpu_count := os.environ.get("GPU_COUNT")) is not None:
        return int(gpu_count)
    visible_devices = os.environ.get("HIP_VISIBLE_DEVICES")
    if visible_devices is None:
        return 0
    return len([device for device in visible_devices.split(",") if device.strip()])


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def _quantize_per_channel(
    shape: tuple[int, ...], seed: int, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor]:
    torch.manual_seed(seed)
    source = torch.randn(shape, dtype=torch.bfloat16, device=device).mul_(0.02)
    scale = source.abs().amax(dim=-1, keepdim=True).clamp_(min=1e-8)
    scale = scale.div_(_FP8_MAX).to(torch.float32)
    quantized = (source / scale).to(torch.float8_e4m3fnuz)
    quantized_bits = quantized.view(torch.int8)
    quantized_bits[quantized_bits == -128] = 0
    return quantized_bits.view(torch.float8_e4m3fnuz), scale.mul_(2.0)


def _parallelism_config(rank: int) -> ParallelismConfig:
    config = ParallelismConfig()
    config.tp_size = _WORLD_SIZE
    config.tp_rank = rank
    config.ffn_tp_size = _WORLD_SIZE
    config.ffn_tp_rank = rank
    config.ep_size = 1
    config.ep_rank = 0
    config.dp_size = 1
    config.dp_rank = 0
    config.world_size = _WORLD_SIZE
    config.world_rank = rank
    config.local_rank = rank
    config.local_world_size = _WORLD_SIZE
    return config


def _build_executor(rank: int, device: torch.device) -> RocmExpertsFp8PerChannel:
    runtime_device = RocmImpl.__new__(RocmImpl)
    w1_quantized, w1_scale = _quantize_per_channel(
        (_EXPERTS, 2 * _LOCAL_INTER_SIZE, _HIDDEN_SIZE), 1000 + rank, device
    )
    w2_quantized, w2_scale = _quantize_per_channel(
        (_EXPERTS, _HIDDEN_SIZE, _LOCAL_INTER_SIZE), 2000 + rank, device
    )
    w1_quantized = runtime_device.cat_0(
        [
            w1_quantized[:, _LOCAL_INTER_SIZE:, :],
            w1_quantized[:, :_LOCAL_INTER_SIZE, :],
        ],
        dim=1,
    )
    w1_scale = torch.cat(
        [
            w1_scale[:, _LOCAL_INTER_SIZE:, :],
            w1_scale[:, :_LOCAL_INTER_SIZE, :],
        ],
        dim=1,
    )
    weights = {
        W.moe_w1: runtime_device.shuffle_moe_weight(
            w1_quantized, w1_quantized.dtype, W.moe_w1
        ),
        W.moe_w2: runtime_device.shuffle_moe_weight(
            w2_quantized, w2_quantized.dtype, W.moe_w2
        ),
        W.moe_s1: runtime_device.shuffle_moe_weight(w1_scale, w1_scale.dtype, W.moe_s1),
        W.moe_s2: w2_scale,
    }

    model_config = ModelConfig()
    model_config.attn_config.head_num = 4
    model_config.attn_config.size_per_head = 64
    model_config.num_layers = 2
    model_config.max_seq_len = 2048
    model_config.vocab_size = 32000
    model_config.expert_num = _EXPERTS
    model_config.moe_k = _TOPK
    model_config.inter_size = _LOCAL_INTER_SIZE * _WORLD_SIZE
    model_config.activation_type = "SiGLU"
    model_config.data_type = "bf16"
    config = MoEConfigAdapter(
        model_config=model_config,
        parallelism_config=_parallelism_config(rank),
        moe_config=MoeConfig(),
    )
    return RocmExpertsFp8PerChannel(config, FusedMoEQuantConfig(), weights)


def _payload(device: torch.device) -> ExpertForwardPayload:
    torch.manual_seed(2026)
    hidden_states = torch.randn(
        _TOKENS, _HIDDEN_SIZE, dtype=torch.bfloat16, device=device
    )
    topk_ids = torch.arange(_TOPK, dtype=torch.int32, device=device).repeat(_TOKENS, 1)
    topk_weights = torch.softmax(
        torch.randn(_TOKENS, _TOPK, dtype=torch.float32, device=device), dim=-1
    )
    dist.broadcast(hidden_states, src=0)
    dist.broadcast(topk_ids, src=0)
    dist.broadcast(topk_weights, src=0)
    return ExpertForwardPayload(
        expert_x=hidden_states,
        expert_x_origin_dtype=hidden_states.dtype,
        expert_x_scale=None,
        expert_tokens_meta=ExpertTokensMetadata(None, None, None),
        expert_topk_ids=topk_ids,
        expert_topk_weights=topk_weights,
    )


def _execute(executor, payload) -> torch.Tensor:
    return executor.execute(
        payload=payload,
        activation="SiGLU",
        expert_map=None,
        a2_scale=None,
        apply_router_weight_on_input=False,
        extra_expert_args=None,
    ).fused_expert_output


def _tp2_worker(rank: int, port: int) -> None:
    initialized = False
    try:
        os.environ[ROCM_FP8_MOE_DETERMINISTIC_REDUCE_ENV] = "1"
        torch.cuda.set_device(rank)
        device = torch.device(f"cuda:{rank}")
        dist.init_process_group(
            "nccl",
            init_method=f"tcp://127.0.0.1:{port}",
            rank=rank,
            world_size=_WORLD_SIZE,
            timeout=timedelta(seconds=300),
            device_id=device,
        )
        initialized = True
        executor = _build_executor(rank, device)
        payload = _payload(device)

        route_local_calls = 0
        route_local_stage2 = deterministic_fp8_moe._route_local_stage2

        def observed_route_local_stage2(*args, **kwargs):
            nonlocal route_local_calls
            route_local_calls += 1
            return route_local_stage2(*args, **kwargs)

        def reject_legacy_fallback(*args, **kwargs):
            raise AssertionError("deterministic TP2 test unexpectedly used fused_moe")

        deterministic_fp8_moe._make_metadata_transform.cache_clear()
        deterministic_fp8_moe._runtime_unsupported_reason.cache_clear()
        with (
            patch.object(
                deterministic_fp8_moe,
                "_route_local_stage2",
                side_effect=observed_route_local_stage2,
            ),
            patch.object(
                rocm_moe_module, "fused_moe", side_effect=reject_legacy_fallback
            ),
        ):
            baseline = _execute(executor, payload)
            if route_local_calls == 0:
                raise AssertionError(
                    "route-local deterministic stage2 was not selected"
                )
            dist.all_reduce(baseline)
            torch.cuda.synchronize(device)
            baseline = baseline.clone()

            for _ in range(_REPEATS):
                eager_output = _execute(executor, payload)
                dist.all_reduce(eager_output)
                torch.cuda.synchronize(device)
                if not torch.equal(eager_output, baseline):
                    raise AssertionError(
                        "TP2 eager output changed across identical runs"
                    )

            capture_stream = torch.cuda.Stream(device=device)
            capture_stream.wait_stream(torch.cuda.current_stream(device))
            with torch.cuda.stream(capture_stream):
                _execute(executor, payload)
            capture_stream.synchronize()
            dist.barrier()

            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, stream=capture_stream):
                graph_local_output = _execute(executor, payload)
            capture_stream.synchronize()
            dist.barrier()

            for _ in range(_REPEATS):
                graph.replay()
                torch.cuda.synchronize(device)
                graph_output = graph_local_output.clone()
                dist.all_reduce(graph_output)
                torch.cuda.synchronize(device)
                if not torch.equal(graph_output, baseline):
                    raise AssertionError(
                        "TP2 HIP graph output changed across identical replays"
                    )
    finally:
        if initialized:
            dist.destroy_process_group()


def _launch_tp2() -> None:
    context = mp.get_context("spawn")
    port = _free_port()
    processes = [
        context.Process(target=_tp2_worker, args=(rank, port), name=f"tp2-rank-{rank}")
        for rank in range(_WORLD_SIZE)
    ]
    try:
        for process in processes:
            process.start()
        for process in processes:
            process.join(timeout=600)
        failed = [process for process in processes if process.exitcode != 0]
        if failed:
            raise RuntimeError(
                "deterministic ROCm FP8 MoE TP2 worker failed: "
                + ", ".join(
                    f"{process.name} exitcode={process.exitcode}" for process in failed
                )
            )
    finally:
        for process in processes:
            if process.is_alive():
                process.terminate()
                process.join(timeout=5)


@unittest.skipUnless(torch.version.hip is not None, "requires ROCm PyTorch")
class DeterministicFp8MoeTp2Test(unittest.TestCase):
    def test_real_tp2_eager_and_hip_graph_are_bitwise_stable(self):
        self.assertGreaterEqual(
            _configured_gpu_count(),
            _WORLD_SIZE,
            "test target must allocate two MI308X GPUs",
        )
        _launch_tp2()


if __name__ == "__main__":
    unittest.main()
