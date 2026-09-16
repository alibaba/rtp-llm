"""Manual performance guard for the full MegaMoE gate-to-expert chain."""

from __future__ import annotations

import os
import tempfile
import unittest
from datetime import timedelta
from statistics import median

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from rtp_llm.models_py.modules.factory.fused_moe.defs.fused_moe import FusedMoe
from rtp_llm.models_py.modules.factory.fused_moe.impl.cuda.executors.test.mega_moe_routed_only_execution_test import (
    _DIM,
    _EXPERTS,
    _MAX_TOKENS,
    _TOPK,
    _WORLD_SIZE,
    _make_global_weights,
    _make_moe,
)
from rtp_llm.utils.model_weight import W


class _GatePackDisabledFusedMoe(FusedMoe):
    """Production FusedMoe path with only gate-pack capability disabled."""

    @property
    def supports_gate_pack(self) -> bool:
        return False


def _disable_gate_pack(layer) -> None:
    fused_moe = layer.fused_moe
    layer.fused_moe = _GatePackDisabledFusedMoe(
        fused_moe.router,
        fused_moe.fused_experts,
        fused_moe.expert_num,
        fused_moe.strategy_name,
    )


def _local_mega_weights(
    global_weights: dict,
    rank: int,
    world_size: int,
    *,
    include_shared: bool,
    hash_routing: bool,
) -> dict:
    local_start = rank * (_EXPERTS // world_size)
    local_end = local_start + (_EXPERTS // world_size)
    weights = {
        W.moe_gate: global_weights[W.moe_gate],
        W.moe_gate_bias: global_weights[W.moe_gate_bias],
    }
    if hash_routing:
        weights[W.moe_gate_tid2eid] = global_weights[W.moe_gate_tid2eid]
    for key in (
        W.moe_w1,
        W.moe_s1,
        W.moe_w2,
        W.moe_s2,
    ):
        weights[key] = global_weights[key][local_start:local_end].clone()
    if include_shared:
        for key in (
            W.ffn_w13,
            W.ffn_s13,
            W.ffn_w2,
            W.ffn_s2,
        ):
            weights[key] = global_weights[key].clone()
    return weights


def _time_batch(fn, iters: int, device: torch.device) -> float:
    dist.barrier()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    end.synchronize()
    dist.barrier()
    return start.elapsed_time(end) / iters


def _bench_pair(fused, separated, device: torch.device) -> tuple[float, float, float]:
    for _ in range(4):
        fused()
        separated()
    torch.cuda.synchronize(device)
    fused_samples = []
    separated_samples = []
    ratios = []
    for round_index in range(5):
        if round_index % 2 == 0:
            fused_ms = _time_batch(fused, 20, device)
            separated_ms = _time_batch(separated, 20, device)
        else:
            separated_ms = _time_batch(separated, 20, device)
            fused_ms = _time_batch(fused, 20, device)
        fused_samples.append(fused_ms)
        separated_samples.append(separated_ms)
        ratios.append(fused_ms / separated_ms)
    return median(fused_samples), median(separated_samples), median(ratios)


def _perf_worker(rank: int, world_size: int, rendezvous_path: str) -> None:
    os.environ["MODEL_WARM_UP"] = "0"
    os.environ["MEGA_MOE_INPUT_PACKER_IMPL"] = "optimized"
    torch.cuda.set_device(rank)
    device = torch.device("cuda", rank)
    dist.init_process_group(
        backend="nccl",
        init_method=f"file://{rendezvous_path}",
        rank=rank,
        world_size=world_size,
        device_id=device,
        timeout=timedelta(seconds=180),
    )
    try:
        for hash_routing, include_shared in (
            (False, False),
            (True, False),
            (False, True),
        ):
            torch.manual_seed(20260905)
            global_weights = _make_global_weights(
                device,
                include_shared=include_shared,
                hash_routing=hash_routing,
            )
            fused_layer = _make_moe(
                _local_mega_weights(
                    global_weights,
                    rank,
                    world_size,
                    include_shared=include_shared,
                    hash_routing=hash_routing,
                ),
                ep_size=world_size,
                ep_rank=rank,
                strategy="mega_moe",
                n_shared_experts=int(include_shared),
                hash_routing=hash_routing,
            )._moe
            separated_layer = _make_moe(
                _local_mega_weights(
                    global_weights,
                    rank,
                    world_size,
                    include_shared=include_shared,
                    hash_routing=hash_routing,
                ),
                ep_size=world_size,
                ep_rank=rank,
                strategy="mega_moe",
                n_shared_experts=int(include_shared),
                hash_routing=hash_routing,
            )._moe
            _disable_gate_pack(separated_layer)
            del global_weights
            for tokens in (1, _MAX_TOKENS):
                torch.manual_seed(20260950 + tokens + rank)
                x = torch.randn(tokens, _DIM, dtype=torch.bfloat16, device=device)
                input_ids = torch.arange(tokens, dtype=torch.long, device=device)

                def fused():
                    return fused_layer(x, input_ids)

                def separated():
                    return separated_layer(x, input_ids)

                with torch.inference_mode():
                    fused_output = fused().clone()
                    separated_output = separated().clone()
                    error = (fused_output.float() - separated_output.float()).abs()
                    baseline = separated_output.float().abs().mean().item() + 1e-6
                    assert error.mean().item() / baseline < 0.05
                    fused_ms, separated_ms, paired_ratio = _bench_pair(
                        fused, separated, device
                    )
                worst_ratio = torch.tensor(paired_ratio, device=device)
                dist.all_reduce(worst_ratio, op=dist.ReduceOp.MAX)
                if rank == 0:
                    print(
                        "[MegaMoE end-to-end] "
                        f"tokens={tokens} hash={hash_routing} "
                        f"shared={include_shared} fused={fused_ms:.3f}ms "
                        f"separated={separated_ms:.3f}ms "
                        f"worst_paired_ratio={worst_ratio.item():.3f}"
                    )
                assert worst_ratio.item() <= 1.10, (
                    "MegaMoE gate-pack chain regressed by more than 10%: "
                    f"tokens={tokens} hash={hash_routing} shared={include_shared} "
                    f"ratio={worst_ratio.item():.3f}"
                )
            del fused_layer, separated_layer
            torch.cuda.empty_cache()
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


class MegaMoeEndToEndPerfTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        super().setUpClass()
        if not torch.cuda.is_available():
            raise AssertionError("CUDA is required by this dedicated target")
        if torch.cuda.device_count() < _WORLD_SIZE:
            raise AssertionError(
                f"{_WORLD_SIZE} GPUs are required by this dedicated target"
            )

    def test_gate_pack_chain_vs_separated_path(self) -> None:
        test_tmpdir = os.environ.get("TEST_TMPDIR")
        with tempfile.TemporaryDirectory(dir=test_tmpdir) as tmpdir:
            rendezvous_path = os.path.join(tmpdir, "torch_dist_rendezvous")
            mp.spawn(
                _perf_worker,
                args=(_WORLD_SIZE, rendezvous_path),
                nprocs=_WORLD_SIZE,
                join=True,
            )


if __name__ == "__main__":
    unittest.main()
