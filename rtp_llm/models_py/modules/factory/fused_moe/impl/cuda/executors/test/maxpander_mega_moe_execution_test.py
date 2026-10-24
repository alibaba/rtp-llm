import os
import tempfile
import unittest
from datetime import timedelta
from unittest.mock import PropertyMock, patch

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from rtp_llm.config.maxpander_config import MaxPanderConfig
from rtp_llm.eplb.maxpander_plan import build_circulant_plan
from rtp_llm.models_py.modules.factory.fused_moe.defs.fused_moe import FusedMoe
from rtp_llm.models_py.modules.factory.fused_moe.impl.cuda.executors.test.mega_moe_routed_only_execution_test import (
    _DIM,
    _EXPERTS,
    _MAX_TOKENS,
    _WORLD_SIZE,
    _make_global_weights,
    _make_moe,
    _sync_ranks_for_mega,
)
from rtp_llm.utils.model_weight import W


def _worker(rank, rendezvous_path, strategy):
    os.environ["MODEL_WARM_UP"] = "0"
    os.environ["MEGA_MOE_INPUT_PACKER_IMPL"] = "optimized"
    os.environ["MEGA_MOE_INPUT_PACKER"] = "fused"
    torch.cuda.set_device(rank)
    device = torch.device("cuda", rank)
    dist.init_process_group(
        "nccl",
        init_method=f"file://{rendezvous_path}",
        rank=rank,
        world_size=_WORLD_SIZE,
        device_id=device,
        timeout=timedelta(seconds=180),
    )
    try:
        config = MaxPanderConfig(enable=True).resolve("PREFILL")
        for hashed in (False, True):
            torch.manual_seed(20261016)
            global_weights = _make_global_weights(
                device, include_shared=True, hash_routing=hashed
            )
            reference = _make_moe(
                dict(global_weights),
                ep_size=1,
                ep_rank=0,
                strategy="local_loop",
                n_shared_experts=1,
                hash_routing=hashed,
            )
            for redundant in (0, 4, 8, 16):
                large_tiles = strategy == "mega_moe" and not hashed and redundant == 4
                physical_count = _EXPERTS + redundant
                placement, candidates, replicas = build_circulant_plan(
                    _EXPERTS, physical_count, _WORLD_SIZE
                )
                local_count = physical_count // _WORLD_SIZE
                selected = torch.tensor(
                    placement[rank * local_count : (rank + 1) * local_count],
                    device=device,
                    dtype=torch.long,
                )
                local_weights = dict(global_weights)
                for key in (W.moe_w1, W.moe_s1, W.moe_w2, W.moe_s2):
                    tensor = global_weights[key]
                    local_weights[key] = (
                        tensor.view(torch.uint8)
                        .index_select(0, selected)
                        .view(tensor.dtype)
                    )
                local_weights[W.log2phy] = torch.tensor(
                    candidates, device=device, dtype=torch.int32
                )
                local_weights[W.logic_expert_cnt] = torch.tensor(
                    replicas, device=device, dtype=torch.int32
                )
                layer = _make_moe(
                    local_weights,
                    ep_size=_WORLD_SIZE,
                    ep_rank=rank,
                    strategy=strategy,
                    n_shared_experts=1,
                    hash_routing=hashed,
                    maxpander_config=config,
                    physical_expert_num=physical_count,
                    max_tokens_per_rank=512 if large_tiles else _MAX_TOKENS,
                )
                assert layer.fused_moe.supports_gate_pack
                assert layer.fused_moe.fused_experts.physical_router is not None
                token_count_cases = ((7, 5), (5, 0))
                if large_tiles:
                    token_count_cases += ((256, 256), (512, 512))
                for token_counts in token_count_cases:
                    tokens = token_counts[rank]
                    inputs = torch.randn(
                        (tokens, _DIM), device=device, dtype=torch.bfloat16
                    )
                    input_ids = torch.arange(tokens, device=device)
                    with torch.inference_mode():
                        expected = reference(inputs, input_ids).clone()
                        _sync_ranks_for_mega(device)
                        fused = layer(inputs, input_ids).clone()
                        _sync_ranks_for_mega(device)
                        with patch.object(
                            FusedMoe,
                            "supports_gate_pack",
                            new_callable=PropertyMock,
                            return_value=False,
                        ):
                            ordinary = layer(inputs, input_ids).clone()
                    torch.cuda.synchronize(device)
                    for actual in (fused, ordinary):
                        assert actual.shape == inputs.shape
                        assert torch.isfinite(actual).all()
                        if tokens:
                            relative_error = (
                                actual.float() - expected.float()
                            ).abs().mean() / (expected.float().abs().mean() + 1e-6)
                            assert relative_error.item() < 0.05, (
                                strategy,
                                hashed,
                                redundant,
                                rank,
                                relative_error.item(),
                            )
                    dist.barrier()
                del layer
            del reference
    finally:
        dist.destroy_process_group()


class MaxPanderMegaMoEExecutionTest(unittest.TestCase):
    def test_two_rank_logical_routing_matches_both_packing_paths(self):
        self.assertGreaterEqual(torch.cuda.device_count(), _WORLD_SIZE)
        for strategy in ("mega_moe", "mega_moe_se"):
            with self.subTest(strategy=strategy), tempfile.TemporaryDirectory(
                dir=os.environ.get("TEST_TMPDIR")
            ) as directory:
                mp.spawn(
                    _worker,
                    args=(os.path.join(directory, "rendezvous"), strategy),
                    nprocs=_WORLD_SIZE,
                    join=True,
                )


if __name__ == "__main__":
    unittest.main()
