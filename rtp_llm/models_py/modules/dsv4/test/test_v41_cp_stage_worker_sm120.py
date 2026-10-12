"""Native CEP4PP2 worker ownership, initialization and stage-local agreement."""

import json
import os
import socket
import tempfile
import unittest
from dataclasses import replace
from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from rtp_llm.models_py.distributed import collective_torch
from rtp_llm.models_py.distributed.ep_stage_context import EpStageContext
from rtp_llm.models_py.distributed.rank_layout import Group
from rtp_llm.models_py.model_desc.deepseek_v4_model import Dsv4SharedRuntimeBufferStore
from rtp_llm.models_py.model_desc.deepseek_v41_model import DeepSeekV41Model
from rtp_llm.models_py.model_desc.v41_errors import V41InputRejected
from rtp_llm.models_py.modules.dsv4.fp8.test.test_v41_worker_sm120 import (
    checkpoint,
    model_config,
    resources,
    tagged_cache,
)
from rtp_llm.models_py.modules.dsv4.test.test_v41_stage_slices_sm120 import wide_fixture
from rtp_llm.ops import (
    CPRotateMethod,
    HWKernelConfig,
    NcclCommConfig,
    ParallelismConfig,
    RoleType,
)
from rtp_llm.ops.compute_ops import PyModelInputs
from rtp_llm.utils.model_weight import W


def parallelism_for(rank):
    config = ParallelismConfig()
    config.world_size = config.local_world_size = 8
    config.world_rank = config.local_rank = rank
    config.pp_size, config.dp_size, config.tp_size, config.ep_size = 2, 1, 4, 4
    config.pp_rank, config.tp_rank, config.ep_rank = rank // 4, rank % 4, rank % 4
    config.ffn_tp_size, config.ffn_tp_rank = 4, rank % 4
    config.pp_stage_layer_counts = [21, 19]
    config.role_type = RoleType.PREFILL
    config.prefill_cp_config.method = CPRotateMethod.PREFILL_CP
    config.prefill_cp_config.prefill_cp_size = 4
    config.pp_ep_enabled = True
    config.pp_ep_backend = "fork_nccl_mxfp8"
    config.resolve_local_cp("deepseek_v41", False, False, False)
    assert config.local_cp_enabled()
    return config


def make_worker(rank, config):
    torch.manual_seed(41701)
    args, weights = wide_fixture()
    pp = rank // 4
    owned = range(21) if pp == 0 else range(21, 40)
    globals_owned = {W.embedding} if pp == 0 else {W.final_ln_gamma, W.lm_head}
    shard = SimpleNamespace(
        global_weights={
            key: value
            for key, value in weights.global_weights.items()
            if key in globals_owned
        },
        weights=[
            dict(layer) if i in owned else {} for i, layer in enumerate(weights.weights)
        ],
    )
    for layer in shard.weights:
        if not layer:
            continue
        for projection in ("w1", "w2", "w3"):
            for kind in ("w", "s"):
                key = getattr(W, f"v4_routed_{projection}_{kind}")
                layer[key] = layer[key][rank % 4 : rank % 4 + 1].contiguous()
    pcfg = parallelism_for(rank)
    hw = HWKernelConfig()
    hw.enable_cuda_graph = False
    hw.enable_native_cuda_graph = False
    return DeepSeekV41Model(
        model_config(args, config), pcfg, shard, None, 2, py_hw_kernel_config=hw
    )


def run_rank(rank, port, config):
    torch.cuda.set_device(rank)
    os.environ.update(
        {
            "DSV41_ENGRAM_HOST_BYTES": str(2**20),
            "DSV41_ENGRAM_CACHE_DIR": "/dev/shm",
            "DSV4_PP_EP_ENABLE": "1",
            "DSV4_PP_EP_BACKEND": "fork_nccl_mxfp8",
            "DSV4_MOE_STRATEGY": "auto",
            "MOEDBG": "0",
        }
    )
    dist.init_process_group(
        "nccl",
        init_method=f"tcp://127.0.0.1:{port}",
        rank=rank,
        world_size=8,
        timeout=timedelta(seconds=120),
    )
    model = None
    try:
        collective_torch.init_distributed_environment(
            parallelism_for(rank), NcclCommConfig(nccl_ip="127.0.0.1"), port
        )
        model = make_worker(rank, config)
        assert model._pp_ep_grouped_kernel_available()
        with patch("importlib.metadata.version", return_value="2.5.0+8a4dfba"):
            assert not model._pp_ep_grouped_kernel_available()
        import deep_gemm

        with patch.object(deep_gemm, "m_grouped_fp8_fp4_gemm_nt_masked", None):
            assert not model._pp_ep_grouped_kernel_available()
        with patch.dict(
            os.environ, {"DSV41_ENGRAM_HOST_BYTES": "0" if rank == 0 else str(2**20)}
        ):
            if rank < 4:
                try:
                    model.initialize(resources())
                except (ValueError, RuntimeError):
                    pass
                else:
                    raise AssertionError("stage-zero peer host rejection was ignored")
                assert model.v4 is None and hasattr(model, "weight")
                assert not model._engram_tables
            else:
                assert model.initialize(resources())
                assert not model._engram_tables and model._engram_hash_state is None
        assert model.initialize(resources())
        context = model._v4_args.moe_stage_context
        assert isinstance(context, EpStageContext)
        assert context == model._stage_agreement
        assert model._dp_group is collective_torch._get_group(Group.STAGE)
        assert tuple(context.world_ranks) == tuple(
            range(rank // 4 * 4, rank // 4 * 4 + 4)
        )
        assert [layer.layer_id for layer in model.v4.layers] == model.pp_layer_ids()
        assert [layer.layer_id for layer, _ in model._engram_layers] == (
            [1, 14] if rank < 4 else []
        )
        assert not hasattr(model, "weight")
        for layer in model.v4.layers:
            assert layer.ffn._strategy.cfg.stage_context == context
        assert model.initialize(resources(tagged_cache(model)))
        device = model._weight_device()
        assert device == torch.device("cuda", rank)
        if rank >= 4:
            assert model.v4.embed is None
            assert device == model.v4.head_weight.device
            assert not list(model.v4.parameters())
        # Stage phases deliberately differ, as do peer-local EP/PP ranks.
        model._agree_dp(None, ("stage-independent", rank // 4), device)
        # A recoverable dispatch failure must reject this whole stage only.
        error = ValueError("fixture-peer-rejection") if rank == 0 else None
        if rank < 4:
            try:
                model._agree_dp(error, "native-preflight", device, native_dispatch=True)
            except V41InputRejected:
                pass
            else:
                raise AssertionError("peer rejection was ignored")
        else:
            model._agree_dp(None, "other-stage-ready", device, native_dispatch=True)
        assert model._worker_state == "ready"
        # Differing model options reject collectively, without consuming sources.
        with patch.object(
            model._v4_args,
            "norm_eps",
            0.1 if rank % 4 == 0 else model._v4_args.norm_eps,
        ):
            try:
                model._agree_dp(None, "mismatched-signature", device)
            except RuntimeError:
                pass
            else:
                raise AssertionError("peer signature mismatch was ignored")
        model._agree_dp(None, "retry-after-signature", device)
        for field, bad in (("ep_rank", 8), ("gen_num_per_cycle", 1)):
            with patch.object(model._v4_args, field, bad):
                try:
                    model._configure_dp_group()
                except (ValueError, NotImplementedError):
                    pass
                else:
                    raise AssertionError("unsupported stage topology was admitted")
        for bad in (
            replace(context, process_group=dist.group.WORLD),
            replace(context, generation=context.generation + 1),
            replace(context, group_rank=(context.group_rank + 1) % 4),
        ):
            with patch.object(model._v4_args, "moe_stage_context", bad):
                try:
                    model._agreement_config()
                except RuntimeError:
                    pass
                else:
                    raise AssertionError("changed expert agreement was admitted")
        # DP1 resolves STAGE through its registered TP alias, not Group.STAGE.
        replaced = {
            key: dist.group.WORLD
            for key, value in collective_torch._group_map.items()
            if value is context.process_group
        }
        assert replaced
        with patch.dict(collective_torch._group_map, replaced):
            assert collective_torch._get_group(Group.STAGE) is dist.group.WORLD
            try:
                model._agreement_config()
            except RuntimeError:
                pass
            else:
                raise AssertionError("changed registered group was admitted")
        try:
            model.forward(PyModelInputs())
        except NotImplementedError as error:
            assert "native frame transport" in str(error)
        else:
            raise AssertionError("unqualified PP worker forward was admitted")
        assert model._worker_state == "ready"
        model.close()
        assert not model._engram_tables
        model.close()
        print(
            "DSV41_STAGE_WORKER "
            + json.dumps(
                {
                    "rank": rank,
                    "stage": rank // 4,
                    "owned_layers": len(model.v4.layers),
                    "initialization": True,
                    "stage_agreement": True,
                    "peer_retry": True,
                    "engram_ownership": True,
                    "forward_gate": True,
                    "cleanup": True,
                }
            ),
            flush=True,
        )
        dist.barrier()
    finally:
        if model is not None:
            model.close()
        Dsv4SharedRuntimeBufferStore._reset_for_test()
        collective_torch.destroy_distributed_environment()


class V41CPStageWorkerSM120Test(unittest.TestCase):
    def test_eight_workers_initialize_and_agree_in_owned_stage_only(self):
        self.assertGreaterEqual(torch.cuda.device_count(), 8)
        for rank in range(8):
            self.assertEqual(torch.cuda.get_device_capability(rank), (12, 0))
        with tempfile.TemporaryDirectory() as root, socket.socket() as sock:
            torch.manual_seed(41700)
            config = checkpoint(Path(root), full_schedule=True, hidden_size=5120)
            sock.bind(("127.0.0.1", 0))
            port = sock.getsockname()[1]
            sock.close()
            mp.spawn(run_rank, args=(port, config), nprocs=8, join=True)


if __name__ == "__main__":
    unittest.main()
