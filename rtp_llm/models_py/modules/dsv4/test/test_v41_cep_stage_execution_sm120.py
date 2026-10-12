"""Maintained CP4/EP4/Engram owner composition before PP worker transport."""

import json
import os
import socket
import tempfile
import unittest
from contextlib import contextmanager
from dataclasses import replace
from datetime import timedelta
from pathlib import Path
from unittest.mock import patch

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from rtp_llm.models_py.distributed import collective_torch
from rtp_llm.models_py.distributed.ep_stage_context import EpStageContext
from rtp_llm.models_py.distributed.rank_layout import Group
from rtp_llm.models_py.model_desc.deepseek_v4_model import Dsv4SharedRuntimeBufferStore
from rtp_llm.models_py.model_desc.deepseek_v41_model import DeepSeekV41Model
from rtp_llm.models_py.modules.dsv4.engram import make_token_windows
from rtp_llm.models_py.modules.dsv4.fp8.test.test_v41_cp_owner_sm120 import zigzag
from rtp_llm.models_py.modules.dsv4.fp8.test.test_v41_worker_sm120 import (
    checkpoint,
    model_config,
    resources,
)
from rtp_llm.models_py.modules.dsv4.test.test_v41_cp_stage_slices_sm120 import (
    run_controls,
)
from rtp_llm.models_py.modules.dsv4.test.test_v41_cp_stage_worker_sm120 import (
    parallelism_for,
)
from rtp_llm.models_py.modules.dsv4.test.test_v41_stage_slices_sm120 import wide_fixture
from rtp_llm.ops import NcclCommConfig, ParallelismConfig, RoleType
from rtp_llm.utils.model_weight import W

ROUNDS = ([1, 1], [6, 4], [122, 124])


class EngramControls:
    def __init__(self, reference, rank, context):
        self.reference = reference
        self.rank = rank
        self.context = context
        self.owners = []
        self.calls = 0
        self.modules = {
            layer.layer_id: (index, layer.engram)
            for layer, index in reference._engram_layers
        }
        assert set(self.modules) == {1, 14}

    def configure(self, model):
        owned = []
        for layer in model.layers:
            assert layer.ffn._strategy.name == "v41_nccl_ep"
            assert layer.ffn._strategy.cfg.stage_context == self.context
            if layer.layer_id in self.modules:
                _, layer.engram = self.modules[layer.layer_id]
                owned.append(layer.layer_id)
        assert owned == ([] if model.pp_has_embedding is False else [1, 14])
        model._test_prefill_scope = self.scope
        self.owners.append(model)

    def ids_for(self, step):
        return (
            torch.arange(sum(ROUNDS[step])) + step * 17 + (self.rank // 4) * 23
        ).remainder(128)

    def windows(self, lengths, prefixes):
        history = [[], []]
        for step, sizes in enumerate(ROUNDS):
            if [len(h) for h in history] == list(prefixes):
                assert list(lengths) == list(sizes)
                lookback = [list(reversed(h[-3:])) for h in history]
                lookback = [h + [-1] * (3 - len(h)) for h in lookback]
                return make_token_windows(
                    self.ids_for(step),
                    torch.tensor([0, sizes[0], sum(sizes)]),
                    torch.tensor(lookback),
                ).to(torch.int32)
            ids = self.ids_for(step).tolist()
            history[0].extend(ids[: sizes[0]])
            history[1].extend(ids[sizes[0] :])
        raise AssertionError("unexpected continuation history")

    @contextmanager
    def scope(self, model, ids, cp, lengths, prefixes):
        active = [layer for layer in model.layers if layer.engram is not None]
        try:
            if active:
                windows = self.windows(lengths, prefixes)
                if cp is not None:
                    _, indices, _ = zigzag(lengths, prefixes, cp.cp_rank, ids.device)
                    windows = windows.index_select(0, indices.cpu())
                    windows[~cp.local_is_real.cpu()] = -1
                windows = windows.to(ids.device).contiguous()
                real = windows[:, 0] >= 0
                torch.testing.assert_close(
                    windows[real, 0].long(), ids[real].long(), rtol=0, atol=0
                )
                hashes = self.reference._engram_hash_state.hash_token_windows(windows)
                for layer in active:
                    index, _ = self.modules[layer.layer_id]
                    layer.engram_hashes = hashes[:, index].contiguous()
                    layer.engram_token_mask = real
                self.calls += 1
            yield
        finally:
            for layer in model.layers:
                layer.engram_hashes = None
                layer.engram_token_mask = None

    def require_cleanup(self):
        assert len(self.owners) == 4 and self.calls >= 9
        for model in self.owners:
            for layer in model.layers:
                assert layer.engram_hashes is None and layer.engram_token_mask is None
                assert layer.ffn.gate._v41_cp_ctx is None
                assert layer.attn._prefill_meta_shared is None


def run_rank(rank, port, config, frame_exporter=None):
    torch.cuda.set_device(rank)
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    os.environ.update(
        {
            "NCCL_P2P_DISABLE": "1",
            "DSV41_ENGRAM_HOST_BYTES": str(2**20),
            "DSV41_ENGRAM_CACHE_DIR": "/dev/shm",
            "DSV4_PP_EP_ENABLE": "1",
            "DSV4_PP_EP_BACKEND": "fork_nccl_mxfp8",
            "DSV4_MOE_STRATEGY": "auto",
            "DSV4_PREFILL_FAST_PATH": "0",
            "RTP_V41_BATCHED_PRODUCER": "0",
            "MOEDBG": "0",
        }
    )
    dist.init_process_group(
        "nccl",
        init_method=f"tcp://127.0.0.1:{port}",
        rank=rank,
        world_size=8,
        timeout=timedelta(seconds=180),
    )
    reference = None
    try:
        pcfg = parallelism_for(rank)
        collective_torch.init_distributed_environment(
            pcfg, NcclCommConfig(nccl_ip="127.0.0.1"), port
        )
        context = EpStageContext.build(pcfg, backend="fork_nccl_mxfp8")
        torch.manual_seed(41600 + (rank // 4) * 100)
        args, weights = wide_fixture()
        local = ParallelismConfig()
        local.role_type = RoleType.PREFILL
        reference = DeepSeekV41Model(
            model_config(args, config), local, weights, None, 2
        )
        with patch.dict(
            os.environ, {"DSV4_PP_EP_ENABLE": "0", "DSV4_PP_EP_BACKEND": ""}
        ):
            assert reference.initialize(resources())
        controls = EngramControls(reference, rank, context)

        def fixture_fn():
            torch.manual_seed(41600 + (rank // 4) * 100)
            args, weights = wide_fixture()
            for layer in weights.weights:
                for projection in ("w1", "w2", "w3"):
                    for kind in ("w", "s"):
                        key = getattr(W, f"v4_routed_{projection}_{kind}")
                        layer[key] = layer[key][rank % 4 : rank % 4 + 1].contiguous()
            return (
                replace(
                    args,
                    ep_size=4,
                    ep_rank=rank % 4,
                    world_size=8,
                    world_rank=rank,
                    moe_tp_size=4,
                    moe_tp_rank=rank % 4,
                    moe_cp_size=4,
                    moe_cp_enabled=True,
                    moe_stage_context=context,
                    v41_config={**args.v41_config, **config},
                ),
                weights,
            )

        with torch.inference_mode():
            run_controls(
                rank,
                collective_torch._get_group(Group.TP),
                True,
                fixture_fn=fixture_fn,
                configure_owner=controls.configure,
                capture_stage_frame=frame_exporter,
            )
        controls.require_cleanup()
        torch.cuda.synchronize()
        reference.close()
        assert not reference._engram_tables
        print(
            "DSV41_CEP_ENGRAM "
            + json.dumps(
                {
                    "rank": rank,
                    "ep_size": 4,
                    "engram_prepared_calls": controls.calls,
                    "stage_one_engram_absent": True,
                    "history_padding_cleanup": True,
                }
            ),
            flush=True,
        )
        dist.barrier()
    finally:
        if reference is not None:
            reference.close()
        Dsv4SharedRuntimeBufferStore._reset_for_test()
        collective_torch.destroy_distributed_environment()


class V41CEPStageExecutionSM120Test(unittest.TestCase):
    def test_cp4_ep4_engram_with_native_21_19_owner_execution(self):
        self.assertGreaterEqual(torch.cuda.device_count(), 8)
        for rank in range(8):
            self.assertEqual(torch.cuda.get_device_capability(rank), (12, 0))
        with tempfile.TemporaryDirectory() as root, socket.socket() as sock:
            torch.manual_seed(41700)
            config = checkpoint(Path(root), full_schedule=True, hidden_size=5120)
            sock.bind(("127.0.0.1", 0))
            port = sock.getsockname()[1]
            sock.close()
            # The checkpoint must outlive all spawned host-table owners.
            mp.spawn(run_rank, args=(port, config), nprocs=8, join=True)


if __name__ == "__main__":
    unittest.main()
