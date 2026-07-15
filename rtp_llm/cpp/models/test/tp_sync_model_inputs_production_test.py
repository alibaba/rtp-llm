"""Production-boundary TP metadata sync test.

This test intentionally imports the shipped ``librtp_compute_ops`` module for
bootstrap and calls ``tpSyncModelInputs`` from the shipped
``libth_transformer.so`` through a thin test bridge.  The target symbols are not
redefined here or in the bridge.
"""

import multiprocessing as mp
import os
import tempfile
import time
import traceback
import unittest

import torch

import librtp_compute_ops
import libth_transformer
from rtp_llm.cpp.models.test import (
    libtp_sync_model_inputs_production_test_bridge as production_bridge,
)
from rtp_llm.models_py.distributed import collective_torch as ct
from rtp_llm.ops import NcclCommConfig, ParallelismConfig
from rtp_llm.test.utils.port_util import PortManager


_EXPECTED = {
    "combo_tokens": [11, 12, 13],
    "input_lengths": [2, 1],
    "sequence_lengths": [5, 6],
    "prefix_lengths": [0, 2],
    "request_id": [101, 202],
    "request_pd_separation": [False, True],
    "lm_output_indexes": [1, 2],
    "lm_output_lengths": [7, 8],
    "combo_position_ids": [0, 1, 2],
    "text_tokens_mask": [1, 1, 0],
}
_EXPECTED_GPU = {"last_hidden_states": [[1.25, 2.5], [3.75, 4.5], [5.25, 6.5]]}


def _parallelism_config(rank: int) -> ParallelismConfig:
    config = ParallelismConfig()
    config.world_rank = rank
    config.world_size = 2
    config.local_rank = rank
    config.local_world_size = 2
    config.tp_rank = rank
    config.tp_size = 2
    config.dp_rank = 0
    config.dp_size = 1
    return config


def _assert_synced(result, phase: str, rank: int, include_gpu: bool) -> None:
    expected_values = _EXPECTED | (_EXPECTED_GPU if include_gpu else {})
    for name, expected in expected_values.items():
        actual = result[name].cpu()
        expected_tensor = torch.tensor(expected, dtype=actual.dtype)
        if not torch.equal(actual, expected_tensor):
            raise AssertionError(
                f"{phase}: rank {rank} got bad {name}: {actual} != {expected_tensor}"
            )


def _run_real_tp_sync(rank: int, phase: str, include_gpu: bool = True) -> None:
    result = production_bridge.run_tp_sync_model_inputs(
        libth_transformer.__file__, rank, include_gpu
    )
    _assert_synced(result, phase, rank, include_gpu)


def _production_boundary_worker(rank: int, nccl_port: int, uds_dir: str) -> None:
    initialized = False
    try:
        os.environ["RTP_LLM_CPU_TP_BROADCASTER_DIR"] = uds_dir
        os.environ["RTP_LLM_CPU_TP_BROADCASTER_ID"] = "production-boundary"
        torch.cuda.set_device(rank)

        config = _parallelism_config(rank)
        comm_config = NcclCommConfig(
            nccl_ip="127.0.0.1",
            tp_nccl_port=nccl_port + 9,
            dp_tp_nccl_port=nccl_port + 1,
            ffn_tp_nccl_port=nccl_port + 6,
        )
        ct.init_distributed_environment(
            config,
            nccl_comm_config=comm_config,
            nccl_init_port=nccl_port,
            backend="nccl",
            timeout=60,
        )
        initialized = True

        # 1. Default-on production bootstrap initializes UDS on every TP rank;
        # tpSyncModelInputs then crosses libth_transformer -> librtp_compute_ops.
        if ct._cpu_tp_broadcaster_base_path is None:
            raise AssertionError(f"rank {rank}: production UDS bootstrap was skipped")
        # Exercise the default mixed path first: packed CPU metadata uses UDS
        # while the GPU payload remains on the registered c10d/NCCL callback.
        _run_real_tp_sync(rank, "uds-with-gpu")
        torch.distributed.barrier()

        # Then clear c10d callbacks and use CPU-only inputs so a duplicate or
        # uninitialized DSO singleton cannot silently fall back and pass.
        librtp_compute_ops.clear_comm_ops()
        _run_real_tp_sync(rank, "uds-without-fallback", include_gpu=False)
        ct._register_process_groups_to_cpp()
        torch.distributed.barrier()

        # 2. With the production singleton uninitialized, the same real wrapper
        # must fall back to the registered c10d/NCCL callback.
        librtp_compute_ops.destroy_cpu_tp_broadcaster()
        ct._cpu_tp_broadcaster_base_path = None
        torch.distributed.barrier()
        _run_real_tp_sync(rank, "uninitialized-fallback")
        torch.distributed.barrier()

        # Recreate a valid group before injecting a one-rank re-init failure.
        ct._init_cpu_tp_broadcaster_if_needed(librtp_compute_ops)
        valid_base_path = ct._cpu_tp_broadcaster_base_path
        if valid_base_path is None:
            raise AssertionError(f"rank {rank}: failed to recreate production UDS state")
        torch.distributed.barrier()

        # 3. Rank 1 owns a deliberately incompatible *real* singleton state.
        # Its production init call fails immediately while rank 0's matching
        # re-init succeeds.  The bootstrap all-gather must reset the whole group
        # so the following real tpSyncModelInputs call consistently uses c10d.
        if rank == 1:
            librtp_compute_ops.destroy_cpu_tp_broadcaster()
            librtp_compute_ops.init_cpu_tp_broadcaster(
                0, 1, valid_base_path + "_rank1_fault"
            )
        torch.distributed.barrier()

        init_succeeded = True
        try:
            librtp_compute_ops.init_cpu_tp_broadcaster(
                rank, 2, valid_base_path
            )
        except RuntimeError:
            init_succeeded = False
        init_results = [False, False]
        torch.distributed.all_gather_object(init_results, init_succeeded)
        if init_results != [True, False]:
            raise AssertionError(
                f"expected a rank-local production init failure, got {init_results}"
            )

        ct._init_cpu_tp_broadcaster_if_needed(librtp_compute_ops)
        if ct._cpu_tp_broadcaster_base_path is not None:
            raise AssertionError(
                f"rank {rank}: divergent init did not force group fallback"
            )
        _run_real_tp_sync(rank, "group-consistent-fallback")
        torch.distributed.barrier()
    except Exception:
        traceback.print_exc()
        raise
    finally:
        if initialized and torch.distributed.is_initialized():
            ct.destroy_distributed_environment()


class TpSyncModelInputsProductionTest(unittest.TestCase):
    def test_real_bootstrap_uds_and_fallbacks_tp2(self):
        self.assertTrue(torch.cuda.is_available(), "CUDA is required")
        self.assertGreaterEqual(
            torch.cuda.device_count(), 2, "the H20 test target requires two GPUs"
        )

        port_manager = PortManager()
        ports, locks = port_manager.get_consecutive_ports(1)
        context = mp.get_context("spawn")
        processes = []
        try:
            with tempfile.TemporaryDirectory(
                prefix="tp_sync_production_boundary."
            ) as uds_dir:
                for rank in range(2):
                    process = context.Process(
                        target=_production_boundary_worker,
                        args=(rank, ports[0], uds_dir),
                        name=f"tp-rank-{rank}",
                    )
                    process.start()
                    processes.append(process)

                deadline = time.monotonic() + 180
                for process in processes:
                    process.join(timeout=max(0, deadline - time.monotonic()))
                timed_out = [process for process in processes if process.is_alive()]
                for process in timed_out:
                    process.terminate()
                    process.join(timeout=10)
                failures = [
                    f"{process.name} exit={process.exitcode}"
                    for process in processes
                    if process.exitcode != 0
                ]
                failures.extend(f"{process.name} timed out" for process in timed_out)
                if failures:
                    self.fail("; ".join(failures))
        finally:
            for process in processes:
                if process.is_alive():
                    process.terminate()
                    process.join(timeout=10)
            for lock in locks:
                lock.__exit__(None, None, None)


if __name__ == "__main__":
    unittest.main()
