"""Production-boundary TP metadata sync test.

This test intentionally imports the shipped ``librtp_compute_ops`` module for
bootstrap and calls ``tpSyncModelInputs`` from the shipped
``libth_transformer.so`` through a thin test bridge.  The target symbols are not
redefined here or in the bridge.
"""

import multiprocessing as mp
import os
import tempfile
import threading
import time
import traceback
import unittest

# The shipped native modules rely on symbols loaded by torch.  Preserve this
# order: importing either extension first can segfault in the dynamic loader.
# isort: off
import torch
import librtp_compute_ops
import libth_transformer

# isort: on

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
_EXPECTED_GPU = {
    "kv_cache_kernel_block_id": [[[1, 2], [3, 4]]],
    "kv_cache_block_id": [[[10, 11, 12], [20, 21, 22]]],
    "kv_cache_layer_to_group": [0, 0],
    "kv_cache_group_types": [1],
    "kv_cache_update_mapping": [[4, 5], [6, 7]],
    "cache_keys": [[1001, 1002, 1003], [2001, 2002, 2003]],
    "mm_features_locs": [0, 2],
    "multimodal_feature_0": [[10.5, 11.5], [12.5, 13.5]],
    "multimodal_feature_1": [[20.5, 21.5]],
    "mm_extra_input_0": [30.5, 31.5, 32.5],
    "mm_extra_input_1": [40.5],
    "last_hidden_states": [[1.25, 2.5], [3.75, 4.5], [5.25, 6.5]],
}


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


def _run_real_tp_sync_with_gil_progress(rank: int) -> None:
    """Delay rank 1 and verify rank 0's blocked UDS call releases the GIL."""
    if rank == 1:
        time.sleep(0.5)
        _run_real_tp_sync(rank, "uds-gil-release", include_gpu=False)
        return

    helper_progress_times = []

    def record_helper_progress() -> None:
        time.sleep(0.1)
        helper_progress_times.append(time.monotonic())

    helper = threading.Thread(target=record_helper_progress, name="gil-progress")
    helper.start()
    _run_real_tp_sync(rank, "uds-gil-release", include_gpu=False)
    broadcast_completed_at = time.monotonic()
    helper.join(timeout=5)
    if not helper_progress_times:
        raise AssertionError("Python helper thread did not run during UDS broadcast")
    if helper_progress_times[0] >= broadcast_completed_at:
        raise AssertionError("blocked UDS broadcast retained the Python GIL")


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
        torch.distributed.barrier()
        _run_real_tp_sync_with_gil_progress(rank)
        torch.distributed.barrier()
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
            raise AssertionError(
                f"rank {rank}: failed to recreate production UDS state"
            )
        torch.distributed.barrier()

        # 3. Rank 1 owns a deliberately incompatible *real* singleton state.
        # A retry must reset old state on every rank before entering blocking C++
        # init; otherwise one side can reject re-init while the other waits for
        # a peer until the full initialization timeout.
        if rank == 1:
            librtp_compute_ops.destroy_cpu_tp_broadcaster()
            librtp_compute_ops.init_cpu_tp_broadcaster(
                0, 1, valid_base_path + "_rank1_fault"
            )
        torch.distributed.barrier()

        ct._init_cpu_tp_broadcaster_if_needed(librtp_compute_ops)
        if ct._cpu_tp_broadcaster_base_path is None:
            raise AssertionError(
                f"rank {rank}: stale singleton state was not reset before retry"
            )
        _run_real_tp_sync(rank, "stale-singleton-reset")
        torch.distributed.barrier()

        # 4. Non-empty hidden states with zero combo tokens must raise a useful
        # shape error rather than divide by zero on non-root. Root observes the
        # peer abort through the next UDS metadata broadcast.
        try:
            production_bridge.run_tp_sync_model_inputs(
                libth_transformer.__file__,
                rank,
                True,
                empty_combo_tokens=True,
            )
        except RuntimeError as error:
            if rank == 1:
                # This rank rejected the shape before entering root's next UDS
                # broadcast. Close its idle endpoint so root observes the
                # terminal peer abort immediately instead of waiting timeout.
                librtp_compute_ops.destroy_cpu_tp_broadcaster()
            expected_error = "non-zero combo tokens" if rank == 1 else "CpuBroadcaster"
            if expected_error not in str(error):
                raise AssertionError(
                    f"rank {rank}: unexpected empty-combo error: {error}"
                ) from error
        else:
            raise AssertionError(
                f"rank {rank}: empty combo tokens did not reject hidden states"
            )

        # The preceding runtime failure is intentionally terminal. Coordinate
        # reset/re-init before testing the next independent terminal condition.
        torch.distributed.barrier()
        ct._init_cpu_tp_broadcaster_if_needed(librtp_compute_ops)
        if ct._cpu_tp_broadcaster_base_path is None:
            raise AssertionError(
                f"rank {rank}: failed to reset after empty-combo rejection"
            )
        torch.distributed.barrier()

        # 5. Deliberately put a tensor on CUDA at root while non-root allocates
        # that logical tensor on CPU. Device-layout metadata must reject the
        # split before either packed payload can be consumed. Rank 1 reports
        # the explicit mismatch; root observes the peer abort as a terminal UDS
        # error instead of silently continuing into NCCL.
        try:
            production_bridge.run_tp_sync_model_inputs(
                libth_transformer.__file__,
                rank,
                False,
                root_combo_tokens_on_gpu=True,
            )
        except RuntimeError as error:
            expected_error = "device mismatch" if rank == 1 else "CpuBroadcaster"
            if expected_error not in str(error):
                raise AssertionError(
                    f"rank {rank}: unexpected device mismatch error: {error}"
                ) from error
        else:
            raise AssertionError(
                f"rank {rank}: mismatched tensor devices did not fail fast"
            )
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
