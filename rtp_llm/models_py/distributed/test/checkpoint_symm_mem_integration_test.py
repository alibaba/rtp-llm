"""Two-rank CUDA checkpoint integration for NCCL and symmetric memory.

The parent process deliberately never imports torch or creates a CUDA context.
Only spawned workers own CUDA state; the parent drives the external CUDA
process-checkpoint API after both workers have torn down their communicators.
"""

import datetime
import multiprocessing
import os
import tempfile
import time
import traceback
import unittest
from multiprocessing.connection import wait

from rtp_llm.utils.checkpoint_controller import (
    CUDA_SUCCESS,
    DRIVER_STATE_CHECKPOINTED,
    DRIVER_STATE_LOCKED,
    DRIVER_STATE_RUNNING,
    CheckpointController,
    CheckpointError,
    CheckpointTarget,
    LibCudaCheckpointDriver,
    read_process_starttime,
)

_WORLD_SIZE = 2
_CHECKPOINT_CYCLES = 3
_PHASE_TIMEOUT_SECONDS = 180
_EXIT_TIMEOUT_SECONDS = 30
_LOCK_TIMEOUT_MS = 60_000
_SYMM_MEM_MODE = os.environ.get("RTP_LLM_CHECKPOINT_TEST_SYMM_MEM", "all")


def _symmetric_memory_enabled(generation):
    if _SYMM_MEM_MODE == "all":
        return True
    if _SYMM_MEM_MODE in ("initial", "p2p"):
        return generation == 0
    if _SYMM_MEM_MODE in ("p2p-all", "none"):
        return False
    raise ValueError(
        "RTP_LLM_CHECKPOINT_TEST_SYMM_MEM must be all, initial, p2p, "
        "p2p-all, or none"
    )


def _p2p_symmetric_memory_enabled(generation):
    return _SYMM_MEM_MODE == "p2p-all" or (_SYMM_MEM_MODE == "p2p" and generation == 1)


class _WorkerSkip(RuntimeError):
    pass


def _run_collective_generation(torch, dist, symm_mem, rank, init_path, generation):
    """Build communicators, run both real collectives, then fully tear down."""
    communicator = None
    symm_buffer = None
    symm_handle = None
    expected = float(
        sum(
            generation * _WORLD_SIZE + peer_rank + 1 for peer_rank in range(_WORLD_SIZE)
        )
    )
    try:
        # CUDA process restore does not preserve the runtime's current-device
        # selection. Production rebuild does this in collective_torch before
        # constructing ProcessGroupNCCL, so mirror that contract here.
        torch.cuda.set_device(rank)
        dist.init_process_group(
            backend="nccl",
            init_method=f"file://{os.path.abspath(init_path)}",
            rank=rank,
            world_size=_WORLD_SIZE,
            timeout=datetime.timedelta(seconds=_PHASE_TIMEOUT_SECONDS),
            device_id=torch.device("cuda", rank),
        )

        local_value = float(generation * _WORLD_SIZE + rank + 1)
        nccl_value = torch.tensor(local_value, dtype=torch.float32, device="cuda")
        dist.all_reduce(nccl_value, op=dist.ReduceOp.SUM)

        symm_result = None
        if _symmetric_memory_enabled(generation):
            communicator = symm_mem.init_symm_mem_communicator(dist.group.WORLD)
            if communicator is None or communicator.disabled:
                raise RuntimeError("Torch symmetric-memory communicator is disabled")

            symm_value = torch.full(
                (4,), local_value, dtype=torch.bfloat16, device="cuda"
            )
            if not communicator.should_torch_symm_mem_allreduce(symm_value):
                raise RuntimeError("symmetric-memory all-reduce rejected test payload")
            symm_result = communicator.all_reduce(symm_value)
            if symm_result is None:
                raise RuntimeError("symmetric-memory all-reduce returned no result")
        elif _p2p_symmetric_memory_enabled(generation):
            # Mega's symmetric buffer needs peer mappings, but not the multicast
            # pointer required by the fast all-reduce communicator. Exercise that
            # lower-level contract independently after CUDA restore.
            symm_buffer = symm_mem.torch_symm_mem.empty(
                4, device=torch.device("cuda", rank), dtype=torch.bfloat16
            )
            symm_handle = symm_mem.torch_symm_mem.rendezvous(
                symm_buffer, group=dist.group.WORLD.group_name
            )
            symm_buffer.fill_(local_value)
            symm_handle.barrier()
            symm_result = torch.zeros_like(symm_buffer)
            for peer_rank in range(_WORLD_SIZE):
                symm_result.add_(
                    symm_handle.get_buffer(
                        peer_rank, symm_buffer.shape, symm_buffer.dtype
                    )
                )
            symm_handle.barrier()

        torch.cuda.synchronize(rank)
        nccl_result = float(nccl_value.item())
        symm_result_value = (
            float(symm_result[0].item()) if symm_result is not None else None
        )
        if nccl_result != expected:
            raise AssertionError(
                f"generation {generation} NCCL result {nccl_result}, expected {expected}"
            )
        if symm_result is not None and not bool(
            torch.all(symm_result == expected).item()
        ):
            raise AssertionError(
                "generation "
                f"{generation} symmetric-memory result {symm_result_value}, "
                f"expected {expected}"
            )

        symm_handle = None
        symm_buffer = None
        dist.barrier()
        torch.cuda.synchronize(rank)
        return {"nccl": nccl_result, "symm": symm_result_value}
    finally:
        if dist.is_initialized():
            # Symmetric-memory mappings retain the process group. They must be
            # released before ProcessGroupNCCL is destroyed.
            symm_mem.destroy_symm_mem_communicator()
            dist.destroy_process_group()
            if dist.is_initialized():
                raise RuntimeError("process group survived generation teardown")
        torch.cuda.empty_cache()


def _checkpoint_worker(rank, init_paths, connection):
    """Own all CUDA work and wait on a CPU pipe while checkpointed."""
    try:
        # CUDA process checkpoint cannot restore multicast state on the tested
        # CUDA 13 driver. Level 3 applies both settings before its first
        # production process-group or symmetric-memory initialization.
        os.environ["NCCL_NVLS_ENABLE"] = "0"
        os.environ["TORCH_SYMM_MEM_DISABLE_MULTICAST"] = "1"
        import torch
        import torch.distributed as dist

        if not torch.cuda.is_available() or not torch.version.cuda:
            raise _WorkerSkip("PyTorch CUDA is unavailable")
        visible_devices = torch.cuda.device_count()
        if visible_devices < _WORLD_SIZE:
            raise _WorkerSkip(
                f"test requires {_WORLD_SIZE} visible GPUs, found {visible_devices}"
            )

        torch.cuda.set_device(rank)
        from rtp_llm.models_py.distributed import symm_mem

        capability = torch.cuda.get_device_capability(rank)
        if _SYMM_MEM_MODE != "none" and not symm_mem.torch_symm_mem_available:
            raise _WorkerSkip("PyTorch symmetric memory is unavailable")
        if _SYMM_MEM_MODE != "none" and (
            capability[0] not in symm_mem.TORCH_SYMM_MEM_ALL_REDUCE_MAX_SIZES
            or _WORLD_SIZE
            not in symm_mem.TORCH_SYMM_MEM_ALL_REDUCE_MAX_SIZES[capability[0]]
        ):
            raise _WorkerSkip(
                "Torch symmetric-memory all-reduce does not support "
                f"compute capability {capability} with world size {_WORLD_SIZE}"
            )

        connection.send(
            {
                "kind": "ready",
                "rank": rank,
                "pid": os.getpid(),
                "capability": capability,
                "device_name": torch.cuda.get_device_name(rank),
            }
        )
        command = connection.recv()
        if command == "stop":
            return
        if command != "run-initial":
            raise RuntimeError(f"unexpected parent command: {command!r}")

        for generation, init_path in enumerate(init_paths):
            if generation > 0:
                # This recv is CPU-only. The parent checkpoints and restores both
                # CUDA processes while they are blocked here with all communicators
                # gone.
                command = connection.recv()
                if command == "stop":
                    return
                if command != "rebuild":
                    raise RuntimeError(f"unexpected parent command: {command!r}")

            results = _run_collective_generation(
                torch, dist, symm_mem, rank, init_path, generation=generation
            )
            connection.send(
                {
                    "kind": (
                        "complete" if generation == _CHECKPOINT_CYCLES else "quiesced"
                    ),
                    "rank": rank,
                    "pid": os.getpid(),
                    "generation": generation,
                    "results": results,
                }
            )
    except _WorkerSkip as error:
        try:
            connection.send({"kind": "skip", "rank": rank, "reason": str(error)})
        except (BrokenPipeError, EOFError, OSError):
            pass
    except BaseException as error:
        try:
            connection.send(
                {
                    "kind": "error",
                    "rank": rank,
                    "error": f"{type(error).__name__}: {error}",
                    "traceback": traceback.format_exc(),
                }
            )
        except (BrokenPipeError, EOFError, OSError):
            pass
    finally:
        connection.close()


def _receive_all(connections, processes, expected_kind, timeout_seconds):
    deadline = time.monotonic() + timeout_seconds
    pending = set(connections)
    messages = {}
    while pending:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            states = {rank: processes[rank].exitcode for rank in pending}
            raise TimeoutError(
                f"timed out waiting for {expected_kind}; rank exit codes={states}"
            )

        waitables = [connections[rank] for rank in pending]
        waitables.extend(processes[rank].sentinel for rank in pending)
        ready = set(wait(waitables, timeout=remaining))
        if not ready:
            continue

        for rank in list(pending):
            connection = connections[rank]
            process = processes[rank]
            if connection in ready or (process.sentinel in ready and connection.poll()):
                try:
                    message = connection.recv()
                except EOFError as error:
                    raise RuntimeError(
                        f"rank {rank} exited before {expected_kind}; "
                        f"exitcode={process.exitcode}"
                    ) from error
                kind = message.get("kind")
                if kind == "skip":
                    raise unittest.SkipTest(message["reason"])
                if kind == "error":
                    raise RuntimeError(
                        f"rank {rank} failed: {message['error']}\n"
                        f"{message['traceback']}"
                    )
                if kind != expected_kind:
                    raise RuntimeError(
                        f"rank {rank} sent {kind!r}, expected {expected_kind!r}"
                    )
                if message.get("rank") != rank:
                    raise RuntimeError(
                        f"rank {rank} sent mismatched identity {message.get('rank')!r}"
                    )
                messages[rank] = message
                pending.remove(rank)
            elif process.sentinel in ready:
                raise RuntimeError(
                    f"rank {rank} exited before {expected_kind}; "
                    f"exitcode={process.exitcode}"
                )
    return messages


def _checkpoint_api_unavailable(error):
    message = str(error).lower()
    return any(
        marker in message for marker in ("unavailable", "not supported", "unsupported")
    )


def _restore_before_cleanup(controller, driver, processes):
    """Restore every live target before any stop signal or termination."""
    errors = []
    if controller is not None:
        try:
            controller.restore_all()
        except Exception as error:
            errors.append(f"controller restore failed: {error}")

    if driver is None:
        return {rank: True for rank in processes}, errors

    # Direct recovery is a fallback for a partially completed controller
    # transaction. Restore every checkpointed rank before unlocking any rank.
    for rank, process in processes.items():
        if not process.is_alive() or process.pid is None:
            continue
        try:
            if driver.get_state(process.pid) == DRIVER_STATE_CHECKPOINTED:
                result = driver.restore(process.pid)
                if result != CUDA_SUCCESS:
                    errors.append(
                        f"rank {rank} direct restore failed: "
                        f"{driver.error_string(result)}"
                    )
        except Exception as error:
            errors.append(f"rank {rank} restore-state check failed: {error}")

    running = {}
    for rank, process in processes.items():
        if not process.is_alive() or process.pid is None:
            running[rank] = True
            continue
        try:
            state = driver.get_state(process.pid)
            if state == DRIVER_STATE_LOCKED:
                result = driver.unlock(process.pid)
                if result != CUDA_SUCCESS:
                    errors.append(
                        f"rank {rank} direct unlock failed: "
                        f"{driver.error_string(result)}"
                    )
                state = driver.get_state(process.pid)
            running[rank] = state == DRIVER_STATE_RUNNING
            if not running[rank]:
                errors.append(f"rank {rank} remains in CUDA driver state {state}")
        except Exception as error:
            running[rank] = False
            errors.append(f"rank {rank} final driver-state check failed: {error}")
    return running, errors


def _cleanup_workers(connections, processes, controller, driver):
    running, errors = _restore_before_cleanup(controller, driver, processes)

    for rank, process in processes.items():
        if process.is_alive() and running.get(rank, True):
            try:
                connections[rank].send("stop")
            except (BrokenPipeError, EOFError, OSError):
                pass
    for process in processes.values():
        if process.is_alive():
            process.join(_EXIT_TIMEOUT_SECONDS)
    for rank, process in processes.items():
        if process.is_alive() and running.get(rank, True):
            process.terminate()
            process.join(_EXIT_TIMEOUT_SECONDS)
        if process.is_alive():
            errors.append(
                f"rank {rank} left alive because CUDA RUNNING was not confirmed"
            )

    for connection in connections.values():
        connection.close()
    for process in processes.values():
        if not process.is_alive():
            process.close()
    if errors:
        raise RuntimeError("; ".join(errors))


class CheckpointSymmetricMemoryIntegrationTest(unittest.TestCase):
    def test_two_rank_three_checkpoint_restore_collective_cycles(self):
        context = multiprocessing.get_context("spawn")
        connections = {}
        processes = {}
        controller = None
        driver = None

        with tempfile.TemporaryDirectory() as tempdir:
            init_paths = tuple(
                os.path.join(tempdir, f"generation-{generation}.store")
                for generation in range(_CHECKPOINT_CYCLES + 1)
            )
            manifest_path = os.path.join(tempdir, "checkpoint.json")
            try:
                for rank in range(_WORLD_SIZE):
                    parent_connection, child_connection = context.Pipe(duplex=True)
                    process = context.Process(
                        target=_checkpoint_worker,
                        args=(rank, init_paths, child_connection),
                        name=f"checkpoint-symm-mem-rank-{rank}",
                    )
                    connections[rank] = parent_connection
                    processes[rank] = process
                    try:
                        process.start()
                    finally:
                        child_connection.close()

                ready = _receive_all(
                    connections, processes, "ready", _PHASE_TIMEOUT_SECONDS
                )
                for rank, message in ready.items():
                    self.assertEqual(message["pid"], processes[rank].pid)

                for connection in connections.values():
                    connection.send("run-initial")
                quiesced = _receive_all(
                    connections, processes, "quiesced", _PHASE_TIMEOUT_SECONDS
                )
                self._assert_generation_results(
                    quiesced, generation=0, processes=processes
                )

                try:
                    # This initializes only the parent-side driver control API;
                    # the parent never imports torch or creates a CUDA context.
                    driver = LibCudaCheckpointDriver()
                except CheckpointError as error:
                    if _checkpoint_api_unavailable(error):
                        self.skipTest(str(error))
                    raise

                controller = CheckpointController(
                    manifest_path,
                    driver=driver,
                    lock_timeout_ms=_LOCK_TIMEOUT_MS,
                )
                targets = [
                    CheckpointTarget(
                        pid=processes[rank].pid,
                        rank=rank,
                        address=f"local-rank://{rank}",
                        expected_starttime=read_process_starttime(processes[rank].pid),
                    )
                    for rank in range(_WORLD_SIZE)
                ]
                for cycle in range(_CHECKPOINT_CYCLES):
                    try:
                        checkpoint_status = controller.checkpoint_all(
                            targets, f"nccl-symm-mem-cycle-{cycle}"
                        )
                    except CheckpointError as error:
                        if _checkpoint_api_unavailable(error):
                            self.skipTest(str(error))
                        raise
                    self.assertTrue(checkpoint_status.checkpoint_complete)
                    self.assertEqual(checkpoint_status.phase, "CHECKPOINTED")
                    self.assertEqual(len(checkpoint_status.processes), _WORLD_SIZE)

                    restore_status = controller.restore_all()
                    self.assertTrue(restore_status.restore_complete)
                    self.assertEqual(restore_status.phase, "UNLOCKED")
                    self.assertEqual(len(restore_status.processes), _WORLD_SIZE)

                    for connection in connections.values():
                        connection.send("rebuild")
                    generation = cycle + 1
                    kind = (
                        "complete" if generation == _CHECKPOINT_CYCLES else "quiesced"
                    )
                    messages = _receive_all(
                        connections, processes, kind, _PHASE_TIMEOUT_SECONDS
                    )
                    self._assert_generation_results(
                        messages, generation=generation, processes=processes
                    )

                for process in processes.values():
                    process.join(_EXIT_TIMEOUT_SECONDS)
                    self.assertFalse(process.is_alive())
                    self.assertEqual(process.exitcode, 0)
            finally:
                _cleanup_workers(
                    connections, processes, controller=controller, driver=driver
                )

    def _assert_generation_results(self, messages, generation, processes):
        expected = float(
            sum(
                generation * _WORLD_SIZE + peer_rank + 1
                for peer_rank in range(_WORLD_SIZE)
            )
        )
        self.assertEqual(set(messages), set(range(_WORLD_SIZE)))
        for rank, message in messages.items():
            self.assertEqual(message["rank"], rank)
            self.assertEqual(message["pid"], processes[rank].pid)
            self.assertEqual(message["generation"], generation)
            self.assertEqual(message["results"]["nccl"], expected)
            if _symmetric_memory_enabled(generation) or _p2p_symmetric_memory_enabled(
                generation
            ):
                self.assertEqual(message["results"]["symm"], expected)
            else:
                self.assertIsNone(message["results"]["symm"])


if __name__ == "__main__":
    unittest.main()
