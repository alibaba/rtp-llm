import csv
import fcntl
import multiprocessing
import os
import subprocess
import tempfile
import time
import unittest

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

_ALLOCATION_BYTES = 512 * 1024 * 1024
_TENSOR_VALUE = 37
_CHILD_READY_TIMEOUT_SECONDS = 120
_CHILD_VERIFY_TIMEOUT_SECONDS = 120
_GPU_MEMORY_TIMEOUT_SECONDS = 30
_CHILD_EXIT_TIMEOUT_SECONDS = 30
_MAX_IDLE_GPU_MEMORY_MIB = 1024


def _cuda_child(connection, gpu_selector, allocation_bytes):
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_selector
    try:
        import torch

        visible_device_count = torch.cuda.device_count()
        if not torch.cuda.is_available():
            connection.send(
                {
                    "kind": "skip",
                    "reason": (
                        "PyTorch CUDA runtime is unavailable "
                        f"(torch={torch.__version__}, torch.cuda={torch.version.cuda}, "
                        f"CUDA_VISIBLE_DEVICES="
                        f"{os.environ.get('CUDA_VISIBLE_DEVICES')!r})"
                    ),
                }
            )
            return
        if visible_device_count != 1:
            raise RuntimeError(
                "selected GPU is not the child's only visible CUDA device "
                f"(CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')!r}, "
                f"device_count={visible_device_count})"
            )

        torch.cuda.set_device(0)
        tensor = torch.full(
            (allocation_bytes,), _TENSOR_VALUE, dtype=torch.uint8, device="cuda"
        )
        expected_checksum = _TENSOR_VALUE * allocation_bytes
        initial_checksum = int(tensor.sum(dtype=torch.int64).item())
        torch.cuda.synchronize()
        connection.send(
            {
                "kind": "ready",
                "pid": os.getpid(),
                "starttime": read_process_starttime(os.getpid()),
                "allocation_bytes": tensor.numel() * tensor.element_size(),
                "checksum": initial_checksum,
                "expected_checksum": expected_checksum,
            }
        )

        command = connection.recv()
        if command != "verify":
            if command != "exit":
                raise RuntimeError(f"unexpected parent command: {command!r}")
            return

        restored_checksum = int(tensor.sum(dtype=torch.int64).item())
        content_matches = bool(torch.all(tensor == _TENSOR_VALUE).item())
        torch.cuda.synchronize()
        connection.send(
            {
                "kind": "verified",
                "checksum": restored_checksum,
                "content_matches": content_matches,
            }
        )
    except BaseException as error:
        try:
            connection.send(
                {"kind": "error", "error": f"{type(error).__name__}: {error}"}
            )
        except (BrokenPipeError, EOFError, OSError):
            pass
    finally:
        connection.close()


def _run_nvidia_smi(query, timeout_seconds=10):
    try:
        return subprocess.run(
            [
                "nvidia-smi",
                f"--query-{query}",
                "--format=csv,noheader,nounits",
            ],
            check=False,
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return None


def _csv_rows(output):
    return [
        [field.strip() for field in row]
        for row in csv.reader(output.splitlines())
        if row
    ]


def _compute_processes():
    result = _run_nvidia_smi("compute-apps=gpu_uuid,pid,used_gpu_memory")
    if result is None or result.returncode != 0:
        return None

    processes = {}
    for row in _csv_rows(result.stdout):
        if len(row) != 3:
            return None
        try:
            pid = int(row[1])
            used_mib = int(row[2])
        except ValueError:
            return None
        processes[(row[0], pid)] = used_mib
    return processes


def _reserve_idle_gpu(test_case):
    result = _run_nvidia_smi(
        "gpu=index,uuid,name,memory.total,memory.used,persistence_mode"
    )
    if result is None:
        test_case.skipTest("nvidia-smi is unavailable; cannot select an idle GPU")
    if result.returncode != 0:
        reason = result.stderr.strip() or result.stdout.strip() or "unknown error"
        test_case.skipTest(f"nvidia-smi cannot enumerate GPUs: {reason}")

    processes = _compute_processes()
    if processes is None:
        test_case.skipTest(
            "nvidia-smi compute-process accounting is unavailable; "
            "cannot prove that a GPU is idle"
        )
    busy_uuids = {gpu_uuid for gpu_uuid, _ in processes}
    requested_gpu = os.environ.get("RTP_LLM_CHECKPOINT_TEST_GPU")
    candidates = []
    for row in _csv_rows(result.stdout):
        if len(row) != 6:
            continue
        index, gpu_uuid, name, total_mib, used_mib, persistence_mode = row
        if requested_gpu and requested_gpu not in (index, gpu_uuid):
            continue
        if gpu_uuid in busy_uuids:
            continue
        try:
            parsed_used_mib = int(used_mib)
            if parsed_used_mib > _MAX_IDLE_GPU_MEMORY_MIB:
                continue
            candidates.append(
                {
                    "index": index,
                    "uuid": gpu_uuid,
                    "name": name,
                    "total_mib": int(total_mib),
                    "used_mib": parsed_used_mib,
                    "persistence_mode": persistence_mode,
                }
            )
        except ValueError:
            continue

    if not candidates:
        requested = (
            f" matching RTP_LLM_CHECKPOINT_TEST_GPU={requested_gpu!r}"
            if requested_gpu
            else ""
        )
        test_case.skipTest(f"no otherwise idle NVIDIA GPU is available{requested}")

    candidates.sort(key=lambda gpu: gpu["used_mib"])
    persistence_failures = []
    for gpu in candidates:
        lock_path = os.path.join(
            tempfile.gettempdir(),
            f"rtp_llm_checkpoint_gpu_{gpu['uuid'].replace('/', '_')}.lock",
        )
        lock_fd = os.open(lock_path, os.O_CREAT | os.O_RDWR, 0o600)
        try:
            fcntl.flock(lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            os.close(lock_fd)
            continue

        processes = _compute_processes()
        if processes is not None and not any(
            gpu_uuid == gpu["uuid"] for gpu_uuid, _ in processes
        ):
            if gpu["persistence_mode"].lower() != "enabled":
                os.close(lock_fd)
                persistence_failures.append(
                    f"GPU {gpu['index']} ({gpu['uuid']}) has persistence mode "
                    f"{gpu['persistence_mode']}; CUDA process restore requires it"
                )
                continue
            return gpu, lock_fd
        os.close(lock_fd)

    if persistence_failures:
        test_case.skipTest("; ".join(persistence_failures))
    test_case.skipTest("idle GPUs were claimed before the test could reserve one")


def _receive(connection, process, timeout_seconds, phase):
    if not connection.poll(timeout_seconds):
        raise TimeoutError(
            f"CUDA child timed out during {phase}; exitcode={process.exitcode}"
        )
    try:
        message = connection.recv()
    except EOFError as error:
        raise RuntimeError(
            f"CUDA child exited during {phase}; exitcode={process.exitcode}"
        ) from error
    if message.get("kind") == "error":
        raise RuntimeError(f"CUDA child failed during {phase}: {message['error']}")
    if message.get("kind") == "skip":
        raise unittest.SkipTest(message["reason"])
    return message


def _wait_for_pid_memory(gpu_uuid, pid, predicate, timeout_seconds):
    deadline = time.monotonic() + timeout_seconds
    last_memory = None
    while time.monotonic() < deadline:
        processes = _compute_processes()
        if processes is None:
            return None
        last_memory = processes.get((gpu_uuid, pid), 0)
        if predicate(last_memory):
            return last_memory
        time.sleep(0.25)
    return last_memory


def _checkpoint_api_unavailable(error):
    message = str(error).lower()
    return "not supported" in message or "unsupported" in message


def _recover_child_driver_state(controller, driver, pid):
    errors = []
    if controller is None:
        # No mutating checkpoint call could have happened yet.
        return True, errors
    try:
        controller.restore_all()
    except Exception as error:
        errors.append(f"controller restore failed: {error}")

    if driver is None:
        return True, errors
    try:
        state = driver.get_state(pid)
        if state == DRIVER_STATE_CHECKPOINTED:
            result = driver.restore(pid)
            if result != CUDA_SUCCESS:
                errors.append(f"direct restore failed: {driver.error_string(result)}")
            state = driver.get_state(pid)
        if state == DRIVER_STATE_LOCKED:
            result = driver.unlock(pid)
            if result != CUDA_SUCCESS:
                errors.append(f"direct unlock failed: {driver.error_string(result)}")
            state = driver.get_state(pid)
        if state != DRIVER_STATE_RUNNING:
            errors.append(f"CUDA child remains in driver state {state}")
            return False, errors
        return True, errors
    except Exception as error:
        errors.append(f"driver-state cleanup failed: {error}")
        return False, errors


def _cleanup_child(process, connection, controller, driver, pid):
    if process is None:
        return
    if not process.is_alive():
        connection.close()
        process.close()
        return

    running, errors = _recover_child_driver_state(controller, driver, pid)
    if process.is_alive() and running:
        try:
            connection.send("exit")
        except (BrokenPipeError, EOFError, OSError):
            pass
        process.join(_CHILD_EXIT_TIMEOUT_SECONDS)
    if process.is_alive() and running:
        # terminate() sends SIGTERM. Deliberately never call Process.kill().
        process.terminate()
        process.join(_CHILD_EXIT_TIMEOUT_SECONDS)
    if process.is_alive():
        errors.append(
            "CUDA child was left alive because it could not be confirmed RUNNING"
        )
    connection.close()
    if not process.is_alive():
        process.close()
    if errors:
        raise RuntimeError("; ".join(errors))


class CheckpointControllerGpuTest(unittest.TestCase):
    def test_real_gpu_checkpoint_releases_and_restores_cuda_state(self):
        gpu, gpu_lock_fd = _reserve_idle_gpu(self)
        driver = None
        controller = None
        child = None
        parent_connection = None
        child_pid = None

        try:
            try:
                driver = LibCudaCheckpointDriver()
            except CheckpointError as error:
                self.skipTest(
                    f"CUDA checkpoint symbols/driver are unavailable: {error}"
                )

            context = multiprocessing.get_context("spawn")
            parent_connection, child_connection = context.Pipe(duplex=True)
            child = context.Process(
                target=_cuda_child,
                args=(child_connection, gpu["index"], _ALLOCATION_BYTES),
                name="cuda-checkpoint-test-child",
            )
            previous_visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES")
            previous_device_order = os.environ.get("CUDA_DEVICE_ORDER")
            os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
            os.environ["CUDA_VISIBLE_DEVICES"] = gpu["index"]
            try:
                child.start()
            finally:
                if previous_visible_devices is None:
                    os.environ.pop("CUDA_VISIBLE_DEVICES", None)
                else:
                    os.environ["CUDA_VISIBLE_DEVICES"] = previous_visible_devices
                if previous_device_order is None:
                    os.environ.pop("CUDA_DEVICE_ORDER", None)
                else:
                    os.environ["CUDA_DEVICE_ORDER"] = previous_device_order
            child_connection.close()

            ready = _receive(
                parent_connection, child, _CHILD_READY_TIMEOUT_SECONDS, "CUDA setup"
            )
            self.assertEqual(ready["kind"], "ready")
            child_pid = ready["pid"]
            self.assertEqual(ready["starttime"], read_process_starttime(child_pid))
            self.assertEqual(ready["allocation_bytes"], _ALLOCATION_BYTES)
            self.assertEqual(ready["checksum"], ready["expected_checksum"])

            memory_before = _wait_for_pid_memory(
                gpu["uuid"],
                child_pid,
                lambda used_mib: used_mib >= _ALLOCATION_BYTES // (1024 * 1024),
                _GPU_MEMORY_TIMEOUT_SECONDS,
            )
            if memory_before is not None:
                self.assertGreaterEqual(
                    memory_before, _ALLOCATION_BYTES // (1024 * 1024)
                )

            try:
                initial_state = driver.get_state(child_pid)
            except CheckpointError as error:
                self.skipTest(
                    "CUDA process checkpoint API is unavailable for the selected "
                    f"GPU/driver: {error}"
                )
            self.assertEqual(initial_state, DRIVER_STATE_RUNNING)

            with tempfile.TemporaryDirectory() as tempdir:
                manifest_path = os.path.join(tempdir, "checkpoint.json")
                controller = CheckpointController(
                    manifest_path, driver=driver, lock_timeout_ms=60000
                )
                target = CheckpointTarget(
                    pid=child_pid,
                    rank=0,
                    address=f"gpu://{gpu['uuid']}",
                    expected_starttime=ready["starttime"],
                )
                try:
                    checkpoint_status = controller.checkpoint_all(
                        [target], "real-gpu-integration"
                    )
                except CheckpointError as error:
                    if _checkpoint_api_unavailable(error):
                        self.skipTest(
                            "CUDA process checkpoint is unsupported by the selected "
                            f"GPU/driver: {error}"
                        )
                    raise

                self.assertTrue(checkpoint_status.checkpoint_complete)
                self.assertEqual(checkpoint_status.phase, "CHECKPOINTED")
                self.assertEqual(driver.get_state(child_pid), DRIVER_STATE_CHECKPOINTED)

                memory_checkpointed = _wait_for_pid_memory(
                    gpu["uuid"],
                    child_pid,
                    lambda used_mib: used_mib == 0,
                    _GPU_MEMORY_TIMEOUT_SECONDS,
                )
                if memory_checkpointed is not None:
                    self.assertEqual(memory_checkpointed, 0)

                restore_status = controller.restore_all()
                self.assertTrue(restore_status.restore_complete)
                self.assertEqual(restore_status.phase, "UNLOCKED")
                self.assertEqual(driver.get_state(child_pid), DRIVER_STATE_RUNNING)

                parent_connection.send("verify")
                verified = _receive(
                    parent_connection,
                    child,
                    _CHILD_VERIFY_TIMEOUT_SECONDS,
                    "post-restore verification",
                )
                self.assertEqual(verified["kind"], "verified")
                self.assertEqual(verified["checksum"], ready["checksum"])
                self.assertTrue(verified["content_matches"])
                print(
                    "CUDA checkpoint integration: "
                    f"gpu={gpu['index']} uuid={gpu['uuid']} pid={child_pid} "
                    f"memory_before_mib={memory_before} "
                    f"memory_checkpointed_mib={memory_checkpointed}"
                )
                child.join(_CHILD_EXIT_TIMEOUT_SECONDS)
                self.assertFalse(child.is_alive())
                self.assertEqual(child.exitcode, 0)
        finally:
            try:
                if child is not None:
                    _cleanup_child(
                        child,
                        parent_connection,
                        controller,
                        driver,
                        child_pid,
                    )
            finally:
                fcntl.flock(gpu_lock_fd, fcntl.LOCK_UN)
                os.close(gpu_lock_fd)


if __name__ == "__main__":
    unittest.main()
