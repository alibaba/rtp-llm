# ============================================================================
# xdist Per-Worker GPU Slicing (MUST be first — before any torch import)
# ============================================================================
import os as _os
import re as _re
import sys as _sys

_xdist_worker = _os.environ.get("PYTEST_XDIST_WORKER")
if _xdist_worker:
    _os.environ["_RTP_TORCH_BEFORE_SLICE"] = "1" if "torch" in _sys.modules else "0"

    # Worker name parse: pytest-xdist standard is "gw0", "gw1", ...
    # Custom runners (remote sessions, controller-only modes) may pass other
    # names. Parse defensively — fall back to slice 0 + warn rather than
    # raising ValueError at module-import time (which surfaces as a confusing
    # ImportError to the test runner).
    _m = _re.match(r"^gw(\d+)$", _xdist_worker)
    if _m:
        _wn = int(_m.group(1))
    else:
        _wn = 0
        _sys.stderr.write(
            f"[conftest_gpu_slice] WARN: unparseable worker '{_xdist_worker}', "
            f"defaulting to slice 0; GPU pool affinity may be incorrect\n"
        )

    _cvd = _os.environ.get("CUDA_VISIBLE_DEVICES")
    _hvd = _os.environ.get("HIP_VISIBLE_DEVICES")
    _pool = _cvd or _hvd or ""
    if _pool:
        _all_gpus = [g.strip() for g in _pool.split(",") if g.strip()]
        _gpu_per_worker = int(_os.environ.get("GPU_COUNT_PER_WORKER", "1"))
        _start = _wn * _gpu_per_worker
        _my_gpus = _all_gpus[_start : _start + _gpu_per_worker]
        # Fail-fast on pool exhaustion. Silently writing CVD="" hides the
        # misconfiguration: tests would collect 0 items / pass trivially while
        # the real problem (worker count > pool size) goes unnoticed.
        if not _my_gpus:
            _sys.stderr.write(
                f"[conftest_gpu_slice] FATAL: GPU pool '{_pool}' exhausted: "
                f"worker '{_xdist_worker}' (idx={_wn}) needs slice "
                f"[{_start}:{_start + _gpu_per_worker}] but pool has only "
                f"{len(_all_gpus)} GPU(s). Reduce -n WORKER_COUNT or expand "
                f"the GPU pool (CUDA_VISIBLE_DEVICES / HIP_VISIBLE_DEVICES).\n"
            )
            _sys.stderr.flush()
            _sys.exit(2)  # 2 = pytest "command-line usage error"
        _slice = ",".join(_my_gpus)
        if _cvd is not None:
            _os.environ["CUDA_VISIBLE_DEVICES"] = _slice
        if _hvd is not None:
            _os.environ["HIP_VISIBLE_DEVICES"] = _slice
        _sys.stderr.write(
            f"[conftest_gpu_slice] {_xdist_worker}: CVD={_os.environ.get('CUDA_VISIBLE_DEVICES', 'unset')} "
            f"HVD={_os.environ.get('HIP_VISIBLE_DEVICES', 'unset')} "
            f"torch_before_slice={_os.environ['_RTP_TORCH_BEFORE_SLICE']} "
            f"(from pool {_pool}, per_worker={_gpu_per_worker})\n"
        )
        _sys.stderr.flush()

    # File-based faulthandler: write crash stacks to per-worker files so they
    # survive even when xdist swallows worker stderr (which it always does on
    # crash — see xdist/workermanage.py:406 "Not properly terminated").
    import faulthandler as _fh

    _fault_dir = "/tmp/rtp_xdist_crash"
    _os.makedirs(_fault_dir, exist_ok=True)
    _fault_path = f"{_fault_dir}/{_xdist_worker}.fault"
    _fault_file = open(_fault_path, "w")
    _fh.enable(file=_fault_file, all_threads=True)
    _sys.stderr.write(f"[conftest] faulthandler → {_fault_path}\n")
    _sys.stderr.flush()

# ============================================================================
# GPU isolation is handled by:
#   - conftest.py module-level code (above): xdist workers slice inherited CVD
#   - device_resource.py __main__: per-test remote / smoke wraps pytest
# Both set CUDA_VISIBLE_DEVICES BEFORE entry-point plugins trigger cuInit().
# ============================================================================
import logging
import os
import re

import pytest

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


# ============================================================================
# Session-scoped diagnostic — prints GPU assignment visible in xdist output
# ============================================================================


@pytest.fixture(scope="session", autouse=True)
def _log_gpu_assignment():
    """Log GPU assignment at session start (visible in xdist worker output)."""
    worker = os.environ.get("PYTEST_XDIST_WORKER", "controller")
    cvd = os.environ.get("CUDA_VISIBLE_DEVICES", "unset")
    hvd = os.environ.get("HIP_VISIBLE_DEVICES", "unset")
    gpus = f"CUDA={cvd}" if hvd == "unset" else f"HIP={hvd}"
    print(f"\n[GPU_ASSIGN] {worker} pid={os.getpid()} {gpus}")
    try:
        import torch

        if torch.cuda.is_available():
            dc = torch.cuda.device_count()
            free, total = torch.cuda.mem_get_info(0)
            name = torch.cuda.get_device_name(0)
            print(
                f"[GPU_VERIFY] {worker} device_count={dc} name={name} "
                f"free={free/1e9:.1f}GB total={total/1e9:.1f}GB"
            )
    except Exception as e:
        print(f"[GPU_VERIFY] {worker} error: {e}")
    yield


# ============================================================================
# Per-test GPU memory monitoring + cleanup
# ============================================================================


def _get_gpu_mem_mb():
    """Return (allocated_MB, reserved_MB) for current default CUDA device, or None."""
    try:
        import torch

        if not torch.cuda.is_available():
            return None
        return (
            torch.cuda.memory_allocated() / (1024 * 1024),
            torch.cuda.memory_reserved() / (1024 * 1024),
        )
    except Exception:
        return None


@pytest.fixture(scope="function", autouse=True)
def _gpu_mem_monitor(request):
    """Per-test GPU memory tracking and aggressive cleanup between tests."""
    before = _get_gpu_mem_mb()
    yield

    try:
        import gc

        gc.collect()
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        # Reset default device to CPU after every test.  Tests that call
        # torch.set_default_device("cuda") (e.g. torch_symm_mem_test,
        # collective_torch_test) would otherwise pollute subsequent tests,
        # causing CP attention tests to hang or crash.
        torch.set_default_device("cpu")
    except Exception:
        pass

    after = _get_gpu_mem_mb()

    if before is not None and after is not None:
        alloc_before, reserved_before = before
        alloc_after, reserved_after = after
        delta_alloc = alloc_after - alloc_before
        delta_reserved = reserved_after - reserved_before
        worker = os.environ.get("PYTEST_XDIST_WORKER", "main")
        if abs(delta_alloc) > 10 or abs(delta_reserved) > 100:
            logger.warning(
                "[GPU_MEM] %s %s: alloc %.0f->%.0f MB (d%+.0f), "
                "reserved %.0f->%.0f MB (d%+.0f)",
                worker,
                request.node.nodeid,
                alloc_before,
                alloc_after,
                delta_alloc,
                reserved_before,
                reserved_after,
                delta_reserved,
            )


# ============================================================================
# Per-test child process cleanup (prevents GPU memory leaks from orphans)
# ============================================================================


def _get_child_pids(parent_pid: int) -> set:
    """Return set of all descendant PIDs of parent_pid."""
    try:
        import psutil

        return {c.pid for c in psutil.Process(parent_pid).children(recursive=True)}
    except Exception:
        return set()


def _is_safe_to_kill(pid: int) -> bool:
    """Check if a child process is safe to kill.

    resource_tracker and similar daemon processes hold CUDA IPC state (symmetric
    memory, NCCL shared memory). Killing them corrupts the CUDA driver for the
    parent process, causing SIGSEGV in subsequent tests. These are harmless
    daemon processes that die automatically when pytest exits.
    """
    try:
        import psutil

        cmdline = " ".join(psutil.Process(pid).cmdline())
        if "resource_tracker" in cmdline:
            return False
    except Exception:
        pass
    return True


@pytest.fixture(scope="function", autouse=True)
def _cleanup_child_processes():
    """Safety net: kill child processes spawned during a test to prevent GPU memory leaks.

    Multi-GPU tests spawn worker processes via mp.Process. If a test fails (exception,
    timeout, assertion), workers may survive and become orphans holding GPU memory.
    This fixture tracks child PIDs before/after each test and terminates stragglers.

    Skips resource_tracker processes — killing them corrupts CUDA IPC state and
    causes SIGSEGV in subsequent tests. They are harmless daemons that die at exit.

    NOTE: Does NOT protect against SIGSEGV (which bypasses Python teardown).
    For SIGSEGV resilience, test helpers should also call terminate() in finally blocks.
    """
    import signal
    import time

    before = _get_child_pids(os.getpid())
    yield
    after = _get_child_pids(os.getpid())
    orphans = {pid for pid in (after - before) if _is_safe_to_kill(pid)}
    if not orphans:
        return
    logger.warning(
        "[ORPHAN_CLEANUP] killing %d orphaned child process(es): %s",
        len(orphans),
        orphans,
    )
    for pid in orphans:
        try:
            os.kill(pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
    time.sleep(2)
    for pid in orphans:
        try:
            os.kill(pid, signal.SIGKILL)
        except ProcessLookupError:
            pass


# ============================================================================
# Marker Registration & Rewriting
# ============================================================================


def _register_synthetic_gpu_marker(config, count: int) -> str:
    synthetic_name = f"gpu_count_{count}"
    registered = getattr(config, "_synthetic_gpu_markers", set())
    if synthetic_name not in registered:
        config.addinivalue_line(
            "markers",
            f"{synthetic_name}: synthetic marker for gpu(count={count}) filtering",
        )
        registered.add(synthetic_name)
        config._synthetic_gpu_markers = registered
    return synthetic_name


def pytest_configure(config):
    """Rewrite gpu(count=N) in -m expressions and register needed synthetic markers."""
    config.addinivalue_line(
        "markers", "manual: test requires manual execution (deselected by default)"
    )
    config.addinivalue_line(
        "markers",
        "remote_cache: OSS smoke using KV cache manager remote cache (see smoke_remote_cache_oss profile)",
    )
    config._synthetic_gpu_markers = set()

    marker_expr = config.option.markexpr
    if marker_expr:
        rewritten = re.sub(
            r"(?<!\w)gpu\(count\s*=\s*(\d+)\)",
            r"gpu_count_\1",
            marker_expr,
        )
        config.option.markexpr = rewritten
        for count in re.findall(r"(?<!\w)gpu_count_(\d+)\b", rewritten):
            _register_synthetic_gpu_marker(config, int(count))
        logger.debug(f"Modified marker expression: {marker_expr} -> {rewritten}")


# ============================================================================
# Collection hooks
# ============================================================================


@pytest.hookimpl(tryfirst=True)
def pytest_collection_modifyitems(config, items):
    """
    - Deselect tests marked @pytest.mark.manual (require manual execution).
    - Add synthetic gpu_count_N markers before pytest applies -m selection.
    """
    marker_expr = getattr(config.option, "markexpr", "") or ""
    if "manual" not in marker_expr:
        manual_items = []
        remaining = []
        for item in items:
            if item.get_closest_marker("manual"):
                manual_items.append(item)
            else:
                remaining.append(item)
        if manual_items:
            config.hook.pytest_deselected(items=manual_items)
            items[:] = remaining

    for item in items:
        gpu_marker = item.get_closest_marker("gpu")
        if not gpu_marker:
            continue

        gpu_type = gpu_marker.kwargs.get("type")
        if gpu_type:
            item.add_marker(getattr(pytest.mark, gpu_type))

        count = gpu_marker.kwargs.get("count", 1)
        try:
            count = int(count)
        except (TypeError, ValueError):
            count = 1

        synthetic_name = _register_synthetic_gpu_marker(config, count)
        item.add_marker(getattr(pytest.mark, synthetic_name))
