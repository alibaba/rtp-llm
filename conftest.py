
import logging
import os
import pytest
import re

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


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
    config.addinivalue_line("markers", "manual: test requires manual execution (deselected by default)")
    config._synthetic_gpu_markers = set()

    marker_expr = config.option.markexpr
    if marker_expr:
        rewritten = re.sub(
            r'(?<!\w)gpu\(count\s*=\s*(\d+)\)',
            r'gpu_count_\1',
            marker_expr,
        )
        config.option.markexpr = rewritten
        for count in re.findall(r'(?<!\w)gpu_count_(\d+)\b', rewritten):
            _register_synthetic_gpu_marker(config, int(count))
        logger.debug(f"Modified marker expression: {marker_expr} -> {rewritten}")


# ============================================================================
# GPU Lock — delegates to DeviceResource (file-lock based, cross-process safe)
# ============================================================================

def _get_gpu_count_from_markers(node) -> int:
    """Get required GPU count from @pytest.mark.gpu(count=N), GPU_COUNT env, or default 1."""
    gpu_marker = node.get_closest_marker("gpu")
    if gpu_marker:
        if "count" in gpu_marker.kwargs:
            return int(gpu_marker.kwargs["count"])
        return 1

    gpu_count_env = os.environ.get("GPU_COUNT")
    if gpu_count_env:
        try:
            return int(gpu_count_env)
        except ValueError:
            logger.warning(f"Invalid GPU_COUNT env: {gpu_count_env}, using default 1")

    return 1


@pytest.fixture(scope="session", autouse=True)
def _worker_gpu_lock(request):
    """No-op session fixture.

    GPU locking is fully handled by the function-scoped ``gpu_lock`` fixture.
    Each test dynamically acquires N GPUs via DeviceResource file locks.

    On REAPI remote workers, GPU_COUNT is set by the scheduler to ensure
    the worker has enough GPUs, but CUDA_VISIBLE_DEVICES is NOT pre-set —
    DeviceResource discovers available GPUs at runtime and uses file locks
    for cross-process isolation (multiple sessions on the same machine).
    """
    yield


@pytest.fixture(scope="function")
def gpu_lock(request):
    """
    Function-scoped GPU lock — acquires N GPUs for this test.

    N is determined by @pytest.mark.gpu(count=N) or GPU_COUNT env.
    Uses DeviceResource file locks for cross-process safety:
    - xdist workers on the same session compete for GPUs
    - Multiple sessions on the same machine compete via the same file locks
    - REAPI ensures the machine has enough GPUs (via gpu_count scheduling)

    count=1: lock 1 GPU, other workers/sessions can use remaining GPUs
    count=2: lock 2 GPUs
    count=4: lock 4 GPUs, other tests wait
    """
    if request.node.get_closest_marker("no_gpu_lock"):
        yield None
        return

    gpu_count = _get_gpu_count_from_markers(request.node)
    if gpu_count < 1:
        yield None
        return

    from rtp_llm.test.utils.device_resource import (
        DeviceResource,
        get_device_info,
        _get_visible_devices_env,
    )

    device_info = get_device_info()
    if not device_info:
        yield None
        return

    device_name, _ = device_info
    env_name = _get_visible_devices_env(device_name)

    with DeviceResource(required_gpu_count=gpu_count) as gpu_resource:
        os.environ[env_name] = ",".join(gpu_resource.gpu_ids)
        logger.info(f"gpu_lock: {env_name}={os.environ[env_name]} (count={gpu_count})")
        yield gpu_resource


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
