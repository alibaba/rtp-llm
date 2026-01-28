
import fcntl
import logging
import os
import pytest
import re
from contextlib import contextmanager

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# GPU Lock configuration
GPU_LOCK_FILE_PREFIX = os.environ.get("GPU_LOCK_FILE", "/tmp/gpu_lock")

# ============================================================================
# GPU Lock Context Manager
# ============================================================================

@contextmanager
def gpu_lock_context(gpu_count: int = 1):
    """
    Context manager for GPU lock acquisition and release.
    
    Args:
        gpu_count: Number of GPUs required (default: 1).
                   Each GPU has its own lock file to enable parallel test execution
                   when tests require different GPU counts.
    
    Lock Strategy:
        - gpu_count=1: Acquires lock on GPU 0 only
        - gpu_count=2: Acquires locks on GPU 0 and GPU 1
        - gpu_count=N: Acquires locks on GPU 0 through GPU N-1
        
    This allows tests requiring 1 GPU to run in parallel with each other
    (on different GPUs), while tests requiring all GPUs will block others.
    """
    lock_fds = []
    try:
        for i in range(gpu_count):
            lock_file = f"{GPU_LOCK_FILE_PREFIX}_{i}"
            fd = open(lock_file, 'w')
            logger.info(f"Acquiring GPU lock {i}: {lock_file}")
            fcntl.flock(fd, fcntl.LOCK_EX)
            lock_fds.append(fd)
        logger.info(f"GPU lock acquired for {gpu_count} GPU(s)")
        yield lock_fds
    finally:
        for fd in lock_fds:
            fcntl.flock(fd, fcntl.LOCK_UN)
            fd.close()
        logger.info(f"GPU lock released for {gpu_count} GPU(s)")


def _get_gpu_count_from_markers(node):
    """
    从 pytest markers 或环境变量中获取所需的 GPU 数量。
    
    优先级：
    1. @pytest.mark.gpu(count=N) - 显式指定
    3. GPU_COUNT 环境变量 - 从 Bazel exec_properties 传递
    4. 默认: 1 GPU
    
    Args:
        node: pytest 测试节点
        
    Returns:
        int: 所需的 GPU 数量
    """
    # 检查 @pytest.mark.gpu(count=N)
    gpu_marker = node.get_closest_marker("gpu")
    if gpu_marker:
        if "count" in gpu_marker.kwargs:
            return int(gpu_marker.kwargs["count"])
        # 如果只有 @pytest.mark.gpu 没有 count，默认 1
        return 1

    # 检查环境变量 GPU_COUNT（从 Bazel exec_properties 传递）
    gpu_count_env = os.environ.get("GPU_COUNT")
    if gpu_count_env:
        try:
            return int(gpu_count_env)
        except ValueError:
            logger.warning(f"Invalid GPU_COUNT environment variable: {gpu_count_env}, using default 1")
    
    # 默认 1 GPU
    return 1


def pytest_configure(config):
    """
    Make marker expressions like "gpu(count=4)" work with pytest -m.

    Approach (minimal + let pytest handle and/or/not):
    - Rewrite "gpu(count=N)" in `-m` into a synthetic marker name: "gpu_count_N"
    - Also treat bare "gpu" as "gpu(count=1)" (rewrite to "gpu_count_1")
    - During collection, for each test marked with @pytest.mark.gpu(count=N),
      add the synthetic marker "gpu_count_N" (default N=1 when not provided)

    Then pytest's own marker expression evaluator can handle:
      - "H20 and not gpu(count=4) and not smoke"
      - "gpu(count=1) or gpu(count=2)"
      - "gpu and not gpu(count=4)"
    """
    # With --strict-markers enabled, any marker name used on tests must be
    # registered. Since we synthesize markers like "gpu_count_4" at runtime,
    # register a reasonable range up front.
    #
    # You can override the range via env var if needed:
    #   PYTEST_GPU_COUNT_MAX=32 pytest ...
    max_count = os.environ.get("PYTEST_GPU_COUNT_MAX", "16")
    try:
        max_count_int = int(max_count)
    except ValueError:
        max_count_int = 16
    max_count_int = max(1, max_count_int)

    for n in range(1, max_count_int + 1):
        config.addinivalue_line(
            "markers",
            f"gpu_count_{n}: synthetic marker for gpu(count={n}) filtering",
        )

    marker_expr = config.option.markexpr
    if not marker_expr:
        return

    # Only support gpu(count=N) for now (easy to extend later).
    # Allow optional spaces around "=".
    rewritten = re.sub(
        r'(?<!\w)gpu\(count\s*=\s*(\d+)\)',
        r'gpu_count_\1',
        marker_expr,
    )

    config.option.markexpr = rewritten
    logger.debug(f"Modified marker expression: {marker_expr} -> {rewritten}")


# @pytest.fixture(scope="function")
def gpu_lock(request):
    """
    GPU lock fixture for test isolation.
    
    Automatically applied to tests marked with @pytest.mark.gpu, @pytest.mark.smoke,
    @pytest.mark.ppu, @pytest.mark.cuda, or @pytest.mark.rocm.
    
    GPU count can be specified via:
        - @pytest.mark.gpu(count=N): Require N GPUs 
        - GPU_COUNT environment variable: From Bazel exec_properties
        - Default: 1 GPU
    
    Can be disabled with @pytest.mark.no_gpu_lock.
    
    Usage:
        @pytest.mark.gpu
        def test_single_gpu(gpu_lock):
            # Requires 1 GPU (default)
            ...
        
        @pytest.mark.gpu(count=2)
        def test_dual_gpu(gpu_lock):
            # Requires 2 GPUs
            ...
            
        @pytest.mark.no_gpu_lock
        def test_cpu_only():
            # No GPU lock needed
            ...
    """
    # Check if GPU lock should be disabled
    if request.node.get_closest_marker("no_gpu_lock"):
        logger.debug("GPU lock disabled for this test")
        yield None
        return
    
    # Get required GPU count from markers
    gpu_count = _get_gpu_count_from_markers(request.node)
    logger.debug(f"Test requires {gpu_count} GPU(s)")
    
    # Use GPU lock with specified count
    with gpu_lock_context(gpu_count) as lock_fds:
        yield lock_fds


@pytest.hookimpl(tryfirst=True)
def pytest_collection_modifyitems(config, items):
    """
    Add synthetic gpu_count_N markers before pytest applies -m selection.

    - @pytest.mark.gpu           -> also add @pytest.mark.gpu_count_1
    - @pytest.mark.gpu(count=4)  -> also add @pytest.mark.gpu_count_4
    """
    for item in items:
        gpu_marker = item.get_closest_marker("gpu")
        if not gpu_marker:
            continue

        count = gpu_marker.kwargs.get("count", 1)
        try:
            count = int(count)
        except (TypeError, ValueError):
            count = 1

        synthetic_name = f"gpu_count_{count}"
        item.add_marker(getattr(pytest.mark, synthetic_name))
