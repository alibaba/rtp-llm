
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
    Configure pytest to support marker parameter filtering.
    
    This hook runs before collection and modifies the marker expression
    to make it compatible with pytest's standard filtering.
    
    Strategy:
    1. Extract parameter filters like "gpu(count=1)" from the expression
    2. Replace them with simple marker names (e.g., "gpu(count=1)" -> "gpu")
    3. Store the original expression for later use in pytest_collection_modifyitems
    """
    marker_expr = config.option.markexpr
    if not marker_expr:
        return
    
    # Check if marker expression contains parameter filters
    param_pattern = r'(\w+)\(count=(\d+)\)'
    matches = re.findall(param_pattern, marker_expr)
    
    if not matches:
        return
    
    # Store original expression and filters for later filtering
    config._original_marker_expr = marker_expr
    config._marker_param_filters = matches
    
    # Replace parameter filters with simple marker names for pytest's standard filtering
    # e.g., "A10 and gpu(count=1) and not smoke" -> "A10 and gpu and not smoke"
    modified_expr = marker_expr
    for marker_name, count in matches:
        # Replace marker(count=N) with marker
        modified_expr = re.sub(
            rf'\b{re.escape(marker_name)}\(count={re.escape(count)}\)',
            marker_name,
            modified_expr
        )
    
    # Update the marker expression
    config.option.markexpr = modified_expr
    logger.debug(f"Modified marker expression: {marker_expr} -> {modified_expr}")


@pytest.fixture(scope="function")
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


def pytest_collection_modifyitems(config, items):
    """
    Modify test collection to support filtering by marker parameters.
    
    This hook extends pytest's marker filtering to support parameter-based filtering.
    
    Key feature: If a marker like 'gpu' is used without count parameter,
    it defaults to count=1. So:
        - @pytest.mark.gpu  → count=1 (default)
        - @pytest.mark.gpu(count=2)  → count=2
    
    Strategy:
    1. pytest_configure already modified the marker expression for standard filtering
    2. This hook applies parameter-based filtering on the remaining items
    
    Usage examples:
        pytest -m "gpu(count=1)"           # Only tests with gpu(count=1) or gpu (default count=1)
        pytest -m "gpu and not gpu(count=2)"  # GPU tests but not count=2
        pytest -m "A10 and gpu(count=1)"    # A10 tests with gpu count=1
    """
    # Get parameter filters stored in pytest_configure
    if not hasattr(config, '_marker_param_filters'):
        return
    
    matches = config._marker_param_filters
    if not matches:
        return
    
    # Get original marker expression (before modification)
    original_expr = getattr(config, '_original_marker_expr', None)
    
    # Apply parameter-based filtering
    # Note: This runs AFTER pytest's standard marker filtering
    # So items here already match the standard marker expression (like "A10 and gpu and not smoke")
    filtered_items = []
    for item in items:
        should_include = True
        
        for marker_name, expected_count_str in matches:
            marker = item.get_closest_marker(marker_name)
            expected_count = int(expected_count_str)
            
            if marker:
                # Get actual count from marker, default to 1 if not specified
                actual_count = marker.kwargs.get("count")
                if actual_count is not None:
                    actual_count = int(actual_count)
                else:
                    # Default count is 1 for markers without count parameter
                    actual_count = 1
                
                # Check if this is a positive or negative filter
                # We need to check the original expression for "not" patterns
                if original_expr:
                    not_pattern = f"not {marker_name}\\(count={expected_count}\\)"
                    is_negative = re.search(not_pattern, original_expr)
                else:
                    # If we don't have original expr, assume positive filter
                    is_negative = False
                
                if is_negative:
                    # Negative filter: exclude if count matches
                    if actual_count == expected_count:
                        should_include = False
                        break
                else:
                    # Positive filter: include only if count matches
                    if actual_count != expected_count:
                        should_include = False
                        break
            else:
                # Item doesn't have this marker
                # For positive filter, exclude items without the marker
                # For negative filter, include items without the marker
                if original_expr:
                    not_pattern = f"not {marker_name}\\(count={expected_count}\\)"
                    is_negative = re.search(not_pattern, original_expr)
                else:
                    is_negative = False
                
                if not is_negative:
                    # Positive filter: item doesn't have marker, exclude it
                    should_include = False
                    break
        
        if should_include:
            filtered_items.append(item)
    
    # Replace items with filtered list
    items[:] = filtered_items
