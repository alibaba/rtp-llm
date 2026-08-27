"""Generic Triton autotune cache for rtp-llm.

Public surface:

  cuda_cached_autotune(...)   - drop-in replacement for `triton.autotune` that
                                consults checked-in JSON configs under
                                `autotune_cache/configs/{GPU}/{kernel}.json`
                                before falling back to Triton benchmark.
  cached_autotune(...)        - same as above but assumes CUDA is available
                                (no torch.cuda.is_available guard).
  CacheMode                   - enum of lookup modes (disabled / cached).
  KernelConfigFile            - validated in-memory representation of a
                                {kernel}.json file.
  CachedAutotuner             - Triton Autotuner subclass; usually you want
                                the decorator instead.
  get_config_dir() / get_gpu_info()
                              - resolve where this process reads JSON from.
  load_cached_default_config(kernel_name)
                              - return the default_config dict for a kernel,
                                or None. Exposed for tests/tooling that want
                                the parsed JSON without launching a kernel.
  autotune_cache_kwargs       - spread at autotune call sites to opt in to
                                Triton's on-disk benchmark cache when
                                supported (Triton >= 3.4); no-op otherwise.
  SUPPORTS_AUTOTUNE_CACHE     - bool: whether triton.autotune accepts a
                                `cache_results` kwarg in this Triton version.

Environment variables:
  TRITON_AUTOTUNE_CACHE_MODE     - "disabled" (default) | "cached"
  TRITON_AUTOTUNE_CONFIG_DIR     - override JSON root
  TRITON_AUTOTUNE_GPU_NAME       - override GPU model id used for path lookup
"""

from rtp_llm.models_py.triton_kernels.autotune_cache.cache import (
    SUPPORTS_AUTOTUNE_CACHE,
    CachedAutotuner,
    CacheMode,
    KernelConfigFile,
    autotune_cache_kwargs,
    cached_autotune,
    get_config_dir,
    get_gpu_info,
    load_cached_default_config,
    sanitize_gpu_name,
)
from rtp_llm.models_py.triton_kernels.autotune_cache.decorator import (
    cuda_cached_autotune,
)

__all__ = [
    "CacheMode",
    "CachedAutotuner",
    "KernelConfigFile",
    "SUPPORTS_AUTOTUNE_CACHE",
    "autotune_cache_kwargs",
    "cached_autotune",
    "cuda_cached_autotune",
    "get_config_dir",
    "get_gpu_info",
    "load_cached_default_config",
    "sanitize_gpu_name",
]
