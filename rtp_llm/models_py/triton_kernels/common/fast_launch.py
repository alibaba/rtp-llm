"""Local Triton launch cache with a guarded CUDA launcher fast path.

Only explicit inference wrappers use this helper. No global JIT monkeypatch,
no retained tensor arguments, no stream caching, and no changed GPU program.
"""

from collections import OrderedDict

import torch
import triton
from rtp_llm.models_py import prefill_optimization as _prefill_optimization
from rtp_llm.models_py.kernels.cuda import prefill_cpu_ops as _cpu_ops
from triton.runtime.jit import JITFunction

try:
    from triton import knobs
except ImportError:
    knobs = None

ENABLED = _prefill_optimization.ENABLED
TRACE = False  # Diagnostic requests only; disabled during performance sampling.
counts = {"raw": 0, "compiled_fallback": 0, "jit_miss": 0}
_cache = OrderedDict()
_launchers = OrderedDict()
_native_launchers = OrderedDict()
_raw_stream = getattr(torch._C, "_cuda_getCurrentRawStream", None)
_hook_chain_type = getattr(knobs, "HookChain", None)


def hooks_idle():
    if knobs is None:
        return False
    enter, leave = knobs.runtime.launch_enter_hook, knobs.runtime.launch_exit_hook
    return (
        enter is None or (type(enter) is _hook_chain_type and not enter.calls)
    ) and (leave is None or (type(leave) is _hook_chain_type and not leave.calls))


def native_launcher(kernel):
    """Prepare only the standard CUDA ABI; retain unsupported launch modes."""
    cached = _native_launchers.get(kernel)
    if cached is not None:
        return cached
    runner = kernel.run
    cached = False
    if (
        triton.__version__.split(".")[:2] == ["3", "6"]
        and torch.version.hip is None
        and getattr(runner, "global_scratch_size", -1) == 0
        and getattr(runner, "profile_scratch_size", -1) == 0
        and not getattr(kernel.metadata, "tensordesc_meta", None)
        and not getattr(runner, "launch_cooperative_grid", True)
        and not getattr(runner, "launch_pdl", True)
        and getattr(kernel.metadata, "num_ctas", 1) == 1
    ):
        names = kernel.src.fn.arg_names

        def position(key):
            key = key[0] if isinstance(key, tuple) else key
            return names.index(key) if isinstance(key, str) else key

        constants = {position(k) for k in kernel.src.constants}
        types = {"i32": 1, "u32": 2, "i64": 3, "u64": 4, "fp32": 5, "fp64": 6}
        slots, codes = [], []
        for key, dtype in sorted(
            kernel.src.signature.items(), key=lambda item: position(item[0])
        ):
            index = position(key)
            if index in constants or dtype == "constexpr":
                continue
            if not isinstance(dtype, str):
                break
            code = (
                0
                if isinstance(dtype, str) and dtype.startswith("*")
                else types.get(dtype)
            )
            if code is None:
                break
            slots.append(index)
            codes.append(code)
        else:
            library = _cpu_ops.native()
            if library is not None and len(slots) <= 64:
                cached = library.NativeLauncher(
                    kernel.function,
                    kernel.metadata.shared,
                    kernel.metadata.num_warps,
                    slots,
                    codes,
                    kernel,
                )
    if len(_native_launchers) >= 512:
        _native_launchers.popitem(last=False)
    _native_launchers[kernel] = cached
    return cached


def launch_compiled(kernel, grid, args):
    grid = tuple(grid) + (1,) * (3 - len(grid))
    if ENABLED and _raw_stream is not None and knobs is not None:
        if _cpu_ops.NATIVE_LAUNCH and hooks_idle():
            launcher = native_launcher(kernel)
            if launcher:
                device = next(
                    a.device.index for a in args if isinstance(a, torch.Tensor)
                )
                launcher.launch(args, grid, _raw_stream(device))
                if TRACE:
                    counts["raw"] += 1
                return
        prepared = _launchers.get(kernel)
        if prepared is None:
            runner = kernel.run
            prepared = False
            # Prepare only the known CUDA ABI without scratch/TMA arguments.
            if (
                hasattr(runner, "launch")
                and getattr(runner, "global_scratch_size", -1) == 0
                and getattr(runner, "profile_scratch_size", -1) == 0
                and not getattr(kernel.metadata, "tensordesc_meta", None)
            ):
                device = next(
                    (a.device.index for a in args if isinstance(a, torch.Tensor)), None
                )
                if device is not None:
                    prepared = (
                        runner.launch,
                        device,
                        kernel.function,
                        runner.launch_cooperative_grid,
                        runner.launch_pdl,
                        kernel.packed_metadata,
                    )
            if len(_launchers) >= 512:
                _launchers.popitem(last=False)
            _launchers[kernel] = prepared
        # Instrumentation may change after warmup, so check its hooks per call.
        if prepared and hooks_idle():
            run, device, function, cooperative, pdl, metadata = prepared
            run(
                *grid,
                _raw_stream(device),
                function,
                cooperative,
                pdl,
                None,
                None,
                metadata,
                None,
                None,
                None,
                *args,
            )
            if TRACE:
                counts["raw"] += 1
            return
    if TRACE:
        counts["compiled_fallback"] += 1
    kernel[grid](*args)


def launch(jit, grid, args, **options):
    if (
        not ENABLED
        or callable(grid)
        or not isinstance(jit, JITFunction)
        or getattr(jit, "pre_run_hooks", None)
        or torch.compiler.is_compiling()
    ):
        return jit[grid](*args, **options)
    # Preserve every scalar specialization. Tensor argument types, device and
    # pointer alignment are the JIT specialization properties at these sites.
    library = _cpu_ops.native() if _cpu_ops.NATIVE_LAUNCH else None
    signature = (
        library.specialization_key(args)
        if library is not None
        else tuple(
            (
                (a.dtype, a.device.index, a.data_ptr() % 16)
                if isinstance(a, torch.Tensor)
                else (type(a), a)
            )
            for a in args
        )
    )
    key = (jit, signature, tuple(options.items()))
    kernel = _cache.get(key)
    if kernel is None:
        if TRACE:
            counts["jit_miss"] += 1
        kernel = jit[grid](*args, **options)
        if len(_cache) >= 512:
            _cache.popitem(last=False)
        _cache[key] = kernel
    else:
        _cache.move_to_end(key)
        launch_compiled(kernel, grid, args)
