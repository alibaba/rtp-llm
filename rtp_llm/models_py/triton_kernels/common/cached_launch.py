"""Bounded warm-launch cache for fixed-signature CUDA layout kernels.

Only tensor pointers followed by fixed-type numeric scalars are supported.
Every scalar (including strides/constexprs), pointer dtype/alignment, grid and
current device participates in the key. Neither tensors nor streams are cached.
The compiled Triton launcher still obtains the current stream and runs launch
hooks. Debug, instrumentation and JIT pre-run hooks use the normal JIT path.
"""

import torch
import triton

try:
    from triton import knobs
except ImportError:  # Older supported CUDA12 environments lack this API.
    knobs = None


class CachedLaunch:
    def __init__(self, kernel, **options):
        self.kernel = kernel
        self.options = options
        self.cache = {}
        self.enabled = (
            knobs is not None
            and triton.__version__.split(".")[:2] == ["3", "6"]
            and isinstance(kernel, triton.JITFunction)
        )

    def __call__(self, grid, tensors, scalars):
        kernel = self.kernel
        if (
            not self.enabled
            or kernel.pre_run_hooks
            or knobs.runtime.debug
            or knobs.compilation.instrumentation_mode
            or knobs.runtime.jit_cache_hook is not None
        ):
            kernel[grid](*tensors, *scalars, **self.options)
            return
        device = torch.cuda.current_device()
        key = (
            device,
            grid,
            tuple((x.dtype, x.data_ptr() % 16) for x in tensors),
            scalars,
        )
        entry = self.cache.get(key)
        if entry is None:
            compiled = kernel[grid](*tensors, *scalars, **self.options)
            # Bound retained metadata even for unbounded eager request shapes.
            if len(self.cache) >= 256:
                self.cache.clear()
            self.cache[key] = (compiled, compiled.run, compiled[grid])
        else:
            compiled, run, launch = entry
            stream = triton.runtime.driver.active.get_current_stream(device)
            enter = knobs.runtime.launch_enter_hook
            leave = knobs.runtime.launch_exit_hook
            if (
                type(enter) is knobs.HookChain
                and type(leave) is knobs.HookChain
                and not enter.calls
                and not leave.calls
            ):
                # Triton 3.6's normal runner creates lazy metadata and invokes
                # two empty HookChains even with no profiler. Keep its backend
                # launcher (including scratch allocation), but omit those no-ops
                # and the redundant current-device lookup. Never cache streams.
                run(
                    *grid,
                    stream,
                    compiled.function,
                    compiled.packed_metadata,
                    None,
                    None,
                    None,
                    *tensors,
                    *scalars,
                )
            else:
                launch(*tensors, *scalars, stream=stream)
