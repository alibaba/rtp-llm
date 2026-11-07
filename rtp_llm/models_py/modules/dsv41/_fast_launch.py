"""Direct-launch cache for hot 1D-grid triton kernels.

JITFunction.run re-binds arguments, re-checks specialization and re-hashes
the cache key on every call. Kernels launched thousands of times per second
per rank with stable constexprs and 16B-aligned torch tensors can instead
cache the CompiledKernel per specialization class and launch through
CompiledKernel.__getitem__(grid), skipping all of it.

Any deviation from a warmed class (misaligned tensor, unusual int value)
simply lands in a new cache entry warmed on first sight — never wrong, just
slower for that one call. The warmed kernel is produced by the same
JITFunction binding path (``warmup``), so the binary is identical to what
the regular call would select.
"""

import torch


class FastKernel1D:
    """Cache and direct-launch a 1D-grid triton JITFunction.

    Args come in full signature order (constexpr positions included); the
    helper splits them by the kernel's own parameter metadata, so the
    constexpr values always land in the cache key. (They are still passed
    through to the compiled launcher — triton's generated C expects the
    full bound signature and skips constexpr slots by position.)
    """

    __slots__ = ("_jit_fn", "_launch_kw", "_constexpr_idx", "_cache")

    def __init__(self, jit_fn, **launch_kw):
        self._jit_fn = jit_fn
        self._launch_kw = launch_kw
        self._constexpr_idx = {
            i for i, p in enumerate(jit_fn.params) if p.is_constexpr
        }
        self._cache = {}

    def _split(self, args):
        key = []
        runtime = []
        for i, a in enumerate(args):
            if i in self._constexpr_idx:
                key.append(a)
                continue
            runtime.append(a)
            if isinstance(a, torch.Tensor):
                # dtype and 16B alignment mirror triton's specialization key.
                key.append((a.dtype, a.data_ptr() % 16 == 0))
            elif isinstance(a, bool):
                key.append(a)
            elif isinstance(a, int):
                # triton specializes ints on ==1 and %16==0.
                key.append((a == 1, a % 16 == 0))
        return tuple(key), runtime

    def __call__(self, rows, *args):
        if rows == 0:
            return
        device = next(a.device for a in args if isinstance(a, torch.Tensor))
        key, runtime = self._split(args)
        cache_key = (device.index, key)
        ck = self._cache.get(cache_key)
        if ck is None:
            ck = self._jit_fn.warmup(*args, grid=(1,), **self._launch_kw)
            self._cache[cache_key] = ck
        stream = torch.cuda.current_stream(device).cuda_stream
        # triton's launcher expects the full bound signature (constexpr
        # slots included); the C side skips them by position.
        ck[(rows, 1, 1)](*args, stream=stream)
