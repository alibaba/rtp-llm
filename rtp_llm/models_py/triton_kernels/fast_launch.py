"""Low-overhead steady-state launcher for warmed-up Triton ``JITFunction`` kernels.

On warmed CUDA kernels (no compile miss, no CUDA graph capture, no debug hooks,
no TMA tensor descriptors or scratch allocations), calls Triton's C driver
launch entrypoint directly via a specialized per-specialization MRU closure
bound directly onto ``jit_fn._rtp_fast_fn`` on repeated hits, with zero
per-call cache-key building, signature binding, or attribute lookup churn.

Every cache miss, cache clear (``device_caches.clear()`` or per-device
``triton_cache.clear()``), CUDA graph capture, special argument type, scratch
allocation, or runtime/debug hook automatically falls back to standard
``jit_fn[grid](*args, **kwargs)``.
"""

from __future__ import annotations

import contextvars
from typing import Any, Callable, Dict, Optional, Tuple

_CAPTURING_HINT: contextvars.ContextVar[Optional[bool]] = contextvars.ContextVar(
    "rtp_fast_launch_capturing_hint", default=None
)

_RUNTIME_INITIALIZED = False
_TORCH_TENSOR_CLS: Any = None
_TORCH_CUDA: Any = None
_JIT_FUNCTION_CLS: Any = None
_KNOBS_RT: Any = None
_KNOBS_COMP: Any = None
_C_LAUNCH: Any = None
_GET_DEVICE: Any = None
_GET_STREAM: Any = None
_FALLBACK_DISPATCH_FN: Any = None

# Triton CUDA kernel launch kwargs that are valid as launch options (never kernel
# params unless shadowed by kernel signature). Keep in sync with CUDAOptions.
_KNOWN_LAUNCH_OPTIONS = frozenset(
    {
        "num_warps",
        "num_ctas",
        "num_stages",
        "num_buffers_warp_spec",
        "num_consumer_groups",
        "reg_dec_producer",
        "reg_inc_consumer",
        "maxnreg",
        "enable_fp_fusion",
        "backend_compile_args",
        "extern_libs",
        "debug",
        "sanitize_overflow",
        "instrumentation_mode",
        "launch_cooperative_grid",
        "launch_pdl",
        "supported_fp8_dtypes",
        "deprecated_fp8_dot_operand_dtypes",
        "default_dot_input_precision",
        "allowed_dot_input_precisions",
        "max_num_imprecise_acc_default",
        "ftz",
    }
)


_HAS_SHARED_DRIVER_LAUNCH = False


def _init_runtime() -> bool:
    global _RUNTIME_INITIALIZED
    global _TORCH_TENSOR_CLS
    global _TORCH_CUDA
    global _JIT_FUNCTION_CLS
    global _KNOBS_RT
    global _KNOBS_COMP
    global _C_LAUNCH
    global _HAS_SHARED_DRIVER_LAUNCH
    global _GET_DEVICE
    global _GET_STREAM

    if _RUNTIME_INITIALIZED:
        return True
    try:
        import torch
        from triton import knobs
        from triton.runtime.driver import driver
        from triton.runtime.jit import JITFunction
    except ImportError:
        return False

    try:
        active_driver = driver.active
        c_launch = getattr(getattr(active_driver, "utils", None), "launch", None)
        get_device = active_driver.get_current_device
        get_stream = active_driver.get_current_stream
    except Exception:
        return False

    _TORCH_TENSOR_CLS = torch.Tensor
    _TORCH_CUDA = torch.cuda
    _JIT_FUNCTION_CLS = JITFunction
    _KNOBS_RT = knobs.runtime
    _KNOBS_COMP = knobs.compilation
    _C_LAUNCH = c_launch
    _HAS_SHARED_DRIVER_LAUNCH = c_launch is not None
    _GET_DEVICE = get_device
    _GET_STREAM = get_stream
    _RUNTIME_INITIALIZED = True
    return True


def is_cuda_stream_capturing() -> bool:
    """Return whether CUDA stream capture is active, honoring ``fast_launch_capture_scope``."""
    hint = _CAPTURING_HINT.get()
    if hint is not None:
        return hint
    cuda_mod = _TORCH_CUDA
    if cuda_mod is None:
        import torch

        cuda_mod = torch.cuda
    return bool(cuda_mod.is_available() and cuda_mod.is_current_stream_capturing())


class _FastLaunchCaptureScope:
    """Reusable context manager setting ``_CAPTURING_HINT`` across sequential or nested regions."""

    __slots__ = ("_capturing", "_tokens")

    def __init__(self, capturing: Optional[bool]) -> None:
        self._capturing = capturing
        self._tokens: list[contextvars.Token[Optional[bool]]] = []

    def __enter__(self) -> None:
        self._tokens.append(_CAPTURING_HINT.set(self._capturing))

    def __exit__(self, _exc_type: Any, _exc_val: Any, _exc_tb: Any) -> bool:
        _CAPTURING_HINT.reset(self._tokens.pop())
        return False


def fast_launch_capture_scope(capturing: Optional[bool]) -> _FastLaunchCaptureScope:
    """Scoped hint telling ``fast_launch`` / ``is_cuda_stream_capturing`` whether capture is active.

    Passing ``False`` after checking ``torch.cuda.is_current_stream_capturing()``
    once at forward entry eliminates per-launch ``cudaStreamIsCapturing`` calls
    while ``True`` enforces fallback to ``jit_fn[grid](...)``.
    """
    return _FastLaunchCaptureScope(capturing)


def clear_fast_launch_cache(jit_fn: Optional[Any] = None) -> None:
    """Clear compiled fast-launch dispatch state for ``jit_fn``."""
    if jit_fn is not None:
        if hasattr(jit_fn, "_rtp_fl"):
            delattr(jit_fn, "_rtp_fl")
        if hasattr(jit_fn, "_rtp_fast_fn"):
            delattr(jit_fn, "_rtp_fast_fn")


def _has_hooks_or_debug(jit_fn: Any) -> bool:
    rt = _KNOBS_RT
    return bool(
        jit_fn.debug
        or jit_fn.pre_run_hooks
        or jit_fn.used_global_vals
        or jit_fn.launch_metadata is not None
        or rt.debug
        or bool(
            getattr(
                rt.add_stages_inspection_hook, "calls", rt.add_stages_inspection_hook
            )
        )
        or bool(getattr(rt.launch_enter_hook, "calls", rt.launch_enter_hook))
        or bool(getattr(rt.launch_exit_hook, "calls", rt.launch_exit_hook))
        or _KNOBS_COMP.instrumentation_mode
    )


def _bind_call_shape(
    jit_fn: Any, n_pos: int, kw_names: Tuple[str, ...]
) -> Optional[Tuple[list[Tuple[str, Any]], list[str]]]:
    params = jit_fn.params
    param_by_name = {p.name: p for p in params}
    n_params = len(params)

    slot_source: list[Optional[Tuple[str, Any]]] = [None] * n_params
    if n_pos > n_params:
        return None
    for i in range(n_pos):
        slot_source[i] = ("pos", i)

    option_kws: list[str] = []
    for k in kw_names:
        p = param_by_name.get(k)
        if p is not None:
            if slot_source[p.num] is not None:
                return None
            slot_source[p.num] = ("kw", k)
        elif k in _KNOWN_LAUNCH_OPTIONS:
            if k in ("debug", "instrumentation_mode", "extern_libs"):
                return None
            option_kws.append(k)
        else:
            return None

    resolved: list[Tuple[str, Any]] = []
    for i, src in enumerate(slot_source):
        if src is None:
            if not params[i].has_default:
                return None
            resolved.append(("default", i))
        else:
            resolved.append(src)
    option_kws.sort()
    return resolved, option_kws


def _compile_mru_fast_closure(
    jit_fn: Any,
    n_pos: int,
    kw_names: Tuple[str, ...],
    device: int,
    dev_entry: Any,
    triton_cache: Any,
    record: Tuple[Any, ...],
    mru_kinds: Tuple[Any, ...],
    mru_vals: Tuple[Any, ...],
) -> Optional[Callable[..., Any]]:
    bound = _bind_call_shape(jit_fn, n_pos, kw_names)
    if bound is None:
        return None
    slot_source, option_kws = bound
    params = jit_fn.params
    is_shared, fn_handle, coop, pdl, pmeta, ann, sig, ck, c_launch = record

    def _fallback_call(grid: Any, args: Tuple[Any, ...], kwargs: Dict[str, Any]) -> Any:
        return _fast_launch_slow(jit_fn, grid, args, kwargs)

    ns: Dict[str, Any] = {
        "_cap_get": _CAPTURING_HINT.get,
        "_cuda_mod": _TORCH_CUDA,
        "_rt": _KNOBS_RT,
        "_comp": _KNOBS_COMP,
        "_jit_fn": jit_fn,
        "_dev_caches": jit_fn.device_caches,
        "_m_dev": device,
        "_m_dev_entry": dev_entry,
        "_m_tc": triton_cache,
        "_get_dev": _GET_DEVICE,
        "_get_stream": _GET_STREAM,
        "_c_launch": c_launch,
        "_m_fn": fn_handle,
        "_m_coop": coop,
        "_m_pdl": pdl,
        "_m_pmeta": pmeta,
        "_m_ann": ann,
        "_m_sig": sig,
        "_m_ck": ck,
        "_fallback": _fallback_call,
        "_Tensor": _TORCH_TENSOR_CLS,
        "_getattr": getattr,
    }
    for i, src in enumerate(slot_source):
        if src[0] == "default":
            ns[f"_d{i}"] = params[i].default

    bind_lines: list[str] = []
    if n_pos == 1:
        bind_lines.append(
            "    (v0,) = args" if slot_source[0] == ("pos", 0) else "    (p0,) = args"
        )
    elif n_pos > 1:
        pos_names = [f"p{j}" for j in range(n_pos)]
        bind_lines.append(f"    {', '.join(pos_names)} = args")

    for i, src in enumerate(slot_source):
        kind, ref = src
        if kind == "pos":
            if not (n_pos == 1 and i == 0):
                bind_lines.append(f"    v{i} = p{ref}")
        elif kind == "kw":
            bind_lines.append(f"    v{i} = kwargs[{ref!r}]")
        else:
            bind_lines.append(f"    v{i} = _d{i}")

    for oi, ok in enumerate(option_kws):
        bind_lines.append(f"    o{oi} = kwargs[{ok!r}]")

    check_lines: list[str] = []
    arg_exprs: list[str] = []
    val_idx = 0
    for i, p in enumerate(params):
        v = f"v{i}"
        if p.is_constexpr:
            ktag = mru_kinds[i]
            val = mru_vals[val_idx]
            val_idx += 1
            ct_name = f"_ct{i}"
            cv_name = f"_cv{i}"
            ns[ct_name] = type(val)
            ns[cv_name] = val
            check_lines.append(
                f"    if {v} is not {cv_name} and (type({v}) is not {ct_name} or {v} != {cv_name}): return _fallback(grid, args, kwargs)"
            )
            arg_exprs.append(v)
        else:
            ktag = mru_kinds[i]
            val = mru_vals[val_idx]
            val_idx += 1
            if ktag == "tensor_aligned":
                dt_name = f"_dt{i}"
                ns[dt_name] = val
                check_lines.extend(
                    [
                        f"    if type({v}) is not _Tensor or not {v}.is_cuda or {v}.dtype is not {dt_name}: return _fallback(grid, args, kwargs)",
                        f"    _a{i} = {v}.data_ptr()",
                        f"    if _a{i} & 15: return _fallback(grid, args, kwargs)",
                    ]
                )
                arg_exprs.append(f"_a{i}")
            elif ktag == "tensor_unaligned":
                dt_name = f"_dt{i}"
                ns[dt_name] = val
                check_lines.extend(
                    [
                        f"    if type({v}) is not _Tensor or not {v}.is_cuda or {v}.dtype is not {dt_name}: return _fallback(grid, args, kwargs)",
                        f"    _a{i} = {v}.data_ptr()",
                        f"    if not (_a{i} & 15): return _fallback(grid, args, kwargs)",
                    ]
                )
                arg_exprs.append(f"_a{i}")
            elif ktag == "tensor_dns":
                dt_name = f"_dt{i}"
                ns[dt_name] = val
                check_lines.extend(
                    [
                        f"    if type({v}) is not _Tensor or not {v}.is_cuda or {v}.dtype is not {dt_name}: return _fallback(grid, args, kwargs)",
                        f"    _a{i} = {v}.data_ptr()",
                    ]
                )
                arg_exprs.append(f"_a{i}")
            elif ktag == "int_eq1":
                check_lines.append(
                    f"    if type({v}) is not int or {v} != 1: return _fallback(grid, args, kwargs)"
                )
                arg_exprs.append(v)
            elif ktag == "int_spec":
                spec_code = int(val)
                width = spec_code >> 2
                align_tag = spec_code & 3
                if width == 0:
                    w_cond = f"-2147483648 <= {v} <= 2147483647"
                elif width == 1:
                    w_cond = f"-9223372036854775808 <= {v} <= 9223372036854775807"
                else:
                    w_cond = (
                        f"({v} < -9223372036854775808 or {v} > 9223372036854775807)"
                    )
                if p.do_not_specialize:
                    check_lines.append(
                        f"    if type({v}) is not int or not ({w_cond}): return _fallback(grid, args, kwargs)"
                    )
                elif p.do_not_specialize_on_alignment:
                    # align_tag == 3 (not 1)
                    check_lines.append(
                        f"    if type({v}) is not int or not ({w_cond}) or {v} == 1: return _fallback(grid, args, kwargs)"
                    )
                elif align_tag == 2:
                    check_lines.append(
                        f"    if type({v}) is not int or not ({w_cond}) or ({v} & 15): return _fallback(grid, args, kwargs)"
                    )
                else:
                    # align_tag == 3: neither 1 nor %16==0
                    check_lines.append(
                        f"    if type({v}) is not int or not ({w_cond}) or {v} == 1 or not ({v} & 15): return _fallback(grid, args, kwargs)"
                    )
                arg_exprs.append(v)
            elif ktag == "none":
                check_lines.append(
                    f"    if {v} is not None: return _fallback(grid, args, kwargs)"
                )
                arg_exprs.append("None")
            elif ktag == "bool":
                check_lines.append(
                    f"    if type({v}) is not bool: return _fallback(grid, args, kwargs)"
                )
                arg_exprs.append(v)
            elif ktag == "float":
                check_lines.append(
                    f"    if type({v}) is not float: return _fallback(grid, args, kwargs)"
                )
                arg_exprs.append(v)
            else:
                return None

    for oi in range(len(option_kws)):
        val = mru_vals[val_idx]
        val_idx += 1
        ot_name = f"_okt{oi}"
        ov_name = f"_okv{oi}"
        ns[ot_name] = type(val)
        ns[ov_name] = val
        check_lines.append(
            f"    if o{oi} is not {ov_name} and (type(o{oi}) is not {ot_name} or o{oi} != {ov_name}): return _fallback(grid, args, kwargs)"
        )

    if is_shared:
        if len(arg_exprs) == 0:
            kargs_expr = "()"
        elif len(arg_exprs) == 1:
            kargs_expr = f"({arg_exprs[0]},)"
        else:
            kargs_expr = f"({', '.join(arg_exprs)})"
    else:
        flat_kargs_expr = ", ".join(arg_exprs)

    meta_dict_expr = (
        "{" + ", ".join(f"{p.name!r}: v{i}" for i, p in enumerate(params)) + "}"
    )

    # Move all namespace items into LOAD_FAST defaults when possible (Python 3.10
    # accesses defaults via fastlocals index instead of globals dict hash).
    default_params = ", ".join(f"{k}={k}" for k in ns.keys())

    lines = [
        f"def _mru_fast(grid, args, kwargs, {default_params}):",
        "    _cap = _cap_get()",
        "    if _cap or (_cap is None and _cuda_mod.is_current_stream_capturing()):",
        "        return _fallback(grid, args, kwargs)",
        "    if (",
        "        _jit_fn.debug",
        "        or _jit_fn.pre_run_hooks",
        "        or _jit_fn.used_global_vals",
        "        or _jit_fn.launch_metadata is not None",
        "        or _rt.debug",
        "        or _getattr(_rt.add_stages_inspection_hook, 'calls', _rt.add_stages_inspection_hook)",
        "        or _getattr(_rt.launch_enter_hook, 'calls', _rt.launch_enter_hook)",
        "        or _getattr(_rt.launch_exit_hook, 'calls', _rt.launch_exit_hook)",
        "        or _comp.instrumentation_mode",
        "    ):",
        "        return _fallback(grid, args, kwargs)",
        f"    if len(args) != {n_pos} or len(kwargs) != {len(kw_names)}:",
        "        return _fallback(grid, args, kwargs)",
        "    _dev = _get_dev()",
        "    if _dev != _m_dev or _dev_caches.get(_dev) is not _m_dev_entry or not _m_tc:",
        "        return _fallback(grid, args, kwargs)",
    ]
    if option_kws or any(s[0] == "kw" for s in slot_source):
        lines.append("    try:")
        lines.extend("    " + ln for ln in bind_lines)
        lines.append("    except KeyError:")
        lines.append("        return _fallback(grid, args, kwargs)")
    else:
        lines.extend(bind_lines)

    lines.extend(check_lines)
    lines.extend(
        [
            "    _gt = type(grid)",
            "    if _gt is tuple:",
            "        _gn = len(grid)",
            "        if _gn == 1:",
            "            _g0 = grid[0]; _g1 = 1; _g2 = 1",
            "        elif _gn == 2:",
            "            _g0, _g1 = grid; _g2 = 1",
            "        elif _gn == 3:",
            "            _g0, _g1, _g2 = grid",
            "        else:",
            "            return _fallback(grid, args, kwargs)",
            "    elif callable(grid):",
            f"        _g = grid({meta_dict_expr})",
            "        _gn = len(_g)",
            "        if _gn == 1:",
            "            _g0 = _g[0]; _g1 = 1; _g2 = 1",
            "        elif _gn == 2:",
            "            _g0, _g1 = _g; _g2 = 1",
            "        elif _gn == 3:",
            "            _g0, _g1, _g2 = _g",
            "        else:",
            "            return _fallback(grid, args, kwargs)",
            "    elif _gt is list:",
            "        _gn = len(grid)",
            "        if _gn == 1:",
            "            _g0 = grid[0]; _g1 = 1; _g2 = 1",
            "        elif _gn == 2:",
            "            _g0, _g1 = grid; _g2 = 1",
            "        elif _gn == 3:",
            "            _g0, _g1, _g2 = grid",
            "        else:",
            "            return _fallback(grid, args, kwargs)",
            "    else:",
            "        return _fallback(grid, args, kwargs)",
        ]
    )
    if is_shared:
        lines.extend(
            [
                "    _c_launch(",
                "        _g0,",
                "        _g1,",
                "        _g2,",
                "        _get_stream(_m_dev),",
                "        _m_fn,",
                "        _m_coop,",
                "        _m_pdl,",
                "        _m_pmeta,",
                "        None,",
                "        None,",
                "        None,",
                "        None,",
                "        None,",
                "        _m_ann,",
                "        _m_sig,",
                f"        {kargs_expr},",
                "    )",
                "    return _m_ck",
            ]
        )
    else:
        call_lines = [
            "    _c_launch(",
            "        _g0,",
            "        _g1,",
            "        _g2,",
            "        _get_stream(_m_dev),",
            "        _m_fn,",
            "        _m_coop,",
            "        _m_pdl,",
            "        None,",
            "        None,",
            "        _m_pmeta,",
            "        None,",
            "        None,",
            "        None,",
        ]
        if flat_kargs_expr:
            call_lines.append(f"        {flat_kargs_expr},")
        call_lines.extend(
            [
                "    )",
                "    return _m_ck",
            ]
        )
        lines.extend(call_lines)

    code = "\n".join(lines)
    exec(code, ns)
    return ns["_mru_fast"]


def _fallback_record(
    jit_fn: Any,
    grid: Any,
    args: Tuple[Any, ...],
    kwargs: Dict[str, Any],
    device: int,
    cache_key: Any,
    entries: Dict[int, Tuple[Any, Dict[Any, Any]]],
    mru_meta: Optional[Tuple[Any, ...]],
):
    ck = jit_fn[grid](*args, **kwargs)
    jit_fn._rtp_fast_fn = None
    if ck is None or cache_key is None:
        return ck
    dev_entry = jit_fn.device_caches.get(device)
    if dev_entry is None:
        return ck
    triton_cache = dev_entry[0]
    if not triton_cache:
        return ck
    launcher = getattr(ck, "_run", None)
    fn_handle = getattr(ck, "function", None)
    if (
        launcher is None
        or fn_handle is None
        or bool(getattr(ck.metadata, "tensordesc_meta", None))
        or launcher.global_scratch_size != 0
        or launcher.profile_scratch_size != 0
        or getattr(launcher, "gsan_enabled", False)
    ):
        return ck
    src = getattr(ck, "src", None)
    sig = getattr(src, "signature", None)
    if sig is None or len(sig) != len(jit_fn.params):
        return ck
    if any(isinstance(s, str) and s.startswith("tensordesc") for s in sig.values()):
        return ck
    c_launch = getattr(launcher, "launch", None)
    if not callable(c_launch):
        return ck

    is_shared = hasattr(launcher, "arg_annotations") and hasattr(
        launcher, "kernel_signature"
    )
    if is_shared:
        if not _HAS_SHARED_DRIVER_LAUNCH or c_launch is not _C_LAUNCH:
            return ck
        record = (
            True,
            fn_handle,
            launcher.launch_cooperative_grid,
            launcher.launch_pdl,
            ck.packed_metadata,
            launcher.arg_annotations,
            launcher.kernel_signature,
            ck,
            c_launch,
        )
    else:
        record = (
            False,
            fn_handle,
            launcher.launch_cooperative_grid,
            launcher.launch_pdl,
            ck.packed_metadata,
            None,
            None,
            ck,
            c_launch,
        )

    dev_bucket = entries.get(device)
    if dev_bucket is None or dev_bucket[0] is not triton_cache:
        cache_map: Dict[Any, Any] = {}
        entries[device] = (triton_cache, cache_map)
    else:
        cache_map = dev_bucket[1]

    # [record, seen_count, mru_by_shape_and_kw, mru_meta]
    cache_map[cache_key] = [record, 1, {}, mru_meta]
    return ck


def _compile_shape_fallback_dispatcher(
    jit_fn: Any, n_pos: int, kw_names: Tuple[str, ...], entries: Dict[int, Any]
) -> Optional[Callable[..., Any]]:
    bound = _bind_call_shape(jit_fn, n_pos, kw_names)
    if bound is None:
        return None
    slot_source, option_kws = bound
    params = jit_fn.params
    shape_key = (n_pos, kw_names)

    ns: Dict[str, Any] = {
        "Tensor": _TORCH_TENSOR_CLS,
        "_c_launch": _C_LAUNCH,
        "_get_device": _GET_DEVICE,
        "_get_stream": _GET_STREAM,
        "_fallback_record": _fallback_record,
        "_compile_mru_fast_closure": _compile_mru_fast_closure,
        "jit_fn": jit_fn,
        "dev_caches": jit_fn.device_caches,
        "entries": entries,
        "n_pos": n_pos,
        "kw_names": kw_names,
        "shape_key": shape_key,
    }
    for i, src in enumerate(slot_source):
        if src[0] == "default":
            ns[f"_d{i}"] = params[i].default

    lines = ["def _shape_dispatch(grid, args, kwargs):"]
    if n_pos == 1:
        lines.append(
            "    (v0,) = args" if slot_source[0] == ("pos", 0) else "    (p0,) = args"
        )
    elif n_pos > 1:
        pos_names = [f"p{j}" for j in range(n_pos)]
        lines.append(f"    {', '.join(pos_names)} = args")

    for i, src in enumerate(slot_source):
        kind, ref = src
        if kind == "pos":
            if not (n_pos == 1 and i == 0):
                lines.append(f"    v{i} = p{ref}")
        elif kind == "kw":
            lines.append(f"    v{i} = kwargs[{ref!r}]")
        else:
            lines.append(f"    v{i} = _d{i}")

    for oi, ok in enumerate(option_kws):
        lines.append(f"    o{oi} = kwargs[{ok!r}]")

    key_vars: list[str] = []
    arg_vars: list[str] = []
    mru_kind_exprs: list[str] = []
    mru_val_exprs: list[str] = []
    for i, p in enumerate(params):
        v = f"v{i}"
        kv = f"k{i}"
        av = f"a{i}"
        arg_vars.append(av)
        if p.is_constexpr:
            lines.append(f"    {av} = {v}")
            lines.append(f"    {kv} = (type({v}), {v})")
            key_vars.append(kv)
            mru_kind_exprs.append("'constexpr'")
            mru_val_exprs.append(v)
        else:
            dns = p.do_not_specialize
            dns_oa = p.do_not_specialize_on_alignment
            lines.append(f"    _t{i} = type({v})")
            lines.append(f"    if _t{i} is Tensor:")
            lines.append(f"        if not {v}.is_cuda:")
            lines.append("            return jit_fn[grid](*args, **kwargs)")
            lines.append(f"        _ptr{i} = {v}.data_ptr()")
            lines.append(f"        {av} = _ptr{i}")
            lines.append(f"        _dt{i} = {v}.dtype")
            lines.append(f"        _mv{i} = _dt{i}")
            if dns or dns_oa:
                lines.append(f"        {kv} = _dt{i}")
                lines.append(f"        _mk{i} = 'tensor_dns'")
            else:
                lines.append(f"        _al{i} = (_ptr{i} & 15) == 0")
                lines.append(f"        {kv} = (_dt{i}, _al{i})")
                lines.append(
                    f"        _mk{i} = 'tensor_aligned' if _al{i} else 'tensor_unaligned'"
                )
            lines.append(f"    elif _t{i} is int:")
            lines.append(f"        {av} = {v}")
            lines.append(
                f"        _w{i} = 0 if -2147483648 <= {v} <= 2147483647 else "
                f"(1 if -9223372036854775808 <= {v} <= 9223372036854775807 else 2)"
            )
            if dns:
                lines.append(f"        _sc{i} = _w{i} << 2")
                lines.append(f"        _mk{i} = 'int_spec'")
            elif dns_oa:
                lines.append(f"        _at{i} = 1 if {v} == 1 else 3")
                lines.append(f"        _sc{i} = (_w{i} << 2) | _at{i}")
                lines.append(
                    f"        _mk{i} = 'int_eq1' if _at{i} == 1 else 'int_spec'"
                )
            else:
                lines.append(
                    f"        _at{i} = 1 if {v} == 1 else (2 if ({v} & 15) == 0 else 3)"
                )
                lines.append(f"        _sc{i} = (_w{i} << 2) | _at{i}")
                lines.append(
                    f"        _mk{i} = 'int_eq1' if _at{i} == 1 else 'int_spec'"
                )
            lines.append(f"        {kv} = (int, _sc{i})")
            lines.append(f"        _mv{i} = _sc{i}")
            lines.append(f"    elif {v} is None:")
            lines.append(f"        {av} = None")
            lines.append(f"        {kv} = None")
            lines.append(f"        _mk{i} = 'none'")
            lines.append(f"        _mv{i} = None")
            lines.append(f"    elif _t{i} is bool:")
            lines.append(f"        {av} = {v}")
            lines.append(f"        {kv} = bool")
            lines.append(f"        _mk{i} = 'bool'")
            lines.append(f"        _mv{i} = None")
            lines.append(f"    elif _t{i} is float:")
            lines.append(f"        {av} = {v}")
            lines.append(f"        {kv} = float")
            lines.append(f"        _mk{i} = 'float'")
            lines.append(f"        _mv{i} = None")
            lines.append("    else:")
            lines.append("        return jit_fn[grid](*args, **kwargs)")
            key_vars.append(kv)
            mru_kind_exprs.append(f"_mk{i}")
            mru_val_exprs.append(f"_mv{i}")

    option_exprs = [f"(type(o{oi}), o{oi})" for oi in range(len(option_kws))]
    for oi in range(len(option_kws)):
        mru_val_exprs.append(f"o{oi}")

    all_key_parts = key_vars + option_exprs
    if len(all_key_parts) == 0:
        lines.append("    cache_key = ()")
    elif len(all_key_parts) == 1:
        lines.append(f"    cache_key = ({all_key_parts[0]},)")
    else:
        lines.append(f"    cache_key = ({', '.join(all_key_parts)})")

    if len(arg_vars) == 0:
        lines.append("    kernel_args = ()")
    elif len(arg_vars) == 1:
        lines.append(f"    kernel_args = ({arg_vars[0]},)")
    else:
        lines.append(f"    kernel_args = ({', '.join(arg_vars)})")

    kinds_expr = (
        "()"
        if not mru_kind_exprs
        else (
            f"({mru_kind_exprs[0]},)"
            if len(mru_kind_exprs) == 1
            else f"({', '.join(mru_kind_exprs)})"
        )
    )
    vals_expr = (
        "()"
        if not mru_val_exprs
        else (
            f"({mru_val_exprs[0]},)"
            if len(mru_val_exprs) == 1
            else f"({', '.join(mru_val_exprs)})"
        )
    )
    meta_dict_expr = (
        "{" + ", ".join(f"{p.name!r}: v{i}" for i, p in enumerate(params)) + "}"
    )

    lines.extend(
        [
            f"    mru_meta = ({kinds_expr}, {vals_expr})",
            "    device = _get_device()",
            "    dev_entry = dev_caches.get(device)",
            "    bucket = entries.get(device)",
            "    if dev_entry is None or bucket is None:",
            "        if bucket is not None and dev_entry is None:",
            "            entries.pop(device, None)",
            "        try:",
            "            hash(cache_key)",
            "        except TypeError:",
            "            cache_key = None",
            "            mru_meta = None",
            "        return _fallback_record(jit_fn, grid, args, kwargs, device, cache_key, entries, mru_meta)",
            "    triton_cache = dev_entry[0]",
            "    if bucket[0] is not triton_cache or not triton_cache:",
            "        entries.pop(device, None)",
            "        jit_fn._rtp_fast_fn = None",
            "        try:",
            "            hash(cache_key)",
            "        except TypeError:",
            "            cache_key = None",
            "            mru_meta = None",
            "        return _fallback_record(jit_fn, grid, args, kwargs, device, cache_key, entries, mru_meta)",
            "    try:",
            "        entry_box = bucket[1].get(cache_key)",
            "    except TypeError:",
            "        return jit_fn[grid](*args, **kwargs)",
            "    if entry_box is None:",
            "        return _fallback_record(jit_fn, grid, args, kwargs, device, cache_key, entries, mru_meta)",
            "    rec = entry_box[0]",
            "    entry_box[1] += 1",
            "    mru_by_shape = entry_box[2]",
            "    mru_fn = mru_by_shape.get(shape_key)",
            "    if mru_fn is None and entry_box[1] >= 2 and mru_meta is not None:",
            "        mru_fn = _compile_mru_fast_closure(jit_fn, n_pos, kw_names, device, dev_entry, triton_cache, rec, mru_meta[0], mru_meta[1])",
            "        if mru_fn is not None:",
            "            mru_by_shape[shape_key] = mru_fn",
            "    if mru_fn is not None:",
            "        jit_fn._rtp_fast_fn = mru_fn",
            "    _gt = type(grid)",
            "    if _gt is tuple:",
            "        _gn = len(grid)",
            "        if _gn == 1:",
            "            _g0 = grid[0]; _g1 = 1; _g2 = 1",
            "        elif _gn == 2:",
            "            _g0, _g1 = grid; _g2 = 1",
            "        elif _gn == 3:",
            "            _g0, _g1, _g2 = grid",
            "        else:",
            "            return jit_fn[grid](*args, **kwargs)",
            "    elif callable(grid):",
            f"        _g = grid({meta_dict_expr})",
            "        _gn = len(_g)",
            "        if _gn == 1:",
            "            _g0 = _g[0]; _g1 = 1; _g2 = 1",
            "        elif _gn == 2:",
            "            _g0, _g1 = _g; _g2 = 1",
            "        elif _gn == 3:",
            "            _g0, _g1, _g2 = _g",
            "        else:",
            "            return jit_fn[grid](*args, **kwargs)",
            "    elif _gt is list:",
            "        _gn = len(grid)",
            "        if _gn == 1:",
            "            _g0 = grid[0]; _g1 = 1; _g2 = 1",
            "        elif _gn == 2:",
            "            _g0, _g1 = grid; _g2 = 1",
            "        elif _gn == 3:",
            "            _g0, _g1, _g2 = grid",
            "        else:",
            "            return jit_fn[grid](*args, **kwargs)",
            "    else:",
            "        return jit_fn[grid](*args, **kwargs)",
            "    if rec[0]:",
            "        rec[8](",
            "            _g0,",
            "            _g1,",
            "            _g2,",
            "            _get_stream(device),",
            "            rec[1],",
            "            rec[2],",
            "            rec[3],",
            "            rec[4],",
            "            None,",
            "            None,",
            "            None,",
            "            None,",
            "            None,",
            "            rec[5],",
            "            rec[6],",
            "            kernel_args,",
            "        )",
            "    else:",
            "        rec[8](",
            "            _g0,",
            "            _g1,",
            "            _g2,",
            "            _get_stream(device),",
            "            rec[1],",
            "            rec[2],",
            "            rec[3],",
            "            None,",
            "            None,",
            "            rec[4],",
            "            None,",
            "            None,",
            "            None,",
            "            *kernel_args,",
            "        )",
            "    return rec[7]",
        ]
    )

    code = "\n".join(lines)
    exec(code, ns)
    return ns["_shape_dispatch"]


def _fast_launch_slow(
    jit_fn: Any,
    grid: Any,
    args: Tuple[Any, ...],
    kwargs: Dict[str, Any],
) -> Any:
    if not _RUNTIME_INITIALIZED and not _init_runtime():
        return jit_fn[grid](*args, **kwargs)
    if type(jit_fn) is not _JIT_FUNCTION_CLS:
        return jit_fn[grid](*args, **kwargs)
    capturing = _CAPTURING_HINT.get()
    if capturing or (
        capturing is None
        and _TORCH_CUDA.is_available()
        and _TORCH_CUDA.is_current_stream_capturing()
    ):
        return jit_fn[grid](*args, **kwargs)
    if _has_hooks_or_debug(jit_fn):
        return jit_fn[grid](*args, **kwargs)

    fl = getattr(jit_fn, "_rtp_fl", None)
    if fl is None:
        # [plans, entries]
        fl = [{}, {}]
        jit_fn._rtp_fl = fl
    plans: Dict[Any, Any] = fl[0]
    entries: Dict[int, Any] = fl[1]
    kw_names = tuple(kwargs)
    shape_key = (len(args), kw_names)
    dispatcher = plans.get(shape_key)
    if dispatcher is None:
        if shape_key in plans:
            return jit_fn[grid](*args, **kwargs)
        dispatcher = _compile_shape_fallback_dispatcher(
            jit_fn, shape_key[0], kw_names, entries
        )
        plans[shape_key] = dispatcher
        if dispatcher is None:
            return jit_fn[grid](*args, **kwargs)
    return dispatcher(grid, args, kwargs)


def fast_launch(jit_fn: Any, grid: Any, *args: Any, **kwargs: Any) -> Any:
    """Drop-in fast replacement for ``jit_fn[grid](*args, **kwargs)``."""
    if type(jit_fn) is _JIT_FUNCTION_CLS:
        fn = getattr(jit_fn, "_rtp_fast_fn", None)
        if fn is not None:
            return fn(grid, args, kwargs)
    return _fast_launch_slow(jit_fn, grid, args, kwargs)
