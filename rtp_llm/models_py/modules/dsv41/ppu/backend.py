"""PPU provider entry for the V4.1 backend contract.

The shared compute ops (L1) and the model/weight wiring (L2, this batch) are
integrated: ``get_backend()`` returns the complete ``V41Backend`` —
``name="ppu"``, ``strategy=PPU_STRATEGY``, bound ops, ``model_cls =
DeepSeekV41SharedPPUModel`` and ``weight_cls = DeepSeekV41SharedPPUWeight`` — and
``check_capabilities`` probes the accelerator kernel surface exactly as the
preflight does.

``get_backend`` refuses to register the provider when the PPU runtime surface
is incomplete (the model wiring dispatches on
``rtp_llm.device.device_type.is_ppu``, so a partial/simulated device module
means the provider is not integrated on that host) or when the model/weight
modules cannot import; both failures raise ``RuntimeError`` with
"not integrated" in the message instead of registering a half-integrated
backend. No NVIDIA or CPU fallback; PPU890 MegaMoE is an explicit option.

This module stays import-light: importing it never imports torch, the ops
package, or an accelerator library, so the device-dispatch contract tests
run it on a CPU host.
"""

from __future__ import annotations

import importlib
import os

from rtp_llm.models_py.modules.dsv41.ppu.moe_precision import (
    moe_backend,
    moe_kernel_name,
    moe_precision,
)

PROVIDER_NAME = "ppu"
MODEL_PATH_ENV = "RTP_V41_PPU_MODEL_PATH"


def model_path(env=None):
    """Validate the inherited runtime selector used by existing launchers."""
    env = os.environ if env is None else env
    value = env.get(MODEL_PATH_ENV, "shared").strip().lower()
    if value != "shared":
        raise ValueError(f"{MODEL_PATH_ENV} must be shared, got {value!r}")
    return value


COMMON_KERNEL_SYMBOLS = (
    ("deep_ep", ("Buffer",)),
    ("flash_mla", ("flash_mla_sparse_fwd",)),
    # Wheel package is fast-hadamard-transform (import name and symbol per
    # tools/v41/preflight.py and the 2026-11-03 8-rank hw probe).
    ("fast_hadamard_transform", ("hadamard_transform",)),
)


def kernel_symbols(env=None):
    precision = moe_precision(env)
    backend = moe_backend(env)
    functions = (moe_kernel_name(precision, backend),)
    if backend == "megamoe":
        functions += (
            "get_symm_buffer_for_mega_moe",
            "transform_weights_for_mega_moe",
            "mega_moe_pre_dispatch",
        )
    elif precision == "w4a8":
        functions += ("preprocess_weights_and_scales",)
    return (("deep_gemm", functions),) + COMMON_KERNEL_SYMBOLS


# Compatibility for callers inspecting the default kernel surface. Runtime
# probes use kernel_symbols() so an explicit W4A4 environment is honored.
KERNEL_SYMBOLS = kernel_symbols()


def check_moe_preprocess(module, env=None):
    if moe_precision(env) == "w4a8" and not hasattr(
        getattr(module, "PreprocessMode", None), "W4A8_MXFP4_MMA"
    ):
        raise RuntimeError("deep_gemm.PreprocessMode.W4A8_MXFP4_MMA is missing")


def check_capabilities(runtime) -> None:
    """Raise unless the PPU device and the required kernel symbols exist.

    Import probes only: no process groups, no allocations. The rank-local
    collective probes run in the model initializer once A1 creates groups.
    """
    from rtp_llm.device.device_type import is_ppu

    if not is_ppu():
        raise RuntimeError(
            "V4.1 PPU backend requires a PPU device; this host is not one"
        )
    if runtime is not None:
        from rtp_llm.config.v41_runtime import PPU_STRATEGY

        strategy = getattr(runtime, "strategy", None)
        if strategy is not None and strategy != PPU_STRATEGY:
            raise ValueError(
                f"unapproved V4.1 PPU strategy {strategy!r}; expected {PPU_STRATEGY}"
            )
    missing: list[str] = []
    for module_name, symbols in kernel_symbols():
        try:
            module = importlib.import_module(module_name)
        except Exception as error:  # noqa: BLE001 - report any import failure
            missing.append(f"{module_name} (import failed: {error})")
            continue
        for symbol in symbols:
            if not callable(getattr(module, symbol, None)):
                missing.append(f"{module_name}.{symbol}")
        if module_name == "deep_gemm":
            try:
                check_moe_preprocess(module)
            except RuntimeError as error:
                missing.append(str(error))
    if missing:
        raise RuntimeError("V4.1 PPU kernels missing: " + ", ".join(missing))


def get_ops(**kwargs):
    """Construct the integrated L1 shared-compute ops (see PPUOps)."""
    from rtp_llm.models_py.modules.dsv41.ppu.ops import PPUOps

    return PPUOps(**kwargs)


def _not_integrated(reason: str) -> RuntimeError:
    return RuntimeError(
        f"V4.1 ppu backend is not integrated on this host: {reason}. "
        "The L1 shared compute ops remain available via "
        "rtp_llm.models_py.modules.dsv41.ppu.backend.get_ops. "
        "No NVIDIA or CPU fallback."
    )


def _native_graph_smoke_selected(env=None):
    """Exact isolated fixture opt-in; no change to normal provider selection."""
    env = os.environ if env is None else env
    selector = "RTP_V41_N4_NATIVE_GRAPH_SMOKE"
    if selector not in env:
        return False
    expected = {
        selector: "1", "RTP_V41_BACKEND": "ppu", MODEL_PATH_ENV: "shared",
        "MODEL_TYPE": "deepseek_v41", "ROLE_TYPE": "DECODE",
        "RTP_V41_PPU_MOE_BACKEND": "megamoe", "RTP_V41_PPU_MOE_PRECISION": "w4a4",
    }
    if any(env.get(key) != value for key, value in expected.items()):
        raise ValueError("N4 native Graph smoke requires exact explicit PPU/shared/deepseek_v41/DECODE/MegaMoE/W4A4 selection")
    return True


def _native_optimized_smoke_selected(env=None):
    """N5 fixture uses the same explicit canonical-provider seam as N4."""
    env = os.environ if env is None else env
    selector = "RTP_V41_N5_NATIVE_OPTIMIZED_SMOKE"
    if selector not in env:
        return False
    expected = {
        selector: "1", "RTP_V41_BACKEND": "ppu", MODEL_PATH_ENV: "shared",
        "MODEL_TYPE": "deepseek_v41", "ROLE_TYPE": "DECODE",
        "RTP_V41_PPU_MOE_BACKEND": "megamoe", "RTP_V41_PPU_MOE_PRECISION": "w4a4",
    }
    if any(env.get(key) != value for key, value in expected.items()):
        raise ValueError("N5 native smoke requires exact explicit PPU/shared/deepseek_v41/DECODE/MegaMoE/W4A4 selection")
    if "RTP_V41_N4_NATIVE_GRAPH_SMOKE" in env:
        raise ValueError("N4 and N5 isolated native fixture selectors are exclusive")
    return True


def get_backend():
    """Return the complete V41Backend for the PPU provider.

    Fails loudly (never registers a partial backend) when the device runtime
    surface is incomplete or the model/weight wiring cannot import.
    """
    try:
        from rtp_llm.device.device_type import is_ppu  # noqa: F401
    except Exception as error:  # noqa: BLE001 - report any import failure
        raise _not_integrated(
            f"the device dispatch surface is incomplete (is_ppu unavailable: "
            f"{error})"
        ) from error
    from rtp_llm.config.v41_runtime import PPU_STRATEGY
    from rtp_llm.models_py.modules.dsv41.backend import V41Backend

    model_path()
    native_graph_smoke = _native_graph_smoke_selected()
    native_optimized_smoke = _native_optimized_smoke_selected()
    try:
        from rtp_llm.models_py.modules.dsv41.ppu.shared_model import DeepSeekV41SharedPPUModel
        from rtp_llm.models_py.modules.dsv41.ppu.shared_weight import DeepSeekV41SharedPPUWeight

        model_cls = DeepSeekV41SharedPPUModel
        if native_graph_smoke:
            if not is_ppu():
                raise ValueError("native Graph smoke requires an actual PPU device")
            from rtp_llm.models_py.modules.dsv41.ppu.test.native_shared_graph_role import native_smoke_model_class

            model_cls = native_smoke_model_class()
        elif native_optimized_smoke:
            if not is_ppu():
                raise ValueError("native optimized smoke requires an actual PPU device")
            from rtp_llm.models_py.modules.dsv41.ppu.test.native_shared_optimized_role import native_smoke_model_class

            model_cls = native_smoke_model_class()
    except Exception as error:  # noqa: BLE001 - report any import failure
        raise _not_integrated(
            f"shared model/weight wiring import failed ({error})"
        ) from error
    if not DeepSeekV41SharedPPUWeight.framework_base_available():
        raise _not_integrated("framework weight descriptor base is unavailable")
    return V41Backend(
        name=PROVIDER_NAME,
        strategy=PPU_STRATEGY,
        # Representative ops instance for contract validation; the model
        # constructor builds the rank-configured PPUOps (real EP/ENP groups).
        ops=get_ops(ep_size=8),
        model_cls=model_cls,
        weight_cls=DeepSeekV41SharedPPUWeight,
        check_capabilities=check_capabilities,
    )
