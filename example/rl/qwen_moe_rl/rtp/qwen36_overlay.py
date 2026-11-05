"""Compatibility bootstrap for the former isolated Qwen text-MoE deployment.

RTP model, loader, cache and score fixes now live in the normal source tree.
This module keeps the deployment's import API without replacing RTP modules.
"""
from __future__ import annotations

import importlib
import os
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
SOURCE = Path(os.environ.get("RTP_SOURCE", HERE.parents[3])).resolve()
MODULES = {
    name: SOURCE.joinpath(*name.split(".")).with_suffix(".py")
    for name in (
        "rtp_llm.config.server_config_setup",
        "rtp_llm.model_loader.loader",
        "rtp_llm.model_loader.weight_manager",
        "rtp_llm.model_loader.rl_weight_update",
        "rtp_llm.models_py.triton_kernels.common.qk_norm_rope_cache",
    )
}
_installed = False


def _native_log_policy(config_module) -> str:
    """Keep example-owned native logs outside the source checkout."""
    directory = Path(os.environ.get("LOG_PATH", HERE.parent / "logs/rtp-native"))
    directory.mkdir(parents=True, exist_ok=True)
    source = Path(config_module.__file__).resolve().with_name("alog.conf")
    lines = []
    for line in source.read_text().splitlines():
        if ".fileName=logs/" in line:
            prefix, filename = line.split(".fileName=logs/", 1)
            line = prefix + ".fileName=" + str((directory / filename).resolve())
        lines.append(line)
    output = directory / "native-alog.conf"
    temporary = output.with_name(output.name + f".writing-{os.getpid()}")
    temporary.write_text("\n".join(lines) + "\n")
    temporary.replace(output)
    return str(output)


def install() -> None:
    global _installed
    if _installed:
        return
    native = Path(os.environ.get("RTP_NATIVE_LIBRARY_DIR", HERE.parent / "build/rtp-native"))
    if (native / "libth_transformer.so").is_file():
        sys.path.insert(0, str(native))
    kernels = Path(os.environ.get("RTP_KERNEL_PYTHON_PATH", native / "python-deps"))
    if (kernels / "rtp_kernel/__init__.py").is_file():
        sys.path.insert(0, str(kernels))
    from rtp_llm.model_factory_register import ensure_model_registered
    from rtp_llm.frontend.tokenizer_factory.tokenizer_factory_register import ensure_tokenizer_registered

    for name in ("qwen35_moe_text", "qwen36_moe_text"):
        if not ensure_model_registered(name) or not ensure_tokenizer_registered(name):
            raise RuntimeError(f"RTP checkout does not contain text-MoE registration: {name}")
    config = importlib.import_module("rtp_llm.config.server_config_setup")
    original = config.setup_default_args

    def setup_default_args(options):
        if options.model_args.model_type in ("qwen35_moe_text", "qwen36_moe_text"):
            logging = options.profiling_debug_logging_config
            if not logging.ft_alog_conf_path:
                logging.ft_alog_conf_path = _native_log_policy(config)
        return original(options)

    config.setup_default_args = setup_default_args
    _installed = True


def assert_overlays_loaded() -> dict[str, str]:
    """Legacy probe name: verify canonical files rather than install overlays."""
    result = {}
    for name, expected in MODULES.items():
        module = importlib.import_module(name)
        actual = Path(module.__file__).resolve()
        if actual != expected:
            raise RuntimeError(f"Wrong RTP checkout for {name}: {actual}, expected {expected}")
        result[name] = str(actual)
    return result


def preflight_required_kernels() -> dict:
    """Resolve the required Python/native API before spawning TP workers."""
    try:
        wrapper = importlib.import_module("rtp_kernel.fused_rope_kvcache")
        native = importlib.import_module("rtp_ops")
        required = ("prefill_fused_rope_kvcache", "decode_fused_rope_kvcache")
        for name in required:
            if not callable(getattr(wrapper, name, None)) or not callable(getattr(native, name, None)):
                raise ImportError(f"RTP kernel binding is missing: {name}")
    except Exception as error:
        raise RuntimeError(
            "The example requires compatible rtp_kernel/rtp_ops fused RoPE APIs. "
            "Build/install for the selected native interpreter and GPU architecture."
        ) from error
    return {"passed": True, "wrapper": wrapper.__file__, "native": native.__file__,
            "required_apis": list(required)}
