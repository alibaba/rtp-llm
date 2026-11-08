"""Independent, target-only NVFP4 projection switches."""

import os

KDA_NVFP4_MODULES = ("kda.input", "kda.forget")
DSA_NVFP4_MODULES = ("dsa.input", "dsa.query")
NVFP4_MODULES = frozenset(KDA_NVFP4_MODULES + DSA_NVFP4_MODULES)


def uses_fp8_output(module, selected_modules):
    return (
        (module == "kda.output" and "kda.input" in selected_modules)
        or (module == "dsa.output" and "dsa.input" in selected_modules)
    )


def parse_nvfp4_flag(name, value=None, *, default=False):
    """Explicit smoke booleans override the environment; services default off."""
    if value is not None:
        if type(value) is not bool:
            raise ValueError(f"{name} must be boolean in smoke configuration")
        return value
    raw = os.environ.get(name, str(int(default)))
    if raw not in ("0", "1"):
        raise ValueError(f"{name} must be 0 or 1, got {raw!r}")
    return raw == "1"


def selected_nvfp4_modules(kda, dsa, *, is_draft=False):
    if type(kda) is not bool or type(dsa) is not bool:
        raise ValueError("KDA_NVFP4 and DSA_NVFP4 must be booleans")
    if is_draft:
        return ()
    return (KDA_NVFP4_MODULES if kda else ()) + (DSA_NVFP4_MODULES if dsa else ())


def reject_legacy_nvfp4(args):
    if "k31_nvfp4_modules" in args:
        raise ValueError(
            "k31_nvfp4_modules was replaced by KDA_NVFP4/DSA_NVFP4; "
            "use the native checkpoint with online NVFP4 quantization"
        )
    if {"kda_nvfp4", "dsa_nvfp4"} & args.keys():
        raise ValueError(
            "Set KDA_NVFP4/DSA_NVFP4 in the service environment; "
            "kda_nvfp4/dsa_nvfp4 JSON booleans belong to smoke configuration, "
            "not JSON_MODEL_OVERRIDE_ARGS"
        )


def validate_nvfp4_backend(modules):
    if modules and os.environ.get("RTP_LLM_FP4_GEMM_BACKEND", "cutlass") != "cutlass":
        raise ValueError("K3.1 NVFP4 requires RTP_LLM_FP4_GEMM_BACKEND=cutlass")
