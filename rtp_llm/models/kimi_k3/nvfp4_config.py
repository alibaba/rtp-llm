"""K3.1 projection policy over the shared dynamic NVFP4 implementation."""

import os

KDA_NVFP4_MODULES = ("kda.input", "kda.forget")
DSA_NVFP4_MODULES = ("dsa.input", "dsa.query")
NVFP4_MODULES = KDA_NVFP4_MODULES + DSA_NVFP4_MODULES


def uses_fp8_output(module, selected_modules):
    return (module == "kda.output" and "kda.input" in selected_modules) or (
        module == "dsa.output" and "dsa.input" in selected_modules
    )


def selected_nvfp4_modules(quantization):
    return NVFP4_MODULES if quantization == "KIMI_K31_NVFP4" else ()


def reject_legacy_nvfp4_options(options):
    if "kda_nvfp4" in options or "dsa_nvfp4" in options:
        raise ValueError(
            "kda_nvfp4/dsa_nvfp4 were removed; use --quantization KIMI_K31_NVFP4"
        )


def validate_nvfp4_backend(modules):
    if modules and os.environ.get("RTP_LLM_FP4_GEMM_BACKEND", "cutlass") != "cutlass":
        raise ValueError("Dynamic NVFP4 projections require the cutlass backend")
