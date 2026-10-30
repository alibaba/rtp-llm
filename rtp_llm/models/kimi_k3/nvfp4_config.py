"""K3.1 projection policy over the shared dynamic NVFP4 implementation."""

import os

KDA_NVFP4_MODULES = ("kda.input", "kda.forget")
DSA_NVFP4_MODULES = ("dsa.input", "dsa.query")
NVFP4_MODULES = frozenset(KDA_NVFP4_MODULES + DSA_NVFP4_MODULES)


def uses_fp8_output(module, selected_modules):
    return (module == "kda.output" and "kda.input" in selected_modules) or (
        module == "dsa.output" and "dsa.input" in selected_modules
    )


def selected_nvfp4_modules(kda, dsa, *, is_draft=False):
    if type(kda) is not bool or type(dsa) is not bool:
        raise ValueError("kda_nvfp4 and dsa_nvfp4 must be booleans")
    if is_draft:
        return ()
    return (KDA_NVFP4_MODULES if kda else ()) + (DSA_NVFP4_MODULES if dsa else ())


def validate_nvfp4_backend(modules):
    if modules and os.environ.get("RTP_LLM_FP4_GEMM_BACKEND", "cutlass") != "cutlass":
        raise ValueError("Dynamic NVFP4 projections require the cutlass backend")
