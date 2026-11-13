"""Triton compiler defaults configured by server startup, before model loading."""

import logging
import os
import re
import subprocess

_DEFAULT_PTXAS_PATHS = (
    "/usr/local/cuda-13.2/bin/ptxas",
    "/usr/local/cuda/bin/ptxas",
)


def configure_sm120_triton_ptxas() -> None:
    """Choose an installed CUDA 13.2+ ptxas for an SM120 server before any JIT.

    Triton 3.6 selects a separate Blackwell compiler, so TRITON_PTXAS_PATH
    alone does not select the compiler used by SM120 QSA FP8 kernels. Preserve
    an explicit Blackwell selection and leave unsupported toolchains to the
    model's compatibility checks; other models can still start without it.
    """
    if "TRITON_PTXAS_BLACKWELL_PATH" in os.environ:
        return

    candidates = [os.environ.get("TRITON_PTXAS_PATH")]
    cuda_home = os.environ.get("CUDA_HOME")
    if cuda_home:
        candidates.append(os.path.join(cuda_home, "bin", "ptxas"))
    candidates.extend(_DEFAULT_PTXAS_PATHS)
    for path in dict.fromkeys(candidates):
        if not path:
            continue
        try:
            output = subprocess.check_output(
                [path, "--version"],
                stderr=subprocess.STDOUT,
                text=True,
                timeout=5,
            )
        except (OSError, subprocess.SubprocessError):
            continue
        match = re.search(r"release\s+(\d+)\.(\d+)", output)
        if match is None or tuple(map(int, match.groups())) < (13, 2):
            continue
        os.environ["TRITON_PTXAS_BLACKWELL_PATH"] = path
        logging.info(
            "SM120 startup selected TRITON_PTXAS_BLACKWELL_PATH=%s (ptxas %s.%s)",
            path,
            *match.groups(),
        )
        return
