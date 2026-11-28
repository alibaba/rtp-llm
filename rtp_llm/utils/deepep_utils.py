"""Configure DeepEP without importing its GPU extension or model operators."""

import logging
import os
from pathlib import Path


def configure_deepep_nccl_root() -> None:
    """Pin EP_NCCL_ROOT_DIR to the NCCL library that is actually loaded.

    deep_ep byte-compares the loaded libnccl against find_nccl_root()'s pick
    and EP_NCCL_ROOT_DIR takes precedence in find_nccl_root, so the env must
    name the root of the LOADED library (torch's, usually the wheel from the
    runfiles/venv).  Never guess from installed package metadata: a
    conda/site-packages copy can differ byte-wise from the loaded wheel.  If
    no loaded libnccl is mapped yet, leave the env unset and let deep_ep
    auto-detect with its own sys.path-priority logic.
    """
    if "EP_NCCL_ROOT_DIR" in os.environ:
        return
    try:
        loaded_nccl = sorted(
            {
                Path(line.rsplit(maxsplit=1)[-1]).resolve()
                for line in Path("/proc/self/maps").read_text().splitlines()
                if "/libnccl.so" in line
            }
        )
    except OSError:
        return
    for nccl in loaded_nccl:
        nccl_root = nccl.parent.parent
        if (nccl_root / "lib").is_dir() and (nccl_root / "include").is_dir():
            os.environ["EP_NCCL_ROOT_DIR"] = str(nccl_root)
            logging.info("DeepEP will use NCCL from %s", nccl_root)
            return
