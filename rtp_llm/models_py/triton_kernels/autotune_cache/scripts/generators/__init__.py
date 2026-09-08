"""Registry of op-specific cache generators.

Each generator is a callable ``generate(device) -> None`` that exercises the
kernels of one op family under ``triton.autotune`` so that Triton's on-disk
cache is populated.  The ``extract`` driver then reads that cache and writes
per-kernel JSON.

Shape parameters (head_dim, chunk_size, etc.) are kernel-specific and owned
by each generator.

To add a new op family:
  1. Drop a ``<op>.py`` next to this file with a generator class.
  2. Register a launcher in ``REGISTRY`` below.
"""

from typing import Callable, Dict


def _kda_generate(device):
    from rtp_llm.models_py.triton_kernels.autotune_cache.scripts.generators.kda import (
        KDACacheGenerator,
    )

    KDACacheGenerator(head_dim=128, device=device).generate_kda()


REGISTRY: Dict[str, Callable] = {
    "kda": _kda_generate,
}


def available_ops() -> tuple[str, ...]:
    return tuple(sorted(REGISTRY))
