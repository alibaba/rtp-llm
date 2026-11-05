"""R26-TRACE: env-gated NVTX phase ranges for DSv4.1F prefill (default OFF).

Flag ``DSV41_NVTX_PHASES`` (default "0" = every ``nvtx_phase`` is a pure
no-op context manager and the marked code paths are byte-identical to the
uninstrumented ones). Under "1" each ``nvtx_phase(name)`` emits a paired
``torch.cuda.nvtx.range_push/range_pop`` around the phase, captured by
``nsys -t nvtx`` (and, redundantly, visible to torch-profiler via the
existing record_function sites that already cover the same regions).

Names are stage-independent (both PP stages execute the same code paths),
which is what makes stage-0/stage-1 timelines comparable:

  v41.admission         validate + compact-CP prepare + DP agree
  v41.chunk_setup       engram prep (stage 0) / PP frame import + premix (stage 1)
  v41.layers            the full per-layer forward_prefill block
  v41.layer             one layer (20 per chunk per stage)
  v41.attn / v41.moe    per-layer attention / FFN-MoE sections
  v41.moe.router        Gate.forward (routing scores -> topk)
  v41.moe.counts_sync   host-visible EP counts preparation
  v41.moe.dispatch_prep pack_dispatch (wire packet build)
  v41.moe.dispatch      dispatch all_to_all exchange
  v41.moe.experts       local expert compute (incl. combine or fused pack)
  v41.moe.return        return pack + return all_to_all
  v41.moe.combine       deferred mxfp8 peer-sum + shared epilogue
  v41.hc.stats          mHC boundary stats (prenorm/fn_mul, split, sinkhorn)
  v41.hc.post           mHC post (collapse/combine into residual)
  v41.pp_export         stage-0 PP frame pack (pre_mix crosses the wire)
  v41.pp.send/pp.recv   PP p2p transport callbacks (cpp_isend/cpp_irecv)
  v41.result            post-forward output packaging

The flag state is cached at import time; workers read it after the harness
readmission delivers the env, so per-run toggling works through the normal
DSV41_* env path. No CUDA calls happen unless the flag is on.
"""

import os

import torch

NVTX_PHASES_FLAG = "DSV41_NVTX_PHASES"
_ENABLED = os.environ.get(NVTX_PHASES_FLAG, "0") == "1"
_ANNOUNCED = [False]


def nvtx_phases_enabled() -> bool:
    return _ENABLED


class _NvtxPhase:
    """Reentrant-pairing context manager; zero CUDA calls when disabled."""

    __slots__ = ("name",)

    def __init__(self, name: str) -> None:
        self.name = name

    def __enter__(self):
        if _ENABLED:
            _announce()
            torch.cuda.nvtx.range_push(self.name)
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        if _ENABLED:
            torch.cuda.nvtx.range_pop()
        return False


def _announce() -> None:
    if not _ANNOUNCED[0]:
        _ANNOUNCED[0] = True
        import sys

        print(
            f"[NVTXPHASE] engaged ({NVTX_PHASES_FLAG}=1): v41.* phase ranges",
            file=sys.stderr,
            flush=True,
        )


def nvtx_phase(name: str) -> _NvtxPhase:
    return _NvtxPhase(name)


def nvtx_push(name: str) -> None:
    """Unbalanced-pair variant for wrapping large existing blocks without
    re-indentation. Callers must pair with nvtx_pop() on every path."""
    if _ENABLED:
        _announce()
        torch.cuda.nvtx.range_push(name)


def nvtx_pop() -> None:
    if _ENABLED:
        torch.cuda.nvtx.range_pop()
