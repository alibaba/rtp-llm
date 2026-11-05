"""R32/F2: DSV41_F2_GATEUP_EPILOGUE (default "0") — the DeepGEMM gate/up
GEMM epilogue absorbs SwiGLU + the LUT FP8 quantize, removing the ~113 MB
BF16 intermediate round trip per layer.

Byte-exactness: the w13 weights are interleaved at load time so (gate_j,
up_j) land as adjacent output channels (each channel's value identical,
reordered); the fused epilogue applies the reference chain verbatim
(bf16 round -> clamp -> LUT silu -> weight*(act*up) -> bf16 re-round ->
group-32 amax with 1e-4 floor -> UE8M0 round-up -> div -> clamp -> E4M3
RTNE) and emits the down GEMM's exact input layout. The chain is proven
bitwise by r32_f2_bitwise.py before any timing arm runs.
"""

import os
import sys

F2_GATEUP_EPILOGUE_FLAG = "DSV41_F2_GATEUP_EPILOGUE"
_ANNOUNCED = [False]

# R39: same fused entry, but wired with zero added per-call host work (cached
# arange + pinned-ring async H2D for the expert layout; no pageable syncs) and
# the R39 warp-parallel epilogue kernel. F2 stays 0 forever (kept for R32
# reproduction); this flag gates the R39 engagement.
ROUTED_GATEUP_EPI_V2_FLAG = "DSV41_ROUTED_GATEUP_EPI_V2"
_ANNOUNCED_V2 = [False]


def f2_gateup_epilogue_enabled() -> bool:
    if os.environ.get(F2_GATEUP_EPILOGUE_FLAG, "0") == "1":
        if not _ANNOUNCED[0]:
            _ANNOUNCED[0] = True
            print(
                f"[F2GATEUP] engaged ({F2_GATEUP_EPILOGUE_FLAG}=1): interleaved w13 "
                "+ fused SwiGLU+quant epilogue",
                file=sys.stderr,
                flush=True,
            )
        return True
    return False


def routed_gateup_epi_v2_enabled() -> bool:
    if os.environ.get(ROUTED_GATEUP_EPI_V2_FLAG, "0") == "1":
        if not _ANNOUNCED_V2[0]:
            _ANNOUNCED_V2[0] = True
            print(
                f"[RGEPIV2] engaged ({ROUTED_GATEUP_EPI_V2_FLAG}=1): interleaved w13 "
                "+ fused SwiGLU+quant epilogue (warp-parallel phase 2, "
                "cached-arange + pinned-ring layout build)",
                file=sys.stderr,
                flush=True,
            )
        return True
    return False
