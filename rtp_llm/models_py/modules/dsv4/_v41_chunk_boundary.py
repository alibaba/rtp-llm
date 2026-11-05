"""R28/C: DSV41_CHUNK_BOUNDARY_OVERLAP flag (default "0").

Under "1", execute_native_pp_prefill issues the DP admission agreement
asynchronously (issue at admission start; wait+check right before the
chunk's layers — still before any of this chunk's collectives, so the
fault semantics are unchanged, only the gather latency is overlapped).
"""

import os
import sys

CHUNK_BOUNDARY_OVERLAP_FLAG = "DSV41_CHUNK_BOUNDARY_OVERLAP"
_ANNOUNCED = [False]


def chunk_boundary_overlap_enabled() -> bool:
    if os.environ.get(CHUNK_BOUNDARY_OVERLAP_FLAG, "0") == "1":
        if not _ANNOUNCED[0]:
            _ANNOUNCED[0] = True
            print(
                f"[CHUNKOV] engaged ({CHUNK_BOUNDARY_OVERLAP_FLAG}=1): async DP "
                "agreement issued at admission, waited before layers",
                file=sys.stderr,
                flush=True,
            )
        return True
    return False
