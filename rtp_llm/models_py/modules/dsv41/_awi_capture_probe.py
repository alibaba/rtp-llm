"""Diagnostic hipGetLastError checkpoints for HIP graph capture bring-up.

Model-load evidence (loads 25-31): every decode graph capture records an
illegal kernel configuration (sticky hipErrorInvalidConfiguration) that
surfaces only at the NEXT error-checked torch op (hc/delayed.py _post_impl),
regardless of AMD_SERIALIZE_KERNEL, because the offending launchers (Triton,
RCCL, hipBLASLt wrappers) do not check hipGetLastError. Standalone probing
proved raw-RCCL capture is NOT the culprit (awi_rccl_capture_test.py: all
sizes/scenarios capture+replay OK on 8 ranks).

probe(tag) reads AND CLEARS the thread's last HIP error at instrumented
segment boundaries, so the first probe after a bad recording names the
segment, and clearing lets the capture continue and enumerate further bad
segments in the same run. Enabled with AWI_CAPTURE_PROBE=1; zero-cost
otherwise (one module-level env read).
"""

from __future__ import annotations

import ctypes
import logging
import os
import sys

_ENABLED = os.environ.get("AWI_CAPTURE_PROBE", "0") == "1"
_lib = None
_logger = logging.getLogger("awi.capture_probe")


def _runtime():
    global _lib
    if _lib is None:
        lib = ctypes.CDLL("libamdhip64.so")
        lib.hipGetLastError.restype = ctypes.c_int
        lib.hipGetErrorString.restype = ctypes.c_char_p
        lib.hipGetErrorString.argtypes = [ctypes.c_int]
        _lib = lib
    return _lib


def probe(tag: str) -> int:
    """Return (and clear) the pending HIP error, logging it against tag."""
    if not _ENABLED:
        return 0
    try:
        lib = _runtime()
        err = lib.hipGetLastError()
        if err:
            raw = lib.hipGetErrorString(err)
            text = raw.decode(errors="replace") if raw else "?"
            capturing = "unknown"
            try:
                import torch

                capturing = torch.cuda.is_current_stream_capturing()
            except Exception:  # noqa: BLE001
                pass
            msg = (
                f"AWI_CAPTURE_PROBE {tag}: hipError={err} ({text}) "
                f"capturing={capturing}"
            )
            print(msg, file=sys.stderr, flush=True)
            _logger.error(msg)
        return err
    except Exception as exc:  # noqa: BLE001
        print(
            f"AWI_CAPTURE_PROBE {tag}: probe failed {exc}",
            file=sys.stderr,
            flush=True,
        )
        return -1
