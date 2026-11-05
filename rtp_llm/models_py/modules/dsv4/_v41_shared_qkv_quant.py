"""E6 (Batch-4b part 2): shared SM12x QKV input-quant flag.

On consumer SM12x the DeepGEMM-based ``_can_reuse_qkv_input_quant`` gate is
closed, so wq_a and wkv each re-quantize the SAME normed attention input
with the SAME internal quantizer (``V41MXFP8Linear._quantize_input`` =
sgl group32 UE8M0). Under ``DSV41_E6_SHARED_QKV_QUANT`` the prefill path
quantizes once and feeds both projections via ``forward_quantized`` —
byte-exact by construction, since ``forward`` IS
``forward_quantized(*_quantize_input(x))`` for that class.
"""

import os
import sys

_E6_FLAG = "DSV41_E6_SHARED_QKV_QUANT"
_E6_ANNOUNCED = False


def e6_shared_qkv_quant_enabled() -> bool:
    """E6: share the SM12x QKV input quantization. Default ON since
    Batch-5 (R16): byte-exact by construction (forward IS
    forward_quantized∘_quantize_input; sgl determinism bench PASS),
    GPU parity byte-exact with engagement proven x8, A/B
    median −4.5 ms 2/3. ``=0`` restores the double quantization."""
    return os.environ.get(_E6_FLAG, "1") == "1"


def announce_e6() -> None:
    global _E6_ANNOUNCED
    if not _E6_ANNOUNCED:
        _E6_ANNOUNCED = True
        print(
            "[E6SHQ] shared SM12x QKV input quant engaged", file=sys.stderr, flush=True
        )
