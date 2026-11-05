"""E5b (Batch-4b part 2): Triton inverse RoPE for the attention output.

``_project_output`` applies the conjugate rotation to the RoPE tail of the
attention output through the eager 4-kernel complex path (float cast,
view_as_complex, conj-multiply, copy back). The existing Triton
``rope_only_inplace(inverse=True)`` computes the same conjugate rotation
in one kernel. Under ``DSV41_E5B_INV_ROPE_TRITON`` the Triton path is used
when the layout/dtype guards hold; otherwise (and whenever the flag is
off) the eager path is kept. GPU byte-compare gated: if the kernel's FMA
contraction does not reproduce the eager complex multiply bitwise, the
item is dropped.
"""

import os
import sys

import torch

_E5B_FLAG = "DSV41_E5B_INV_ROPE_TRITON"
_E5B_ANNOUNCED = False


def e5b_inv_rope_enabled() -> bool:
    """E5b: Triton inverse RoPE on the attention output (default 0)."""
    return os.environ.get(_E5B_FLAG, "0") == "1"


def _announce_e5b() -> None:
    global _E5B_ANNOUNCED
    if not _E5B_ANNOUNCED:
        _E5B_ANNOUNCED = True
        print("[E5BROPE] Triton inverse RoPE engaged", file=sys.stderr, flush=True)


def apply_inverse_rope_output(tail, freqs):
    """Inverse RoPE in place on ``tail`` = ``o[..., -rope_dim:]``.

    ``tail`` is a [t, h, rd] view with unit last stride; ``freqs`` is
    complex64 [t, rd/2]. Falls back to the eager path on any mismatch.
    """
    from rtp_llm.models_py.modules.dsv4._rope_only_triton import rope_only_inplace
    from rtp_llm.models_py.modules.dsv4.rope import apply_rotary_emb

    rd = tail.shape[-1]
    if (
        e5b_inv_rope_enabled()
        and tail.dtype in (torch.bfloat16, torch.float16, torch.float32)
        and tail.is_cuda
        and tail.stride(-1) == 1
        and freqs.is_complex
        and freqs.dtype == torch.complex64
        and freqs.shape[-1] == rd // 2
        and freqs.numel() == tail.shape[0] * (rd // 2)
        and not torch.cuda.is_current_stream_capturing()
    ):
        _announce_e5b()
        rope_only_inplace(tail, freqs, inverse=True)
        return
    apply_rotary_emb(tail.unsqueeze(0), freqs, inverse=True)
