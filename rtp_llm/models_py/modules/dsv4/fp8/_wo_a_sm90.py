"""SM90 replacement for DSV4-Flash's wo_a batched FP8 einsum.

The production wo_a projection is one
``deep_gemm.fp8_einsum("bhr,hdr->bhd", ..., recipe=(1, 1, 128))`` launch.
That kernel is Blackwell-only.  On a device of compute capability 9.0 it
fails twice over: handed the int32-packed UE8M0 scale layout it raises
``csrc/apis/layout.hpp:46: Unknown SF transformation``, and handed a layout
the dispatcher *does* recognise it raises
``csrc/apis/einsum.hpp:174: Unsupported architecture``.  There is no SM90
``fp8_einsum``, so the einsum has to be expressed as GEMMs.  Both messages
reproduce by calling ``fp8_einsum`` directly with this module's operands --
see ``wo_a_sm90_gemm_test.py`` for the shapes.

``"bhr,hdr->bhd"`` with ``G`` groups is ``G`` independent
``[M, K] x [R, K]^T`` GEMMs that share the token axis, so this module runs
one ``fp8_gemm_nt`` per group over *strided views* of the activation:
``o_fp8[:, h, :]`` (stride ``(G*K, 1)``) and ``out[:, h, :]``.  DeepGEMM
accepts both, which is what makes this a drop-in — no activation
transpose, no extra buffers, and the output lands in the same
``[M, G, R]`` tensor ``wo_b`` already expects.  The alternative,
``m_grouped_fp8_gemm_nt_contiguous`` with the groups laid out along M, is
one launch instead of ``G`` but needs a full ``[M, G, K]`` fp8 transpose
copy first (512 MB at a 16k-token chunk), so it loses more to bandwidth
than it saves in launch overhead.

Two layout consequences, both of which *simplify* the init path:

* The SM90 GEMM takes ``recipe=(1, 128, 128)`` — activation per-token
  1x128, weight per-128x128-block — which is the checkpoint's *native*
  scale grid.  So unlike ``_prepare_wo_a_stacked``, this module does not
  row-repeat the weight scale by 128 and does not pack it: a plain
  ``.float().view(G, R/128, K/128)`` is what the kernel wants.
* The activation scale still arrives int32-packed, because
  ``fused_inv_rope_fp8_quant`` emits the UE8M0 form the einsum wanted.
  Unpacking it (4 exponent bytes per word -> fp32) is a handful of
  elementwise ops on a tensor 1/128th the size of the activation, so it is
  cheaper than changing the Triton kernel's output contract, and it keeps
  the FP8 activation values bit-identical between architectures.
"""

from typing import Optional, Tuple

import torch

# ``deep_gemm`` is imported inside :func:`wo_a_grouped_gemm` rather than here: the
# scale unpack and the weight regrouping below are plain tensor algebra, and
# keeping them importable without DeepGEMM (or a GPU) is what lets them be covered
# by a host test.
_UE8M0_BIAS = 127.0

# Bytes per int32 word in the packed UE8M0 scale layout.
_UE8M0_BYTES_PER_WORD = 4


def prepare_wo_a_grouped(
    weight_fp8: torch.Tensor,
    scale_raw: torch.Tensor,
    G: int,
    R: int,
    K: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """SM90 counterpart of ``_prepare_wo_a_stacked``.

    Takes the V4 checkpoint form (``[G*R, K]`` fp8 + ``[G*R/128, K/128]``
    e8m0fnu) to ``([G, R, K]`` fp8, ``[G, R/128, K/128]`` fp32), the
    operand pair ``fp8_gemm_nt(..., recipe=(1, 128, 128))`` consumes.
    """
    assert scale_raw.dtype == torch.float8_e8m0fnu, (
        f"expected raw e8m0fnu ckpt scale, got {scale_raw.dtype}; the packed "
        "int32 form is the SM100 path's and cannot be un-packed per-block here"
    )
    w = weight_fp8.view(G, R, K).contiguous()
    s = scale_raw.float().view(G, R // 128, K // 128).contiguous()
    return w, s


def unpack_ue8m0_int32_scale(
    packed: torch.Tensor,
    k_blocks: int,
    *,
    bytes_per_word: int = _UE8M0_BYTES_PER_WORD,
) -> torch.Tensor:
    """Packed int32 UE8M0 exponents -> ``[..., k_blocks]`` fp32.

    Each int32 word carries ``bytes_per_word`` biased exponents,
    least-significant byte first, so the fp32 scale is ``2 ** (byte - 127)``.
    Written with shifts rather than a ``view(torch.uint8)`` because the packed
    tensor is MN-major (stride ``(1, K/512 * tma_M, tma_M)``) and a bit-view
    would require contiguity.  The shift is arithmetic, so a word whose top byte
    is ``>= 0x80`` reads back negative; the ``& 0xFF`` after it is what makes the
    byte extraction correct in that case rather than incidental.

    ``k_blocks`` trims the trailing bytes of the last word, which the packer
    rounds up to a whole word.

    ``bytes_per_word`` is explicit because two packers with *different* grouping
    semantics feed this function.  The shared-expert packer groups four
    consecutive K-blocks per word; the wo_a activation packer groups per head,
    and only coincides with the first because DSV4-Flash's ``head_dim`` of 512
    makes ``chunks_per_head`` exactly 4.  The assertion below is what keeps a
    change to either shape from silently producing wrong scales.
    """
    words_present = packed.shape[-1]
    expected_words = -(-int(k_blocks) // int(bytes_per_word))  # ceil
    assert words_present == expected_words, (
        f"packed last dim {words_present} cannot hold {k_blocks} blocks at "
        f"{bytes_per_word} bytes/word (expected {expected_words}); the packer's "
        "grouping and this unpack disagree"
    )
    words = torch.stack(
        [(packed >> (8 * i)) & 0xFF for i in range(bytes_per_word)], dim=-1
    )  # [..., words, bytes_per_word]
    exps = words.reshape(*packed.shape[:-1], -1)[..., :k_blocks]
    return torch.exp2(exps.float() - _UE8M0_BIAS)


def wo_a_grouped_gemm(
    o_fp8: torch.Tensor,
    o_scale_packed: torch.Tensor,
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """``out[m, h, :] = o[m, h, :] @ weight[h].T`` for all ``h``, in FP8.

    ``o_fp8`` is ``[M, G, K]``, ``o_scale_packed`` the int32 UE8M0 scale
    ``fused_inv_rope_fp8_quant`` returns, and ``(weight, weight_scale)``
    the pair from :func:`prepare_wo_a_grouped`.  Returns ``[M, G, R]``
    bf16.
    """
    from deep_gemm.utils.layout import get_mn_major_tma_aligned_tensor

    import deep_gemm

    M, G, K = o_fp8.shape
    R = weight.shape[1]
    assert weight.shape == (G, R, K), f"weight {tuple(weight.shape)} vs o {(M, G, K)}"
    sf = unpack_ue8m0_int32_scale(o_scale_packed, K // 128)  # [M, G, K/128]
    if out is None:
        out = torch.empty(M, G, R, dtype=torch.bfloat16, device=o_fp8.device)
    for h in range(G):
        deep_gemm.fp8_gemm_nt(
            (o_fp8[:, h, :], get_mn_major_tma_aligned_tensor(sf[:, h, :].contiguous())),
            (weight[h], weight_scale[h]),
            out[:, h, :],
            recipe=(1, 128, 128),
        )
    return out
