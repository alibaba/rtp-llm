"""TopK dispatch with explicit caller semantics and unchanged providers.

Scores keep their producer's dtype and candidate domain. In particular,
SG output is never passed through local Prefill's sanitizer/finite finish.
"""

import torch

from rtp_llm.models_py.modules.dsv41.ppu import _quant


def select_tokens(scores, lengths, *, contract, **options):
    """Select in the caller's existing domain; never retry another contract.

    Local returns int32 token IDs or None before unsupported work. SG dense
    returns (token IDs, published/private blocks); SG sparse returns native
    page-translated IDs. Graph retains its short-row and valid-mask rules.
    Provider-specific arguments and failures keep their original contracts.
    """
    if contract == "local":
        from .prefill_token_topk import try_select_tokens

        return try_select_tokens(scores, lengths, **options)
    if contract == "sg_dense":
        from .local_prefill_sg_topk import select_dense

        return select_dense(scores, lengths, **options)
    if contract == "sg_sparse":
        from .local_prefill_sg_topk import select_sparse

        return select_sparse(scores, lengths, **options)
    if contract == "graph":
        from .graph_indexer import _select_dense_topk

        return _select_dense_topk(scores, lengths, **options)
    raise ValueError(f"unsupported TopK contract: {contract!r}")


def native_topk_op(rows, device, topk, *, bf16_provenance=False):
    """Use the BF16 key only with the original typed producer proof.

    Both APIs consume FP32 storage. No scan or FP32-to-BF16 conversion can
    substitute for a BF16 prefix and the qualified Verify tail reducer.
    """
    from rtp_llm.ops.compute_ops import rtp_llm_ops

    use_bf16 = (
        bf16_provenance
        and device.type == "cuda"
        and _quant.is_ppu()
        and 0 < rows <= 32
        and topk == 512
        and hasattr(rtp_llm_ops, "dsv4_persistent_topk_bf16_scores")
    )
    return (
        (
            rtp_llm_ops.dsv4_persistent_topk_bf16_scores
            if use_bf16
            else rtp_llm_ops.dsv4_persistent_topk
        ),
        use_bf16,
    )


def raw_tensor(value):
    return (
        isinstance(value, torch.Tensor)
        and value.layout == torch.strided
        and not value.is_neg()
        and not value.is_conj()
        and not value.requires_grad
    )


def overlaps(left, right):
    """Conservative storage/byte-range overlap, including score row padding."""
    if left.untyped_storage().data_ptr() == right.untyped_storage().data_ptr():
        return True

    def span(value):
        count = (
            (value.shape[0] - 1) * value.stride(0) + value.shape[1]
            if value.ndim == 2
            else value.numel()
        )
        return value.data_ptr(), value.data_ptr() + count * value.element_size()

    a, b = span(left), span(right)
    return a[0] < b[1] and b[0] < a[1]


def local_device_supported(scores):
    """The shared SG/hierarchical default-stream gate, with no device reads."""
    if not scores.is_cuda or torch.version.hip is not None or not _quant.is_ppu():
        return False
    try:
        return (
            torch.cuda.current_device() == scores.device.index
            and torch.cuda.get_device_capability(scores.device) == (8, 9)
            and not torch.cuda.is_current_stream_capturing()
            and torch.cuda.current_stream(scores.device)
            == torch.cuda.default_stream(scores.device)
        )
    except (AssertionError, RuntimeError):
        return False
