"""Register the PPU FA3 attention backends with the open-source factory.

The open-source attention package picks its implementations in a
``get_device_type()`` branch, which must not grow a PPU arm. Instead it drains
the ``attention`` backend slot after building its lists, so register a hook
here and let it hand the lists over.

Ordering in those lists is priority: earlier wins. FA3 inserts at the front to
outrank the FlashInfer impls the open-source CUDA branch appends (PPU is not
ROCm, so it takes that branch). Each impl still self-filters via
``support_parallelism_config`` / its own availability check, so FA3 losing
flash_attn_3 falls back to FlashInfer rather than failing.

Every backend is env-gated: set PPU_USE_FA3_DECODE / PPU_USE_FA3_PREFILL /
PPU_USE_FA3_VERIFY to 0 to disable it.
"""

import logging

from rtp_llm.utils.backend_registry import register_backend_hook

_registered = False


def _register_impls(
    prefill_mha_imps, decode_mha_imps, prefill_mla_imps=None, decode_mla_imps=None
) -> None:
    from rtp_llm.models_py.modules.factory.attention.ppu_impl.fa3_mha import (
        FA3DecodeImpl,
        FA3PrefillPagedImpl,
        _check_fa3_available,
    )

    decode_mha_imps.insert(0, FA3DecodeImpl)
    prefill_mha_imps.insert(0, FA3PrefillPagedImpl)

    logging.info(
        "[ppu attn register] FA3 decode/prefill registered at highest priority "
        "(flash_attn_3 available: %s)",
        _check_fa3_available(),
    )

    # Sparse MLA (GLM-5 / DSV3.2 DSA). The PPU flash_mla 2.0 wheel exposes the
    # same flash_mla_sparse_fwd / flash_mla_with_kvcache API and the C++
    # bindings (SparseMlaParams, concat_and_cache_mla, indexer ops) build for
    # PPU, so the CUDA SparseMlaImpl works unmodified. PPU has no dense MLA
    # backend, so the factory's short-prefill fast path is disabled on PPU and
    # this impl serves every sparse-MLA role.
    if prefill_mla_imps is not None and decode_mla_imps is not None:
        try:
            from rtp_llm.models_py.modules.factory.attention.cuda_mla_impl.flashmla_sparse_impl import (
                SparseMlaImpl,
            )

            prefill_mla_imps.append(SparseMlaImpl)
            decode_mla_imps.append(SparseMlaImpl)
            logging.info(
                "[ppu attn register] SparseMlaImpl registered for sparse MLA"
            )
        except ImportError as e:
            logging.warning(
                "[ppu attn register] flash_mla unavailable, sparse MLA not "
                "registered: %s",
                e,
            )

        # Prefill context parallel. Once prefill_cp_config.is_enabled() the
        # factory filters on support_prefill_cp(), which SparseMlaImpl answers
        # False -- without this the CP list is empty and prefill has no backend.
        # Appended after SparseMlaImpl so non-CP prefill still wins on priority.
        try:
            from rtp_llm.models_py.modules.factory.attention.cuda_mla_impl.flashmla_sparse_cp_impl import (
                SparseMlaCpImpl,
            )

            prefill_mla_imps.append(SparseMlaCpImpl)
            logging.info(
                "[ppu attn register] SparseMlaCpImpl registered for prefill CP"
            )
        except ImportError as e:
            logging.warning(
                "[ppu attn register] prefill CP sparse MLA unavailable: %s",
                e,
            )


def install() -> None:
    """Record the registration hook. Must run before the factory is imported."""
    global _registered
    if _registered:
        return
    _registered = True
    register_backend_hook("attention", _register_impls)
    logging.info("[ppu attn register] Attention backend hook recorded")
