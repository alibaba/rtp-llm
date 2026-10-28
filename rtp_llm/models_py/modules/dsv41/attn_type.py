"""V4.1 cache tags, shared with declarative cache specifications."""

SWA_KV = "swa_kv"
DECODER_SWA_KV = "decoder_swa_kv"
CSA_KV = "global_kv_2"
HCA_KV = "global_kv_1"
INDEXER_KV = "indexer_kv"
INDEXER_KV_2 = "indexer_kv_2"


def sm90_indexer_mode() -> str:
    """Default to FP8 DeepGEMM on SM90; reject FP8 on other devices."""
    import os

    from rtp_llm.device.device_type import is_sm90

    mode = os.environ.get("DSV41_SM90_INDEXER_MODE")
    if mode is None:
        return "fp8_dg" if is_sm90() else "fp4"
    if mode not in ("fp4", "fp8_dg"):
        raise ValueError(f"unsupported DSV41_SM90_INDEXER_MODE={mode!r}")
    if mode == "fp8_dg" and not is_sm90():
        raise ValueError("DSV41_SM90_INDEXER_MODE=fp8_dg requires SM90")
    return mode


def compact_ratio2_indexer_enabled() -> bool:
    """Only the FP8-primary mode uses half-size ratio-2 pages."""
    return sm90_indexer_mode() == "fp8_dg"


def indexer_kv_tag(compress_ratio: int, *, compact_ratio2: bool = False) -> str:
    """Choose the indexer storage group for a source layer.

    The default preserves the production group. The compact tag has separate
    physical stride and must be selected consistently for cache and readers.
    """
    if compress_ratio not in (1, 2):
        raise ValueError("indexer source ratio must be 1 or 2")
    return INDEXER_KV_2 if compact_ratio2 and compress_ratio == 2 else INDEXER_KV


INDEXER_STATE = "indexer_state"
CSA_STATE = "csa_state"
HCA_STATE = "hca_state"
