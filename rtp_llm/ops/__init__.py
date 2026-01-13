import logging
from typing import List

from rtp_llm.libs.libth_transformer_config import *
from rtp_llm.libs.libth_transformer_config import get_block_cache_keys as cpp_get_block_cache_keys

from .compute_ops import rtp_llm_ops
from rtp_llm.libs.librtp_compute_ops import KVCache, PyAttentionInputs, PyModelInputs, PyModelOutputs, PyModelInitResources, PyCacheStoreInputs
from rtp_llm.libs.libth_transformer import RtpEmbeddingOp, RtpLLMOp
from rtp_llm.libs.libth_transformer import EmbeddingCppOutput


def get_block_cache_keys(token_ids: List[int], block_size: int) -> List[int]:
    try:
        # split token_ids into chunks of size block_size, dropping the last chunk if it is smaller than block_size
        token_ids_list: List[List[int]] = []
        for i in range(0, len(token_ids), block_size):
            chunk = token_ids[i : i + block_size]
            if len(chunk) == block_size:
                token_ids_list.append(chunk)
        return cpp_get_block_cache_keys(token_ids_list)  # type: ignore
    except Exception as e:
        logging.error(f"get block ids error: {e}")
        # If an error occurs, return an empty list
        return []
