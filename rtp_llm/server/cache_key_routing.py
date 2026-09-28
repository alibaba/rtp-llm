import os
from typing import List

from rtp_llm.ops import DSV41_SWA_BOUNDED_REPLAY_CACHE_KEY_SEED


def route_cache_key_seed_from_env() -> int:
    # These flags describe the worker cache layout, including on standalone
    # frontends whose local TP/CP topology differs from the prefill workers.
    # Workers validate the V4.1 model and supported topology at startup.
    enabled_values = {"1", "true", "yes", "on"}
    if os.environ.get("DSV41_SWA_BOUNDED_REPLAY", "0").lower() not in enabled_values:
        return 0
    if os.environ.get("DSV41_CED", "0").lower() not in enabled_values:
        raise ValueError("DSV41_SWA_BOUNDED_REPLAY requires DSV41_CED=1 at startup")
    return DSV41_SWA_BOUNDED_REPLAY_CACHE_KEY_SEED


def route_cache_keys_for_page_rr(
    block_cache_keys: List[int], page_rr_enabled: bool, cp_size: int
) -> List[int]:
    # Input keys must be computed at physical-block granularity. Page-RR only
    # changes which existing rolling-hash keys are routed to flexlb; it does not
    # change the request hash block size to virtualBlockSize.
    if not page_rr_enabled or cp_size <= 1:
        return block_cache_keys
    # Device/cache-connector Page-RR uses the last rank's logical block key as
    # the canonical key for one virtual block: K(cp_size-1), K(2*cp_size-1), ...
    return block_cache_keys[cp_size - 1 :: cp_size]
