from typing import List


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
