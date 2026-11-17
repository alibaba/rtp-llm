"""GLM53 request-state capacity expressed through main's cache-pool policy."""

from rtp_llm.ops import RoleType


def configure_decode_request_state_pool(
    kv_cache_config,
    role_type,
    max_generate_batch_size: int,
    gen_num_per_cycle: int,
    *,
    enabled: bool,
):
    """Bound Decode's KDA state pool by live requests instead of KV pages.

    Match the feature branch's two rotating states, retained read state, and
    speculative checkpoints per request. Physical block zero is reserved.
    Explicit main pool settings and shared byte arenas retain their policies.
    """
    if (
        not enabled
        or role_type != RoleType.DECODE
        or kv_cache_config.linear_pool_blocks != 0
        or kv_cache_config.shared_hybrid_cache == 1
    ):
        return None
    batch = int(max_generate_batch_size)
    blocks = 1 + batch * (3 + max(int(gen_num_per_cycle), 0))
    if batch <= 0 or blocks > 2147483647:
        raise ValueError("GLM53 Decode request-state pool must fit [2, INT_MAX]")
    kv_cache_config.linear_pool_blocks = blocks
    return blocks
