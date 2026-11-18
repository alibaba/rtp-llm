from __future__ import annotations

from collections.abc import Iterable
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from vllm.v1.core.kv_cache_utils import KVCacheBlock
    from vllm.v1.request import Request


def promote_pool_blocks(self, blocks: Iterable[KVCacheBlock]) -> int:
    """Move idle cached blocks to the MRU end of the eviction queue.

    Unlike :meth:`touch`, this does not reserve blocks or change their
    reference counts. ``blocks`` must be ordered from first to last eviction
    priority; callers use this to preserve prefix-cache tie breaking.
    """
    promoted_blocks: list[KVCacheBlock] = []
    seen_block_ids: set[int] = set()
    for block in blocks:
        if block.block_id in seen_block_ids:
            continue
        seen_block_ids.add(block.block_id)
        if block.ref_cnt != 0 or block.is_null or block.block_hash is None:
            continue
        promoted_blocks.append(block)

    if not promoted_blocks:
        return 0

    current = self.free_block_queue.fake_free_list_tail.prev_free_block
    for block in reversed(promoted_blocks):
        if current is not block:
            break
        current = block.prev_free_block
    else:
        return len(promoted_blocks)

    for block in promoted_blocks:
        self.free_block_queue.remove(block)

    self.free_block_queue.append_n(promoted_blocks)
    return len(promoted_blocks)


def promote_request_blocks(self, request: Request) -> int:
    """Refresh the eviction priority of a request's cached prefix.

    This is a non-reserving lookup for requests that may wait before actual
    KV allocation. Cache availability is checked again when the request is
    scheduled.
    """
    if not self.enable_caching or request.skip_reading_prefix_cache:
        return 0

    max_cache_hit_length = request.num_tokens - 1
    computed_blocks, _ = self.coordinator.find_longest_cache_hit(
        request.block_hashes, max_cache_hit_length
    )

    # Match free() tie breaking: deeper prefix blocks are evicted before
    # shallower blocks when all of them have the same access time.
    max_group_blocks = max((len(group) for group in computed_blocks), default=0)
    eviction_order = (
        group[block_index]
        for block_index in range(max_group_blocks - 1, -1, -1)
        for group in computed_blocks
        if block_index < len(group)
    )
    return promote_pool_blocks(self.block_pool, eviction_order)


class VLLMCacheObserver:
    def __init__(self, listener):
        self.listener = listener

    def stored(self, blocks, block_hashes, group_id, request_id):
        hashes = tuple(
            block_hashes[i] for i, block in enumerate(blocks) if not block.is_null
        )
        if hashes:
            self.listener("stored", group_id, hashes, str(request_id), "request_store")

    def removed(self, block_hash, request_id, cause):
        from vllm.v1.core.kv_cache_utils import get_block_hash, get_group_id

        self.listener(
            "removed",
            get_group_id(block_hash),
            (get_block_hash(block_hash),),
            request_id,
            cause,
        )

    def cleared(self):
        self.listener("cleared", None, (), None, "prefix_cache_reset")
