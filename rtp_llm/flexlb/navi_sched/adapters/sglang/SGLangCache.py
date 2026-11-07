from __future__ import annotations

UINT64_MASK = (1 << 64) - 1


def prompt_block_hashes(
    token_ids: list[int], page_size: int, *, extra_key=None, cache_salt=None
) -> list[int]:
    """Use SGLang's Dynamo event identity, including its legacy salt chain."""
    if page_size <= 0:
        raise ValueError("page_size must be positive")
    from sglang.srt.disaggregation.kv_events import (
        kv_event_namespace_seed,
        namespaced_block_hash,
    )
    from sglang.srt.mem_cache.cpp_utils.native_hash import get_native_hash
    from sglang.srt.mem_cache.utils import hash_str_to_int64

    complete_length = len(token_ids) // page_size * page_size
    if not complete_length:
        return []
    prior = None
    cache_salt = cache_salt or None
    if cache_salt is not None:
        import hashlib

        prior = hashlib.sha256(
            b"sglang-cache-salt-v1\0" + cache_salt.encode("utf-8")
        ).digest()
    hashes = get_native_hash(token_ids[:complete_length], prior, page_size)
    seed = kv_event_namespace_seed(extra_key=extra_key, cache_salt=cache_salt)
    return [
        namespaced_block_hash(hash_str_to_int64(value), namespace_seed=seed)
        & UINT64_MASK
        for value in hashes
    ]


class SGLangCacheState:
    def __init__(
        self, bridge, total_blocks: int, page_size: int, *, hybrid_tail_required=False
    ):
        if total_blocks < 0 or page_size <= 0:
            raise ValueError("invalid SGLang cache capacity")
        self._bridge = bridge
        self._total_blocks = total_blocks
        self._page_size = page_size
        self._hashes: set[int] = set()
        self.hybrid_tail_required = hybrid_tail_required
        self._tail_hashes: set[int] = set()
        self._loading_hashes: set[int] = set()

    def finish_loads(self):
        if self._loading_hashes:
            added = self._loading_hashes - self._hashes
            self._publish(added=sorted(added))
            self._hashes.update(added)
            self._loading_hashes.clear()

    def _publish(self, added=(), removed=(), tails_added=(), tails_removed=()):
        if self.hybrid_tail_required:
            self._bridge.report_cache_updated(
                list(added),
                list(removed),
                list(added),
                list(removed),
                list(tails_added),
                list(tails_removed),
            )
        else:
            self._bridge.report_cache_inserted(list(added))
            self._bridge.report_cache_removed(list(removed))

    def invalidate_tails(self):
        if self._tail_hashes:
            self._publish(tails_removed=sorted(self._tail_hashes))
            self._tail_hashes.clear()

    def on_match(self, key, result):
        if not self.hybrid_tail_required:
            return
        if (
            result.host_hit_length
            or result.swa_host_hit_length
            or result.mamba_host_hit_length
        ):
            return
        length = result.device_prefix_len
        if not length or length % self._page_size:
            return
        hashes = prompt_block_hashes(
            key[:length],
            self._page_size,
            extra_key=key.extra_key,
            cache_salt=key.cache_salt,
        )
        if (
            hashes
            and hashes[-1] in self._hashes
            and hashes[-1] not in self._tail_hashes
        ):
            self._publish(tails_added=[hashes[-1]])
            self._tail_hashes.add(hashes[-1])

    def on_event(self, event, *, device_ready=True) -> None:
        from sglang.srt.disaggregation.kv_events import (
            AllBlocksCleared,
            BlockRemoved,
            BlockStored,
            StorageMedium,
        )

        if isinstance(event, AllBlocksCleared):
            removed = sorted(self._hashes)
            self._publish(removed=removed, tails_removed=sorted(self._tail_hashes))
            self._hashes.clear()
            self._tail_hashes.clear()
            self._loading_hashes.clear()
            return
        if event.medium not in (None, StorageMedium.GPU):
            return
        if isinstance(event, BlockStored):
            # Partial tail pages cannot match NaviSched's complete block prefix.
            if event.block_size < self._page_size:
                return
            if event.block_size != self._page_size:
                raise ValueError("SGLang KV event page size changed")
            hashes = {int(value) & UINT64_MASK for value in event.block_hashes}
            added = hashes - self._hashes
            if len(self._hashes | self._loading_hashes | added) > self._total_blocks:
                raise ValueError("SGLang KV events exceed advertised cache capacity")
            if not device_ready:
                self._loading_hashes.update(added)
                return
            self._publish(added=sorted(added))
            self._hashes.update(added)
        elif isinstance(event, BlockRemoved):
            hashes = {int(value) & UINT64_MASK for value in event.block_hashes}
            self._loading_hashes.difference_update(hashes)
            removed = hashes & self._hashes
            tails_removed = removed & self._tail_hashes
            self._publish(removed=sorted(removed), tails_removed=sorted(tails_removed))
            self._hashes.difference_update(removed)
            self._tail_hashes.difference_update(tails_removed)
