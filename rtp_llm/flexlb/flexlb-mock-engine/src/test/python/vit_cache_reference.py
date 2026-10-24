"""Regenerate CPU golden traces using the owning real cache, without CUDA or Mock code.

Run from the repository root: python3 <this file> <output.json>.
"""
import json
import sys
from pathlib import Path
import torch
from rtp_llm.multimodal.mm_embedding_cache import MMEmbeddingCache, MMHashKeyCache, _TensorMemoryPool


def snapshot(cache):
    stats = cache.stats()
    return dict(keys=sorted(cache.resident_tiers()), resident=stats['resident_bytes'],
                used=stats['cpu_pool_used_bytes'], free=stats['cpu_pool_free_bytes'],
                largest=stats['cpu_pool_largest_free_block_bytes'])


def embedding_trace(capacity, operations):
    cache = MMEmbeddingCache(gpu_max_bytes=0, cpu_max_bytes=capacity)
    result = []
    for op in operations:
        action, key = op[:2]
        if action == 'put':
            _, entry = cache.try_acquire(key)
            entry.complete(torch.arange(op[2], dtype=torch.uint8))
        elif action == 'remove':
            cache.remove(key)
        elif action == 'get':
            state, entry = cache.try_acquire(key)
            assert state == 'complete'
            entry.wait()
        result.append(snapshot(cache))
    return dict(capacity=capacity, operations=operations, expected=result)


def hash_trace():
    cache = MMHashKeyCache(max_bytes=1100)
    generation = '0' * 32
    operations = [['put', 'a', 4], ['put', 'é', 4], ['probe', 'a'], ['consume', 'a'],
                  ['put', '中', 4], ['put', '🙂', 4], ['put', 'a', 4096]]
    result = []
    for op in operations:
        if op[0] == 'put':
            cache.put(op[1], [torch.arange(op[2], dtype=torch.int32)], generation)
        elif op[0] == 'probe':
            # metadata() uses the same non-consuming peek; do not call get().
            assert op[1] in cache._entries
        else:
            assert cache.get(op[1], generation) is not None
        result.append(dict(keys=cache.keys(), resident=cache.stats()['resident_bytes']))
    return dict(capacity=1100, operations=operations, expected=result)


def hash_accounting():
    """Measure the owning cache's runtime layout without modifying real code."""
    return dict(entry_overhead=MMHashKeyCache._ENTRY_OVERHEAD,
                ascii_base=sys.getsizeof(""), latin1_base=sys.getsizeof("é") - 1,
                bmp_base=sys.getsizeof("中") - 2, ucs4_base=sys.getsizeof("🙂") - 4,
                hash_list_bytes=sys.getsizeof([item for item in (None,)]),
                tensor_object_bytes=sys.getsizeof(torch.empty(0, dtype=torch.int32)),
                record_bytes=sys.getsizeof((None, None, 0, [], False)), integer_bytes=sys.getsizeof(0))


def main():
    document = dict(hash_cache_accounting=hash_accounting(),
        embedding=[embedding_trace(32, [['put', k, 8] for k in 'abcd'] +
                    [['remove', 'b'], ['remove', 'd'], ['put', 'large', 16]]),
                   embedding_trace(32, [['put', 'small', 16], ['put', 'large', 80]]),
                   embedding_trace(16, [['put', 'a', 1], ['put', 'b', 1], ['remove', 'a'], ['put', 'c', 8]]),
                   embedding_trace(32, [['put', 'a', 16], ['put', 'b', 16], ['get', 'a'], ['put', 'c', 16]])],
        hashes=hash_trace())
    Path(sys.argv[1]).write_text(json.dumps(document, indent=2, ensure_ascii=False) + '\n')


if __name__ == '__main__':
    main()
