#pragma once

#include <memory>

#include "rtp_llm/models_py/bindings/core/OpData.h"

namespace rtp_llm {

class CacheStore;

// Writes per-layer KV blocks of context-batch requests to the given CacheStore
// (used for prefill/decode disaggregation). Skipped on warmup, when pd_separation
// is off, when context_batch_size == 0, or when cache_store is null.
void runtimeWriteCacheStore(const CacheStoreInputs&     inputs,
                            const KvCacheInfo&          kv_cache,
                            bool                        mla_kvcache,
                            std::shared_ptr<CacheStore> cache_store);

void execWriteCacheStore(const CacheStoreInputs&     inputs,
                         const KvCacheInfo&          kv_cache,
                         bool                        mla_kvcache,
                         std::shared_ptr<CacheStore> cache_store);

}  // namespace rtp_llm
