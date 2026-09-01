#pragma once

#include <cstddef>
#include <memory>
#include <torch/torch.h>

namespace torch_ext {
struct LayerKVCache;
struct PyCacheStoreInputs;
}  // namespace torch_ext

namespace rtp_llm {

class CacheConfig;
class CacheStore;

// Writes per-layer KV blocks of context-batch requests to the given CacheStore
// (used for prefill/decode disaggregation). Skipped on warmup, when pd_separation
// is off, when context_batch_size == 0, or when cache_store is null.
void runtimeWriteCacheStore(const torch_ext::PyCacheStoreInputs& cache_store_inputs,
                            const torch_ext::LayerKVCache&       layer_kv,
                            const CacheConfig&                   cache_config,
                            std::shared_ptr<CacheStore>          cache_store,
                            size_t                               cache_model_id,
                            int                                  cp_rank,
                            int                                  cp_size,
                            std::shared_ptr<torch::Event>        pre_created_event);

}  // namespace rtp_llm
