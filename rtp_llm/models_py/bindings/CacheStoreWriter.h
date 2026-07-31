#pragma once

namespace torch_ext {
struct PyCacheStoreInputs;
struct LayerKVCache;
}  // namespace torch_ext

namespace rtp_llm {

// Python-facing write contract. The concrete writer owns scheduling and CacheStore
// state. Keep this interface dependency-free so implementations do not need OpDefs.h.
class CacheStoreWriter {
public:
    virtual ~CacheStoreWriter() = default;

    virtual void write(const torch_ext::PyCacheStoreInputs& cache_store_inputs,
                       const torch_ext::LayerKVCache&       layer_kv) = 0;
};

}  // namespace rtp_llm
