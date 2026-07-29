#pragma once

namespace torch_ext {
struct PyCacheStoreInputs;
struct LayerKVCache;
}  // namespace torch_ext

namespace rtp_llm {

// Python-facing write contract. The concrete writer owns scheduling and CacheStore
// state. Kept dependency-free so implementations do not pull in OpDefs.h for the
// interface alone; the parameter types only need to be complete at the call site.
class CacheStoreWriter {
public:
    virtual ~CacheStoreWriter() = default;

    virtual void write(const torch_ext::PyCacheStoreInputs& cache_store_inputs,
                       const torch_ext::LayerKVCache&       layer_kv) = 0;
};

}  // namespace rtp_llm
