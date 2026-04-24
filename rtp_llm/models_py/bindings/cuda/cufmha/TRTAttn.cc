#include "rtp_llm/models_py/bindings/cuda/cufmha/TRTAttn.h"

namespace rtp_llm {

void TRTAttn::setKvCache(KVBlockArray& kv_block_array, const KvCacheInfo& kv_cache) {
    kv_block_array.mPrimaryPoolPtr = kv_cache.kv_cache_buffer.data_ptr();
    if (kv_cache.kv_scale_buffer.defined()) {
        kv_block_array.scale = kv_cache.kv_scale_buffer.data_ptr();
    }
}

}  // namespace rtp_llm
