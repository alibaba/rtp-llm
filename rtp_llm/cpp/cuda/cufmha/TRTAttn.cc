#include "rtp_llm/cpp/cuda/cufmha/TRTAttn.h"

namespace rtp_llm {

void TRTAttn::setKvCache(KVBlockArray& kv_block_array, const KvCacheInfo& kv_cache) {
    kv_block_array.mPrimaryPoolPtr = kv_cache.k_cache_buffer->data();
    if (kv_cache.k_scale_buffer) {
        kv_block_array.scale = kv_cache.k_scale_buffer->data();
    }
}

}  // namespace rtp_llm
