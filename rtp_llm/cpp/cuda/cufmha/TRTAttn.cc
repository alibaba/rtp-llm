#include "rtp_llm/cpp/cuda/cufmha/TRTAttn.h"

namespace rtp_llm {

void TRTAttn::setKvCache(KVBlockArray& kv_block_array, const KvCacheInfo& kv_cache) {
    kv_block_array.mPrimaryPoolPtr = kv_cache.kv_cache_buffer->data();
}

}  // namespace rtp_llm
