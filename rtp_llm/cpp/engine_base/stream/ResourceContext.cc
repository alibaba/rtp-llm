#include "rtp_llm/cpp/engine_base/stream/ResourceContext.h"
#include "rtp_llm/cpp/utils/Logger.h"

namespace rtp_llm {

void ResourceContext::initCacheConfig(const KVCacheConfig&       kv_cache_config,
                                      const FIFOSchedulerConfig& scheduler_config,
                                      int64_t                    max_seq_len) {
    reuse_cache                = kv_cache_config.reuse_cache;
    enable_memory_cache        = kv_cache_config.enable_memory_cache;
    enable_remote_cache        = kv_cache_config.enable_remote_cache;
    enable_device_cache        = kv_cache_config.enable_device_cache;
    write_cache_sync           = kv_cache_config.write_cache_sync;
    enable_tiered_memory_cache = kv_cache_config.enable_tiered_memory_cache;

    if (kv_cache_config.device_cache_min_free_blocks > 0) {
        device_cache_min_free_blocks = kv_cache_config.device_cache_min_free_blocks;
    } else {
        int64_t       max_prefill_tokens = scheduler_config.max_context_batch_size * max_seq_len;
        const int64_t max_batch_tokens   = scheduler_config.max_batch_tokens_size;
        if (max_batch_tokens > 0) {
            max_prefill_tokens = std::min(max_prefill_tokens, max_batch_tokens);
        }
        const int64_t block_size     = kv_cache_config.seq_size_per_block;
        device_cache_min_free_blocks = (max_prefill_tokens + block_size - 1) / block_size;
        RTP_LLM_LOG_INFO("device_cache_min_free_blocks auto-set to %ld"
                         " (max_context_batch_size=%ld, max_seq_len=%ld,"
                         " max_batch_tokens_size=%ld, seq_size_per_block=%d)",
                         device_cache_min_free_blocks,
                         scheduler_config.max_context_batch_size,
                         max_seq_len,
                         max_batch_tokens,
                         kv_cache_config.seq_size_per_block);
    }
}

}  // namespace rtp_llm
