#pragma once

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "maga_transformer/cpp/system_prompt/SystemPrompt.h"
#include "maga_transformer/cpp/cache/CacheManager.h"
#include "maga_transformer/cpp/cache/BatchKVCacheResource.h"
#include <memory>

namespace rtp_llm {

class GenerateStream;

struct ResourceContext {
    std::shared_ptr<CacheManager>   cache_manager = nullptr;
    std::shared_ptr<CacheManager>   propose_cache_manager = nullptr;
    std::shared_ptr<SystemPrompt>   system_prompt = nullptr;
    bool                            reuse_cache{false};
    bool                            use_cache_store{false};
};

class StreamCacheResource {
public:
    StreamCacheResource(
            GenerateStream* stream,
            const ResourceContext& resource_context,
            bool need_release_resource = true):
        stream_(stream),
        resource_context_(resource_context),
        need_release_resource_(need_release_resource) {}
    ~StreamCacheResource() {
        releaseResource();
    }
    void init(int batch_size);
    void constructCacheKey();
    void reConstructCacheKeys();
    bool hasCacheKeys() const;
    const std::vector<int64_t>& cacheKeys(int32_t batch_id) const;
    absl::StatusOr<int> initKVBlock(int token_capacity, size_t reserve_step = 0);
    absl::StatusOr<int> incrKVBlock(int token_capacity, size_t reserve_step = 0);
    int  tryReleaseKVBlock(size_t nums);
    absl::Status releaseSequenceKVCache(size_t total_seq_len, size_t release_seq_len);
    void freeBatchBlocks(size_t batch_id, std::vector<int>& blocks);
    void releaseResource();
    int  singleBatchNeedBlocks(int seq_len) const;
    int  maxBlockSize() const;

    const BatchKVCacheResource& kvCache() const;
    void                         setKVCache(const BatchKVCacheResource& kv_cache_resource);

    void beamSearchKvCacheUpdate(const std::vector<int>& beam_index);


    const ResourceContext& resourceContext() const {
        return resource_context_;
    }

    int seqSizePerBlock() const {
        return resource_context_.cache_manager->cacheConfig().seq_size_per_block;
    }

    void setNeedReleaseResource(bool need_release_resource) {
        need_release_resource_ = need_release_resource;
    }

    void setStream(GenerateStream* stream) {
        stream_ = stream;
    }

    std::string debugString() const {
        std::stringstream debug_string;
        debug_string << "StreamCacheResource {"
                     << "need_release_resource: " << need_release_resource_
                     << ", batch_resource: [";

        for (size_t i = 0; i < batch_resource_.batchSize(); i++) {
            debug_string << " [";
            for (size_t j = 0; j < batch_resource_.batch_block_id[i].size(); j++) {
                debug_string << batch_resource_.batch_block_id[i][j] << " ";
            }
            debug_string << "],";
        }

        debug_string << "}";
        return debug_string.str();
    }

private:
    GenerateStream*                 stream_;
    BatchKVCacheResource            batch_resource_;
    ResourceContext                 resource_context_;
    bool                            last_block_aligned_ = false;
    bool                            need_release_resource_ = true;
    int                             malloc_failed_times_ = 0;
};

}  // namespace rtp_llm
