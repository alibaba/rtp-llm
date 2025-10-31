#pragma once

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "rtp_llm/cpp/engine_base/system_prompt/SystemPrompt.h"
#include "rtp_llm/cpp/cache/CacheManager.h"
#include "rtp_llm/cpp/cache/BatchKVCacheResource.h"
#include <memory>

namespace rtp_llm {

class GenerateStream;

struct ResourceContext {
    std::shared_ptr<CacheManager>              cache_manager         = nullptr;
    std::shared_ptr<CacheManager>              propose_cache_manager = nullptr;
    std::shared_ptr<SystemPrompt>              system_prompt         = nullptr;
    bool                                       reuse_cache{false};
    bool                                       enable_3fs{false};
    bool                                       enable_memory_block_cache{false};
    bool                                       use_cache_store{false};
    std::vector<std::shared_ptr<CacheManager>> mtp_cache_managers;
};

class StreamCacheResource {
public:
    StreamCacheResource(GenerateStream*        stream,
                        const ResourceContext& resource_context,
                        bool                   need_release_resource = true,
                        const std::string&     adapter_name          = ""):
        stream_(stream),
        resource_context_(resource_context),
        block_update_mapping_(),
        need_release_resource_(need_release_resource),
        adapter_name_(adapter_name) {}
    ~StreamCacheResource() {
        releaseResource();
    }
    void                       init(int batch_size);
    void                       constructCacheKey();
    void                       reConstructCacheKeys();
    bool                       hasCacheKeys() const;
    const std::vector<size_t>& cacheKeys(int32_t batch_id) const;
    absl::StatusOr<int>        initKVBlock(int token_capacity, size_t reserve_step = 0);
    absl::StatusOr<int>        incrKVBlock(int token_capacity, size_t reserve_step = 0);
    void                       fakeInitKVBlock();
    int                        tryReleaseKVBlock(size_t nums);
    absl::Status               releaseSequenceKVCache(size_t total_seq_len, size_t release_seq_len);
    void                       freeBatchBlocks(size_t batch_id, std::vector<int>& blocks);
    void                       releaseResource();
    int                        singleBatchNeedBlocks(int seq_len) const;
    int                        maxBlockSize() const;
    int                        mallocFailedTimes() const;

    const BatchKVCacheResource& kvCache() const;
    void                        setKVCache(const BatchKVCacheResource& kv_cache_resource);

    // update kv block based on the source of new batches and generate block copy mapping.
    // used in beam search or multiple return sequences
    //
    // @params block_src_batch: [new_batch_size] int, indicating the blocks of batch i should be
    //                           forked from old batch block_src_batch[i],
    // @params copy_last_block: bool, if ture, copy the last block from the old batch for each batch
    //
    // Note: This method may allocate and free KV cache blocks, but the caller must
    // execute the block copy maunually (e.g., via `getKVBlockUpdateMapping` and
    // `CacheManager::blockBatchCopy`) before using the cache
    //
    // Example: given old batch size 3, block_src_batch = [1, 2, 2, 2], the copy mapping of
    // old blocks to new blocks is
    //
    // old batch 0 --free  /--- new batch 0
    // old batch 1 -------/ /-- new batch 1
    // old batch 2 --------+--- new batch 2
    //                      \-- new batch 3
    //
    // @returns true if success, false otherwise
    //
    bool updateKVBlock(const std::vector<int>& block_src_batch, bool copy_last_block);

    // clear block copy mapping
    void clearKVBlockUpdateMapping() {
        block_update_mapping_.clear();
    }

    // get block copy mapping of last kv cache update
    const std::vector<BlockIdPair>& getKVBlockUpdateMapping() const {
        return block_update_mapping_;
    }

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

    bool reuseCache() const;
    bool enable3FS() const;
    bool enableMemoryBlockCache() const;

    std::string debugString() const {
        std::stringstream debug_string;
        debug_string << "StreamCacheResource {"
                     << "need_release_resource: " << need_release_resource_ << ", batch_resource: [";

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
    GenerateStream*          stream_;
    BatchKVCacheResource     batch_resource_;
    ResourceContext          resource_context_;
    std::vector<BlockIdPair> block_update_mapping_;
    bool                     last_block_aligned_    = false;
    bool                     need_release_resource_ = true;
    int                      malloc_failed_times_   = 0;
    bool                     fake_inited_           = false;
    const std::string        adapter_name_;
};

}  // namespace rtp_llm
