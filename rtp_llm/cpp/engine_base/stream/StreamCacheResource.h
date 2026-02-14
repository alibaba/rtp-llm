#pragma once

#include <memory>
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "rtp_llm/cpp/engine_base/stream/ResourceContext.h"
#include "rtp_llm/cpp/cache/BatchKVCacheResource.h"

namespace rtp_llm {

class AsyncContext;
class GenerateStream;

class StreamCacheResource {
public:
    StreamCacheResource(GenerateStream*        stream,
                        const ResourceContext& resource_context,
                        bool                   need_release_resource = true,
                        const std::string&     adapter_name          = ""):
        stream_(stream),
        batch_kv_cache_resource_(std::make_shared<BatchKVCacheResource>()),
        resource_context_(resource_context),
        need_release_resource_(need_release_resource) {}

    ~StreamCacheResource() {
        releaseResource();
    }

    void                 init(int batch_size);
    bool                 hasCacheKeys() const;
    const CacheKeysType& cacheKeys(int32_t batch_id) const;
    absl::Status         initKVBlock(size_t reserve_step = 0);
    absl::Status         incrKVBlock(size_t reserve_step = 0);
    void                 fakeInitKVBlock();
    int                  tryReleaseKVBlock(size_t nums);
    void                 freeBatchBlocks(size_t batch_id, std::vector<int>& blocks);
    void                 releaseResource();

    // TODO, remove this after remove fallback
    int singleBatchNeedBlocks(int seq_len, int reserve_step) const;

    int curBlocksNum() const;
    int mallocFailedTimes() const;

    const BatchKVCacheResource& kvCache() const;
    BatchKVCacheResource&       kvCacheMutable();
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
    // `KVCacheManager::blockBatchCopy`) before using the cache
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

    bool reuseCache() const;
    bool enable3FS() const;
    bool enableDeviceCache() const;
    bool enableMemoryCache() const;

    std::string debugString() const {
        std::stringstream debug_string;
        debug_string << "StreamCacheResource {"
                     << "need_release_resource: " << need_release_resource_ << ", batch_resource: ";

        debug_string << batch_kv_cache_resource_->debugString();

        debug_string << "}";
        return debug_string.str();
    }

private:
    void loadCacheSync();
    void waitLoadCacheDone(const std::shared_ptr<AsyncContext>& load_context);
    void storeCacheAsync();
    void waitStoreCacheDone(const std::shared_ptr<AsyncContext>& store_context);

private:
    GenerateStream*          stream_;
    BatchKVCacheResourcePtr  batch_kv_cache_resource_;
    ResourceContext          resource_context_;
    std::vector<BlockIdPair> block_update_mapping_;

    bool need_release_resource_ = true;
    bool last_block_aligned_    = false;
    int  malloc_failed_times_   = 0;
    bool fake_inited_           = false;
};

}  // namespace rtp_llm
