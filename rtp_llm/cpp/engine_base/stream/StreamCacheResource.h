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

    ~StreamCacheResource() = default;

    void                 init(int batch_size);
    bool                 hasCacheKeys() const;
    const CacheKeysType& cacheKeys(int32_t batch_id) const;
    absl::Status         initKVBlock(size_t reserve_step = 0);
    absl::Status         waitForAllocatorLoad();
    // seq_len_override (-1 = unset) is forwarded to MallocInfo::incr_seq_len_override.
    absl::Status incrKVBlock(size_t reserve_step = 0, int seq_len_override = -1);
    void         fakeInitKVBlock(size_t reserved_blocks = 0);
    int          tryReleaseKVBlock(size_t nums);
    void         freeBatchBlocks(size_t batch_id, std::vector<int>& blocks);
    void         releaseResource();
    bool         asyncLoadCache();
    bool         loadCacheDone();

    // swap all linear groups rhs and lhs
    void swapLinearBlocks(int32_t batch_id, size_t rhs, size_t lhs);

    // TODO, remove this after remove fallback
    int singleBatchNeedBlocks(int seq_len, int reserve_step) const;

    int  curBlocksNum() const;
    int  mallocFailedTimes() const;
    bool isContextStream() const;

    const BatchKVCacheResource& kvCache() const;
    BatchKVCacheResource&       kvCacheMutable();
    void                        setKVCache(const BatchKVCacheResource& kv_cache_resource);

    // Rebuild KV block ownership for beam/multiple-return sequences.
    // This records copy mappings; caller must execute them via
    // getKVBlockUpdateMapping/KVCacheManager::blockBatchCopy before reuse.
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

    // Effective block-tokens for converting reuse_blocks → token count.
    // Under CP shard, reuseBlockNum() counts cp-virtual blocks (each spanning
    // cp_size physical blocks), so multiply by virtualBlockSize() instead of
    // raw seq_size_per_block.
    int reuseBlockTokens() const {
        const auto& mapper = resource_context_.cache_manager->cpSlotMapper();
        if (mapper && mapper->isSharded()) {
            return mapper->virtualBlockSize();
        }
        return resource_context_.cache_manager->cacheConfig().seq_size_per_block;
    }

    void setNeedReleaseResource(bool need_release_resource) {
        need_release_resource_ = need_release_resource;
    }

    bool isResourceReleased() const {
        return resource_released_;
    }

    // Borrow the owning stream pointer; used by GenerateStateMachine to query
    // async-bookkeeping state without duplicating accessors here. Lifetime is
    // bound by the GenerateStream that owns this StreamCacheResource.
    GenerateStream* stream() const {
        return stream_;
    }

    bool reuseCache() const;
    bool enableMemoryCache() const;
    bool enableRemoteCache() const;
    bool enableDeviceCache() const;
    bool enableTieredMemoryCache() const;

    void holdKVCacheForPDSep();
    void releaseKVCacheForPDSep();

    std::string debugString() const {
        std::stringstream debug_string;
        debug_string << "StreamCacheResource {"
                     << "need_release_resource: " << need_release_resource_ << ", batch_resource: ";

        debug_string << batch_kv_cache_resource_->debugString();

        debug_string << "}";
        return debug_string.str();
    }

private:
    GenerateStream*          stream_;
    BatchKVCacheResourcePtr  batch_kv_cache_resource_;
    ResourceContext          resource_context_;
    std::vector<BlockIdPair> block_update_mapping_;

    bool                          need_release_resource_ = true;
    bool                          last_block_aligned_    = false;
    int                           malloc_failed_times_   = 0;
    bool                          fake_inited_           = false;
    bool                          resource_released_     = false;
    // BlockTree host/disk load-back is an allocator prerequisite and is not a
    // connector read context. Keep it separate so connector-specific result
    // decoding/retry logic never dynamic-casts it to FusedAsyncReadContext.
    std::shared_ptr<AsyncContext> allocator_load_context_;

    // Connector reference counting for PD separation (RAII auto-release)
    std::shared_ptr<KVCacheResource> pd_kvcache_ref_;
};

}  // namespace rtp_llm
