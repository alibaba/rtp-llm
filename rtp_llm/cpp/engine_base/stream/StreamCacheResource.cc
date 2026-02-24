#include "rtp_llm/cpp/engine_base/stream/StreamCacheResource.h"
#include "rtp_llm/cpp/engine_base/stream/GenerateStream.h"
#include "rtp_llm/cpp/utils/HashUtil.h"
#include "rtp_llm/cpp/core/BufferHelper.h"
#include "rtp_llm/cpp/cache/Types.h"
#include "rtp_llm/cpp/cache/connector/KVCacheConnectorReadWriteContext.h"
#include "rtp_llm/cpp/engine_base/stream/CompleteTokenIds.h"

using namespace std;

namespace rtp_llm {

// ----------------------------- KVCacheConnectorReadWriteContextImpl -----------------------------

class KVCacheConnectorReadWriteContextImpl: public KVCacheConnectorReadWriteContext {
public:
    KVCacheConnectorReadWriteContextImpl(const std::shared_ptr<BatchKVCacheResource>& batch_resource,
                                         const std::shared_ptr<Meta>&                 meta):
        batch_resource_(batch_resource), meta_(meta) {}
    ~KVCacheConnectorReadWriteContextImpl() override = default;

public:
    const KVCacheResource& kvCacheResource() const override {
        return batch_resource_->cacheResource(0);
    }
    const std::shared_ptr<Meta>& meta() const override {
        return meta_;
    }

private:
    std::shared_ptr<BatchKVCacheResource> batch_resource_;
    std::shared_ptr<Meta>                 meta_;
};

class MetaImpl: public Meta {
public:
    MetaImpl(bool enable_memory_cache): enable_memory_cache_(enable_memory_cache) {}
    virtual ~MetaImpl() = default;

public:
    bool enableMemoryCache() const override {
        return enable_memory_cache_;
    }

private:
    bool enable_memory_cache_{true};
};

// ----------------------------- StreamCacheResource -----------------------------

void StreamCacheResource::init(int batch_size) {
    batch_kv_cache_resource_->resetBatchSize(batch_size);
    int layer_all_num = 0;
    if (resource_context_.cache_manager) {  // cache manager is null when warmup
        layer_all_num = resource_context_.cache_manager->cacheConfig().layer_all_num;
    }
    batch_kv_cache_resource_->initGroups(1, layer_all_num);
}

void StreamCacheResource::releaseResource() {
    if (!resource_context_.cache_manager) {
        return;
    }
    // do not reuse cache from stopped beam search streams, whose states are likely corrupted
    if (!need_release_resource_ && (!stream_->hasNumBeams() || !stream_->stoppedWithoutLock())) {
        return;
    }
    tryReleaseKVBlock(curBlocksNum());
    batch_kv_cache_resource_->clearBlocks();
}

int StreamCacheResource::tryReleaseKVBlock(size_t nums) {
    RTP_LLM_LOG_DEBUG("stream [%ld] try release [%lu] blocks", stream_->streamId(), nums);

    if (fake_inited_) {
        int max_blocks_num = curBlocksNum();
        int batch_size     = batch_kv_cache_resource_->batchSize();
        batch_kv_cache_resource_->clearBlocks();
        batch_kv_cache_resource_->resetBatchSize(batch_size);
        fake_inited_ = false;
        return max_blocks_num;
    }

    // NOTE: Currently only support releasing all blocks
    // Partial release (shrink) is not supported yet
    int total_blocks = curBlocksNum();
    RTP_LLM_CHECK(nums == total_blocks);

    if (total_blocks > 0) {
        // if (reuseCache() && (stream_->finishedWithoutLock() || stream_->isRemoteRunningWithoutLock())) {
        //     // save cache to gpu
        //     if (enableDeviceCache()) {
        //         InsertInfo insert_info{batch_kv_cache_resource_, stream_->completeTokenIdsPtr(), false};
        //         resource_context_.cache_manager->insertIntoCache(insert_info);
        //     }
        //     // save cache to connector
        //     storeCacheAsync();
        // }

        FreeInfo free_info{batch_kv_cache_resource_, stream_->completeTokenIdsPtr()};
        free_info.request_id = stream_->streamId();

        resource_context_.cache_manager->free(free_info);
    }

    return total_blocks;
}

// TODO, 等待删除。
int StreamCacheResource::singleBatchNeedBlocks(int seq_len) const {
    return resource_context_.cache_manager->singleBatchNeedBlocks(batch_kv_cache_resource_, seq_len);
}

// TODO(xinfei.sxf) 保证这个函数的原子性
absl::Status StreamCacheResource::initKVBlock(size_t reserve_step) {
    auto status = incrKVBlock(reserve_step);
    if (status.ok() && reuseCache()) {
        insertIntoCache();
    }
    // load cache from connector
    loadCacheSync();
    return absl::OkStatus();
}

absl::Status StreamCacheResource::incrKVBlock(size_t reserve_step) {
    // TODO(xinfei.sxf) add reserver_blocks
    if (fake_inited_) {
        return absl::InternalError("fake inited not allow to incr block");
    }

    MallocInfo malloc_info;
    malloc_info.batch_kv_cache_resource = batch_kv_cache_resource_;
    malloc_info.complete_token_ids      = stream_->completeTokenIdsPtr();
    malloc_info.request_id              = stream_->streamId();
    malloc_info.epoch                   = stream_->batchEpoch();
    malloc_info.verbose                 = malloc_failed_times_ >= 10 ? malloc_failed_times_ % 100 == 0 : true;
    malloc_info.enable_device_cache     = reuseCache() && enableDeviceCache();

    malloc_info.complete_token_ids->setReserveStep(reserve_step);
    auto result = resource_context_.cache_manager->malloc(malloc_info);
    if (!result.success) {
        malloc_failed_times_++;
        return absl::InternalError("malloc failed");
    }

    if (result.reuse_len > 0) {
        stream_->setReuseLength(result.reuse_len);
        stream_->setMtpTokenIndex(result.reuse_len);
        stream_->setInitialReuseLength(result.reuse_len);
        stream_->setLocalReuseLength(result.reuse_len);
    }

    return absl::OkStatus();
}

// TODO, delete it soon
int StreamCacheResource::curBlocksNum() const {
    return batch_kv_cache_resource_->curBlocksNum();
}

const BatchKVCacheResource& StreamCacheResource::kvCache() const {
    batch_kv_cache_resource_->check();
    return *batch_kv_cache_resource_;
}

BatchKVCacheResource& StreamCacheResource::kvCacheMutable() {
    batch_kv_cache_resource_->check();
    return *batch_kv_cache_resource_;
}

void StreamCacheResource::setKVCache(const BatchKVCacheResource& kv_cache_resource) {
    *batch_kv_cache_resource_ = kv_cache_resource;
}

bool StreamCacheResource::updateKVBlock(const std::vector<int>& block_src_batch, bool copy_last_block) {
    return resource_context_.cache_manager->updateKVBlock(
        batch_kv_cache_resource_, block_src_batch, copy_last_block, block_update_mapping_);
}

bool StreamCacheResource::hasCacheKeys() const {
    return batch_kv_cache_resource_->hasCacheKeys();
}

const CacheKeysType& StreamCacheResource::cacheKeys(int32_t batch_id) const {
    return batch_kv_cache_resource_->cacheKeys(batch_id);
}

void StreamCacheResource::fakeInitKVBlock() {
    fake_inited_ = true;
    batch_kv_cache_resource_->resetBatchSize(stream_->maxBatchSize());
    batch_kv_cache_resource_->resizeBlocks(stream_->seqLength(), 0);
}

int StreamCacheResource::mallocFailedTimes() const {
    return malloc_failed_times_;
}

bool StreamCacheResource::reuseCache() const {
    return resource_context_.reuse_cache && stream_->reuseCache();
}

bool StreamCacheResource::enable3FS() const {
    return resource_context_.enable_3fs && stream_->enable3FS();
}

bool StreamCacheResource::enableDeviceCache() const {
    return resource_context_.enable_device_cache && stream_->enableDeviceCache();
}

bool StreamCacheResource::enableMemoryCache() const {
    return resource_context_.enable_memory_cache && stream_->enableMemoryCache();
}

void StreamCacheResource::loadCacheSync() {
    auto meta               = std::make_shared<MetaImpl>(reuseCache() && enableMemoryCache());
    auto connector_context  = std::make_shared<KVCacheConnectorReadWriteContextImpl>(batch_kv_cache_resource_, meta);
    auto load_cache_context = resource_context_.cache_manager->asyncLoadCache(connector_context);
    waitLoadCacheDone(load_cache_context);
}

void StreamCacheResource::waitLoadCacheDone(const std::shared_ptr<AsyncContext>& load_context) {
    if (!load_context) {
        return;
    }
    load_context->waitDone();
    if (!(load_context->success())) {
        RTP_LLM_LOG_WARNING("load cache done but not success, stream: [%ld]", stream_->streamId());
        return;
    }
    auto read_context = std::dynamic_pointer_cast<FusedAsyncReadContext>(load_context);
    if (!read_context) {
        RTP_LLM_LOG_WARNING("load cache success but cast context failed, stream: [%ld]", stream_->streamId());
        return;
    }
    const int total_reuse_len  = read_context->resource()->reuseBlockNum() * seqSizePerBlock();
    const int memory_reuse_len = read_context->resource()->memoryReuseBlockNum() * seqSizePerBlock();
    stream_->setInitialReuseLength(total_reuse_len);
    stream_->setReuseLength(total_reuse_len);
    stream_->setLocalReuseLength(total_reuse_len);
    stream_->setMtpTokenIndex(total_reuse_len);
    stream_->setMemoryReuseLength(memory_reuse_len);
}

void StreamCacheResource::storeCacheAsync() {
    auto meta              = std::make_shared<MetaImpl>(reuseCache() && enableMemoryCache());
    auto connector_context = std::make_shared<KVCacheConnectorReadWriteContextImpl>(batch_kv_cache_resource_, meta);
    auto store_context     = resource_context_.cache_manager->asyncStoreCache(connector_context);
    // wait done is for smoke test only
    if (resource_context_.write_cache_sync) {
        waitStoreCacheDone(store_context);
    }
}

void StreamCacheResource::waitStoreCacheDone(const std::shared_ptr<AsyncContext>& store_context) {
    if (!store_context) {
        return;
    }
    while (!store_context->done()) {
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
}

void StreamCacheResource::insertIntoCache() {
    if (!resource_context_.cache_manager) {
        return;
    }
    if (batch_kv_cache_resource_->curBlocksNum() == 0) {
        return;
    }
    auto       insert_resource = batch_kv_cache_resource_->copy();
    InsertInfo insert_info{insert_resource, stream_->completeTokenIdsPtr(), false};
    insert_info.epoch = stream_->batchEpoch();
    resource_context_.cache_manager->insertIntoCache(insert_info);
}

std::string StreamCacheResource::debugString() const {
    std::stringstream debug_string;
    debug_string << "StreamCacheResource { stream_id: " << stream_->streamId()
                 << ", need_release_resource: " << need_release_resource_ << ", batch_resource: [";

    for (size_t i = 0; i < batch_kv_cache_resource_->batchSize(); i++) {
        debug_string << " [";
        const auto& blocks = batch_kv_cache_resource_->blocks(i);
        for (size_t j = 0; j < blocks.size(); j++) {
            debug_string << blocks[j] << ", ";
        }
        debug_string << "],";
    }
    debug_string << ", cache_keys: ";
    for (size_t i = 0; i < batch_kv_cache_resource_->batchSize(); i++) {
        debug_string << " [";
        const auto& cache_keys = batch_kv_cache_resource_->cacheKeys(i);
        for (size_t j = 0; j < cache_keys.size(); j++) {
            debug_string << cache_keys[j] << ", ";
        }
        debug_string << "],";
    }
    debug_string << "}";
    return debug_string.str();
}

}  // namespace rtp_llm
