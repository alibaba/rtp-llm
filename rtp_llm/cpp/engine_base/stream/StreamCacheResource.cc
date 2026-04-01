#include "rtp_llm/cpp/engine_base/stream/StreamCacheResource.h"
#include "rtp_llm/cpp/engine_base/stream/GenerateStream.h"
#include "rtp_llm/cpp/engine_base/stream/IGenerateStreamImpl.h"
#include "rtp_llm/cpp/utils/HashUtil.h"
#include "rtp_llm/cpp/utils/ProfilingScope.h"
#include "rtp_llm/cpp/cache/Types.h"
#include "rtp_llm/cpp/cache/connector/AsyncContext.h"
#include "rtp_llm/cpp/cache/connector/KVCacheConnectorReadWriteContext.h"
#include "rtp_llm/cpp/config/RoleTypes.h"
#include "rtp_llm/cpp/engine_base/stream/CompleteTokenIds.h"
#include <thread>

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
    MetaImpl(bool enable_memory_cache, bool enable_remote_cache, std::string trace_id):
        enable_memory_cache_(enable_memory_cache), enable_remote_cache_(enable_remote_cache), trace_id_(trace_id) {}
    virtual ~MetaImpl() = default;

public:
    bool enableMemoryCache() const override {
        return enable_memory_cache_;
    }
    bool enableRemoteCache() const override {
        return enable_remote_cache_;
    }
    const std::string& trace_id() const override {
        return trace_id_;
    }
    const std::string& unique_id() const override {
        return unique_id_;
    }
    const std::vector<int64_t>& tokens() const override {
        return tokens_;
    }

    // P2P read 扩展字段
    std::shared_ptr<IGenerateStream> generateStream() const override {
        return generate_stream_;
    }

    std::shared_ptr<IGenerateStream> generate_stream_;

private:
    bool                 enable_memory_cache_{false};
    bool                 enable_remote_cache_{false};
    std::string          trace_id_;
    std::string          unique_id_ = "";
    std::vector<int64_t> tokens_;  // TODO : get tokens (remote connector)
};

// ----------------------------- StreamCacheResource -----------------------------

void StreamCacheResource::init(int batch_size) {
    batch_kv_cache_resource_->resetBatchSize(batch_size);
    int                         group_nums     = 1;
    int                         layer_all_num  = 0;
    std::vector<int>            layer_to_group = {};
    std::vector<CacheGroupType> group_types    = {};

    size_t kernel_blocks_per_kv_block = 1;
    if (resource_context_.cache_manager) {  // cache manager is null when warmup
        const auto& cache_config = resource_context_.cache_manager->cacheConfig();
        group_nums               = cache_config.groupNums();
        layer_all_num            = static_cast<int>(cache_config.layer_all_num);
        layer_to_group           = cache_config.layer_to_group_id;
        group_types              = cache_config.group_types;
        if (cache_config.kernel_seq_size_per_block > 0 && cache_config.seq_size_per_block > 0) {
            kernel_blocks_per_kv_block = cache_config.seq_size_per_block / cache_config.kernel_seq_size_per_block;
        }
    }

    batch_kv_cache_resource_->initGroups(
        group_nums, layer_all_num, layer_to_group, kernel_blocks_per_kv_block, group_types);
    resource_released_ = false;
}

void StreamCacheResource::releaseResource() {
    RTP_LLM_PROFILE_FUNCTION();
    if (!resource_context_.cache_manager) {
        return;
    }
    // Check against double release
    if (resource_released_) {
        RTP_LLM_LOG_ERROR("=== DOUBLE RELEASE CACHE RESOURCE DETECTED ===");
        RTP_LLM_LOG_ERROR("  stream_ ptr:                   %p", static_cast<void*>(stream_));
        RTP_LLM_LOG_ERROR("  stream alive (magic check):    %s",
                          stream_->isStreamAlive() ? "YES" : "NO (stream already destroyed!)");
        if (stream_->isStreamAlive()) {
            RTP_LLM_LOG_ERROR("  stream id:                     %ld", stream_->streamId());
            RTP_LLM_LOG_ERROR("  stream state:                  %s",
                              StreamStateToString(stream_->generate_status_->status).c_str());
            RTP_LLM_LOG_ERROR("  stream hasError:                %d", stream_->hasError());
            RTP_LLM_LOG_ERROR("  stream hasNumBeams:            %d", stream_->hasNumBeams());
        }
        RTP_LLM_LOG_ERROR("  batch_kv_cache_resource_ use_count: %ld", batch_kv_cache_resource_.use_count());
        RTP_LLM_LOG_ERROR("  curBlocksNum:                  %d", curBlocksNum());
        RTP_LLM_LOG_ERROR("  need_release_resource:         %d", need_release_resource_);
        RTP_LLM_LOG_ERROR("  fake_inited:                   %d", fake_inited_);
        RTP_LLM_LOG_ERROR("  batch_kv_cache_resource:       %s", batch_kv_cache_resource_->debugString().c_str());
        RTP_LLM_LOG_ERROR("  thread id:                     %lu",
                          std::hash<std::thread::id>{}(std::this_thread::get_id()));
        abort();
    }
    // do not reuse cache from stopped beam search streams, whose states are likely corrupted
    if (!need_release_resource_ && (!stream_->hasNumBeams() || !stream_->hasError())) {
        return;
    }
    RTP_LLM_LOG_DEBUG("releaseResource: stream=%ld, curBlocksNum=%d, pd_kvcache_ref=%p",
                      stream_->streamId(),
                      curBlocksNum(),
                      pd_kvcache_ref_.get());
    tryReleaseKVBlock(curBlocksNum());
    batch_kv_cache_resource_->clearBlocks();
    resource_released_ = true;
    load_cache_once_.store(false, std::memory_order_release);
}

int StreamCacheResource::tryReleaseKVBlock(size_t nums) {
    RTP_LLM_PROFILE_FUNCTION();
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
        if (reuseCache() && !stream_->hasError() && stream_->getStatus() == StreamState::FINISHED) {
            RTP_LLM_LOG_DEBUG(
                "tryReleaseKVBlock: stream=%ld, storing cache, curBlocksNum=%d", stream_->streamId(), total_blocks);
            // save cache to gpu
            if (enableDeviceCache()) {
                InsertInfo insert_info{batch_kv_cache_resource_, stream_->completeTokenIdsPtr(), false};
                resource_context_.cache_manager->insertIntoCache(insert_info);
            }
            storeCacheAsync(batch_kv_cache_resource_,
                            reuseCache() && enableMemoryCache() && !enableTieredMemoryCache(),
                            reuseCache() && enableRemoteCache());
            // only evict when succeeds
            if (enableTieredMemoryCache()) {
                evictDeviceCacheToMemory();
            }
        } else {
            RTP_LLM_LOG_DEBUG("tryReleaseKVBlock: stream=%ld, NOT storing cache, reuseCache=%d, hasError=%d, status=%s",
                              stream_->streamId(),
                              reuseCache(),
                              stream_->hasError(),
                              StreamStateToString(stream_->getStatus()).c_str());
        }

        FreeInfo free_info{batch_kv_cache_resource_, stream_->completeTokenIdsPtr()};
        free_info.request_id = stream_->streamId();

        resource_context_.cache_manager->free(free_info);
    }

    return total_blocks;
}

// TODO, 等待删除。
int StreamCacheResource::singleBatchNeedBlocks(int seq_len, int reserve_step) const {
    return resource_context_.cache_manager->singleBatchNeedBlocks(batch_kv_cache_resource_, seq_len, reserve_step);
}

// TODO(xinfei.sxf) 保证这个函数的原子性
absl::Status StreamCacheResource::initKVBlock(size_t reserve_step) {
    RTP_LLM_PROFILE_FUNCTION();
    // Decode side: first malloc should NOT use device cache, regardless of runtime config.
    // Follow-up allocations (incrKVBlock) will respect reuseCache() && enableDeviceCache().
    if (fake_inited_) {
        return absl::InternalError("fake inited not allow to incr block");
    }

    MallocInfo malloc_info;
    malloc_info.batch_kv_cache_resource = batch_kv_cache_resource_;
    malloc_info.complete_token_ids      = stream_->completeTokenIdsPtr();
    malloc_info.request_id              = stream_->streamId();
    malloc_info.verbose                 = malloc_failed_times_ >= 10 ? malloc_failed_times_ % 100 == 0 : true;

    const bool is_hybrid       = resource_context_.cache_manager->cacheConfig().groupNums() > 1;
    const bool is_decode_role  = (resource_context_.role_type == RoleType::DECODE);
    const bool is_first_malloc = (batch_kv_cache_resource_->curBlocksNum() == 0);

    if (is_hybrid && is_decode_role && is_first_malloc) {
        malloc_info.reuse_cache         = false;
        malloc_info.enable_device_cache = false;
    } else {
        malloc_info.reuse_cache         = reuseCache();
        malloc_info.enable_device_cache = reuseCache() && enableDeviceCache();
    }

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

absl::Status StreamCacheResource::incrKVBlock(size_t reserve_step) {
    RTP_LLM_PROFILE_FUNCTION();
    // TODO(xinfei.sxf) add reserver_blocks
    if (fake_inited_) {
        return absl::InternalError("fake inited not allow to incr block");
    }

    MallocInfo malloc_info;
    malloc_info.batch_kv_cache_resource = batch_kv_cache_resource_;
    malloc_info.complete_token_ids      = stream_->completeTokenIdsPtr();
    malloc_info.request_id              = stream_->streamId();
    malloc_info.verbose                 = malloc_failed_times_ >= 10 ? malloc_failed_times_ % 100 == 0 : true;
    malloc_info.reuse_cache             = reuseCache();
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

bool StreamCacheResource::asyncLoadCache() {
    // load cache from connector
    if (!reuseCache() || (!enableMemoryCache() && !enableRemoteCache())) {
        return false;
    }

    if (load_cache_context_) {
        return true;  // 已有进行中的 load 任务（幂等）
    }
    assert(reuseCache());
    auto meta              = std::make_shared<MetaImpl>(enableMemoryCache(), enableRemoteCache(), stream_->traceId());
    auto connector_context = std::make_shared<KVCacheConnectorReadWriteContextImpl>(batch_kv_cache_resource_, meta);
    load_cache_context_    = resource_context_.cache_manager->asyncLoadCache(connector_context);
    return load_cache_context_ != nullptr;
}

bool StreamCacheResource::loadCacheDone() {
    if (!load_cache_context_) {
        return true;  // 没有 context，视为已完成
    }
    if (!load_cache_context_->done()) {
        return false;  // coordinator 后台线程尚未处理完
    }
    // 加载完成（无论成功失败），更新 reuse lengths
    waitLoadCacheDone(load_cache_context_);
    if (!load_cache_context_->success()) {
        // 区分匹配失败和传输失败
        auto      read_context = std::dynamic_pointer_cast<FusedAsyncReadContext>(load_cache_context_);
        bool      should_retry = false;
        const int max_retry    = resource_context_.load_cache_retry_times;
        if (read_context && read_context->fusedMatchContext()) {
            // 检查是否有匹配到的块
            size_t matched_blocks = 0;
            for (const auto& match_ctx : read_context->fusedMatchContext()->contexts()) {
                auto async_match_ctx = std::dynamic_pointer_cast<AsyncMatchContext>(match_ctx);
                if (async_match_ctx) {
                    matched_blocks = std::max(matched_blocks, async_match_ctx->matchedBlockCount());
                }
            }
            // 如果匹配到了块（matched_blocks > 0），说明是传输失败，需要重试，否则是匹配失败，不重试
            if (matched_blocks > 0) {
                should_retry = true;
                // 即使传输失败，也更新已匹配到的 reuse lengths
                updateReuseLengthsFromContext(read_context);
                RTP_LLM_LOG_WARNING(
                    "load cache failed (matched %zu blocks but transfer failed), retry count: %d/%d, stream: [%ld]",
                    matched_blocks,
                    load_cache_retry_count_,
                    max_retry,
                    stream_->streamId());
            } else {
                RTP_LLM_LOG_WARNING("load cache failed (no blocks matched), continuing without cache, stream: [%ld]",
                                    stream_->streamId());
            }
        }

        load_cache_context_.reset();

        if (should_retry) {
            // 传输失败：保持重试逻辑
            if (load_cache_retry_count_ >= max_retry) {
                RTP_LLM_LOG_WARNING("load cache failed after %d retries (transfer error), stream: [%ld]",
                                    load_cache_retry_count_,
                                    stream_->streamId());
                stream_->reportEventWithoutLock(StreamEvents::Error,
                                                ErrorCode::LOAD_CACHE_TIMEOUT,
                                                "load cache failed after " + std::to_string(max_retry)
                                                    + " retries (transfer error)");
                releaseResource();
                return true;
            }
            load_cache_retry_count_++;
            asyncLoadCache();
            return false;  // 失败重试
        } else {
            // 匹配失败：不重试，继续执行
            return true;
        }
    }
    load_cache_context_.reset();
    return true;
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

void StreamCacheResource::fakeInitKVBlock(size_t reserved_blocks) {
    fake_inited_ = true;
    batch_kv_cache_resource_->resetBatchSize(stream_->maxBatchSize());
    int                         group_nums                 = 1;
    int                         layer_all_num              = 0;
    size_t                      kernel_blocks_per_kv_block = 1;
    std::vector<int>            layer_to_group             = {};
    std::vector<CacheGroupType> group_types                = {};

    if (resource_context_.cache_manager) {
        const auto& cache_config   = resource_context_.cache_manager->cacheConfig();
        group_nums                 = cache_config.groupNums();
        layer_all_num              = static_cast<int>(cache_config.layer_all_num);
        layer_to_group             = cache_config.layer_to_group_id;
        group_types                = cache_config.group_types;
        kernel_blocks_per_kv_block = cache_config.kernelBlocksPerKvBlock();
    }
    batch_kv_cache_resource_->initGroups(
        group_nums, layer_all_num, layer_to_group, kernel_blocks_per_kv_block, group_types);

    reserved_blocks = std::max(1ul, reserved_blocks);
    batch_kv_cache_resource_->resizeBlocks(reserved_blocks, 0);
}

int StreamCacheResource::mallocFailedTimes() const {
    return malloc_failed_times_;
}

bool StreamCacheResource::reuseCache() const {
    return resource_context_.reuse_cache && stream_->reuseCache();
}

bool StreamCacheResource::enableRemoteCache() const {
    return resource_context_.enable_remote_cache && stream_->enableRemoteCache();
}

bool StreamCacheResource::enableMemoryCache() const {
    return resource_context_.enable_memory_cache && stream_->enableMemoryCache();
}

bool StreamCacheResource::enableDeviceCache() const {
    return resource_context_.enable_device_cache && stream_->enableDeviceCache();
}

bool StreamCacheResource::enableTieredMemoryCache() const {
    return resource_context_.enable_tiered_memory_cache && enableMemoryCache() && enableDeviceCache();
}

void StreamCacheResource::loadCacheSync() {
    if (!resource_context_.cache_manager || !resource_context_.cache_manager->hasActiveConnectors()) {
        return;
    }
    RTP_LLM_PROFILE_FUNCTION();
    // Second+ initKVBlock (same stream): skip — reuse lengths already set on first load.
    if (load_cache_once_.exchange(true)) {
        return;
    }
    auto meta = std::make_shared<MetaImpl>(
        reuseCache() && enableMemoryCache(), reuseCache() && enableRemoteCache(), stream_->traceId());
    meta->generate_stream_  = std::make_shared<IGenerateStreamImpl>(stream_->shared_from_this());
    auto connector_context  = std::make_shared<KVCacheConnectorReadWriteContextImpl>(batch_kv_cache_resource_, meta);
    std::shared_ptr<AsyncContext> load_cache_context;
    {
        RTP_LLM_PROFILE_SCOPE("asyncLoadCache");
        load_cache_context = resource_context_.cache_manager->asyncLoadCache(connector_context);
    }
    waitLoadCacheDone(load_cache_context);
    // TODO: scheduler will call incrkvblock after load cache, or may lack block on p2p connector
}

void StreamCacheResource::waitLoadCacheDone(const std::shared_ptr<AsyncContext>& load_context) {
    RTP_LLM_PROFILE_FUNCTION();
    if (!load_context) {
        return;
    }
    load_context->waitDone();
    if (!(load_context->success())) {
        auto error = load_context->errorInfo();
        RTP_LLM_LOG_WARNING(
            "load cache done but not success, stream: [%ld], error: %s", stream_->streamId(), error.ToString().c_str());
        if (error.hasError()) {
            stream_->setStop(error.code(), error.ToString());
        }
        return;
    }
    auto read_context = std::dynamic_pointer_cast<FusedAsyncReadContext>(load_context);
    if (!read_context) {
        RTP_LLM_LOG_WARNING("load cache success but cast context failed, stream: [%ld]", stream_->streamId());
        return;
    }
    updateReuseLengthsFromContext(read_context);
}

void StreamCacheResource::updateReuseLengthsFromContext(const std::shared_ptr<FusedAsyncReadContext>& read_context) {
    const int total_reuse_len  = read_context->resource()->reuseBlockNum() * seqSizePerBlock();
    const int memory_reuse_len = read_context->resource()->memoryReuseBlockNum() * seqSizePerBlock();
    const int remote_reuse_len = read_context->resource()->remoteReuseBlockNum() * seqSizePerBlock();
    const int device_reuse_len = read_context->resource()->deviceReuseBlockNum() * seqSizePerBlock();
    if (total_reuse_len > 0) {
        stream_->setInitialReuseLength(total_reuse_len);
        stream_->setReuseLength(total_reuse_len);
        stream_->setLocalReuseLength(device_reuse_len + memory_reuse_len);
        stream_->setMtpTokenIndex(total_reuse_len);
        stream_->setMemoryReuseLength(memory_reuse_len);
        stream_->setRemoteReuseLength(remote_reuse_len);
    }
}

std::shared_ptr<AsyncContext> StreamCacheResource::storeCacheAsync(
    const std::shared_ptr<BatchKVCacheResource>& batch_resource, bool enable_memory_cache, bool enable_remote_cache) {
    RTP_LLM_PROFILE_FUNCTION();
    auto meta              = std::make_shared<MetaImpl>(enable_memory_cache, enable_remote_cache, stream_->traceId());
    auto connector_context = std::make_shared<KVCacheConnectorReadWriteContextImpl>(batch_resource, meta);
    auto store_context     = resource_context_.cache_manager->asyncStoreCache(connector_context);
    if (resource_context_.write_cache_sync) {
        waitStoreCacheDone(store_context);
    }
    return store_context;
}

void StreamCacheResource::evictDeviceCacheToMemory() {
    const auto min_free_blocks = resource_context_.device_cache_min_free_blocks;
    if (!reuseCache() || !enableMemoryCache() || min_free_blocks <= 0) {
        return;
    }
    // Use notInUseBlocksNum() instead of freeBlocksNum() to account for
    // in-flight connector blocks (being async-written to memory). These blocks
    // are neither held by requests nor in BlockCache, so they will become free
    // once the async write completes. This prevents concurrent streams from
    // over-evicting when multiple streams finish simultaneously.
    const auto not_in_use_blocks = resource_context_.cache_manager->notInUseBlocksNum();
    if (not_in_use_blocks >= static_cast<size_t>(min_free_blocks)) {
        return;
    }

    const auto need_blocks      = static_cast<size_t>(min_free_blocks) - not_in_use_blocks;
    auto       evicted_resource = resource_context_.cache_manager->popBlocksFromCache(need_blocks);
    if (!evicted_resource || !evicted_resource->hasCacheKeys()) {
        RTP_LLM_LOG_INFO(
            "tiered memory cache skip eviction, stream[%ld], not_in_use_blocks=%zu, min_free_blocks=%ld, need_blocks=%zu",
            stream_->streamId(),
            not_in_use_blocks,
            min_free_blocks,
            need_blocks);
        return;
    }

    RTP_LLM_LOG_INFO(
        "tiered memory cache evict, stream[%ld], not_in_use_blocks=%zu, min_free_blocks=%ld, need_blocks=%zu, evict_keys=%zu",
        stream_->streamId(),
        not_in_use_blocks,
        min_free_blocks,
        need_blocks,
        evicted_resource->cacheKeys(0).size());
    storeCacheAsync(evicted_resource, /*enable_memory_cache=*/true, /*enable_remote_cache=*/false);
    resource_context_.cache_manager->blockCacheFree(evicted_resource);
}

void StreamCacheResource::waitStoreCacheDone(const std::shared_ptr<AsyncContext>& store_context) {
    RTP_LLM_PROFILE_FUNCTION();
    if (!store_context) {
        return;
    }
    while (!store_context->done()) {
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
}

void StreamCacheResource::swapLinearBlocks(int32_t batch_id, size_t rhs, size_t lhs) {
    if (rhs == lhs) {
        return;
    }

    auto type_list = resource_context_.cache_manager->cacheConfig().group_types;

    for (size_t i = 0; i < type_list.size(); i++) {
        if (type_list[i] == CacheGroupType::LINEAR) {
            batch_kv_cache_resource_->swapBlocks(batch_id, i, rhs, lhs);
        }
    }
}

void StreamCacheResource::holdKVCacheForPDSep() {
    auto&       resource   = batch_kv_cache_resource_->cacheResource(0);
    const auto& cache_keys = resource.cacheKeys();
    auto        ref = resource_context_.cache_manager->incrKVCacheRef(resource, cache_keys, /*is_connector=*/true);
    if (ref) {
        pd_kvcache_ref_ = std::move(ref);
    }
}

void StreamCacheResource::releaseKVCacheForPDSep() {
    pd_kvcache_ref_.reset();
}
}  // namespace rtp_llm
