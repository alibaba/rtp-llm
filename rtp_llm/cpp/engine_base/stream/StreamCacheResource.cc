#include "rtp_llm/cpp/engine_base/stream/StreamCacheResource.h"
#include "rtp_llm/cpp/engine_base/stream/GenerateStream.h"
#include "rtp_llm/cpp/utils/AssertUtils.h"
#include "rtp_llm/cpp/utils/ProfilingScope.h"
#include "rtp_llm/cpp/cache/AsyncContext.h"
#include "rtp_llm/cpp/cache/Types.h"
#include "rtp_llm/cpp/config/RoleTypes.h"
#include "rtp_llm/cpp/engine_base/stream/CompleteTokenIds.h"
#include <thread>

using namespace std;

namespace rtp_llm {

// ----------------------------- StreamCacheResource -----------------------------

void StreamCacheResource::init(int batch_size) {
    batch_kv_cache_resource_->resetBatchSize(batch_size);
    int                           group_nums      = 1;
    int                           layer_all_num   = 0;
    std::vector<CacheGroupType>   group_types     = {};
    std::vector<std::vector<int>> layer_to_groups = {};

    size_t kernel_blocks_per_kv_block = 1;
    if (resource_context_.cache_manager) {  // cache manager is null when warmup
        const auto& cache_config   = resource_context_.cache_manager->cacheConfig();
        group_nums                 = cache_config.groupNums();
        layer_all_num              = static_cast<int>(cache_config.layer_all_num);
        group_types                = cache_config.groupTypesSnapshot();
        layer_to_groups            = cache_config.layerGroupIdsSnapshot();
        kernel_blocks_per_kv_block = cache_config.kernelBlocksPerKvBlock();
    }

    batch_kv_cache_resource_->initGroups(
        group_nums, layer_all_num, layer_to_groups, kernel_blocks_per_kv_block, group_types);
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
            RTP_LLM_LOG_ERROR("  stream id:                     %s", stream_->streamLogTag().c_str());
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
    allocator_load_context_.reset();
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
}

int StreamCacheResource::tryReleaseKVBlock(size_t nums) {
    RTP_LLM_PROFILE_FUNCTION();
    RTP_LLM_LOG_DEBUG("stream [%s] try release [%lu] blocks", stream_->streamLogTag().c_str(), nums);

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

    const bool disable_first_malloc_reuse =
        resource_context_.cache_manager->cacheConfig().disable_decode_first_malloc_device_reuse;
    const bool is_decode_role  = (resource_context_.role_type == RoleType::DECODE);
    const bool is_first_malloc = (batch_kv_cache_resource_->curBlocksNum() == 0);

    if (disable_first_malloc_reuse && is_decode_role && is_first_malloc) {
        malloc_info.reuse_cache         = false;
        malloc_info.enable_device_cache = false;
    } else {
        malloc_info.reuse_cache         = reuseCache();
        malloc_info.enable_device_cache = reuseCache() && enableDeviceCache();
    }

    malloc_info.enable_remove_skipped_blocks = false;

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
    allocator_load_context_ = std::move(result.async_context);
    return absl::OkStatus();
}

absl::Status StreamCacheResource::waitForAllocatorLoad() {
    if (!allocator_load_context_) {
        return absl::OkStatus();
    }

    auto load_context = allocator_load_context_;
    load_context->waitDone();
    if (!load_context->done()) {
        return absl::InternalError("allocator load context is non-terminal after waitDone");
    }

    const bool      load_success = load_context->success();
    const ErrorInfo error        = load_context->errorInfo();
    allocator_load_context_.reset();
    if (load_success) {
        return absl::OkStatus();
    }

    const std::string& error_text = error.ToString();
    return absl::InternalError(error_text.empty() ? "allocator load-back failed" :
                                                    "allocator load-back failed: " + error_text);
}

absl::Status StreamCacheResource::incrKVBlock(size_t reserve_step, int seq_len_override) {
    RTP_LLM_PROFILE_FUNCTION();
    // TODO(xinfei.sxf) add reserver_blocks
    if (batch_kv_cache_resource_->curBlocksNum() == 0) {
        return absl::FailedPreconditionError("incrKVBlock requires an initialized KV cache resource");
    }

    if (fake_inited_) {
        return absl::InternalError("fake inited not allow to incr block");
    }

    MallocInfo malloc_info;
    malloc_info.batch_kv_cache_resource      = batch_kv_cache_resource_;
    malloc_info.complete_token_ids           = stream_->completeTokenIdsPtr();
    malloc_info.request_id                   = stream_->streamId();
    malloc_info.verbose                      = malloc_failed_times_ >= 10 ? malloc_failed_times_ % 100 == 0 : true;
    malloc_info.reuse_cache                  = reuseCache();
    malloc_info.enable_device_cache          = reuseCache() && enableDeviceCache();
    malloc_info.enable_remove_skipped_blocks = true;
    malloc_info.incr_seq_len_override        = seq_len_override;

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

    if (result.async_context) {
        allocator_load_context_ = std::move(result.async_context);
        return absl::FailedPreconditionError("async incremental KV block allocation is unsupported");
    }

    return absl::OkStatus();
}

bool StreamCacheResource::asyncLoadCache() {
    RTP_LLM_PROFILE_FUNCTION();
    return allocator_load_context_ != nullptr;
}

bool StreamCacheResource::loadCacheDone() {
    if (allocator_load_context_) {
        if (!allocator_load_context_->done()) {
            return false;
        }
        const bool allocator_load_success = allocator_load_context_->success();
        if (!allocator_load_success) {
            const ErrorInfo error = allocator_load_context_->errorInfo();
            RTP_LLM_LOG_WARNING("block tree load_back failed, stream: [%s], error: %s",
                                stream_->streamLogTag().c_str(),
                                error.ToString().c_str());
            allocator_load_context_.reset();
            stream_->reportEventWithoutLock(
                StreamEvents::Error, ErrorCode::LOAD_CACHE_TIMEOUT, "block tree cache load_back failed");
            return true;
        }
        allocator_load_context_.reset();
    }
    return true;
}

// TODO, delete it soon
int StreamCacheResource::curBlocksNum() const {
    return batch_kv_cache_resource_->curBlocksNum();
}

bool StreamCacheResource::isContextStream() const {
    RTP_LLM_CHECK_WITH_INFO(stream_ != nullptr, "StreamCacheResource::isContextStream called with null stream");
    return stream_->isContextStream();
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
    int                           group_nums                 = 1;
    int                           layer_all_num              = 0;
    size_t                        kernel_blocks_per_kv_block = 1;
    std::vector<CacheGroupType>   group_types                = {};
    std::vector<std::vector<int>> layer_to_groups            = {};

    if (resource_context_.cache_manager) {
        const auto& cache_config   = resource_context_.cache_manager->cacheConfig();
        group_nums                 = cache_config.groupNums();
        layer_all_num              = static_cast<int>(cache_config.layer_all_num);
        group_types                = cache_config.groupTypesSnapshot();
        layer_to_groups            = cache_config.layerGroupIdsSnapshot();
        kernel_blocks_per_kv_block = cache_config.kernelBlocksPerKvBlock();
    }
    batch_kv_cache_resource_->initGroups(
        group_nums, layer_all_num, layer_to_groups, kernel_blocks_per_kv_block, group_types);

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

void StreamCacheResource::swapLinearBlocks(int32_t batch_id, size_t rhs, size_t lhs) {
    if (rhs == lhs) {
        return;
    }

    auto type_list = resource_context_.cache_manager->cacheConfig().groupTypesSnapshot();

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
