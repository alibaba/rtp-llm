#include "rtp_llm/cpp/cache/connector/memory/KVCacheMemoryConnector.h"

#include "rtp_llm/cpp/cache/BlockPool.h"
#include "rtp_llm/cpp/cache/BlockPoolConfigHelper.h"
#include "rtp_llm/cpp/cache/connector/memory/MemoryAsyncContext.h"
#include "rtp_llm/cpp/cache/connector/Meta.h"
#include "rtp_llm/cpp/cache/KVCacheAllocator.h"
#include "rtp_llm/models_py/bindings/core/ExecOps.h"
#include "rtp_llm/models_py/bindings/NoBlockCopy.h"
#include "rtp_llm/cpp/utils/Logger.h"
#include "rtp_llm/cpp/metrics/RtpLLMMetrics.h"
#include "rtp_llm/cpp/utils/ProfilingScope.h"

#include <cstdlib>

namespace rtp_llm {

constexpr size_t kMemoryCopyConcurrency = 8;
constexpr int    kMemoryCopyProtocolVersion = 1;

// Eligibility for the split scatter/gather path is decided by enable_memory_cache_sm_copy. The implementation falls
// back to the generic copy path when the buffer layout does not match.
static void applySplitKvMultiCopyFieldsIfEligible(bool enable_sm_copy, const CacheConfig& cfg, MultiCopyParams& out) {
    if (!enable_sm_copy) {
        return;
    }
    out.split_kv_layer_num          = static_cast<int>(cfg.layer_all_num);
    out.split_kv_cache_stride_bytes = cfg.kv_block_stride_bytes;
    out.split_kv_scale_stride_bytes = cfg.kv_scale_stride_bytes;
}

RemoteLoadLeaseRetainer::Config makeMemoryCopyLeaseConfig(int64_t sync_timeout_ms) {
    const auto bounded_timeout_ms = std::max<int64_t>(
        1, std::min<int64_t>(sync_timeout_ms, MemoryCopyDeadline::kMaxWireDurationMs));
    return RemoteLoadLeaseRetainer::Config{
        /*max_jobs=*/1000,
        /*initial_backoff=*/std::chrono::milliseconds(10),
        /*max_backoff=*/std::chrono::milliseconds(std::min<int64_t>(1000, bounded_timeout_ms)),
        /*stop_grace=*/std::chrono::milliseconds(bounded_timeout_ms),
        /*worker_count=*/kMemoryCopyConcurrency,
    };
}

KVCacheMemoryConnector::KVCacheMemoryConnector(const CacheConfig&                       cache_config,
                                               const KVCacheConfig&                     kv_cache_config,
                                               const std::shared_ptr<KVCacheAllocator>& allocator,
                                               const std::vector<std::string>&          tp_addrs,
                                               const kmonitor::MetricsReporterPtr&      metrics_reporter):
    cache_config_(cache_config),
    kv_cache_config_(kv_cache_config),
    allocator_(allocator),
    tp_addrs_(tp_addrs),
    pending_copy_leases_(makeMemoryCopyLeaseConfig(kv_cache_config.memory_cache_sync_timeout_ms)),
    metrics_reporter_(metrics_reporter) {}

KVCacheMemoryConnector::~KVCacheMemoryConnector() {
    RTP_LLM_LOG_INFO("KVCacheMemoryConnector destructor");
    {
        std::lock_guard<std::mutex> lock(copy_protocol_thread_mutex_);
        stop_.store(true, std::memory_order_release);
        if (copy_protocol_thread_.joinable()) {
            copy_protocol_thread_.join();
        }
    }
    if (metrics_reporter_thread_) {
        metrics_reporter_thread_->join();
        metrics_reporter_thread_.reset();
    }
    if (wait_done_thread_pool_) {
        wait_done_thread_pool_->stop();
        wait_done_thread_pool_.reset();
    }
    const auto drain_timeout_ms = std::max<int64_t>(1,
                                                    std::min<int64_t>(kv_cache_config_.memory_cache_sync_timeout_ms,
                                                                      MemoryCopyDeadline::kMaxWireDurationMs));
    const auto drain_timeout = std::chrono::milliseconds(drain_timeout_ms);
    if (!pending_copy_leases_.stop(drain_timeout)) {
        RTP_LLM_LOG_ERROR("memory copy plans did not quiesce before connector destruction");
        std::abort();
    }
    if (!copy_fence_.stopAndWait(drain_timeout)) {
        RTP_LLM_LOG_ERROR("memory copy handlers did not quiesce before connector destruction");
        std::abort();
    }
    broadcast_manager_.reset();
    block_pool_.reset();
    block_cache_.reset();
}

bool KVCacheMemoryConnector::init() {
    const auto memory_cache_sync_timeout_ms = kv_cache_config_.memory_cache_sync_timeout_ms;
    RTP_LLM_CHECK_WITH_INFO(MemoryCopyDeadline::validWireDuration(memory_cache_sync_timeout_ms),
                            "init failed, sync timeout is invalid, sync timeout: %ld ms",
                            memory_cache_sync_timeout_ms);

    checkLayerBlockStrideBytes();

    initBlockPool();
    block_cache_ = std::make_shared<MemoryBlockCache>();

    broadcast_manager_ = std::make_shared<BroadcastManager>(tp_addrs_);
    RTP_LLM_CHECK_WITH_INFO(broadcast_manager_->init(), "init failed, broadcast manager init failed");

    wait_done_thread_pool_ =
        std::make_shared<autil::LockFreeThreadPool>(kMemoryCopyConcurrency, 1000, nullptr, "WaitDoneThreadPool");
    RTP_LLM_CHECK_WITH_INFO(wait_done_thread_pool_->start(), "init failed, wait done thread pool start failed");

    if (metrics_reporter_) {
        metrics_reporter_thread_ = std::make_shared<std::thread>([this]() { reportMetricsLoop(); });
    }
    return true;
}

void KVCacheMemoryConnector::checkLayerBlockStrideBytes() const {
    const size_t layer_num          = cache_config_.layer_all_num;
    const auto&  layer_block_stride = cache_config_.layer_to_block_stride_bytes;
    RTP_LLM_CHECK_WITH_INFO(layer_block_stride.size() == layer_num,
                            "layer block stride size must equal to layer num, got=%zu need=%zu",
                            layer_block_stride.size(),
                            layer_num);
    for (size_t i = 0; i < layer_num; ++i) {
        RTP_LLM_CHECK_WITH_INFO(
            layer_block_stride[i] > 0, "invalid block stride bytes at layer=%zu: %d", i, layer_block_stride[i]);
    }
}

void KVCacheMemoryConnector::initBlockPool() {
    const auto memory_cache_size_mb = kv_cache_config_.memory_cache_size_mb;
    RTP_LLM_CHECK_WITH_INFO(memory_cache_size_mb > 0,
                            "init block pool failed, memory size is invalid, memory size: %ld MB",
                            memory_cache_size_mb);

    const auto& layer_block_stride = cache_config_.layer_to_block_stride_bytes;

    // block_size here means "one cache-key across all layers" total bytes (kv + scale).
    // Use per-layer block strides so NULL_BLOCK_IDX layers still occupy space in merged layout.
    size_t block_size = std::accumulate(layer_block_stride.begin(), layer_block_stride.end(), 0);
    RTP_LLM_CHECK_WITH_INFO(block_size > 0, "block size is invalid: %zu", block_size);

    block_pool_ = createBlockPool(block_size, memory_cache_size_mb);
    RTP_LLM_CHECK_WITH_INFO(block_pool_ != nullptr, "init block pool failed, create block pool failed");
}

std::shared_ptr<AsyncMatchContext> KVCacheMemoryConnector::asyncMatch(const std::shared_ptr<KVCacheResource>& resource,
                                                                      const std::shared_ptr<Meta>&            meta) {
    RTP_LLM_PROFILE_FUNCTION();
    RTP_LLM_CHECK_WITH_INFO(meta != nullptr, "async match failed, meta is null");
    RTP_LLM_CHECK_WITH_INFO(resource != nullptr, "async match failed, resource is null");
    if (!meta->enableMemoryCache()) {
        return nullptr;
    }
    if (!copyProtocolReadyOrStartProbe()) {
        return nullptr;
    }

    const auto& cache_keys = resource->cacheKeys();
    // do not match last block, whether it is aligned or not, otherwise may cause core dump in computing ops.
    const auto cache_keys_size = cache_keys.empty() ? 0 : cache_keys.size() - 1;
    if (cache_keys_size == 0) {
        RTP_LLM_LOG_DEBUG("async match skip, cache keys is empty");
        return nullptr;
    }

    const auto& layer_block_ids = resource->layerBlocks();
    if (!checkLayerBlocks(layer_block_ids, cache_keys_size)) {
        RTP_LLM_LOG_WARNING("async match failed, invalid layer_block_ids, cache_keys_size=%zu", cache_keys_size);
        return nullptr;
    }

    const size_t already_reuse_num = resource->reuseBlockNum();
    if (already_reuse_num >= cache_keys_size) {
        // gpu has already matched all cache keys, no need to match in memory
        RTP_LLM_LOG_DEBUG(
            "async match skip, already reuse num is greater than cache keys size, cache_keys size: %zu, already_reuse_num: %zu",
            cache_keys_size,
            already_reuse_num);
        return nullptr;
    }

    autil::ScopedTime2 timer;

    // matched_num must end at a key that satisfies BOTH:
    // - memory cache key is complete
    // - all gpu blocks for this key are valid (non-null)
    //
    // Notes:
    // - If a key is complete, we allow gpu blocks to be partially invalid and keep matching further.
    // - If all gpu blocks are valid, the final matched key must be complete.
    size_t matched_num = already_reuse_num;
    for (size_t i = already_reuse_num; i < cache_keys_size; ++i) {
        const auto cache_key    = cache_keys.at(i);
        const auto match_result = block_cache_->match(static_cast<CacheKeyType>(cache_key));
        if (isNullBlockIdx(match_result.matched_index)) {
            break;  // only continuous prefix
        }
        if (match_result.is_complete && gpuBlocksAllValid(layer_block_ids, i)) {
            matched_num = i + 1;
        }
    }

    if (matched_num <= already_reuse_num) {
        RTP_LLM_LOG_DEBUG("not matched cache in memory, cache keys size: %zu, already_reuse_num: %zu",
                          cache_keys_size,
                          already_reuse_num);
        reportMatchMetrics(/*success=*/false, timer.done_us(), cache_keys_size, matched_num);
        return nullptr;
    }
    RTP_LLM_LOG_INFO("memory cache matched blocks: already_reuse=%zu matched=%zu cache_keys=%zu",
                     already_reuse_num,
                     matched_num,
                     cache_keys_size);
    reportMatchMetrics(/*success=*/true, timer.done_us(), cache_keys_size, matched_num);
    return std::make_shared<MemoryAsyncMatchContext>(matched_num);
}

bool KVCacheMemoryConnector::gpuBlocksAllValid(const LayerBlockIds& layer_block_ids, size_t key_index) const {
    for (size_t layer = 0; layer < cache_config_.layer_all_num; ++layer) {
        const auto& blocks = layer_block_ids.at(layer)->blocks();
        if (isNullBlockIdx(blocks.at(key_index))) {
            return false;
        }
    }
    return true;
}

std::shared_ptr<AsyncContext> KVCacheMemoryConnector::asyncRead(const std::shared_ptr<KVCacheResource>&   resource,
                                                                const std::shared_ptr<Meta>&              meta,
                                                                const std::shared_ptr<AsyncMatchContext>& match_context,
                                                                int start_read_block_index,
                                                                int read_block_num) {
    RTP_LLM_PROFILE_FUNCTION();
    RTP_LLM_CHECK_WITH_INFO(meta != nullptr, "async read failed, meta is null");
    RTP_LLM_CHECK_WITH_INFO(resource != nullptr, "async read failed, resource is null");
    if (copy_protocol_state_.load(std::memory_order_acquire) != CopyProtocolState::READY) {
        return nullptr;
    }
    const int64_t operation_deadline_unix_ms =
        resolveOperationDeadline(*meta, MemoryCopyDeadline::unixMillisNow());
    if (operation_deadline_unix_ms <= 0) {
        RTP_LLM_LOG_WARNING("async read failed, request deadline has expired");
        return nullptr;
    }
    const auto& cache_keys      = resource->cacheKeys();
    const auto  cache_keys_size = cache_keys.empty() ? 0 : cache_keys.size() - 1;
    if (cache_keys_size == 0) {
        RTP_LLM_LOG_DEBUG("async read skip, cache keys is empty");
        return nullptr;
    }

    autil::ScopedTime2 timer;

    const auto& layer_block_ids = resource->layerBlocks();
    if (!checkLayerBlocks(layer_block_ids, cache_keys_size)) {
        reportReadMetrics(false, timer.done_us(), cache_keys_size, 0);
        return nullptr;
    }

    if (start_read_block_index < 0 || read_block_num <= 0
        || start_read_block_index + read_block_num > cache_keys_size) {
        RTP_LLM_LOG_WARNING(
            "async read failed, invalid block range, start_read_block_index: %d, read_block_num: %d, cache_keys size: %zu",
            start_read_block_index,
            read_block_num,
            cache_keys_size);
        reportReadMetrics(false, timer.done_us(), cache_keys_size, 0);
        return nullptr;
    }

    auto copy_plan = buildCopyPlanForRead(cache_keys, layer_block_ids, start_read_block_index, read_block_num);
    if (!copy_plan || copy_plan->copy_infos.empty()) {
        reportReadMetrics(false, timer.done_us(), cache_keys_size, 0);
        return nullptr;
    }
    copy_plan->resource_lease = resource;

    const auto total_block_num = cache_keys_size;
    auto       read_done = [resource, copy_plan, total_block_num, read_block_num, timer, this](bool success) mutable {
        RTP_LLM_LOG_DEBUG("async read done, success: %d", success);
        if (success) {
            resource->setMemoryReuseBlockNum(read_block_num);
            for (const auto& copy_info : copy_plan->copy_infos) {
                const auto removed_item = block_cache_->removeIfMatch(copy_info.cache_key, copy_info.mem_block);
                if (!removed_item.has_value()) {
                    continue;
                }
                freeBlocks({removed_item->block_index}, /*cache_free=*/true);
            }
            RTP_LLM_LOG_INFO("memory cache read success: read_blocks=%d released_blocks=%zu total_blocks=%zu",
                             read_block_num,
                             copy_plan->copy_infos.size(),
                             total_block_num);
        }
        // reset ptr to release memory block refs
        copy_plan.reset();
        reportReadMetrics(success, timer.done_us(), total_block_num, read_block_num);
    };

    auto context = std::make_shared<MemoryAsyncContext>(read_done);
    if (!startCopyAsync(context, copy_plan, operation_deadline_unix_ms)) {
        RTP_LLM_LOG_WARNING("async read failed, start copy plan async failed");
        return nullptr;
    }
    return context;
}

std::shared_ptr<KVCacheMemoryConnector::CopyPlan> KVCacheMemoryConnector::buildCopyPlanForRead(
    const CacheKeysType& cache_keys, const LayerBlockIds& layer_block_ids, int start_index, int read_num) {
    std::vector<CopyInfoPerKey> copy_infos;
    const auto                  layer_num = cache_config_.layer_all_num;
    bool                        success   = true;

    for (int i = start_index; i < start_index + read_num; ++i) {
        const auto cache_key = cache_keys.at(i);
        const auto match_result =
            block_cache_->matchAndRequestReference(static_cast<CacheKeyType>(cache_key), *block_pool_);
        if (isNullBlockIdx(match_result.matched_index)) {
            RTP_LLM_LOG_WARNING("build copy plan for read failed, cache key not found, cache key: %ld", cache_key);
            success = false;
            break;
        }
        CopyInfoPerKey copy_info;
        copy_info.cache_key = cache_key;
        copy_info.mem_block = match_result.matched_index;
        copy_info.gpu_blocks.reserve(layer_num);
        for (size_t layer = 0; layer < layer_num; ++layer) {
            // Do NOT skip NULL_BLOCK_IDX here. The merged memory block layout requires reserving
            // per-layer stride even when this layer has no gpu block (-1).
            copy_info.gpu_blocks.push_back(layer_block_ids.at(layer)->blocks().at(i));
        }
        copy_info.is_complete = match_result.is_complete;
        copy_infos.emplace_back(std::move(copy_info));
    }

    // 在match时已经保证了最后一个key是complete, 这里再校验下
    if (success && !copy_infos.empty() && !copy_infos.back().is_complete) {
        RTP_LLM_LOG_WARNING("build copy plan for read failed, last key is not complete, cache key: %ld",
                            copy_infos.back().cache_key);
        success = false;
    }

    // free blocks in destructor
    auto plan = createCopyPlan(copy_infos, CopyDirection::H2D);
    return success ? plan : nullptr;
}

std::shared_ptr<AsyncContext> KVCacheMemoryConnector::asyncWrite(const std::shared_ptr<KVCacheResource>& resource,
                                                                 const std::shared_ptr<Meta>&            meta) {
    RTP_LLM_PROFILE_FUNCTION();
    RTP_LLM_CHECK_WITH_INFO(meta != nullptr, "async write failed, meta is null");
    RTP_LLM_CHECK_WITH_INFO(resource != nullptr, "async write failed, resource is null");
    if (!meta->enableMemoryCache()) {
        return nullptr;
    }
    if (!copyProtocolReadyOrStartProbe()) {
        return nullptr;
    }
    const int64_t operation_deadline_unix_ms =
        resolveOperationDeadline(*meta, MemoryCopyDeadline::unixMillisNow());
    if (operation_deadline_unix_ms <= 0) {
        RTP_LLM_LOG_WARNING("async write failed, request deadline has expired");
        return nullptr;
    }

    const auto& cache_keys = resource->cacheKeys();
    const auto  cache_keys_size =
        cache_keys.empty() ? 0 : (resource->lastBlockAligned() ? cache_keys.size() : cache_keys.size() - 1);
    if (cache_keys_size == 0) {
        RTP_LLM_LOG_DEBUG("async write skip, cache keys is empty");
        return nullptr;
    }

    autil::ScopedTime2 timer;

    const auto& layer_block_ids = resource->layerBlocks();
    if (!checkLayerBlocks(layer_block_ids, cache_keys_size)) {
        reportWriteMetrics(false, timer.done_us(), cache_keys_size, 0);
        return nullptr;
    }

    // 计算内存中已存在的前缀长度
    size_t mem_matched_num = 0;
    for (; mem_matched_num < cache_keys_size; ++mem_matched_num) {
        if (!block_cache_->contains(static_cast<CacheKeyType>(cache_keys[mem_matched_num]))) {
            break;
        }
    }
    if (mem_matched_num == cache_keys_size) {
        RTP_LLM_LOG_DEBUG(
            "async write skip, all cache keys already in memory cache, matched num: %zu, cache keys size: %zu",
            mem_matched_num,
            cache_keys_size);
        reportWriteMetrics(true, timer.done_us(), static_cast<int64_t>(cache_keys_size), 0);
        return nullptr;
    }

    bool no_need_write = false;
    auto copy_plan     = buildCopyPlanForWrite(
        cache_keys, layer_block_ids, mem_matched_num, cache_keys_size - mem_matched_num, no_need_write);
    if (!copy_plan || copy_plan->copy_infos.empty()) {
        reportWriteMetrics(no_need_write, timer.done_us(), static_cast<int64_t>(cache_keys_size), 0);
        return nullptr;
    }
    copy_plan->resource_lease = resource;

    auto write_done =
        [copy_plan, resource_copy = resource, timer, total_block_num = cache_keys_size, this](bool success) mutable {
            RTP_LLM_LOG_DEBUG("async write done, success: %d", success);

            if (success) {
                for (const auto& copy_info : copy_plan->copy_infos) {
                    MemoryBlockCache::CacheItem item;
                    item.cache_key   = copy_info.cache_key;
                    item.block_index = copy_info.mem_block;
                    item.is_resident = false;
                    item.is_complete = copy_info.is_complete;
                    putToCache(item);
                }
                // reset resource to decrease block ref count in destructor
                resource_copy.reset();
            }
            const int64_t write_block_num = success ? static_cast<int64_t>(copy_plan->copy_infos.size()) : 0;
            // reset copy plan to release memory block refs
            copy_plan.reset();
            reportWriteMetrics(success, timer.done_us(), total_block_num, write_block_num);
        };

    auto context = std::make_shared<MemoryAsyncContext>(write_done);
    if (!startCopyAsync(context, copy_plan, operation_deadline_unix_ms)) {
        RTP_LLM_LOG_WARNING("async write failed, start copy plan async failed");
        return nullptr;
    }
    return context;
}

std::shared_ptr<KVCacheMemoryConnector::CopyPlan>
KVCacheMemoryConnector::buildCopyPlanForWrite(const CacheKeysType& cache_keys,
                                              const LayerBlockIds& layer_block_ids,
                                              int                  start_index,
                                              int                  write_num,
                                              bool&                no_need_write) {
    const auto                  layer_num = cache_config_.layer_all_num;
    std::vector<CopyInfoPerKey> copy_infos;
    copy_infos.reserve(write_num);

    // Hybrid-attn support:
    // We allow writing "partial" keys (incomplete KV) to keep prefix continuity,
    // BUT the final written key MUST be "complete" (complete KV on all layers),
    // otherwise the written tail cannot be reused by asyncMatch.
    int last_complete_index = -1;  // cache_key index in [start_index, start_index + write_num)

    for (int i = start_index; i < start_index + write_num; ++i) {
        const auto                cache_key = cache_keys.at(i);
        std::vector<BlockIdxType> gpu_blocks;
        gpu_blocks.reserve(layer_num);
        size_t null_block_num = 0;
        for (size_t layer = 0; layer < layer_num; ++layer) {
            const int gpu_block_idx = layer_block_ids.at(layer)->blocks().at(i);
            // Do NOT skip NULL_BLOCK_IDX here. We must keep per-layer stride slots in the merged big block.
            if (isNullBlockIdx(gpu_block_idx)) {
                ++null_block_num;
            }
            gpu_blocks.push_back(gpu_block_idx);
        }

        const bool is_complete = null_block_num == 0;
        if (is_complete) {
            last_complete_index = i;
        }

        CopyInfoPerKey copy_info;
        copy_info.cache_key   = cache_key;
        copy_info.mem_block   = NULL_BLOCK_IDX;
        copy_info.gpu_blocks  = std::move(gpu_blocks);
        copy_info.is_complete = is_complete;
        copy_infos.emplace_back(std::move(copy_info));
    }

    // ensure the final written key is complete
    no_need_write = last_complete_index < start_index;
    if (no_need_write) {
        return nullptr;
    }

    // drop keys behind the last complete key
    const size_t keep_cnt = static_cast<size_t>(last_complete_index - start_index + 1);
    copy_infos.resize(keep_cnt);

    std::vector<BlockIdxType> mem_blocks;
    if (!mallocBlocks(copy_infos.size(), mem_blocks)) {
        RTP_LLM_LOG_WARNING("build copy plan for write failed, malloc blocks failed, need blocks: %zu",
                            copy_infos.size());
        return nullptr;
    }
    for (size_t i = 0; i < copy_infos.size(); ++i) {
        copy_infos[i].mem_block = mem_blocks[i];
    }

    // free blocks in destructor
    auto plan = createCopyPlan(copy_infos, CopyDirection::D2H);
    return plan;
}

std::shared_ptr<KVCacheMemoryConnector::CopyPlan>
KVCacheMemoryConnector::createCopyPlan(const std::vector<CopyInfoPerKey>& copy_infos, const CopyDirection& direction) {
    auto plan        = new CopyPlan();
    plan->copy_infos = copy_infos;
    plan->direction  = direction;
    plan->operation_id = makeCopyOperationId();
    auto deleter     = [this](CopyPlan* plan) {
        std::vector<BlockIdxType> blocks;
        blocks.reserve(plan->copy_infos.size());
        for (const auto& copy_info : plan->copy_infos) {
            blocks.push_back(copy_info.mem_block);
        }
        freeBlocks(blocks, /*cache_free=*/false);
        delete plan;
    };
    return std::shared_ptr<CopyPlan>(plan, deleter);
}

bool KVCacheMemoryConnector::startCopyAsync(const std::shared_ptr<MemoryAsyncContext>& context,
                                            const std::shared_ptr<CopyPlan>&           copy_plan,
                                            int64_t                                    operation_deadline_unix_ms) {
    if (stop_.load()) {
        context->complete(false);
        return false;
    }
    if (copy_protocol_state_.load(std::memory_order_acquire) != CopyProtocolState::READY) {
        context->complete(false);
        return false;
    }
    const auto initial_admission = MemoryCopyDeadline::evaluateCopy(operation_deadline_unix_ms,
                                                                    kv_cache_config_.memory_cache_sync_timeout_ms,
                                                                    kv_cache_config_.memory_cache_sync_timeout_ms,
                                                                    MemoryCopyDeadline::unixMillisNow());
    if (!initial_admission) {
        context->complete(false);
        return false;
    }
    copy_plan->operation_deadline_unix_ms = operation_deadline_unix_ms;
    auto ticket_status = pending_copy_leases_.reserve(
        copy_plan->operation_id,
        copy_plan,
        [this, operation_id = copy_plan->operation_id, operation_deadline_unix_ms]() {
            return quiesceCopy(operation_id, operation_deadline_unix_ms);
        });
    if (!ticket_status.ok()) {
        RTP_LLM_LOG_WARNING("start copy plan async failed, cannot reserve copy lease: %s",
                            ticket_status.status().ToString().c_str());
        context->complete(false);
        return false;
    }
    auto ticket = std::move(*ticket_status);

    auto task_guard = std::make_shared<MemoryCopyTaskGuard>(context, std::move(ticket));
    auto code = wait_done_thread_pool_->pushTask(
        [this, context, copy_plan, operation_deadline_unix_ms, task_guard]() mutable {
        if (!task_guard->enterBeforeDeadline(operation_deadline_unix_ms,
                                             kv_cache_config_.memory_cache_sync_timeout_ms,
                                             kv_cache_config_.memory_cache_sync_timeout_ms,
                                             MemoryCopyDeadline::unixMillisNow())) {
            return;
        }
        if (!task_guard->markStarted()) {
            return;
        }

        try {
            auto send_result = sendCopyPlan(copy_plan);
            if (send_result == nullptr) {
                task_guard->abandon();
                return;
            }
            const bool copy_succeeded = waitForCopyResult(send_result, copy_plan->operation_id);
            send_result.reset();
            if (copy_succeeded) {
                task_guard->finish(true);
                return;
            }
            task_guard->abandon();
        } catch (const std::exception& e) {
            RTP_LLM_LOG_WARNING("memory copy task failed: %s", e.what());
            task_guard->abandon();
        } catch (...) {
            RTP_LLM_LOG_WARNING("memory copy task failed with unknown exception");
            task_guard->abandon();
        }
    });
    if (code != autil::ThreadPoolBase::ERROR_NONE) {
        task_guard->cancelBeforeDispatch();
        RTP_LLM_LOG_WARNING("start copy plan async failed, push send+wait task failed, code=%d", code);
        return false;
    }
    return true;
}

int64_t KVCacheMemoryConnector::resolveOperationDeadline(const Meta& meta, int64_t now_unix_ms) const {
    const auto routing = meta.p2pRouting();
    const auto request_deadline =
        routing.has_value() && routing->request_deadline_enabled ? routing->deadline_ms : 0;
    return MemoryCopyDeadline::resolve(now_unix_ms,
                                       kv_cache_config_.memory_cache_sync_timeout_ms,
                                       request_deadline);
}

bool KVCacheMemoryConnector::waitForCopyResult(
    const std::shared_ptr<BroadcastResult<FunctionRequestPB, FunctionResponsePB>>& result,
    const std::string&                                                           operation_id) const {
    if (result == nullptr) {
        return false;
    }
    try {
        result->waitDone();
    } catch (const std::exception& e) {
        RTP_LLM_LOG_WARNING("memory copy rpc failed: %s", e.what());
        result->cancelAndDrain();
        return false;
    } catch (...) {
        RTP_LLM_LOG_WARNING("memory copy rpc failed with unknown exception");
        result->cancelAndDrain();
        return false;
    }
    if (!result->success()) {
        return false;
    }
    const auto responses = result->responses();
    if (responses.size() != broadcast_manager_->workerNum()) {
        return false;
    }
    return std::all_of(responses.begin(), responses.end(), [&operation_id](const auto& response) {
        return response.has_mem_response() && response.mem_response().operation_id() == operation_id
               && response.mem_response().success() && response.mem_response().quiesced()
               && response.mem_response().protocol_version() == kMemoryCopyProtocolVersion;
    });
}

bool KVCacheMemoryConnector::copyProtocolReadyOrStartProbe() {
    auto state = copy_protocol_state_.load(std::memory_order_acquire);
    if (state == CopyProtocolState::READY) {
        return true;
    }
    if (state != CopyProtocolState::UNKNOWN) {
        return false;
    }
    std::lock_guard<std::mutex> lock(copy_protocol_thread_mutex_);
    if (stop_.load(std::memory_order_acquire)) {
        return false;
    }
    state = copy_protocol_state_.load(std::memory_order_acquire);
    if (state == CopyProtocolState::READY) {
        return true;
    }
    if (state != CopyProtocolState::UNKNOWN) {
        return false;
    }
    const auto now = MemoryCopyDeadline::unixMillisNow();
    if (now < copy_protocol_next_probe_unix_ms_.load(std::memory_order_acquire)) {
        return false;
    }
    if (!copy_protocol_state_.compare_exchange_strong(
            state, CopyProtocolState::PROBING, std::memory_order_acq_rel, std::memory_order_acquire)) {
        return false;
    }

    try {
        const auto operation_id = makeCopyOperationId();
        const auto deadline = MemoryCopyDeadline::make(now, kv_cache_config_.memory_cache_sync_timeout_ms);
        if (copy_protocol_thread_.joinable()) {
            copy_protocol_thread_.join();
        }
        copy_protocol_thread_ = std::thread([this, operation_id, deadline]() {
            finishCopyProtocolProbe(
                [this, operation_id, deadline]() { return probeCopyProtocol(operation_id, deadline); });
        });
    } catch (const std::exception& e) {
        RTP_LLM_LOG_WARNING("memory copy capability probe could not start: %s", e.what());
        copy_protocol_next_probe_unix_ms_.store(now + 1000, std::memory_order_release);
        copy_protocol_state_.store(CopyProtocolState::UNKNOWN, std::memory_order_release);
    } catch (...) {
        RTP_LLM_LOG_WARNING("memory copy capability probe could not start");
        copy_protocol_next_probe_unix_ms_.store(now + 1000, std::memory_order_release);
        copy_protocol_state_.store(CopyProtocolState::UNKNOWN, std::memory_order_release);
    }
    return false;
}

void KVCacheMemoryConnector::finishCopyProtocolProbe(
    const std::function<CapabilityProbeResult()>& probe) noexcept {
    CapabilityProbeResult result = CapabilityProbeResult::TRANSIENT_FAILURE;
    try {
        result = probe();
    } catch (const std::exception& e) {
        RTP_LLM_LOG_WARNING("memory copy capability probe failed: %s", e.what());
    } catch (...) {
        RTP_LLM_LOG_WARNING("memory copy capability probe failed with unknown exception");
    }

    if (result == CapabilityProbeResult::READY) {
        copy_protocol_state_.store(CopyProtocolState::READY, std::memory_order_release);
    } else if (result == CapabilityProbeResult::UNSUPPORTED) {
        copy_protocol_state_.store(CopyProtocolState::UNSUPPORTED, std::memory_order_release);
    } else {
        copy_protocol_next_probe_unix_ms_.store(
            MemoryCopyDeadline::unixMillisNow() + 1000, std::memory_order_release);
        copy_protocol_state_.store(CopyProtocolState::UNKNOWN, std::memory_order_release);
    }
}

KVCacheMemoryConnector::CapabilityProbeResult
KVCacheMemoryConnector::probeCopyProtocol(const std::string& operation_id, int64_t operation_deadline_unix_ms) {
    auto result = sendCapabilityProbe(operation_id, operation_deadline_unix_ms);
    if (result == nullptr) {
        return CapabilityProbeResult::TRANSIENT_FAILURE;
    }
    try {
        result->waitDone();
    } catch (...) {
        result->cancelAndDrain();
        return CapabilityProbeResult::TRANSIENT_FAILURE;
    }
    if (!result->success()) {
        return CapabilityProbeResult::TRANSIENT_FAILURE;
    }
    const auto responses = result->responses();
    if (responses.size() != broadcast_manager_->workerNum()) {
        return CapabilityProbeResult::TRANSIENT_FAILURE;
    }
    return classifyCapabilityResponses(responses, operation_id);
}

KVCacheMemoryConnector::CapabilityProbeResult KVCacheMemoryConnector::classifyCapabilityResponses(
    const std::vector<FunctionResponsePB>& responses, const std::string& operation_id) {
    if (responses.empty()) {
        return CapabilityProbeResult::TRANSIENT_FAILURE;
    }

    bool all_current      = true;
    bool all_incompatible = true;
    for (const auto& response : responses) {
        if (!response.has_mem_response()) {
            return CapabilityProbeResult::TRANSIENT_FAILURE;
        }
        const auto& mem_response = response.mem_response();
        const bool current = mem_response.protocol_version() == kMemoryCopyProtocolVersion
                             && mem_response.operation_id() == operation_id && mem_response.success()
                             && mem_response.quiesced();
        const bool legacy = mem_response.protocol_version() == 0 && mem_response.operation_id().empty()
                            && mem_response.success() && !mem_response.quiesced();
        const bool known_incompatible =
            legacy
            || (mem_response.protocol_version() > 0
                && mem_response.protocol_version() != kMemoryCopyProtocolVersion
                && mem_response.operation_id() == operation_id && mem_response.success()
                && mem_response.quiesced());

        if (!current && !known_incompatible) {
            return CapabilityProbeResult::TRANSIENT_FAILURE;
        }
        all_current      = all_current && current;
        all_incompatible = all_incompatible && known_incompatible;
    }

    if (all_current) {
        return CapabilityProbeResult::READY;
    }
    if (all_incompatible) {
        return CapabilityProbeResult::UNSUPPORTED;
    }
    return CapabilityProbeResult::TRANSIENT_FAILURE;
}

std::shared_ptr<BroadcastResult<FunctionRequestPB, FunctionResponsePB>>
KVCacheMemoryConnector::sendCapabilityProbe(const std::string& operation_id,
                                            int64_t            operation_deadline_unix_ms) const {
    const int64_t rpc_timeout_ms = MemoryCopyDeadline::rpcTimeout(operation_deadline_unix_ms,
                                                                  kv_cache_config_.memory_cache_sync_timeout_ms,
                                                                  MemoryCopyDeadline::unixMillisNow());
    if (rpc_timeout_ms <= 0) {
        return nullptr;
    }
    MemoryOperationRequestPB mem_req;
    mem_req.set_operation_kind(MemoryOperationRequestPB::CAPABILITY);
    mem_req.set_operation_id(operation_id);
    mem_req.set_protocol_version(kMemoryCopyProtocolVersion);

    std::vector<FunctionRequestPB> requests(broadcast_manager_->workerNum());
    for (auto& request : requests) {
        request.mutable_mem_request()->CopyFrom(mem_req);
    }
    auto rpc_call = [](const std::shared_ptr<RpcService::Stub>&    stub,
                       const std::shared_ptr<grpc::ClientContext>& context,
                       const FunctionRequestPB&                    request,
                       grpc::CompletionQueue*                      completion_queue) {
        return stub->PrepareAsyncExecuteFunction(context.get(), request, completion_queue);
    };
    return broadcast_manager_->broadcastPrepared<FunctionRequestPB, FunctionResponsePB>(
        requests, static_cast<int>(rpc_timeout_ms), rpc_call);
}

bool KVCacheMemoryConnector::quiesceCopy(const std::string& operation_id,
                                         int64_t            operation_deadline_unix_ms) const {
    MemoryOperationRequestPB mem_req;
    mem_req.set_operation_kind(MemoryOperationRequestPB::QUIESCE);
    mem_req.set_operation_id(operation_id);
    mem_req.set_retention_timeout_ms(kv_cache_config_.memory_cache_sync_timeout_ms);
    mem_req.set_quiesce_timeout_ms(kv_cache_config_.memory_cache_sync_timeout_ms);
    mem_req.set_operation_deadline_unix_ms(operation_deadline_unix_ms);
    mem_req.set_protocol_version(kMemoryCopyProtocolVersion);

    std::vector<FunctionRequestPB> requests(broadcast_manager_->workerNum());
    for (auto& request : requests) {
        request.mutable_mem_request()->CopyFrom(mem_req);
    }
    auto rpc_call = [](const std::shared_ptr<RpcService::Stub>&    stub,
                       const std::shared_ptr<grpc::ClientContext>& context,
                       const FunctionRequestPB&                    request,
                       grpc::CompletionQueue*                      completion_queue) {
        return stub->PrepareAsyncExecuteFunction(context.get(), request, completion_queue);
    };
    auto result = broadcast_manager_->broadcastPrepared<FunctionRequestPB, FunctionResponsePB>(
        requests, kv_cache_config_.memory_cache_sync_timeout_ms, rpc_call);
    return waitForCopyResult(result, operation_id);
}

std::string KVCacheMemoryConnector::makeCopyOperationId() {
    return copy_operation_ids_.next();
}

std::shared_ptr<BroadcastResult<FunctionRequestPB, FunctionResponsePB>>
KVCacheMemoryConnector::sendCopyPlan(const std::shared_ptr<CopyPlan>& copy_plan) const {
    const int64_t rpc_timeout_ms = MemoryCopyDeadline::rpcTimeout(copy_plan->operation_deadline_unix_ms,
                                                                  kv_cache_config_.memory_cache_sync_timeout_ms,
                                                                  MemoryCopyDeadline::unixMillisNow());
    if (rpc_timeout_ms <= 0) {
        return nullptr;
    }
    MemoryOperationRequestPB mem_req;
    mem_req.set_operation_kind(MemoryOperationRequestPB::COPY);
    mem_req.set_operation_id(copy_plan->operation_id);
    mem_req.set_retention_timeout_ms(kv_cache_config_.memory_cache_sync_timeout_ms);
    mem_req.set_operation_deadline_unix_ms(copy_plan->operation_deadline_unix_ms);
    mem_req.set_protocol_version(kMemoryCopyProtocolVersion);
    mem_req.set_copy_direction(copy_plan->direction == CopyDirection::H2D ? MemoryOperationRequestPB::H2D :
                                                                            MemoryOperationRequestPB::D2H);
    for (const auto& copy_info : copy_plan->copy_infos) {
        auto* item = mem_req.add_copy_items();
        item->set_mem_block(copy_info.mem_block);
        for (const auto& block : copy_info.gpu_blocks) {
            item->add_gpu_blocks(block);
        }
    }

    std::vector<FunctionRequestPB> requests;
    requests.reserve(broadcast_manager_->workerNum());
    for (size_t i = 0; i < broadcast_manager_->workerNum(); ++i) {
        FunctionRequestPB req;
        req.mutable_mem_request()->CopyFrom(mem_req);
        requests.emplace_back(std::move(req));
    }

    auto rpc_call = [](const std::shared_ptr<RpcService::Stub>&    stub,
                       const std::shared_ptr<grpc::ClientContext>& context,
                       const FunctionRequestPB&                    request,
                       grpc::CompletionQueue*                      completion_queue) {
        return stub->PrepareAsyncExecuteFunction(context.get(), request, completion_queue);
    };
    return broadcast_manager_->broadcastPrepared<FunctionRequestPB, FunctionResponsePB>(
        requests, static_cast<int>(rpc_timeout_ms), rpc_call);
}

void KVCacheMemoryConnector::printCopyPlan(const std::shared_ptr<CopyPlan>& copy_plan) const {
    std::ostringstream oss;
    oss << "copy plan direction: " << (copy_plan->direction == CopyDirection::H2D ? "H2D" : "D2H")
        << ", copy infos size: " << copy_plan->copy_infos.size() << "\n";
    for (int i = 0; i < copy_plan->copy_infos.size(); ++i) {
        const auto& copy_info = copy_plan->copy_infos.at(i);
        oss << "copy info " << i << ": cache key: " << copy_info.cache_key << ", mem block: " << copy_info.mem_block
            << ", gpu layer blocks: [";
        for (const auto& gpu_block : copy_info.gpu_blocks) {
            oss << gpu_block << ", ";
        }
        oss << "]\n";
    }
    RTP_LLM_LOG_INFO("%s", oss.str().c_str());
}

bool KVCacheMemoryConnector::copyCache(const MemoryOperationRequestPB& request, MemoryOperationResponsePB& response) {
    RTP_LLM_PROFILE_FUNCTION();
    autil::ScopedTime2 timer;
    response.set_operation_id(request.operation_id());
    response.set_quiesced(false);
    response.set_protocol_version(kMemoryCopyProtocolVersion);

    if (request.operation_id().empty()) {
        RTP_LLM_LOG_WARNING("memory copy request has no valid operation identity or retention");
        response.set_success(false);
        return true;
    }

    if (request.operation_kind() == MemoryOperationRequestPB::CAPABILITY) {
        const bool supported = request.protocol_version() == kMemoryCopyProtocolVersion;
        response.set_success(supported);
        response.set_quiesced(supported);
        return true;
    }

    if (request.protocol_version() != kMemoryCopyProtocolVersion) {
        RTP_LLM_LOG_WARNING("memory copy request protocol version is unsupported: %d", request.protocol_version());
        response.set_success(false);
        return true;
    }

    if (request.operation_kind() == MemoryOperationRequestPB::QUIESCE) {
        if (!MemoryCopyDeadline::validWireDuration(request.quiesce_timeout_ms())) {
            RTP_LLM_LOG_WARNING("memory copy quiesce timeout is invalid");
            response.set_success(false);
            return true;
        }
        const auto deadline = MemoryCopyDeadline::evaluateQuiesce(request.operation_deadline_unix_ms(),
                                                                  request.retention_timeout_ms(),
                                                                  kv_cache_config_.memory_cache_sync_timeout_ms,
                                                                  MemoryCopyDeadline::unixMillisNow());
        if (!deadline) {
            RTP_LLM_LOG_WARNING("memory copy quiesce rejected: %s", deadline.error.c_str());
            response.set_success(false);
            return true;
        }
        const auto wait_timeout = std::chrono::milliseconds(request.quiesce_timeout_ms());
        const bool quiesced = copy_fence_.sealAndWait(request.operation_id(), wait_timeout, deadline.retention);
        response.set_success(quiesced);
        response.set_quiesced(quiesced);
        return true;
    }

    if (request.operation_kind() != MemoryOperationRequestPB::COPY) {
        RTP_LLM_LOG_WARNING("memory copy request has unsupported operation kind: %d", request.operation_kind());
        response.set_success(false);
        return true;
    }
    if (request.copy_items_size() == 0) {
        RTP_LLM_LOG_WARNING("memory copy request has no copy items");
        response.set_success(false);
        return true;
    }
    if (request.copy_direction() != MemoryOperationRequestPB::H2D
        && request.copy_direction() != MemoryOperationRequestPB::D2H) {
        RTP_LLM_LOG_WARNING("memory copy request has unsupported copy direction: %d", request.copy_direction());
        response.set_success(false);
        return true;
    }

    const auto deadline = MemoryCopyDeadline::evaluateCopy(request.operation_deadline_unix_ms(),
                                                           request.retention_timeout_ms(),
                                                           kv_cache_config_.memory_cache_sync_timeout_ms,
                                                           MemoryCopyDeadline::unixMillisNow());
    if (!deadline) {
        RTP_LLM_LOG_WARNING("memory copy rejected: %s", deadline.error.c_str());
        response.set_success(false);
        return true;
    }
    auto begin = copy_fence_.beginBeforeDeadline(
        request.operation_id(), deadline.retention, request.operation_deadline_unix_ms());
    if (!begin) {
        RTP_LLM_LOG_WARNING("memory copy rejected: %s", begin.error.c_str());
        response.set_success(false);
        return true;
    }
    auto copy_operation = std::move(begin.operation);
    const auto copy_direction =
        (request.copy_direction() == MemoryOperationRequestPB::H2D) ? CopyDirection::H2D : CopyDirection::D2H;
    const bool copy_succeeded = executeCopy(request, copy_direction);
    copy_operation.reset();

    response.set_success(copy_succeeded);
    response.set_quiesced(true);
    reportCopyMetrics(copy_succeeded, timer.done_us(), copy_direction);
    return copy_succeeded;
}

bool KVCacheMemoryConnector::executeCopy(const MemoryOperationRequestPB& request, CopyDirection copy_direction) {
    std::vector<torch::Tensor> dst_buffers;
    std::vector<torch::Tensor> src_buffers;
    for (int i = 0; i < request.copy_items_size(); ++i) {
        const auto&                     item      = request.copy_items(i);
        const auto                      mem_block = static_cast<BlockIdxType>(item.mem_block());
        const std::vector<BlockIdxType> gpu_blocks(item.gpu_blocks().begin(), item.gpu_blocks().end());

        if (!prepareCopyBuffers(mem_block, gpu_blocks, copy_direction, dst_buffers, src_buffers)) {
            RTP_LLM_LOG_WARNING("copy cache failed, prepare copy buffers failed, mem_block=%d, direction=%s",
                                mem_block,
                                copy_direction == CopyDirection::H2D ? "H2D" : "D2H");
            return false;
        }
    }

    if (!dst_buffers.empty()) {
        MultiCopyParams mc{dst_buffers, src_buffers};
        applySplitKvMultiCopyFieldsIfEligible(kv_cache_config_.enable_memory_cache_sm_copy, cache_config_, mc);
        execNoBlockCopy(mc);
    }

    return true;
}

bool KVCacheMemoryConnector::prepareCopyBuffers(BlockIdxType                     mem_block,
                                                const std::vector<BlockIdxType>& gpu_blocks,
                                                CopyDirection                    direction,
                                                std::vector<torch::Tensor>&      dst,
                                                std::vector<torch::Tensor>&      src) {
    RTP_LLM_CHECK_WITH_INFO(mem_block != NULL_BLOCK_IDX, "mem block is null");
    RTP_LLM_CHECK_WITH_INFO(block_pool_ != nullptr, "block pool is null");
    auto mem_buffers = block_pool_->convertIndexToBuffer(/*layer_id=*/0, mem_block);
    if (mem_buffers.empty()) {
        RTP_LLM_LOG_WARNING("prepare copy buffers failed, mem buffers are empty, block=%d, direction=%s",
                            mem_block,
                            direction == CopyDirection::H2D ? "H2D" : "D2H");
        return false;
    }

    // memory has only one buffer
    const auto& mem_buffer = mem_buffers[0];
    RTP_LLM_CHECK_WITH_INFO(mem_buffer.addr != nullptr && mem_buffer.size_bytes > 0,
                            "mem buffer address is null or size is 0, addr=%p, size=%zu, block=%d, direction=%s",
                            mem_buffer.addr,
                            mem_buffer.size_bytes,
                            mem_block,
                            direction == CopyDirection::H2D ? "H2D" : "D2H");

    const size_t layer_num = cache_config_.layer_all_num;
    RTP_LLM_CHECK_WITH_INFO(gpu_blocks.size() == layer_num,
                            "gpu_blocks must contain all layers, got=%zu need=%zu",
                            gpu_blocks.size(),
                            layer_num);

    size_t byte_off = 0;
    for (int layer = 0; layer < layer_num; ++layer) {
        const auto gpu_block    = gpu_blocks.at(layer);
        const auto layer_stride = cache_config_.layer_to_block_stride_bytes[layer];

        if (isNullBlockIdx(gpu_block)) {
            byte_off += layer_stride;
            continue;
        }

        const auto gpu_buffers      = allocator_->convertIndexToBuffer(layer, gpu_block);
        size_t     within_layer_off = 0;
        for (const auto& gpu_buffer : gpu_buffers) {
            if (within_layer_off + gpu_buffer.size_bytes > layer_stride) {
                RTP_LLM_LOG_WARNING("prepare copy buffers failed, gpu buffer overflow: "
                                    "layer=%zu byte_off=%zu within_layer_off=%zu gpu_buffer_size=%zu",
                                    layer,
                                    byte_off,
                                    within_layer_off,
                                    gpu_buffer.size_bytes);
                return false;
            }
            const size_t off = byte_off + within_layer_off;
            if (!appendCopyBytesToBuffers(mem_buffer, gpu_buffer, off, direction, dst, src)) {
                return false;
            }
            within_layer_off += gpu_buffer.size_bytes;
        }
        byte_off += layer_stride;
    }
    return true;
}

bool KVCacheMemoryConnector::appendCopyBytesToBuffers(const BlockInfo&            mem_block,
                                                      const BlockInfo&            gpu_block,
                                                      size_t                      byte_off,
                                                      CopyDirection               direction,
                                                      std::vector<torch::Tensor>& dst,
                                                      std::vector<torch::Tensor>& src) {
    if (!gpu_block.addr || gpu_block.size_bytes == 0) {
        return true;
    }
    if (byte_off + gpu_block.size_bytes > mem_block.size_bytes) {
        RTP_LLM_LOG_WARNING(
            "append copy bytes to buffers failed, mem block overflow: offset=%zu bytes=%zu mem_size=%zu",
            byte_off,
            gpu_block.size_bytes,
            mem_block.size_bytes);
        return false;
    }

    auto mem_device = mem_block.is_cuda ? torch::kCUDA : torch::kCPU;
    auto gpu_device = gpu_block.is_cuda ? torch::kCUDA : torch::kCPU;
    auto mem_tensor = torch::from_blob(static_cast<void*>(static_cast<char*>(mem_block.addr) + byte_off),
                                       {(int64_t)gpu_block.size_bytes},
                                       torch::TensorOptions().dtype(torch::kUInt8).device(mem_device));
    auto gpu_tensor = torch::from_blob(gpu_block.addr,
                                       {(int64_t)gpu_block.size_bytes},
                                       torch::TensorOptions().dtype(torch::kUInt8).device(gpu_device));
    if (direction == CopyDirection::H2D) {
        src.push_back(mem_tensor);
        dst.push_back(gpu_tensor);
    } else {
        src.push_back(gpu_tensor);
        dst.push_back(mem_tensor);
    }
    return true;
}

bool KVCacheMemoryConnector::checkLayerBlocks(const LayerBlockIds& layer_block_ids, size_t required_len) const {
    if (layer_block_ids.empty()) {
        RTP_LLM_LOG_WARNING(
            "check layer blocks failed, layer_block_ids is empty (required_len=%zu, layer_block_ids.size=%zu)",
            required_len,
            layer_block_ids.size());
        return false;
    }

    const auto layer_num = cache_config_.layer_all_num;
    if (layer_block_ids.size() != layer_num) {
        RTP_LLM_LOG_WARNING(
            "check layer blocks failed, layer block ids size is not equal to layer num, layer block ids size: %zu, layer num: %zu",
            layer_block_ids.size(),
            layer_num);
        return false;
    }
    for (const auto& blocks : layer_block_ids) {
        if (blocks->blocksNum() < required_len) {
            RTP_LLM_LOG_WARNING(
                "check layer blocks failed, layer blocksNum is less than required_len, blocksNum: %zu, required_len: %zu",
                blocks->blocksNum(),
                required_len);
            return false;
        }
    }
    return true;
}

bool KVCacheMemoryConnector::mallocBlocks(size_t need_blocks, std::vector<BlockIdxType>& malloced_blocks) {
    RTP_LLM_PROFILE_FUNCTION();
    if (need_blocks == 0) {
        RTP_LLM_LOG_WARNING("malloc memory blocks failed, need blocks cannot be 0");
        return false;
    }

    // make sure `eusure + malloc` is atomic
    std::unique_lock<std::mutex> lock(malloc_mutex_);

    if (!ensureEnoughFreeBlocks(need_blocks)) {
        RTP_LLM_LOG_WARNING(
            "malloc memory blocks failed, ensure enough free blocks failed, need blocks: %zu, free blocks: %zu",
            need_blocks,
            block_pool_->freeBlocksNum());
        return false;
    }

    auto blocks = block_pool_->malloc(need_blocks);
    if (blocks.size() != need_blocks) {
        RTP_LLM_LOG_WARNING("malloc memory blocks failed, malloc failed, need blocks: %zu, allocated blocks: %zu",
                            need_blocks,
                            blocks.size());
        freeBlocks(blocks, /*cache_free=*/false);
        return false;
    }
    malloced_blocks = std::move(blocks);
    return true;
}

bool KVCacheMemoryConnector::freeBlocks(const std::vector<BlockIdxType>& blocks, bool cache_free) {
    RTP_LLM_PROFILE_FUNCTION();
    std::vector<int> need_free_blocks;
    need_free_blocks.reserve(blocks.size());
    for (const auto& block : blocks) {
        if (isNullBlockIdx(block)) {
            continue;
        }
        need_free_blocks.push_back(static_cast<int>(block));
    }
    if (need_free_blocks.empty()) {
        return true;
    }

    RTP_LLM_CHECK_WITH_INFO(block_pool_ != nullptr, "block pool is null");
    if (cache_free) {
        // cache中的block需要blockCacheFree
        block_pool_->blockCacheFree(need_free_blocks);
    } else {
        // malloc的block需要requestFree
        block_pool_->requestFree(need_free_blocks);
    }
    return true;
}

std::shared_ptr<BlockPool> KVCacheMemoryConnector::createBlockPool(size_t block_size, size_t pool_size_mb) const {
    RTP_LLM_CHECK_WITH_INFO(pool_size_mb > 0, "pool size must be > 0");
    const int64_t block_num = pool_size_mb * 1024 * 1024 / static_cast<int64_t>(block_size);
    RTP_LLM_CHECK_WITH_INFO(
        block_num > 0, "pool_size_mb=%ld is too small for block_size=%zu (block_num=0)", pool_size_mb, block_size);
    RTP_LLM_LOG_INFO("create memory block pool, pool size: %ld MB, block num: %ld, block size: %zu",
                     pool_size_mb,
                     block_num,
                     block_size);
    const auto pool_config = BlockPoolConfigHelper::createConfig(
        /*layer_num=*/1, static_cast<uint32_t>(block_num), static_cast<uint32_t>(block_size), rtp_llm::TYPE_INT8);
    auto pool = std::make_shared<BlockPool>(pool_config, AllocationType::HOST);
    RTP_LLM_CHECK_WITH_INFO(pool->init(), "memory block pool init failed, block size: %zu", block_size);
    return pool;
}

std::string KVCacheMemoryConnector::blockPoolDebugString() const {
    std::stringstream oss;
    oss << "total blocks num: " << block_pool_->totalBlocksNum()
        << ", free blocks num: " << block_pool_->freeBlocksNum()
        << ", available blocks num: " << block_pool_->availableBlocksNum();
    return oss.str();
}

void KVCacheMemoryConnector::putToCache(const MemoryBlockCache::CacheItem& item) {
    RTP_LLM_PROFILE_FUNCTION();
    if (auto [success, popped_item_opt] = block_cache_->putAndBlockCacheReference(item, *block_pool_); success) {
        RTP_LLM_LOG_DEBUG("write cache, cache key: %ld, block index: %d, block size: %zu",
                          item.cache_key,
                          item.block_index,
                          item.block_size);
        if (popped_item_opt.has_value()) {
            const auto popped_item = popped_item_opt.value();
            freeBlocks({popped_item.block_index}, /*cache_free=*/true);
        }
    }
}

// this function is called under lock
bool KVCacheMemoryConnector::ensureEnoughFreeBlocks(size_t need_blocks) {
    RTP_LLM_PROFILE_FUNCTION();
    auto free_blocks = block_pool_->freeBlocksNum();
    if (free_blocks >= need_blocks) {
        return true;
    }
    const auto need_evict_blocks = need_blocks - free_blocks;
    const auto evict_blocks      = block_cache_->pop(need_evict_blocks);
    if (!evict_blocks.empty()) {
        freeBlocks(evict_blocks, /*cache_free=*/true);
    }
    return block_pool_->freeBlocksNum() >= need_blocks;
}

std::vector<CacheKeyType> KVCacheMemoryConnector::cacheKeys() const {
    RTP_LLM_CHECK_WITH_INFO(block_cache_ != nullptr, "block cache should not be null");
    return block_cache_->cacheKeys();
}

void KVCacheMemoryConnector::reportMatchMetrics(bool    success,
                                                int64_t latency_us,
                                                int64_t input_block_num,
                                                int64_t matched_block_num) {
    if (!metrics_reporter_) {
        return;
    }

    RtpLLMMemoryCacheMatchMetricsCollector collector;
    collector.failed        = !success;
    collector.latency_us    = latency_us;
    collector.input_token   = input_block_num * cache_config_.seq_size_per_block;
    collector.matched_token = matched_block_num * cache_config_.seq_size_per_block;

    metrics_reporter_->report<RtpLLMMemoryCacheMetrics, RtpLLMMemoryCacheMatchMetricsCollector>(nullptr, &collector);
}

void KVCacheMemoryConnector::reportReadMetrics(bool    success,
                                               int64_t latency_us,
                                               int64_t input_block_num,
                                               int64_t read_block_num) {
    if (!metrics_reporter_) {
        return;
    }

    RtpLLMMemoryCacheReadMetricsCollector collector;
    collector.failed      = !success;
    collector.latency_us  = latency_us;
    collector.input_token = input_block_num * cache_config_.seq_size_per_block;
    collector.read_token  = read_block_num * cache_config_.seq_size_per_block;

    metrics_reporter_->report<RtpLLMMemoryCacheMetrics, RtpLLMMemoryCacheReadMetricsCollector>(nullptr, &collector);
}

void KVCacheMemoryConnector::reportWriteMetrics(bool    success,
                                                int64_t latency_us,
                                                int64_t input_block_num,
                                                int64_t write_block_num) {
    if (!metrics_reporter_) {
        return;
    }

    RtpLLMMemoryCacheWriteMetricsCollector collector;
    collector.failed      = !success;
    collector.latency_us  = latency_us;
    collector.input_token = input_block_num * cache_config_.seq_size_per_block;
    collector.write_token = write_block_num * cache_config_.seq_size_per_block;

    metrics_reporter_->report<RtpLLMMemoryCacheMetrics, RtpLLMMemoryCacheWriteMetricsCollector>(nullptr, &collector);
}

void KVCacheMemoryConnector::reportCopyMetrics(bool success, int64_t latency_us, CopyDirection direction) {
    if (!metrics_reporter_) {
        return;
    }

    RtpLLMMemoryCacheCopyMetricsCollector collector;
    collector.failed     = !success;
    collector.latency_us = latency_us;
    collector.from_gpu   = direction == CopyDirection::D2H;

    metrics_reporter_->report<RtpLLMMemoryCacheMetrics, RtpLLMMemoryCacheCopyMetricsCollector>(nullptr, &collector);
}

void KVCacheMemoryConnector::reportMetricsLoop() {
    while (!stop_.load()) {
        if (metrics_reporter_) {
            if (!block_pool_) {
                std::this_thread::sleep_for(std::chrono::seconds(1));
                continue;
            }

            const auto total_blocks     = block_pool_->totalBlocksNum();
            const auto free_blocks      = block_pool_->freeBlocksNum();
            const auto available_blocks = block_pool_->availableBlocksNum();

            RtpLLMMemoryCacheStatusMetricsCollector collector;
            collector.total_block_num     = total_blocks;
            collector.allocated_block_num = total_blocks - free_blocks;
            collector.available_block_num = available_blocks;

            metrics_reporter_->report<RtpLLMMemoryCacheMetrics, RtpLLMMemoryCacheStatusMetricsCollector>(nullptr,
                                                                                                         &collector);
        }
        std::this_thread::sleep_for(std::chrono::seconds(1));
    }
}

}  // namespace rtp_llm
