#include "rtp_llm/cpp/cache/connector/memory/KVCacheMemoryConnector.h"

#include "rtp_llm/cpp/cache/BlockPool.h"
#include "rtp_llm/cpp/cache/BlockPoolConfigHelper.h"
#include "rtp_llm/cpp/cache/KVCacheAllocator.h"
#include "rtp_llm/cpp/devices/DeviceBase.h"
#include "rtp_llm/cpp/utils/Logger.h"

namespace rtp_llm {

// ----------------------------- MemoryConnectorAsyncContext ---------------------------------

MemoryConnectorAsyncContext::~MemoryConnectorAsyncContext() {
    waitDone();
}

bool MemoryConnectorAsyncContext::done() const {
    return already_done_;
}

bool MemoryConnectorAsyncContext::success() const {
    if (!broadcast_result_ || !broadcast_result_->success()) {
        return false;
    }

    const auto& responses = broadcast_result_->responses();
    for (const auto& response : responses) {
        if (!response.has_mem_response() || !response.mem_response().success()) {
            return false;
        }
    }
    return true;
}

void MemoryConnectorAsyncContext::waitDone() {
    if (already_done_) {
        return;
    }
    std::call_once(wait_done_once_, [this]() {
        if (broadcast_result_) {
            broadcast_result_->waitDone();
        }
        if (done_callback_) {
            done_callback_(success());
        }
        already_done_ = true;
    });
}

// ----------------------------- KVCacheMemoryConnector ---------------------------------

class MemoryAsyncMatchContext: public KVCacheConnector::AsyncMatchContext {
public:
    MemoryAsyncMatchContext(size_t matched_block_count): matched_block_count_(matched_block_count) {}
    ~MemoryAsyncMatchContext() override = default;

    bool done() const override {
        return true;
    }
    bool success() const override {
        return true;
    }
    size_t matchedBlockCount() const override {
        return matched_block_count_;
    }
    KVCacheConnector::ConnectorType connectorType() const override {
        return KVCacheConnector::ConnectorType::Memory;
    }

private:
    size_t matched_block_count_{0};
};

KVCacheMemoryConnector::KVCacheMemoryConnector(const CacheConfig&                       cache_config,
                                               const KVCacheConfig&                     kv_cache_config,
                                               const std::shared_ptr<KVCacheAllocator>& allocator,
                                               rtp_llm::DeviceBase*                     device,
                                               const std::vector<std::string>&          tp_addrs,
                                               const kmonitor::MetricsReporterPtr&      metrics_reporter):
    cache_config_(cache_config),
    kv_cache_config_(kv_cache_config),
    allocator_(allocator),
    device_(device),
    tp_addrs_(tp_addrs),
    metrics_reporter_(metrics_reporter) {}

KVCacheMemoryConnector::~KVCacheMemoryConnector() {
    RTP_LLM_LOG_INFO("KVCacheMemoryConnector destructor");
    stop_.store(true);
    if (metrics_reporter_thread_ && metrics_reporter_thread_->joinable()) {
        metrics_reporter_thread_->join();
        metrics_reporter_thread_.reset();
    }
    if (wait_done_thread_pool_) {
        wait_done_thread_pool_->stop();
        wait_done_thread_pool_.reset();
    }
    tp_broadcast_manager_.reset();
    block_pools_.clear();
    block_cache_.reset();
}

bool KVCacheMemoryConnector::init() {
    const auto memory_cache_sync_timeout_ms = kv_cache_config_.memory_cache_sync_timeout_ms;
    RTP_LLM_CHECK_WITH_INFO(memory_cache_sync_timeout_ms > 0,
                            "init failed, sync timeout is invalid, sync timeout: %ld ms",
                            memory_cache_sync_timeout_ms);

    RTP_LLM_CHECK_WITH_INFO(initBlockPool(), "init block pool failed");
    block_cache_ = std::make_shared<MemoryBlockCache>();

    tp_broadcast_manager_ = std::make_shared<TpBroadcastManager>(tp_addrs_);
    RTP_LLM_CHECK_WITH_INFO(tp_broadcast_manager_->init(), "init failed, tp broadcast manager init failed");

    wait_done_thread_pool_ = std::make_shared<autil::LockFreeThreadPool>(8, 1000, nullptr, "WaitDoneThreadPool");
    RTP_LLM_CHECK_WITH_INFO(wait_done_thread_pool_->start(), "init failed, wait done thread pool start failed");

    if (metrics_reporter_) {
        metrics_reporter_thread_ =
            std::make_shared<std::thread>([self = shared_from_this()]() { self->reportMetricsLoop(); });
    }
    return true;
}

bool KVCacheMemoryConnector::initBlockPool() {
    const auto memory_cache_size_mb = kv_cache_config_.memory_cache_size_mb;
    RTP_LLM_CHECK_WITH_INFO(memory_cache_size_mb > 0,
                            "init block pool failed, memory size is invalid, memory size: %ld MB",
                            memory_cache_size_mb);

    // block_size here means "one cache-key across all layers" total bytes (kv + scale).
    const size_t block_size = cache_config_.block_size_bytes;
    RTP_LLM_CHECK_WITH_INFO(block_size > 0, "init block pool failed, block size is invalid: %zu", block_size);

    auto pool = createBlockPool(block_size, memory_cache_size_mb);
    RTP_LLM_CHECK_WITH_INFO(pool != nullptr, "init block pool failed, create block pool failed");
    {
        std::lock_guard<std::shared_mutex> lock(pool_mutex_);
        block_pools_[block_size] = pool;
    }
    return true;
}

std::shared_ptr<KVCacheConnector::AsyncMatchContext>
KVCacheMemoryConnector::asyncMatch(const std::shared_ptr<KVCacheResource>& resource,
                                   const std::shared_ptr<Meta>&            meta) {
    const auto&  cache_keys    = resource->cacheKeys();
    const size_t gpu_reuse_num = resource->reuseBlocksNum();
    if (gpu_reuse_num >= cache_keys.size()) {
        // gpu has already matched all cache keys, no need to match in memory
        RTP_LLM_LOG_DEBUG(
            "async match skip, gpu reuse len is greater than cache keys size, cache_keys size: %zu, gpu_reuse_num: %zu",
            cache_keys.size(),
            gpu_reuse_num);
        return nullptr;
    }

    autil::ScopedTime2 timer;

    // If last_block_aligned is false, skip matching the last (partial) block
    const size_t matchable_count =
        resource->lastBlockAligned() ? cache_keys.size() : (cache_keys.empty() ? 0 : cache_keys.size() - 1);

    size_t matched_num = 0;
    for (; matched_num < matchable_count; ++matched_num) {
        const auto cache_key    = cache_keys.at(matched_num);
        const auto match_result = block_cache_->match(static_cast<CacheKeyType>(cache_key));
        if (isNullBlockIdx(match_result.matched_index)) {
            break;  // 只处理连续前缀
        }
    }

    if (matched_num == 0) {
        RTP_LLM_LOG_DEBUG(
            "not matched cache in memory, cache keys size: %zu, matchable_count: %zu, last_block_aligned: %d",
            cache_keys.size(),
            matchable_count,
            resource->lastBlockAligned());
        reportMatchMetrics(/*success=*/false, timer.done_us(), cache_keys.size(), matched_num);
        return nullptr;
    }
    reportMatchMetrics(/*success=*/true, timer.done_us(), cache_keys.size(), matched_num);
    return std::make_shared<MemoryAsyncMatchContext>(matched_num);
}

std::shared_ptr<AsyncContext>
KVCacheMemoryConnector::asyncRead(const std::shared_ptr<KVCacheResource>&   resource,
                                  const std::shared_ptr<Meta>&              meta,
                                  const std::shared_ptr<AsyncMatchContext>& match_context) {
    autil::ScopedTime2 timer;

    if (!checkKVCacheResource(resource)) {
        RTP_LLM_LOG_WARNING("async read failed, resource is invalid");
        const size_t cache_keys_num = resource ? resource->cacheKeys().size() : 0;
        reportReadMetrics(false, timer.done_us(), cache_keys_num, 0);
        return nullptr;
    }

    const auto& cache_keys             = resource->cacheKeys();
    const auto& layer_block_ids        = resource->layerBlocks();
    const int   start_read_block_index = meta->start_block_index;
    const int   read_block_num         = meta->block_size;
    const auto  matched_block_num      = match_context->matchedBlockCount();

    if (start_read_block_index < 0 || start_read_block_index > cache_keys.size() || read_block_num <= 0
        || start_read_block_index + read_block_num > cache_keys.size()) {
        RTP_LLM_LOG_WARNING(
            "async read failed, invalid block range, start_read_block_index: %d, read_block_num: %d, cache_keys size: %zu",
            start_read_block_index,
            read_block_num,
            cache_keys.size());
        reportReadMetrics(false, timer.done_us(), cache_keys.size(), 0);
        return nullptr;
    }

    // 因为下文会在线程池中push wait done任务, 所以这里提前检查线程池是否已满
    if (isThreadPoolFull()) {
        RTP_LLM_LOG_WARNING("async read failed, thread pool is full");
        reportReadMetrics(false, timer.done_us(), cache_keys.size(), 0);
        return nullptr;
    }

    auto copy_infos = buildCopyPlanForRead(cache_keys, layer_block_ids, start_read_block_index, read_block_num);
    if (copy_infos.empty()) {
        RTP_LLM_LOG_WARNING(
            "async read failed, build copy plan for read failed, cache keys size: %zu, start_read_block_index: %d, read_block_num: %d",
            cache_keys.size(),
            start_read_block_index,
            read_block_num);
        reportReadMetrics(false, timer.done_us(), cache_keys.size(), 0);
        return nullptr;
    }

    auto send_result = sendCopyPlan(copy_infos, CopyDirection::H2D);
    if (!send_result) {
        RTP_LLM_LOG_WARNING("async read failed, send copy plan to tp failed");
        for (const auto& copy_info : copy_infos) {
            auto block_pool = getBlockPool(copy_info.mem_block_size);
            freeBlocks(block_pool, {copy_info.mem_block_index}, /*cache_free=*/true);
        }
        reportReadMetrics(false, timer.done_us(), cache_keys.size(), 0);
        return nullptr;
    }

    const auto total_block_num = cache_keys.size();
    auto       read_done =
        [resource, copy_infos, total_block_num, matched_block_num, read_block_num, timer, self = shared_from_this()](
            bool success) mutable {
            RTP_LLM_LOG_DEBUG("async read done, success: %d", success);
            if (success) {
                resource->setReuseBlocksNum(matched_block_num);
            }
            for (const auto& copy_info : copy_infos) {
                auto block_pool = self->getBlockPool(copy_info.mem_block_size);
                self->freeBlocks(block_pool, {copy_info.mem_block_index}, /*cache_free=*/true);
            }
            self->reportReadMetrics(success, timer.done_us(), total_block_num, read_block_num);
        };

    auto context = std::make_shared<MemoryConnectorAsyncContext>(send_result, read_done);
    waitContextDoneAsync(context);
    return context;
}

std::vector<KVCacheMemoryConnector::CopyInfoPerKey>
KVCacheMemoryConnector::buildCopyPlanForRead(const std::vector<int64_t>& cache_keys,
                                             const LayerBlockIds&        layer_block_ids,
                                             int                         start_read_block_index,
                                             int                         read_block_num) {
    std::vector<CopyInfoPerKey> copy_infos;
    const auto                  layer_num = cache_config_.layer_all_num;
    bool                        success   = true;

    for (size_t i = start_read_block_index; i < start_read_block_index + read_block_num; ++i) {
        const auto cache_key    = cache_keys.at(i);
        const auto match_result = block_cache_->match(static_cast<CacheKeyType>(cache_key));
        if (isNullBlockIdx(match_result.matched_index)) {
            RTP_LLM_LOG_WARNING(
                "build copy plan for read failed, found null memory block index, cache key: %zu, cache key size: %zu, start_read_block_index: %d, read_block_num: %d",
                cache_key,
                cache_keys.size(),
                start_read_block_index,
                read_block_num);
            success = false;
            break;
        }

        auto block_pool = getBlockPool(match_result.block_size);
        if (!block_pool) {
            RTP_LLM_LOG_WARNING(
                "build copy plan for read failed, get block pool failed, cache key: %zu, block size: %zu, block pool: %s",
                cache_key,
                match_result.block_size,
                blockPoolDebugString().c_str());
            success = false;
            break;
        }
        referenceBlocks(block_pool, {match_result.matched_index});

        CopyInfoPerKey copy_info;
        copy_info.cache_key       = cache_key;
        copy_info.mem_block_index = match_result.matched_index;
        copy_info.mem_block_size  = match_result.block_size;
        copy_info.gpu_layer_blocks.reserve(layer_num);
        for (size_t layer = 0; layer < layer_num; ++layer) {
            const int gpu_block_idx = layer_block_ids.at(layer)->blocks().at(i);
            if (isNullBlockIdx(gpu_block_idx)) {
                RTP_LLM_LOG_DEBUG("build copy plan for read, found null gpu block index, cache key: %zu, layer: %zu",
                                  cache_key,
                                  layer);
                continue;
            }
            LayerBlock lb{static_cast<int>(layer), gpu_block_idx};
            copy_info.gpu_layer_blocks.push_back(lb);
        }
        copy_infos.emplace_back(std::move(copy_info));
    }

    if (!success) {
        for (const auto& copy_info : copy_infos) {
            auto block_pool = getBlockPool(copy_info.mem_block_size);
            freeBlocks(block_pool, {copy_info.mem_block_index}, /*cache_free=*/true);
        }
        return {};
    }
    return copy_infos;
}

std::shared_ptr<AsyncContext> KVCacheMemoryConnector::asyncWrite(const std::shared_ptr<KVCacheResource>& resource,
                                                                 const std::shared_ptr<Meta>&            meta) {
    autil::ScopedTime2 timer;

    if (!checkKVCacheResource(resource)) {
        RTP_LLM_LOG_WARNING("async write failed, resource is invalid");
        const size_t cache_keys_num = resource ? resource->cacheKeys().size() : 0;
        reportWriteMetrics(false, timer.done_us(), cache_keys_num, 0);
        return nullptr;
    }

    const auto& cache_keys      = resource->cacheKeys();
    const auto& layer_block_ids = resource->layerBlocks();
    if (cache_keys.empty() || layer_block_ids.empty()) {
        RTP_LLM_LOG_WARNING(
            "async write failed, cache keys or layer block ids is empty, cache keys size: %zu, layer block ids size: %zu",
            cache_keys.size(),
            layer_block_ids.size());
        reportWriteMetrics(false, timer.done_us(), 0, 0);
        return nullptr;
    }

    // If last_block_aligned is false, skip the last (partial) block for writing
    const size_t writable_count =
        resource->lastBlockAligned() ? cache_keys.size() : (cache_keys.empty() ? 0 : cache_keys.size() - 1);
    if (writable_count == 0) {
        RTP_LLM_LOG_DEBUG("async write skip, no complete blocks to write, last_block_aligned: %d",
                          resource->lastBlockAligned());
        reportWriteMetrics(true, timer.done_us(), static_cast<int64_t>(cache_keys.size()), 0);
        return nullptr;
    }

    // 计算内存中已存在的前缀长度
    size_t cpu_matched_num = 0;
    for (; cpu_matched_num < writable_count; ++cpu_matched_num) {
        // TODO(LXQ): 是否需要提升热度?
        if (!block_cache_->contains(static_cast<CacheKeyType>(cache_keys[cpu_matched_num]))) {
            break;
        }
    }
    if (cpu_matched_num >= writable_count) {
        RTP_LLM_LOG_DEBUG(
            "async write skip, all writable cache keys already in memory cache, matched num: %zu, writable_count: %zu",
            cpu_matched_num,
            writable_count);
        reportWriteMetrics(true, timer.done_us(), static_cast<int64_t>(cache_keys.size()), 0);
        return nullptr;
    }

    // 因为下文会在线程池中push wait done任务, 所以这里提前检查线程池是否已满
    if (isThreadPoolFull()) {
        RTP_LLM_LOG_WARNING("async write failed, thread pool is full");
        reportWriteMetrics(false, timer.done_us(), static_cast<int64_t>(cache_keys.size()), 0);
        return nullptr;
    }

    auto copy_infos = buildCopyPlanForWrite(cache_keys, layer_block_ids, cpu_matched_num, writable_count);
    if (copy_infos.empty()) {
        RTP_LLM_LOG_WARNING("async write failed, build copy plan for write failed");
        reportWriteMetrics(false, timer.done_us(), static_cast<int64_t>(cache_keys.size()), 0);
        return nullptr;
    }

    auto send_result = sendCopyPlan(copy_infos, CopyDirection::D2H);
    if (!send_result) {
        RTP_LLM_LOG_WARNING("async write failed, send copy plan to tp failed");
        for (const auto& copy_info : copy_infos) {
            auto block_pool = getBlockPool(copy_info.mem_block_size);
            freeBlocks(block_pool, {copy_info.mem_block_index}, /*cache_free=*/false);
        }
        reportWriteMetrics(false, timer.done_us(), static_cast<int64_t>(cache_keys.size()), 0);
        return nullptr;
    }

    auto write_done =
        [copy_infos, resource_copy = resource, timer, total_block_num = cache_keys.size(), self = shared_from_this()](
            bool success) mutable {
            RTP_LLM_LOG_DEBUG("async write done, success: %d", success);

            if (success) {
                for (const auto& copy_info : copy_infos) {
                    if (self->block_cache_->contains(copy_info.cache_key)) {
                        continue;
                    }
                    MemoryBlockCache::CacheItem item;
                    item.cache_key   = copy_info.cache_key;
                    item.block_index = static_cast<BlockIdxType>(copy_info.mem_block_index);
                    item.block_size  = copy_info.mem_block_size;
                    item.is_resident = false;
                    self->putToCache(item);
                }
                resource_copy.reset();
            }

            for (const auto& copy_info : copy_infos) {
                auto block_pool = self->getBlockPool(copy_info.mem_block_size);
                self->freeBlocks(block_pool, {copy_info.mem_block_index}, /*cache_free=*/false);
            }

            const int64_t write_block_num = success ? static_cast<int64_t>(copy_infos.size()) : 0;
            self->reportWriteMetrics(success, timer.done_us(), total_block_num, write_block_num);
        };

    auto context = std::make_shared<MemoryConnectorAsyncContext>(send_result, write_done);
    waitContextDoneAsync(context);
    return context;
}

std::vector<KVCacheMemoryConnector::CopyInfoPerKey>
KVCacheMemoryConnector::buildCopyPlanForWrite(const std::vector<int64_t>& cache_keys,
                                              const LayerBlockIds&        layer_block_ids,
                                              size_t                      cpu_matched_num,
                                              size_t                      writable_count) {
    const auto                  layer_num = cache_config_.layer_all_num;
    bool                        success   = true;
    std::vector<CopyInfoPerKey> copy_infos;

    for (size_t i = cpu_matched_num; i < writable_count; ++i) {
        size_t                  total_bytes = 0;
        std::vector<LayerBlock> gpu_layer_blocks;
        for (size_t layer = 0; layer < layer_num; ++layer) {
            const int gpu_block_idx = layer_block_ids.at(layer)->blocks().at(i);
            if (isNullBlockIdx(gpu_block_idx)) {
                continue;
            }
            gpu_layer_blocks.push_back(LayerBlock{static_cast<int>(layer), gpu_block_idx});
            const auto buffers = allocator_->convertIndexToBuffer(static_cast<int>(layer), gpu_block_idx);
            if (buffers.kv_addr) {
                total_bytes += buffers.kv_addr->sizeBytes();
            }
            if (buffers.kv_scale_addr) {
                total_bytes += buffers.kv_scale_addr->sizeBytes();
            }
        }
        if (gpu_layer_blocks.empty() || total_bytes == 0) {
            RTP_LLM_LOG_WARNING(
                "build copy plan for write failed, invalid gpu_layer_blocks or total_bytes, cache key: %zu, gpu blocks: %zu, total bytes: %zu",
                cache_keys.at(i),
                gpu_layer_blocks.size(),
                total_bytes);
            success = false;
            break;
        }

        auto block_pool = getBlockPool(total_bytes);
        if (!block_pool) {
            RTP_LLM_LOG_WARNING(
                "build copy plan for write failed, get block pool failed, layer num: %zu, block size: %zu, block pool: %s",
                layer_num,
                total_bytes,
                blockPoolDebugString().c_str());
            success = false;
            break;
        }

        std::vector<BlockIdxType> mem_blocks;
        if (!mallocBlocks(block_pool, 1, mem_blocks)) {
            const int free_blocks = block_pool ? block_pool->freeBlocksNum() : -1;
            RTP_LLM_LOG_WARNING(
                "build copy plan for write failed, malloc memory blocks failed, maybe no enough free blocks, free blocks: %d",
                free_blocks);
            break;
        }

        CopyInfoPerKey copy_info;
        copy_info.cache_key        = cache_keys.at(i);
        copy_info.mem_block_index  = mem_blocks.front();
        copy_info.mem_block_size   = total_bytes;
        copy_info.gpu_layer_blocks = std::move(gpu_layer_blocks);
        copy_infos.emplace_back(std::move(copy_info));
    }
    if (!success) {
        for (const auto& copy_info : copy_infos) {
            auto block_pool = getBlockPool(copy_info.mem_block_size);
            freeBlocks(block_pool, {copy_info.mem_block_index}, /*cache_free=*/false);
        }
        return {};
    }
    return copy_infos;
}

std::shared_ptr<TPBroadcastResult<FunctionRequestPB, FunctionResponsePB>>
KVCacheMemoryConnector::sendCopyPlan(const std::vector<CopyInfoPerKey>& copy_infos, CopyDirection direction) const {
    if (!tp_broadcast_manager_ || tp_broadcast_manager_->workerNum() == 0) {
        RTP_LLM_LOG_WARNING("send copy plan failed, tp broadcast manager is null or no workers");
        return nullptr;
    }

    MemoryOperationRequestPB mem_req;
    mem_req.set_copy_direction(direction == CopyDirection::H2D ? MemoryOperationRequestPB::H2D :
                                                                 MemoryOperationRequestPB::D2H);
    for (const auto& copy_info : copy_infos) {
        auto* gpu_block = mem_req.add_gpu_blocks();
        for (const auto& lb : copy_info.gpu_layer_blocks) {
            auto* layer_block = gpu_block->add_layer_blocks();
            layer_block->set_layer_id(lb.layer_id);
            layer_block->set_block_id(lb.block_id);
        }
        mem_req.add_mem_block_ids(copy_info.mem_block_index);
        mem_req.add_mem_block_sizes(copy_info.mem_block_size);
    }

    std::vector<FunctionRequestPB> requests;
    requests.reserve(tp_broadcast_manager_->workerNum());
    for (size_t i = 0; i < tp_broadcast_manager_->workerNum(); ++i) {
        FunctionRequestPB req;
        req.mutable_mem_request()->CopyFrom(mem_req);
        requests.emplace_back(std::move(req));
    }

    auto rpc_call = [](const std::shared_ptr<RpcService::Stub>&    stub,
                       const std::shared_ptr<grpc::ClientContext>& context,
                       const FunctionRequestPB&                    request,
                       grpc::CompletionQueue*                      completion_queue) {
        return stub->AsyncExecuteFunction(context.get(), request, completion_queue);
    };
    return tp_broadcast_manager_->broadcast<FunctionRequestPB, FunctionResponsePB>(
        requests, kv_cache_config_.memory_cache_sync_timeout_ms, rpc_call);
}

bool KVCacheMemoryConnector::copyCache(const MemoryOperationRequestPB& request, MemoryOperationResponsePB& response) {
    autil::ScopedTime2 timer;
    const auto         copy_direction =
        (request.copy_direction() == MemoryOperationRequestPB::H2D) ? CopyDirection::H2D : CopyDirection::D2H;

    if (request.gpu_blocks_size() != request.mem_block_ids_size()
        || request.gpu_blocks_size() != request.mem_block_sizes_size()) {
        RTP_LLM_LOG_WARNING(
            "copy cache failed, count not match, gpu blocks: %zu, mem blocks: %zu, mem block sizes: %zu",
            request.gpu_blocks_size(),
            request.mem_block_ids_size(),
            request.mem_block_sizes_size());
        response.set_success(false);
        reportCopyMetrics(false, timer.done_us(), copy_direction);
        return false;
    }

    std::vector<BufferPtr> dst_buffers;
    std::vector<BufferPtr> src_buffers;
    for (int i = 0; i < request.gpu_blocks_size(); ++i) {
        const auto& gpu_block      = request.gpu_blocks(i);
        const auto  mem_block_id   = request.mem_block_ids(i);
        const auto  mem_block_size = request.mem_block_sizes(i);
        if (mem_block_id < 0 || mem_block_size <= 0) {
            RTP_LLM_LOG_WARNING(
                "copy cache failed, invalid mem_block_id or mem_block_size, mem_block_id=%d, mem_block_size=%ld",
                mem_block_id,
                mem_block_size);
            response.set_success(false);
            reportCopyMetrics(false, timer.done_us(), copy_direction);
            return false;
        }

        std::vector<LayerBlock> gpu_layer_blocks;
        gpu_layer_blocks.reserve(gpu_block.layer_blocks_size());
        for (const auto& lb : gpu_block.layer_blocks()) {
            gpu_layer_blocks.push_back(LayerBlock{lb.layer_id(), lb.block_id()});
        }

        if (!prepareCopyBuffers(
                gpu_layer_blocks, mem_block_id, mem_block_size, copy_direction, dst_buffers, src_buffers)) {
            RTP_LLM_LOG_WARNING(
                "copy cache failed, prepare copy buffers failed, mem_block_id=%d, mem_block_size=%zu, direction=%s",
                mem_block_id,
                mem_block_size,
                copy_direction == CopyDirection::H2D ? "H2D" : "D2H");
            response.set_success(false);
            reportCopyMetrics(false, timer.done_us(), copy_direction);
            return false;
        }
    }

    if (!dst_buffers.empty()) {
        device_->noBlockCopy(MultiCopyParams{dst_buffers, src_buffers});
    }
    response.set_success(true);

    reportCopyMetrics(true, timer.done_us(), copy_direction);
    return true;
}

bool KVCacheMemoryConnector::prepareCopyBuffers(const std::vector<LayerBlock>& gpu_layer_blocks,
                                                int                            mem_block_index,
                                                size_t                         mem_block_size,
                                                CopyDirection                  direction,
                                                std::vector<BufferPtr>&        dst,
                                                std::vector<BufferPtr>&        src) {
    auto block_pool = getBlockPool(mem_block_size);
    if (!block_pool) {
        RTP_LLM_LOG_WARNING(
            "prepare copy buffers failed, block pool is null, block_size=%zu, direction=%s, block pool: %s",
            mem_block_size,
            direction == CopyDirection::H2D ? "H2D" : "D2H",
            blockPoolDebugString().c_str());
        return false;
    }

    const auto mem_buffer = block_pool->convertIndexToBuffer(/*layer_id=*/0, mem_block_index);
    if (!mem_buffer.kv_addr) {
        RTP_LLM_LOG_WARNING("prepare copy buffers failed, mem buffer is null, block_idx=%d, direction=%s",
                            mem_block_index,
                            direction == CopyDirection::H2D ? "H2D" : "D2H");
        return false;
    }

    size_t offset = 0;
    for (const auto& lb : gpu_layer_blocks) {
        const int  layer_id      = lb.layer_id;
        const int  gpu_block_idx = lb.block_id;
        const auto layer_num     = cache_config_.layer_all_num;
        if (isNullBlockIdx(gpu_block_idx) || layer_id < 0 || layer_id >= layer_num) {
            RTP_LLM_LOG_WARNING(
                "prepare copy buffers failed, invalid gpu_block_idx or layer_id, gpu_block_idx=%d, layer_id=%d, layer_num=%zu",
                gpu_block_idx,
                layer_id,
                layer_num);
            return false;
        }

        const auto gpu_buffer = allocator_->convertIndexToBuffer(layer_id, gpu_block_idx);
        if (gpu_buffer.kv_addr) {
            const size_t kv_bytes  = gpu_buffer.kv_addr->sizeBytes();
            auto         mem_slice = mem_buffer.kv_addr->slice(offset, kv_bytes, false);
            if (direction == CopyDirection::H2D) {
                src.push_back(mem_slice);
                dst.push_back(gpu_buffer.kv_addr);
            } else {
                src.push_back(gpu_buffer.kv_addr);
                dst.push_back(mem_slice);
            }
            offset += kv_bytes;
        }
        if (gpu_buffer.kv_scale_addr) {
            const size_t scale_bytes = gpu_buffer.kv_scale_addr->sizeBytes();
            auto         mem_slice   = mem_buffer.kv_addr->slice(offset, scale_bytes, false);
            if (direction == CopyDirection::H2D) {
                src.push_back(mem_slice);
                dst.push_back(gpu_buffer.kv_scale_addr);
            } else {
                src.push_back(gpu_buffer.kv_scale_addr);
                dst.push_back(mem_slice);
            }
            offset += scale_bytes;
        }
    }
    return true;
}

bool KVCacheMemoryConnector::checkKVCacheResource(const std::shared_ptr<KVCacheResource>& resource) const {
    if (!resource) {
        RTP_LLM_LOG_WARNING("check kv cache resource failed, resource is null");
        return false;
    }

    const auto& cache_keys      = resource->cacheKeys();
    const auto& layer_block_ids = resource->layerBlocks();
    if (cache_keys.empty() || layer_block_ids.empty()) {
        RTP_LLM_LOG_WARNING(
            "check kv cache resource failed, cache keys or layer block ids is empty, cache keys size: %zu, layer block ids size: %zu",
            cache_keys.size(),
            layer_block_ids.size());
        return false;
    }

    const auto layer_num = cache_config_.layer_all_num;
    if (layer_block_ids.size() != layer_num) {
        RTP_LLM_LOG_WARNING(
            "check kv cache resource failed, layer block ids size is not equal to layer num, layer block ids size: %zu, layer num: %zu",
            layer_block_ids.size(),
            layer_num);
        return false;
    }
    for (const auto& blocks : layer_block_ids) {
        if (blocks->blocksNum() < cache_keys.size()) {
            RTP_LLM_LOG_WARNING(
                "check kv cache resource failed, layer block ids size is less than cache keys size, layer block ids size: %zu, cache keys size: %zu",
                blocks->blocksNum(),
                cache_keys.size());
            return false;
        }
    }
    return true;
}

bool KVCacheMemoryConnector::mallocBlocks(const std::shared_ptr<BlockPool>& block_pool,
                                          size_t                            need_blocks,
                                          std::vector<BlockIdxType>&        malloced_blocks) {
    if (!block_pool) {
        RTP_LLM_LOG_WARNING("malloc memory blocks failed, block pool is null, need blocks: %zu", need_blocks);
        return false;
    }
    if (need_blocks == 0) {
        RTP_LLM_LOG_WARNING("malloc memory blocks failed, need blocks cannot be 0");
        return false;
    }
    if (!ensureEnoughFreeBlocks(block_pool, need_blocks)) {
        RTP_LLM_LOG_WARNING(
            "malloc memory blocks failed, ensure enough free blocks failed, need blocks: %zu, free blocks: %zu",
            need_blocks,
            block_pool->freeBlocksNum());
        return false;
    }
    auto blocks = block_pool->malloc(need_blocks);
    if (blocks.size() != need_blocks) {
        RTP_LLM_LOG_WARNING("malloc memory blocks failed, malloc failed, need blocks: %zu, allocated blocks: %zu",
                            need_blocks,
                            blocks.size());
        freeBlocks(block_pool, blocks, /*cache_free=*/false);
        return false;
    }
    malloced_blocks = std::move(blocks);
    return true;
}

bool KVCacheMemoryConnector::freeBlocks(const std::shared_ptr<BlockPool>& block_pool,
                                        const std::vector<int>&           blocks,
                                        bool                              cache_free) {
    if (blocks.empty()) {
        return true;
    }
    if (!block_pool) {
        RTP_LLM_LOG_WARNING("free blocks failed, memory block pool is null");
        return false;
    }

    std::vector<int> need_free_blocks;
    need_free_blocks.reserve(blocks.size());
    for (const auto& block : blocks) {
        if (isNullBlockIdx(block)) {
            continue;
        }
        need_free_blocks.push_back(block);
    }
    if (need_free_blocks.empty()) {
        return true;
    }

    if (cache_free) {
        // cache中的block需要blockCacheFree
        block_pool->blockCacheFree(need_free_blocks);
    } else {
        // malloc的block需要requestFree
        block_pool->requestFree(need_free_blocks);
    }
    return true;
}

void KVCacheMemoryConnector::referenceBlocks(const std::shared_ptr<BlockPool>& block_pool,
                                             const std::vector<int>&           blocks) {
    if (blocks.empty()) {
        return;
    }
    if (!block_pool) {
        RTP_LLM_LOG_WARNING("reference blocks failed, block pool is null");
        return;
    }
    block_pool->blockCacheReference(blocks);
}

std::shared_ptr<BlockPool> KVCacheMemoryConnector::getBlockPool(size_t block_size) const {
    std::shared_lock<std::shared_mutex> lock(pool_mutex_);
    if (auto it = block_pools_.find(block_size); it != block_pools_.end()) {
        return it->second;
    }
    return nullptr;
}

std::shared_ptr<BlockPool> KVCacheMemoryConnector::createBlockPool(size_t block_size, size_t pool_size_mb) const {
    RTP_LLM_CHECK_WITH_INFO(pool_size_mb > 0, "pool size must be > 0");
    const int64_t block_num = pool_size_mb * 1024 * 1024 / static_cast<int64_t>(block_size);
    RTP_LLM_CHECK_WITH_INFO(
        block_num > 0, "pool_size_mb=%ld is too small for block_size=%zu (block_num=0)", pool_size_mb, block_size);
    RTP_LLM_LOG_INFO("create memory block pool, pool size: %ld MB, block num: %ld, block size: %zu, dtype: %d",
                     pool_size_mb,
                     block_num,
                     block_size,
                     cache_config_.dtype);
    const auto pool_config = BlockPoolConfigHelper::createLayerFirstConfig(
        /*layer_num=*/1, static_cast<uint32_t>(block_num), static_cast<uint32_t>(block_size), cache_config_.dtype);
    auto pool = std::make_shared<BlockPool>(pool_config, device_, AllocationType::HOST);
    RTP_LLM_CHECK_WITH_INFO(pool->init(), "memory block pool init failed, block size: %zu", block_size);
    return pool;
}

std::string KVCacheMemoryConnector::blockPoolDebugString() const {
    std::stringstream                   oss;
    std::shared_lock<std::shared_mutex> lock(pool_mutex_);
    for (const auto& [block_size, block_pool] : block_pools_) {
        oss << "{block size: " << block_size << ", block pool: [total blocks num: " << block_pool->totalBlocksNum()
            << ", free blocks num: " << block_pool->freeBlocksNum()
            << ", available blocks num: " << block_pool->availableBlocksNum() << "]}, ";
    }
    return oss.str();
}

void KVCacheMemoryConnector::putToCache(const MemoryBlockCache::CacheItem& item) {
    if (auto [success, popped_item_opt] = block_cache_->put(item); success) {
        auto block_pool = getBlockPool(item.block_size);
        if (!block_pool) {
            RTP_LLM_LOG_WARNING(
                "put to cache failed, block pool is null, cache key: %ld, block size: %zu, block pool: %s",
                item.cache_key,
                item.block_size,
                blockPoolDebugString().c_str());
            return;
        }
        RTP_LLM_LOG_DEBUG(
            "write cache, cache key: %ld, block index: %d, block size: %zu, free blocks num: %zu, available blocks num: %zu",
            item.cache_key,
            item.block_index,
            item.block_size,
            block_pool->freeBlocksNum(),
            block_pool->availableBlocksNum());
        referenceBlocks(block_pool, {item.block_index});
        if (popped_item_opt.has_value()) {
            const auto popped_item = popped_item_opt.value();
            auto       pool        = getBlockPool(popped_item.block_size);
            freeBlocks(pool, {popped_item.block_index}, /*cache_free=*/true);
        }
    }
}

bool KVCacheMemoryConnector::ensureEnoughFreeBlocks(const std::shared_ptr<BlockPool>& block_pool, size_t need_blocks) {
    const auto free_blocks = block_pool->freeBlocksNum();
    if (free_blocks >= need_blocks) {
        return true;
    }
    const auto need_evict_blocks = need_blocks - free_blocks;
    const auto evict_blocks      = block_cache_->pop(need_evict_blocks);
    if (!evict_blocks.empty()) {
        freeBlocks(block_pool, evict_blocks, /*cache_free=*/true);
    }
    return block_pool->freeBlocksNum() >= need_blocks;
}

bool KVCacheMemoryConnector::waitContextDoneAsync(const std::shared_ptr<MemoryConnectorAsyncContext>& context) {
    if (!wait_done_thread_pool_) {
        RTP_LLM_LOG_WARNING("push async context to thread pool failed, wait done thread pool is null");
        return false;
    }
    auto code = wait_done_thread_pool_->pushTask([context]() { context->waitDone(); });
    if (code != autil::ThreadPoolBase::ERROR_NONE) {
        RTP_LLM_LOG_WARNING("push async context to thread pool failed, push task failed, code: %d, size: %zu",
                            code,
                            wait_done_thread_pool_->getItemCount());
        return false;
    }
    return true;
}

bool KVCacheMemoryConnector::isThreadPoolFull() const {
    if (!wait_done_thread_pool_) {
        RTP_LLM_LOG_WARNING("wait done thread pool is null!");
        return true;
    }
    return wait_done_thread_pool_->isFull();
}

void KVCacheMemoryConnector::printCopyPlan(const std::vector<CopyInfoPerKey>& copy_infos) const {
    RTP_LLM_LOG_INFO("print copy plan, copy infos size: %zu", copy_infos.size());
    for (int i = 0; i < copy_infos.size(); ++i) {
        const auto&        copy_info = copy_infos.at(i);
        std::ostringstream oss;
        oss << "copy info " << i << ": cache key: " << copy_info.cache_key
            << ", mem block size: " << copy_info.mem_block_size << ", mem block index: " << copy_info.mem_block_index
            << ", gpu layer blocks: [";
        for (const auto& gpu_layer_block : copy_info.gpu_layer_blocks) {
            oss << "(layer " << gpu_layer_block.layer_id << ", block " << gpu_layer_block.block_id << "), ";
        }
        oss << "]";
        RTP_LLM_LOG_INFO(oss.str().c_str());
    }
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
            std::shared_ptr<BlockPool> block_pool;
            {
                std::shared_lock<std::shared_mutex> lock(pool_mutex_);
                if (!block_pools_.empty()) {
                    block_pool = block_pools_.begin()->second;
                }
            }
            if (!block_pool) {
                std::this_thread::sleep_for(std::chrono::seconds(1));
                continue;
            }

            const auto total_blocks     = block_pool->totalBlocksNum();
            const auto free_blocks      = block_pool->freeBlocksNum();
            const auto available_blocks = block_pool->availableBlocksNum();

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
