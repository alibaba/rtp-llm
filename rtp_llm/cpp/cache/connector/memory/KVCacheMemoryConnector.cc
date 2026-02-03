#include "rtp_llm/cpp/cache/connector/memory/KVCacheMemoryConnector.h"

#include "rtp_llm/cpp/cache/BlockPool.h"
#include "rtp_llm/cpp/cache/BlockPoolConfigHelper.h"
#include "rtp_llm/cpp/cache/connector/memory/MemoryAsyncContext.h"
#include "rtp_llm/cpp/cache/connector/Meta.h"
#include "rtp_llm/cpp/cache/KVCacheAllocator.h"
#include "rtp_llm/cpp/devices/DeviceBase.h"
#include "rtp_llm/cpp/utils/Logger.h"

namespace rtp_llm {

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
    if (metrics_reporter_thread_) {
        metrics_reporter_thread_->join();
        metrics_reporter_thread_.reset();
    }
    if (wait_done_thread_pool_) {
        wait_done_thread_pool_->stop();
        wait_done_thread_pool_.reset();
    }
    broadcast_manager_.reset();
    block_pools_.clear();
    block_cache_.reset();
}

bool KVCacheMemoryConnector::init() {
    const auto memory_cache_sync_timeout_ms = kv_cache_config_.memory_cache_sync_timeout_ms;
    RTP_LLM_CHECK_WITH_INFO(memory_cache_sync_timeout_ms > 0,
                            "init failed, sync timeout is invalid, sync timeout: %ld ms",
                            memory_cache_sync_timeout_ms);

    initBlockPool();
    block_cache_ = std::make_shared<MemoryBlockCache>();

    broadcast_manager_ = std::make_shared<BroadcastManager>(tp_addrs_);
    RTP_LLM_CHECK_WITH_INFO(broadcast_manager_->init(), "init failed, broadcast manager init failed");

    wait_done_thread_pool_ = std::make_shared<autil::LockFreeThreadPool>(8, 1000, nullptr, "WaitDoneThreadPool");
    RTP_LLM_CHECK_WITH_INFO(wait_done_thread_pool_->start(), "init failed, wait done thread pool start failed");

    if (metrics_reporter_) {
        metrics_reporter_thread_ =
            std::make_shared<std::thread>([self = shared_from_this()]() { self->reportMetricsLoop(); });
    }
    return true;
}

void KVCacheMemoryConnector::initBlockPool() {
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
}

std::shared_ptr<AsyncMatchContext> KVCacheMemoryConnector::asyncMatch(const std::shared_ptr<KVCacheResource>& resource,
                                                                      const std::shared_ptr<Meta>&            meta) {
    RTP_LLM_CHECK_WITH_INFO(meta != nullptr, "async match failed, meta is null");
    RTP_LLM_CHECK_WITH_INFO(resource != nullptr, "async match failed, resource is null");
    if (!meta->enableMemoryCache()) {
        return nullptr;
    }

    const auto& cache_keys = resource->cacheKeys();
    // do not match last block, whether it is aligned or not, otherwise may cause core dump in computing ops.
    const auto cache_keys_size = cache_keys.empty() ? 0 : cache_keys.size() - 1;
    if (cache_keys_size == 0) {
        RTP_LLM_LOG_DEBUG("async match skip, cache keys is empty");
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

    size_t matched_num = 0;  // matched num must end at a big cache_key
    for (size_t i = 0; i < cache_keys_size; ++i) {
        const auto cache_key    = cache_keys.at(i);
        const auto match_result = block_cache_->match(static_cast<CacheKeyType>(cache_key));
        if (isNullBlockIdx(match_result.matched_index)) {
            break;  // only continuous prefix
        }
        if (match_result.is_big) {
            matched_num = i + 1;
        }
    }

    if (matched_num == 0) {
        RTP_LLM_LOG_DEBUG("not matched cache in memory, cache keys size: %zu", cache_keys_size);
        reportMatchMetrics(/*success=*/false, timer.done_us(), cache_keys_size, matched_num);
        return nullptr;
    }
    reportMatchMetrics(/*success=*/true, timer.done_us(), cache_keys_size, matched_num);
    return std::make_shared<MemoryAsyncMatchContext>(matched_num);
}

std::shared_ptr<AsyncContext> KVCacheMemoryConnector::asyncRead(const std::shared_ptr<KVCacheResource>&   resource,
                                                                const std::shared_ptr<Meta>&              meta,
                                                                const std::shared_ptr<AsyncMatchContext>& match_context,
                                                                int start_read_block_index,
                                                                int read_block_num) {
    RTP_LLM_CHECK_WITH_INFO(resource != nullptr, "async read failed, resource is null");
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

    auto copy_infos = buildCopyPlanForRead(cache_keys, layer_block_ids, start_read_block_index, read_block_num);
    if (copy_infos.empty()) {
        RTP_LLM_LOG_WARNING(
            "async read failed, build copy plan for read failed, cache keys size: %zu, start_read_block_index: %d, read_block_num: %d",
            cache_keys_size,
            start_read_block_index,
            read_block_num);
        reportReadMetrics(false, timer.done_us(), cache_keys_size, 0);
        return nullptr;
    }

    const auto total_block_num = cache_keys_size;
    auto       read_done = [resource, copy_infos, total_block_num, read_block_num, timer, self = shared_from_this()](
                         bool success) mutable {
        RTP_LLM_LOG_DEBUG("async read done, success: %d", success);
        if (success) {
            resource->setMemoryReuseBlockNum(read_block_num);
        }
        for (const auto& copy_info : copy_infos) {
            auto block_pool = self->getBlockPool(copy_info.mem_block_size);
            self->freeBlocks(block_pool, {copy_info.mem_block_index}, /*cache_free=*/true);
        }
        self->reportReadMetrics(success, timer.done_us(), total_block_num, read_block_num);
    };

    auto context = std::make_shared<MemoryAsyncContext>(read_done);
    if (!startCopyAsync(context, copy_infos, CopyDirection::H2D)) {
        RTP_LLM_LOG_WARNING("async read failed, start copy plan async failed");
        read_done(false);
        return nullptr;
    }
    return context;
}

std::vector<KVCacheMemoryConnector::CopyInfoPerKey> KVCacheMemoryConnector::buildCopyPlanForRead(
    const CacheKeysType& cache_keys, const LayerBlockIds& layer_block_ids, int start_index, int read_num) {
    std::vector<CopyInfoPerKey> copy_infos;
    const auto                  layer_num = cache_config_.layer_all_num;
    bool                        success   = true;

    for (int i = start_index; i < start_index + read_num; ++i) {
        const auto cache_key    = cache_keys.at(i);
        const auto match_result = block_cache_->match(static_cast<CacheKeyType>(cache_key));
        if (isNullBlockIdx(match_result.matched_index)) {
            success = false;
            break;
        }

        auto block_pool = getBlockPool(match_result.block_size);
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
            LayerBlock lb{static_cast<int>(layer), static_cast<BlockIdxType>(gpu_block_idx)};
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
    RTP_LLM_CHECK_WITH_INFO(meta != nullptr, "async write failed, meta is null");
    RTP_LLM_CHECK_WITH_INFO(resource != nullptr, "async write failed, resource is null");
    if (!meta->enableMemoryCache()) {
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

    auto copy_infos =
        buildCopyPlanForWrite(cache_keys, layer_block_ids, mem_matched_num, cache_keys_size - mem_matched_num);
    if (copy_infos.empty()) {
        RTP_LLM_LOG_WARNING("async write failed, build copy plan for write failed");
        reportWriteMetrics(false, timer.done_us(), static_cast<int64_t>(cache_keys_size), 0);
        return nullptr;
    }

    auto write_done =
        [copy_infos, resource_copy = resource, timer, total_block_num = cache_keys_size, self = shared_from_this()](
            bool success) mutable {
            RTP_LLM_LOG_DEBUG("async write done, success: %d", success);

            if (success) {
                for (const auto& copy_info : copy_infos) {
                    MemoryBlockCache::CacheItem item;
                    item.cache_key   = copy_info.cache_key;
                    item.block_index = static_cast<BlockIdxType>(copy_info.mem_block_index);
                    item.block_size  = copy_info.mem_block_size;
                    item.is_resident = false;
                    item.is_big      = copy_info.is_big;
                    self->putToCache(item);
                }
                // copy resource to decrease block ref count in destructor
                resource_copy.reset();
            }

            for (const auto& copy_info : copy_infos) {
                auto block_pool = self->getBlockPool(copy_info.mem_block_size);
                self->freeBlocks(block_pool, {copy_info.mem_block_index}, /*cache_free=*/false);
            }

            const int64_t write_block_num = success ? static_cast<int64_t>(copy_infos.size()) : 0;
            self->reportWriteMetrics(success, timer.done_us(), total_block_num, write_block_num);
        };

    auto context = std::make_shared<MemoryAsyncContext>(write_done);
    if (!startCopyAsync(context, copy_infos, CopyDirection::D2H)) {
        RTP_LLM_LOG_WARNING("async write failed, start copy plan async failed");
        write_done(false);
        return nullptr;
    }
    return context;
}

std::vector<KVCacheMemoryConnector::CopyInfoPerKey> KVCacheMemoryConnector::buildCopyPlanForWrite(
    const CacheKeysType& cache_keys, const LayerBlockIds& layer_block_ids, int start_index, int write_num) {
    const auto                  layer_num = cache_config_.layer_all_num;
    bool                        success   = true;
    std::vector<CopyInfoPerKey> copy_infos;

    for (int i = start_index; i < start_index + write_num; ++i) {
        const auto              cache_key   = cache_keys.at(i);
        size_t                  total_bytes = 0;
        std::vector<LayerBlock> gpu_layer_blocks;
        for (size_t layer = 0; layer < layer_num; ++layer) {
            const int gpu_block_idx = layer_block_ids.at(layer)->blocks().at(i);
            if (isNullBlockIdx(gpu_block_idx)) {
                continue;
            }
            gpu_layer_blocks.push_back(LayerBlock{static_cast<int>(layer), static_cast<BlockIdxType>(gpu_block_idx)});
            const auto buffers = allocator_->convertIndexToBuffer(static_cast<int>(layer), gpu_block_idx);
            for (const auto& buffer : buffers) {
                if (buffer.addr && buffer.size_bytes > 0) {
                    total_bytes += buffer.size_bytes;
                }
            }
        }
        if (gpu_layer_blocks.empty() || total_bytes == 0) {
            RTP_LLM_LOG_WARNING(
                "build copy plan for write failed, invalid gpu_layer_blocks or total_bytes, cache key: %zu, gpu blocks: %zu, total bytes: %zu",
                cache_key,
                gpu_layer_blocks.size(),
                total_bytes);
            success = false;
            break;
        }

        std::vector<BlockIdxType> mem_blocks;
        auto                      block_pool = getBlockPool(total_bytes);
        if (!mallocBlocks(block_pool, 1, mem_blocks)) {
            break;
        }

        CopyInfoPerKey copy_info;
        copy_info.cache_key        = cache_key;
        copy_info.mem_block_index  = mem_blocks.front();
        copy_info.mem_block_size   = total_bytes;
        copy_info.is_big           = gpu_layer_blocks.size() == layer_num;  // means no null block idx
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

bool KVCacheMemoryConnector::startCopyAsync(const std::shared_ptr<MemoryAsyncContext>& context,
                                            const std::vector<CopyInfoPerKey>&         copy_infos,
                                            CopyDirection                              direction) {
    auto code = wait_done_thread_pool_->pushTask([self = shared_from_this(), context, copy_infos, direction]() mutable {
        auto send_result = self->sendCopyPlan(copy_infos, direction);
        context->setBroadcastResult(send_result);
        context->waitDone();
    });
    if (code != autil::ThreadPoolBase::ERROR_NONE) {
        RTP_LLM_LOG_WARNING("start copy plan async failed, push send+wait task failed, code=%d", code);
        return false;
    }
    return true;
}

std::shared_ptr<BroadcastResult<FunctionRequestPB, FunctionResponsePB>>
KVCacheMemoryConnector::sendCopyPlan(const std::vector<CopyInfoPerKey>& copy_infos, CopyDirection direction) const {
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
        return stub->AsyncExecuteFunction(context.get(), request, completion_queue);
    };
    return broadcast_manager_->broadcast<FunctionRequestPB, FunctionResponsePB>(
        requests, kv_cache_config_.memory_cache_sync_timeout_ms, rpc_call);
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

bool KVCacheMemoryConnector::copyCache(const MemoryOperationRequestPB& request, MemoryOperationResponsePB& response) {
    RTP_LLM_CHECK(request.gpu_blocks_size() == request.mem_block_ids_size());
    RTP_LLM_CHECK(request.gpu_blocks_size() == request.mem_block_sizes_size());

    autil::ScopedTime2 timer;
    const auto         copy_direction =
        (request.copy_direction() == MemoryOperationRequestPB::H2D) ? CopyDirection::H2D : CopyDirection::D2H;

    std::vector<BufferPtr> dst_buffers;
    std::vector<BufferPtr> src_buffers;
    for (int i = 0; i < request.gpu_blocks_size(); ++i) {
        const auto& gpu_block      = request.gpu_blocks(i);
        const auto  mem_block_id   = static_cast<BlockIdxType>(request.mem_block_ids(i));
        const auto  mem_block_size = request.mem_block_sizes(i);

        std::vector<LayerBlock> gpu_layer_blocks;
        gpu_layer_blocks.reserve(gpu_block.layer_blocks_size());
        for (const auto& lb : gpu_block.layer_blocks()) {
            gpu_layer_blocks.push_back(LayerBlock{lb.layer_id(), static_cast<BlockIdxType>(lb.block_id())});
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
                                                BlockIdxType                   mem_block_index,
                                                size_t                         mem_block_size,
                                                CopyDirection                  direction,
                                                std::vector<BufferPtr>&        dst,
                                                std::vector<BufferPtr>&        src) {
    auto block_pool = getBlockPool(mem_block_size);
    RTP_LLM_CHECK_WITH_INFO(block_pool != nullptr, "block pool is null, mem_block_size=%zu", mem_block_size);

    const auto mem_buffers = block_pool->convertIndexToBuffer(/*layer_id=*/0, mem_block_index);
    if (mem_buffers.empty()) {
        RTP_LLM_LOG_WARNING("prepare copy buffers failed, mem buffers are empty, block_idx=%d, direction=%s",
                            mem_block_index,
                            direction == CopyDirection::H2D ? "H2D" : "D2H");
        return false;
    }
    // memory has only one buffer
    const auto& mem_buffer = mem_buffers[0];
    if (!mem_buffer.addr || mem_buffer.size_bytes == 0) {
        RTP_LLM_LOG_WARNING("prepare copy buffers failed, mem buffer is invalid, block_idx=%d, size=%zu, direction=%s",
                            mem_block_index,
                            mem_buffer.size_bytes,
                            direction == CopyDirection::H2D ? "H2D" : "D2H");
        return false;
    }

    size_t byte_off = 0;
    for (const auto& lb : gpu_layer_blocks) {
        const int  layer_id      = lb.layer_id;
        const auto gpu_block_idx = lb.block_id;
        const auto gpu_buffers   = allocator_->convertIndexToBuffer(layer_id, gpu_block_idx);
        for (const auto& gpu_buffer : gpu_buffers) {
            if (!appendCopyBytesToBuffers(mem_buffer, gpu_buffer, byte_off, direction, dst, src)) {
                return false;
            }
            byte_off += gpu_buffer.size_bytes;
        }
    }
    return true;
}

bool KVCacheMemoryConnector::appendCopyBytesToBuffers(const BlockInfo&        mem_block,
                                                      const BlockInfo&        gpu_block,
                                                      size_t                  byte_off,
                                                      CopyDirection           direction,
                                                      std::vector<BufferPtr>& dst,
                                                      std::vector<BufferPtr>& src) {
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

    auto mem_slice = std::make_shared<Buffer>(mem_block.is_cuda ? MemoryType::MEMORY_GPU : MemoryType::MEMORY_CPU,
                                              rtp_llm::DataType::TYPE_INT8,
                                              std::vector<size_t>{gpu_block.size_bytes},
                                              static_cast<void*>(static_cast<char*>(mem_block.addr) + byte_off));
    auto gpu_slice = std::make_shared<Buffer>(gpu_block.is_cuda ? MemoryType::MEMORY_GPU : MemoryType::MEMORY_CPU,
                                              rtp_llm::DataType::TYPE_INT8,
                                              std::vector<size_t>{gpu_block.size_bytes},
                                              gpu_block.addr);
    if (direction == CopyDirection::H2D) {
        src.push_back(mem_slice);
        dst.push_back(gpu_slice);
    } else {
        src.push_back(gpu_slice);
        dst.push_back(mem_slice);
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
                                        const std::vector<BlockIdxType>&  blocks,
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
        need_free_blocks.push_back(static_cast<int>(block));
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
                                             const std::vector<BlockIdxType>&  blocks) {
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
    if (block_pools_.empty()) {
        return nullptr;
    }
    // Forward-compat: choose the smallest configured pool whose block size >= requested block_size.
    auto it = block_pools_.lower_bound(block_size);
    return (it == block_pools_.end()) ? nullptr : it->second;
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
    const auto pool_config = BlockPoolConfigHelper::createLayerFirstConfig(
        /*layer_num=*/1, static_cast<uint32_t>(block_num), static_cast<uint32_t>(block_size), rtp_llm::TYPE_INT8);
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
        RTP_LLM_LOG_DEBUG("write cache, cache key: %ld, block index: %d, block size: %zu",
                          item.cache_key,
                          item.block_index,
                          item.block_size);
        auto block_pool = getBlockPool(item.block_size);
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
