#include "rtp_llm/cpp/cache_new/KVCacheMemoryConnector.h"

#include "rtp_llm/cpp/cache_new/BlockPool.h"
#include "rtp_llm/cpp/cache_new/BlockPoolConfigHelper.h"
#include "rtp_llm/cpp/cache_new/KVCacheAllocator.h"
#include "rtp_llm/cpp/cache_new/MemoryBlockCache.h"
#include "rtp_llm/cpp/cache_new/TpBroadcastManager.h"
#include "rtp_llm/cpp/devices/DeviceBase.h"
#include "rtp_llm/cpp/utils/Logger.h"

namespace rtp_llm {

// ----------------------------- MemoryConnectorAsyncContext ---------------------------------

bool MemoryConnectorAsyncContext::success() const {
    return success_;
}

void MemoryConnectorAsyncContext::cancel() {
    // TODO(LXQ): need cancel tp broadcast
    return;
}

void MemoryConnectorAsyncContext::waitDone() {
    if (already_done_) {
        return;
    }
    if (broadcast_result_) {
        broadcast_result_->waitDone();
        already_done_ = true;

        success_ = broadcast_result_->success();
        if (!success_) {
            return;
        }

        success_ = allResponseSuccess();
    }
    if (done_callback_) {
        done_callback_(success_);
    }
}

bool MemoryConnectorAsyncContext::allResponseSuccess() const {
    const auto responses = broadcast_result_->responses();
    for (const auto& response : responses) {
        if (!response.has_mem_response() || !response.mem_response().success()) {
            return false;
        }
    }
    return true;
}

// ----------------------------- KVCacheMemoryConnector ---------------------------------

KVCacheMemoryConnector::KVCacheMemoryConnector(const CacheConfig&                       cache_config,
                                               const std::shared_ptr<KVCacheAllocator>& allocator,
                                               rtp_llm::DeviceBase*                     device,
                                               const std::vector<std::string>&          tp_addrs):
    cache_config_(cache_config), allocator_(allocator), device_(device), tp_addrs_(tp_addrs) {}

KVCacheMemoryConnector::~KVCacheMemoryConnector() {
    RTP_LLM_LOG_INFO("KVCacheMemoryConnector destructor");
    tp_broadcast_manager_.reset();
    block_pools_.clear();
    block_cache_.reset();
}

bool KVCacheMemoryConnector::init() {
    // 延迟创建不同block_size的BlockPool, 这里只初始化全局block_cache_
    block_pools_.clear();
    block_cache_ = std::make_shared<MemoryBlockCache>();

    tp_broadcast_manager_ = std::make_shared<TpBroadcastManager>(tp_addrs_);
    if (!tp_broadcast_manager_->init()) {
        RTP_LLM_LOG_WARNING("init failed, tp broadcast manager init failed");
        return false;
    }
    return true;
}

std::shared_ptr<KVCacheConnector::AsyncContext>
KVCacheMemoryConnector::asyncRead(const std::shared_ptr<KVCacheResourceV1>& resource,
                                  const std::shared_ptr<Meta>&              meta) {
    if (!resource || resource->cache_keys.empty() || resource->layer_block_ids.empty()) {
        return nullptr;
    }

    const auto&  cache_keys    = resource->cache_keys;
    const size_t gpu_reuse_len = resource->reuse_len;
    if (gpu_reuse_len >= cache_keys.size()) {
        return std::make_shared<MemoryConnectorAsyncContext>(true, true);
    }

    auto copy_infos = buildCopyPlanForRead(cache_keys, resource->layer_block_ids, gpu_reuse_len);
    if (copy_infos.empty()) {
        RTP_LLM_LOG_WARNING("async read failed, build copy plan for read failed, gpu_reuse_len: %zu, cache_keys: %zu",
                            gpu_reuse_len,
                            cache_keys.size());
        return nullptr;
    }

    auto send_result = sendCopyPlan(copy_infos, CopyDirection::H2D);
    if (!send_result) {
        RTP_LLM_LOG_WARNING("async read failed, send copy plan to tp failed");
        return nullptr;
    }

    const auto mem_match_len = copy_infos.size();
    auto       done_cb       = [resource, gpu_reuse_len, mem_match_len](bool success) {
        if (success) {
            resource->reuse_len = gpu_reuse_len + mem_match_len;
        }
    };
    return std::make_shared<MemoryConnectorAsyncContext>(send_result, done_cb);
}

std::vector<KVCacheMemoryConnector::CopyInfoPerKey> KVCacheMemoryConnector::buildCopyPlanForRead(
    const std::vector<size_t>& cache_keys, const LayerBlockIds& layer_block_ids, size_t gpu_reuse_len) const {
    std::vector<CopyInfoPerKey> copy_infos;
    const size_t                layer_num = layer_block_ids.size();

    for (size_t i = gpu_reuse_len; i < cache_keys.size(); ++i) {
        const auto cache_key    = cache_keys.at(i);
        const auto match_result = block_cache_->match(static_cast<CacheKeyType>(cache_key));
        if (isNullBlockIdx(match_result.matched_index)) {
            break;  // 只处理连续前缀
        }

        CopyInfoPerKey copy_info;
        copy_info.cache_key       = cache_key;
        copy_info.mem_block_index = match_result.matched_index;
        copy_info.mem_block_size  = match_result.block_size;
        copy_info.gpu_layer_blocks.reserve(layer_num);
        for (size_t layer = 0; layer < layer_num; ++layer) {
            const int gpu_block_idx = layer_block_ids.at(layer)->block_indices.at(i);
            if (!isNullBlockIdx(gpu_block_idx)) {
                LayerBlock lb{static_cast<int>(layer), gpu_block_idx};
                copy_info.gpu_layer_blocks.push_back(lb);
            }
        }
        copy_infos.emplace_back(std::move(copy_info));
    }
    return copy_infos;
}

std::shared_ptr<KVCacheConnector::AsyncContext>
KVCacheMemoryConnector::asyncWrite(const std::shared_ptr<KVCacheResourceV1>& resource,
                                   const std::shared_ptr<Meta>&              meta) {
    if (!resource || resource->cache_keys.empty() || resource->layer_block_ids.empty()) {
        return nullptr;
    }

    const auto& cache_keys = resource->cache_keys;
    // 计算内存中已存在的前缀长度
    size_t match_len = 0;
    for (; match_len < cache_keys.size(); ++match_len) {
        if (!block_cache_->contains(static_cast<CacheKeyType>(cache_keys[match_len]))) {
            break;
        }
    }
    if (match_len >= cache_keys.size()) {
        return std::make_shared<MemoryConnectorAsyncContext>(true, true);
    }

    auto copy_infos = buildCopyPlanForWrite(cache_keys, resource->layer_block_ids, match_len);
    if (copy_infos.empty()) {
        RTP_LLM_LOG_WARNING("async write failed, build copy plan for write failed");
        return nullptr;
    }

    auto send_result = sendCopyPlan(copy_infos, CopyDirection::D2H);
    if (!send_result) {
        RTP_LLM_LOG_WARNING("async write failed, send copy plan to tp failed");
        for (const auto& copy_info : copy_infos) {
            auto block_pool = getOrCreateMemoryBlockPool(copy_info.mem_block_size);
            freeMemoryBlocks(block_pool, {copy_info.mem_block_index});
        }
        return nullptr;
    }

    auto done_cb = [copy_infos, self = shared_from_this()](bool success) {
        if (!success) {
            for (const auto& copy_info : copy_infos) {
                auto block_pool = self->getOrCreateMemoryBlockPool(copy_info.mem_block_size);
                self->freeMemoryBlocks(block_pool, {copy_info.mem_block_index});
            }
            return;
        }
        for (const auto& copy_info : copy_infos) {
            if (self->block_cache_->contains(copy_info.cache_key)) {
                auto block_pool = self->getOrCreateMemoryBlockPool(copy_info.mem_block_size);
                self->freeMemoryBlocks(block_pool, {copy_info.mem_block_index});
                continue;
            }
            MemoryBlockCache::CacheItem item;
            item.cache_key   = copy_info.cache_key;
            item.block_index = static_cast<BlockIdxType>(copy_info.mem_block_index);
            item.block_size  = copy_info.mem_block_size;
            item.is_resident = false;
            self->block_cache_->put(item);
        }
    };
    return std::make_shared<MemoryConnectorAsyncContext>(send_result, done_cb);
}

std::vector<KVCacheMemoryConnector::CopyInfoPerKey> KVCacheMemoryConnector::buildCopyPlanForWrite(
    const std::vector<size_t>& cache_keys, const LayerBlockIds& layer_block_ids, size_t match_len) {
    const size_t                layer_num = layer_block_ids.size();
    bool                        success   = true;
    std::vector<CopyInfoPerKey> copy_infos;

    for (size_t i = match_len; i < cache_keys.size(); ++i) {
        size_t                  total_bytes = 0;
        std::vector<LayerBlock> gpu_layer_blocks;
        for (size_t layer = 0; layer < layer_num; ++layer) {
            const int gpu_block_idx = layer_block_ids.at(layer)->block_indices.at(i);
            if (isNullBlockIdx(gpu_block_idx)) {
                continue;
            }
            gpu_layer_blocks.push_back(LayerBlock{static_cast<int>(layer), gpu_block_idx});
            const auto buffers = allocator_->convertIndexToBuffer(static_cast<int>(layer), gpu_block_idx);
            if (buffers.k_addr) {
                total_bytes += buffers.k_addr->sizeBytes();
            }
            if (buffers.v_addr) {
                total_bytes += buffers.v_addr->sizeBytes();
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

        const auto&               block_pool = getOrCreateMemoryBlockPool(total_bytes, true);
        std::vector<BlockIdxType> mem_blocks;
        if (!mallocMemoryBlocks(block_pool, 1, mem_blocks)) {
            RTP_LLM_LOG_WARNING(
                "build copy plan for write failed, malloc memory blocks failed, maybe no enough free blocks, free blocks: %zu",
                block_pool->freeBlockNums());
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
            auto block_pool = getOrCreateMemoryBlockPool(copy_info.mem_block_size);
            freeMemoryBlocks(block_pool, {copy_info.mem_block_index});
        }
        return {};
    }
    return copy_infos;
}

std::shared_ptr<TPBroadcastResult> KVCacheMemoryConnector::sendCopyPlan(const std::vector<CopyInfoPerKey>& copy_infos,
                                                                        CopyDirection direction) const {
    if (!tp_broadcast_manager_ || tp_broadcast_manager_->workerNum() == 0) {
        RTP_LLM_LOG_WARNING("send copy plan failed, tp broadcast manager is null or no workers");
        return nullptr;
    }

    MemoryBroadcastTpRequestPB mem_req;
    mem_req.set_copy_direction(direction == CopyDirection::H2D ? MemoryBroadcastTpRequestPB::H2D :
                                                                 MemoryBroadcastTpRequestPB::D2H);
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

    std::vector<BroadcastTpRequestPB> requests;
    requests.reserve(tp_broadcast_manager_->workerNum());
    for (size_t i = 0; i < tp_broadcast_manager_->workerNum(); ++i) {
        BroadcastTpRequestPB req;
        req.mutable_mem_request()->CopyFrom(mem_req);
        requests.emplace_back(std::move(req));
    }

    return tp_broadcast_manager_->broadcast(requests, /*timeout_ms*/ 10000);
}

void KVCacheMemoryConnector::copyCache(const MemoryBroadcastTpRequestPB& request,
                                       MemoryBroadcastTpResponsePB&      response) {
    if (request.gpu_blocks_size() != request.mem_block_ids_size()
        || request.gpu_blocks_size() != request.mem_block_sizes_size()) {
        RTP_LLM_LOG_WARNING(
            "copy cache failed, count not match, gpu blocks: %zu, mem blocks: %zu, mem block sizes: %zu",
            request.gpu_blocks_size(),
            request.mem_block_ids_size(),
            request.mem_block_sizes_size());
        response.set_success(false);
        return;
    }

    const auto copy_direction =
        (request.copy_direction() == MemoryBroadcastTpRequestPB::H2D) ? CopyDirection::H2D : CopyDirection::D2H;

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
            return;
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
            return;
        }
    }

    if (!dst_buffers.empty()) {
        device_->noBlockCopy(MultiCopyParams{dst_buffers, src_buffers});
    }
    response.set_success(true);
}

bool KVCacheMemoryConnector::prepareCopyBuffers(const std::vector<LayerBlock>& gpu_layer_blocks,
                                                int                            mem_block_index,
                                                size_t                         mem_block_size,
                                                CopyDirection                  direction,
                                                std::vector<BufferPtr>&        dst,
                                                std::vector<BufferPtr>&        src) {
    auto mem_pool = getOrCreateMemoryBlockPool(mem_block_size, direction == CopyDirection::D2H);
    if (!mem_pool) {
        RTP_LLM_LOG_WARNING("prepare copy buffers failed, create/get mem pool failed, mem_block_size=%zu, direction=%s",
                            mem_block_size,
                            direction == CopyDirection::H2D ? "H2D" : "D2H");
        return false;
    }

    const auto mem_buffer = mem_pool->convertIndexToBuffer(0, mem_block_index);
    if (!mem_buffer.k_addr) {
        RTP_LLM_LOG_WARNING("prepare copy buffers failed, mem buffer is null, block_idx=%d, direction=%s",
                            mem_block_index,
                            direction == CopyDirection::H2D ? "H2D" : "D2H");
        return false;
    }

    size_t offset = 0;
    for (const auto& lb : gpu_layer_blocks) {
        const int layer_id      = lb.layer_id;
        const int gpu_block_idx = lb.block_id;
        if (isNullBlockIdx(gpu_block_idx) || layer_id < 0 || layer_id >= cache_config_.layer_num) {
            RTP_LLM_LOG_WARNING(
                "prepare copy buffers failed, invalid gpu_block_idx or layer_id, gpu_block_idx=%d, layer_id=%d, layer_num=%d",
                gpu_block_idx,
                layer_id,
                cache_config_.layer_num);
            return false;
        }

        const auto gpu_buffer = allocator_->convertIndexToBuffer(layer_id, gpu_block_idx);
        if (gpu_buffer.k_addr) {
            const size_t k_bytes   = gpu_buffer.k_addr->sizeBytes();
            auto         mem_slice = mem_buffer.k_addr->slice(offset, k_bytes, false);
            if (direction == CopyDirection::H2D) {
                src.push_back(mem_slice);
                dst.push_back(gpu_buffer.k_addr);
            } else {
                src.push_back(gpu_buffer.k_addr);
                dst.push_back(mem_slice);
            }
            offset += k_bytes;
        }
        if (gpu_buffer.v_addr) {
            const size_t v_bytes   = gpu_buffer.v_addr->sizeBytes();
            auto         mem_slice = mem_buffer.k_addr->slice(offset, v_bytes, false);
            if (direction == CopyDirection::H2D) {
                src.push_back(mem_slice);
                dst.push_back(gpu_buffer.v_addr);
            } else {
                src.push_back(gpu_buffer.v_addr);
                dst.push_back(mem_slice);
            }
            offset += v_bytes;
        }
    }
    return true;
}

bool KVCacheMemoryConnector::mallocMemoryBlocks(const std::shared_ptr<BlockPool>& block_pool,
                                                size_t                            need_blocks,
                                                std::vector<BlockIdxType>&        malloced_blocks) const {
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
            block_pool->freeBlockNums());
        return false;
    }
    auto blocks = block_pool->malloc(need_blocks);
    if (blocks.size() != need_blocks) {
        RTP_LLM_LOG_WARNING("malloc memory blocks failed, malloc failed, need blocks: %zu, allocated blocks: %zu",
                            need_blocks,
                            blocks.size());
        block_pool->free(blocks);
        return false;
    }
    malloced_blocks = std::move(blocks);
    return true;
}

bool KVCacheMemoryConnector::freeMemoryBlocks(const std::shared_ptr<BlockPool>& block_pool,
                                              const std::vector<int>&           blocks) {
    if (blocks.empty()) {
        return true;
    }
    if (!block_pool) {
        RTP_LLM_LOG_WARNING("free memory blocks failed, memory block pool is null");
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

    block_pool->free(need_free_blocks);
    return true;
}

std::shared_ptr<BlockPool> KVCacheMemoryConnector::getOrCreateMemoryBlockPool(size_t block_size, bool create) {
    auto it = block_pools_.find(block_size);
    if (it != block_pools_.end()) {
        return it->second;
    }
    if (!create) {
        return nullptr;
    }

    const auto pool_config = BlockPoolConfigHelper::createLayerFirstConfig(
        /*layer_num=*/1, cache_config_.block_num, static_cast<uint32_t>(block_size));
    auto pool = std::make_shared<BlockPool>(pool_config, device_, AllocationType::HOST);
    if (!pool->init()) {
        RTP_LLM_LOG_WARNING("create memory block pool failed, block_size=%zu", block_size);
        return nullptr;
    }
    block_pools_[block_size] = pool;
    return pool;
}

bool KVCacheMemoryConnector::ensureEnoughFreeBlocks(const std::shared_ptr<BlockPool>& block_pool,
                                                    size_t                            need_blocks) const {
    const auto free_blocks = block_pool->freeBlockNums();
    if (free_blocks >= need_blocks) {
        return true;
    }
    const auto need_evict_blocks = need_blocks - free_blocks;
    const auto evict_blocks      = block_cache_->pop(need_evict_blocks);
    if (!evict_blocks.empty()) {
        block_pool->free(evict_blocks);
    }
    return block_pool->freeBlockNums() >= need_blocks;
}

// (removed legacy multi-key plan builder)

}  // namespace rtp_llm
