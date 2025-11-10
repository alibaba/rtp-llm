#include "rtp_llm/cpp/cache_new/MemoryConnector.h"

#include "rtp_llm/cpp/cache_new/BlockCacheV1.h"
#include "rtp_llm/cpp/cache_new/BlockPool.h"
#include "rtp_llm/cpp/cache_new/BlockPoolConfigHelper.h"
#include "rtp_llm/cpp/cache_new/KVCacheAllocator.h"
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
// ----------------------------- MemoryConnector ---------------------------------

MemoryConnector::MemoryConnector(const CacheConfig&                       cache_config,
                                 const std::shared_ptr<KVCacheAllocator>& allocator,
                                 rtp_llm::DeviceBase*                     device,
                                 const std::vector<std::string>&          tp_addrs):
    cache_config_(cache_config), allocator_(allocator), device_(device), tp_addrs_(tp_addrs) {}

MemoryConnector::~MemoryConnector() {
    RTP_LLM_LOG_INFO("MemoryConnector destructor");
    broadcast_manager_.reset();
    device_ = nullptr;
    allocator_.reset();
}

bool MemoryConnector::init() {
    const auto block_size = cache_config_.layer_type_params.at(0)->block_size();
    const auto pool_config =
        BlockPoolConfigHelper::createKVFirstConfig(cache_config_.layer_num, cache_config_.block_num, block_size);
    const auto block_pool = std::make_shared<BlockPool>(pool_config, device_, AllocationType::HOST);
    if (!block_pool->init()) {
        RTP_LLM_LOG_ERROR("failed to init block pool");
        return false;
    }

    const auto layer_layout = allocator_->layerCacheBase();
    for (int layer = 0; layer < static_cast<int>(layer_layout.layer_to_groups.size()); ++layer) {
        const int group_idx    = layer_layout.layer_to_groups.at(layer);
        layer_to_group_[layer] = group_idx;
        if (groups_.count(group_idx) == 0) {
            groups_[group_idx] = Group{GroupType::Invalid, {}, nullptr};
        }
        auto& group                              = groups_.at(group_idx);
        group.type                               = layer_layout.group_id_to_type.at(group_idx);
        group.global_layer_to_local_layer[layer] = group.global_layer_to_local_layer.size();
        if (!group.block_pool) {
            group.block_pool = block_pool;
        }
    }

    broadcast_manager_ = std::make_shared<TpBroadcastManager>(tp_addrs_);
    if (!broadcast_manager_->init()) {
        RTP_LLM_LOG_ERROR("failed to init broadcast manager");
        return false;
    }

    // TODO(LXQ): calculate group_block_stride_
    // group_block_stride_ = 1;

    return true;
}

std::shared_ptr<KVCacheConnector::AsyncContext>
MemoryConnector::asyncRead(const std::shared_ptr<KVCacheResourceV1>& resource, const std::shared_ptr<Meta>& meta) {
    if (!resource || resource->cache_keys.empty() || resource->group_block_ids.empty()) {
        return nullptr;
    }

    const auto   complete_cache_keys = resource->cache_keys;
    const size_t gpu_reuse_len       = resource->reuse_len;
    if (gpu_reuse_len >= complete_cache_keys.size()) {
        return std::make_shared<MemoryConnectorAsyncContext>(true, true);
    }

    std::vector<size_t> cache_keys_to_match(complete_cache_keys.begin() + gpu_reuse_len, complete_cache_keys.end());
    const size_t        memory_match_len = match(cache_keys_to_match);
    if (memory_match_len == 0) {
        return std::make_shared<MemoryConnectorAsyncContext>(true, true);
    }

    std::vector<GroupCopyInfo> group_copy_infos;
    const auto                 group_block_ids = resource->group_block_ids;

    for (const auto& [group_idx, group] : groups_) {
        const auto& block_cache = group.block_pool->blockCache();

        GroupCopyInfo group_copy_info;
        group_copy_info.group_id = group_idx;

        for (size_t cache_key_idx = 0; cache_key_idx < memory_match_len; ++cache_key_idx) {
            const int64_t cache_key  = cache_keys_to_match.at(cache_key_idx);
            const auto gpu_block_idx = group_block_ids.at(group_idx)->block_indices.at(gpu_reuse_len + cache_key_idx);
            if (isNullBlockIdx(gpu_block_idx)) {
                continue;
            }

            const auto match_result = block_cache->match(static_cast<CacheKeyType>(cache_key));
            if (isNullBlockIdx(match_result.matched_index)) {
                RTP_LLM_LOG_WARNING(
                    "memory connector get cache failed, match failed, group: %d, cache key: %zu", group_idx, cache_key);
                return nullptr;
            }
            const auto mem_block_idx = match_result.matched_index;

            group_copy_info.gpu_block_indices.push_back(gpu_block_idx);
            group_copy_info.memory_block_indices.push_back(mem_block_idx);
        }
        if (!group_copy_info.gpu_block_indices.empty()) {
            group_copy_infos.push_back(std::move(group_copy_info));
        }
    }

    if (group_copy_infos.empty()) {
        return std::make_shared<MemoryConnectorAsyncContext>(true, true);
    }

    auto broadcast_result = asyncCopyCache(group_copy_infos, CopyDirection::H2D);
    if (!broadcast_result) {
        RTP_LLM_LOG_WARNING("memory connector get failed, sync rpc call for all rank failed");
        return nullptr;
    }

    auto done_callback = [resource, gpu_reuse_len, memory_match_len](bool success) {
        if (success) {
            resource->reuse_len = gpu_reuse_len + memory_match_len;
        }
    };

    auto async_context = std::make_shared<MemoryConnectorAsyncContext>(broadcast_result, done_callback);
    return async_context;
}

std::shared_ptr<KVCacheConnector::AsyncContext>
MemoryConnector::asyncWrite(const std::shared_ptr<KVCacheResourceV1>& resource, const std::shared_ptr<Meta>& meta) {
    if (!resource || resource->cache_keys.empty() || resource->group_block_ids.empty()) {
        return nullptr;
    }

    const auto   complete_cache_keys = resource->cache_keys;
    const size_t match_len           = match(complete_cache_keys);
    if (match_len >= complete_cache_keys.size()) {
        return std::make_shared<MemoryConnectorAsyncContext>(true, true);
    }
    std::vector<size_t> cache_keys(complete_cache_keys.begin() + match_len, complete_cache_keys.end());

    std::vector<GroupCopyInfo> group_copy_infos;
    group_copy_infos.reserve(groups_.size());
    const auto group_block_ids = resource->group_block_ids;

    for (const auto& [group_idx, group] : groups_) {
        if (!group.block_pool) {
            RTP_LLM_LOG_WARNING("memory connector put failed, block pool is null, group=%d", group_idx);
            return nullptr;
        }
        const auto& block_pool = group.block_pool;

        std::vector<int> gpu_block_indices;
        for (size_t cache_key_idx = 0; cache_key_idx < cache_keys.size(); cache_key_idx++) {
            const auto gpu_block_idx = group_block_ids.at(group_idx)->block_indices.at(match_len + cache_key_idx);
            if (!isNullBlockIdx(gpu_block_idx)) {
                gpu_block_indices.push_back(gpu_block_idx);
            }
        }
        if (gpu_block_indices.empty()) {
            continue;
        }

        const int need_blocks = static_cast<int>(gpu_block_indices.size());
        if (!ensureEnoughFreeBlocks(block_pool, need_blocks)) {
            RTP_LLM_LOG_WARNING(
                "put to memory pool failed, ensure enough free blocks failed, need blocks: %d, free blocks: %zu",
                need_blocks,
                block_pool->freeBlockNums());
            return nullptr;
        }

        auto malloc_blocks = block_pool->malloc(need_blocks);
        if (malloc_blocks.size() != static_cast<size_t>(need_blocks)) {
            RTP_LLM_LOG_WARNING("put to memory pool failed, malloc failed, need blocks: %d, malloc blocks: %zu",
                                need_blocks,
                                malloc_blocks.size());
            if (!malloc_blocks.empty()) {
                block_pool->free(malloc_blocks);
            }
            return nullptr;
        }
        for (const auto& block : malloc_blocks) {
            if (isNullBlockIdx(block)) {
                RTP_LLM_LOG_WARNING(
                    "put to memory pool failed, malloc returned null block, group=%d block=%d", group_idx, block);
                return nullptr;
            }
        }

        GroupCopyInfo group_copy_info;
        group_copy_info.group_id = group_idx;
        group_copy_info.gpu_block_indices.assign(gpu_block_indices.begin(), gpu_block_indices.end());
        group_copy_info.memory_block_indices.assign(malloc_blocks.begin(), malloc_blocks.end());
        group_copy_infos.push_back(std::move(group_copy_info));
    }

    if (group_copy_infos.empty()) {
        return std::make_shared<MemoryConnectorAsyncContext>(true, true);
    }

    auto broadcast_result = asyncCopyCache(group_copy_infos, CopyDirection::D2H);
    if (!broadcast_result) {
        RTP_LLM_LOG_WARNING("memory connector put failed, sync rpc call for all rank failed");
        for (const auto& group_copy_info : group_copy_infos) {
            const auto& group                = groups_.at(group_copy_info.group_id);
            const auto& memory_block_indices = group_copy_info.memory_block_indices;
            group.block_pool->free(memory_block_indices);
        }
        return nullptr;
    }

    std::map<int, std::shared_ptr<BlockPool>> group_block_pools;
    for (const auto& group_copy_info : group_copy_infos) {
        const auto group_idx         = group_copy_info.group_id;
        group_block_pools[group_idx] = groups_.at(group_idx).block_pool;
    }

    auto done = [group_copy_infos, group_block_pools, cache_keys](bool success) {
        if (success) {
            for (const auto& group_copy_info : group_copy_infos) {
                const auto& block_pool           = group_block_pools.at(group_copy_info.group_id);
                const auto& memory_block_indices = group_copy_info.memory_block_indices;
                for (size_t i = 0; i < memory_block_indices.size(); ++i) {
                    const auto mem_block_idx = memory_block_indices.at(i);
                    const auto cache_key     = cache_keys.at(i);

                    BlockCacheV1::CacheItem item;
                    item.cache_key   = static_cast<CacheKeyType>(cache_key);
                    item.block_index = static_cast<BlockIdxType>(mem_block_idx);
                    item.is_resident = false;
                    block_pool->blockCache()->put(item);
                }
                block_pool->reference(memory_block_indices);
            }
        }
    };

    return std::make_shared<MemoryConnectorAsyncContext>(broadcast_result, done);
}

std::shared_ptr<TPBroadcastResult> MemoryConnector::asyncCopyCache(const std::vector<GroupCopyInfo>& group_copy_infos,
                                                                   CopyDirection                     direction) const {
    if (!broadcast_manager_) {
        RTP_LLM_LOG_WARNING("memory connector sync rpc call failed, broadcast manager is null");
        return nullptr;
    }

    MemoryBroadcastTpRequestPB mem_request;
    mem_request.set_direction(direction == CopyDirection::H2D ? MemoryBroadcastTpRequestPB::H2D :
                                                                MemoryBroadcastTpRequestPB::D2H);
    for (const auto& group_copy_info : group_copy_infos) {
        auto* group = mem_request.add_groups();
        if (group == nullptr) {
            RTP_LLM_LOG_WARNING("memory connector sync rpc call failed, add_groups returned nullptr");
            return nullptr;
        }
        group->set_group_id(group_copy_info.group_id);
        for (const auto& gpu_block_idx : group_copy_info.gpu_block_indices) {
            group->add_gpu_block_ids(gpu_block_idx);
        }
        for (const auto& mem_block_idx : group_copy_info.memory_block_indices) {
            group->add_memory_block_ids(mem_block_idx);
        }
    }

    const auto                        worker_num = broadcast_manager_->workerNum();
    std::vector<BroadcastTpRequestPB> requests;
    requests.reserve(worker_num);
    for (size_t i = 0; i < worker_num; ++i) {
        BroadcastTpRequestPB request;
        *request.mutable_mem_request() = mem_request;
        requests.emplace_back(std::move(request));
    }

    const int timeout_ms = 10000;
    return broadcast_manager_->broadcast(requests, timeout_ms);
}

void MemoryConnector::copyCache(const MemoryBroadcastTpRequestPB& request,
                                MemoryBroadcastTpResponsePB&      response) const {
    if (request.groups().empty()) {
        RTP_LLM_LOG_WARNING("copy cache failed, groups is empty");
        response.set_success(false);
        return;
    }

    std::vector<MemoryConnector::GroupCopyInfo> group_copy_infos;
    group_copy_infos.reserve(request.groups().size());
    for (const auto& group : request.groups()) {
        MemoryConnector::GroupCopyInfo group_copy_info;
        group_copy_info.group_id = group.group_id();
        group_copy_info.gpu_block_indices.assign(group.gpu_block_ids().begin(), group.gpu_block_ids().end());
        group_copy_info.memory_block_indices.assign(group.memory_block_ids().begin(), group.memory_block_ids().end());
        group_copy_infos.push_back(std::move(group_copy_info));
    }

    const auto direction = request.direction() == MemoryBroadcastTpRequestPB::H2D ?
                               MemoryConnector::CopyDirection::H2D :
                               MemoryConnector::CopyDirection::D2H;
    if (!copyCache(group_copy_infos, direction)) {
        RTP_LLM_LOG_WARNING("copy cache failed, copy cache for rank failed");
        response.set_success(false);
        return;
    }

    response.set_success(true);
}

bool MemoryConnector::copyCache(const std::vector<GroupCopyInfo>& group_copy_infos, CopyDirection direction) const {
    std::vector<BufferPtr> src_buffers;
    std::vector<BufferPtr> dst_buffers;

    for (const auto& group_copy_info : group_copy_infos) {
        const auto& gpu_block_indices    = group_copy_info.gpu_block_indices;
        const auto& memory_block_indices = group_copy_info.memory_block_indices;

        if (gpu_block_indices.size() != memory_block_indices.size()) {
            RTP_LLM_LOG_WARNING(
                "copy cache for rank failed, gpu block indices size (%zu) != memory block indices size (%zu)",
                gpu_block_indices.size(),
                memory_block_indices.size());
            return false;
        }
        if (gpu_block_indices.empty()) {
            continue;
        }

        const auto group_it = groups_.find(group_copy_info.group_id);
        if (group_it == groups_.end()) {
            RTP_LLM_LOG_WARNING("copy cache for rank failed, unknown group id: %d", group_copy_info.group_id);
            return false;
        }
        const auto& group = group_it->second;
        if (!group.block_pool) {
            RTP_LLM_LOG_WARNING("copy cache for rank failed, block pool is null, group=%d", group_copy_info.group_id);
            return false;
        }
        const auto& global_layer_to_local_layer = group.global_layer_to_local_layer;
        if (global_layer_to_local_layer.empty()) {
            RTP_LLM_LOG_WARNING("copy cache for rank failed, group has no layer mapping, group=%d",
                                group_copy_info.group_id);
            return false;
        }

        for (size_t i = 0; i < memory_block_indices.size(); ++i) {
            const int mem_block_idx = memory_block_indices.at(i);
            const int gpu_block_idx = gpu_block_indices.at(i);
            if (isNullBlockIdx(mem_block_idx) || isNullBlockIdx(gpu_block_idx)) {
                RTP_LLM_LOG_WARNING("copy cache for rank failed, invalid block idx, group=%d, mem=%d, gpu=%d",
                                    group_copy_info.group_id,
                                    mem_block_idx,
                                    gpu_block_idx);
                return false;
            }

            for (const auto& [global_layer, local_layer] : global_layer_to_local_layer) {
                BlockBufferInfo src_buffer_info;
                BlockBufferInfo dst_buffer_info;
                if (direction == CopyDirection::H2D) {
                    src_buffer_info = group.block_pool->convertIndexToBuffer(local_layer, mem_block_idx);
                    dst_buffer_info = allocator_->convertIndexToBuffer(global_layer, gpu_block_idx);
                } else {
                    src_buffer_info = allocator_->convertIndexToBuffer(global_layer, gpu_block_idx);
                    dst_buffer_info = group.block_pool->convertIndexToBuffer(local_layer, mem_block_idx);
                }

                if (!src_buffer_info.k_addr || !dst_buffer_info.k_addr) {
                    RTP_LLM_LOG_WARNING("copy cache for rank failed, k buffer is null, group=%d layer=%d",
                                        group_copy_info.group_id,
                                        global_layer);
                    return false;
                }
                src_buffers.push_back(src_buffer_info.k_addr);
                dst_buffers.push_back(dst_buffer_info.k_addr);

                if (!src_buffer_info.v_addr || !dst_buffer_info.v_addr) {
                    RTP_LLM_LOG_WARNING("copy cache for rank failed, v buffer is null, group=%d layer=%d",
                                        group_copy_info.group_id,
                                        global_layer);
                    return false;
                }
                src_buffers.push_back(src_buffer_info.v_addr);
                dst_buffers.push_back(dst_buffer_info.v_addr);
            }
        }
    }

    if (!dst_buffers.empty()) {
        copyBuffers(dst_buffers, src_buffers);
    }
    return true;
}

size_t MemoryConnector::match(const std::vector<size_t>& keys) const {
    size_t match_len = prefixMatch(keys);
    if (match_len == 0) {
        return 0;
    }

    std::vector<size_t> keys_to_hash_match(keys.begin(), keys.begin() + match_len);
    const auto&         hash_match_results = hashMatch(keys_to_hash_match);
    for (size_t i = match_len - 1; i >= 0; i--) {
        if (hash_match_results[i]) {
            match_len = std::min(i + group_block_stride_, match_len);
            break;
        }
    }
    return match_len;
}

size_t MemoryConnector::prefixMatch(const std::vector<size_t>& keys) const {
    size_t match_len = keys.size();
    for (const auto& [_, group] : groups_) {
        if (group.type == GroupType::Full) {
            std::vector<size_t> keys_to_match(keys.begin(), keys.begin() + match_len);
            const auto          prefix_match_len = prefixMatch(group.block_pool->blockCache(), keys_to_match);
            if (prefix_match_len < match_len) {
                match_len = prefix_match_len;
            }
            if (match_len == 0) {
                break;
            }
        }
    }
    return match_len;
}

size_t MemoryConnector::prefixMatch(const std::shared_ptr<BlockCacheV1>& block_cache,
                                    const std::vector<size_t>&           keys) const {
    for (size_t i = 0; i < keys.size(); i++) {
        if (!block_cache->contains(static_cast<CacheKeyType>(keys[i]))) {
            return i;
        }
    }
    return keys.size();
}

std::vector<bool> MemoryConnector::hashMatch(const std::vector<size_t>& keys) const {
    std::vector<bool> match_result(keys.size(), true);
    for (const auto& [_, group] : groups_) {
        if (group.type == GroupType::Linear) {
            const auto& hash_match_result = hashMatch(group.block_pool->blockCache(), keys);
            for (size_t i = 0; i < match_result.size(); i++) {
                match_result[i] = match_result[i] && hash_match_result[i];
            }
        }
    }
    return match_result;
}

std::vector<bool> MemoryConnector::hashMatch(const std::shared_ptr<BlockCacheV1>& block_cache,
                                             const std::vector<size_t>&           keys) const {
    std::vector<bool> match_result(keys.size(), false);
    for (size_t i = 0; i < keys.size(); i++) {
        match_result[i] = block_cache->contains(static_cast<CacheKeyType>(keys[i]));
    }
    return match_result;
}

void MemoryConnector::copyBuffers(const std::vector<BufferPtr>& dst, const std::vector<BufferPtr>& src) const {
    device_->noBlockCopy(MultiCopyParams{dst, src});
}

bool MemoryConnector::ensureEnoughFreeBlocks(const std::shared_ptr<BlockPool>& block_pool, int need_blocks) const {
    if (!block_pool) {
        RTP_LLM_LOG_WARNING("ensure enough free blocks failed, block pool is null, need blocks: %d", need_blocks);
        return false;
    }
    if (need_blocks <= 0) {
        RTP_LLM_LOG_WARNING(
            "ensure enough free blocks failed, need blocks is less than or equal to zero, need blocks: %d",
            need_blocks);
        return false;
    }

    const auto free_blocks = block_pool->freeBlockNums();
    if (free_blocks >= need_blocks) {
        return true;
    }

    const auto block_cache = block_pool->blockCache();
    if (!block_cache) {
        RTP_LLM_LOG_WARNING("ensure enough free blocks failed, block cache is null");
        return false;
    }

    const auto need_evict_blocks = need_blocks - free_blocks;
    const auto evict_blocks      = block_cache->pop(need_evict_blocks);
    if (!evict_blocks.empty()) {
        block_pool->free(evict_blocks);
    }

    return block_pool->freeBlockNums() >= need_blocks;
}

}  // namespace rtp_llm
