#include <cassert>
#include <iostream>
#include <string>
#include <vector>
#include <set>

#include "rtp_llm/cpp/cache/MemoryBlockCache.h"
#include "rtp_llm/cpp/utils/LRUCache.h"
#include "rtp_llm/cpp/utils/StringUtil.h"
#include "rtp_llm/cpp/core/Buffer.h"
#include "rtp_llm/cpp/core/Types.h"
#include "autil/TimeUtility.h"
#include <chrono>
#include <future>

using namespace std;

namespace rtp_llm {

MemoryBlockCache::MemoryBlockCache(const CacheConfig&                  config,
                                   rtp_llm::DeviceBase*                device,
                                   KVCacheAllocator*                   gpu_kvcache_allocator,
                                   const ParallelismConfig&            parallelism_config,
                                   const KVCacheConfig&                kv_cache_config,
                                   const RuntimeConfig&                runtime_config,
                                   const kmonitor::MetricsReporterPtr& metrics_reporter):
    config_(config),
    device_(device),
    gpu_kvcache_allocator_(gpu_kvcache_allocator),
    parallelism_config_(parallelism_config),
    kv_cache_config_(kv_cache_config),
    runtime_config_(runtime_config),
    metrics_reporter_(metrics_reporter) {}

MemoryBlockCache::~MemoryBlockCache() {
    stop_ = true;
    if (metrics_reporter_thread_.joinable()) {
        metrics_reporter_thread_.join();
    }
}

bool MemoryBlockCache::init() {
    if (!gpu_kvcache_allocator_) {
        RTP_LLM_LOG_ERROR("gpu kvcache allocator is null");
        return false;
    }

    // 初始化内存KVCacheAllocator
    allocator_ = std::make_unique<KVCacheAllocator>(config_, device_, AllocationType::HOST);
    if (!allocator_->init()) {
        RTP_LLM_LOG_ERROR("memory block cache allocator init failed");
        return false;
    }

    // 初始化BlockLRUCache
    block_lru_cache_ = std::make_unique<BlockLRUCache>(config_.block_nums, config_.seq_size_per_block);

    // 初始化RPC连接池
    if (parallelism_config_.tp_size > 1) {
        rpc_pool_ = std::make_shared<RPCPool>();
    }

    if (metrics_reporter_) {
        metrics_reporter_thread_ = std::thread(&MemoryBlockCache::reportMetricsLoop, this);
    }
    RTP_LLM_LOG_INFO("memory block cache init success, memory size %d mb, sync timeout %d ms, config: %s",
                     kv_cache_config_.memory_block_cache_size_mb,
                     kv_cache_config_.memory_block_cache_sync_timeout_ms,
                     config_.debugString().c_str());
    return true;
}

size_t MemoryBlockCache::size() const {
    return block_lru_cache_->size();
}

size_t MemoryBlockCache::capacity() const {
    return allocator_->totalBlocks();
}

size_t MemoryBlockCache::availableBlockNum() const {
    return block_lru_cache_->availableBlockNum();
}

MemoryMatchResult MemoryBlockCache::match(const std::vector<int64_t>& cache_keys,
                                          const std::vector<int>&     gpu_block_ids,
                                          int64_t                     request_id) {
    autil::ScopedTime2 timer;

    MemoryMatchResult result = {0, {}, {}};
    if (cache_keys.size() != gpu_block_ids.size()) {
        RTP_LLM_LOG_DEBUG("MemoryBlockCache::match invalid parameters, request_id=%ld", request_id);
        recordMatchMetrics(false, result, timer.done_us(), static_cast<int64_t>(cache_keys.size()));
        return result;  // 返回0表示没有match到
    }

    std::lock_guard<std::mutex> lock(mutex_);

    // 使用BlockLRUCache进行匹配
    auto match_result = block_lru_cache_->match(cache_keys);

    // 没有匹配到任何block
    if (match_result.matched_len <= 0) {
        recordMatchMetrics(true, result, timer.done_us(), static_cast<int64_t>(cache_keys.size()));
        return result;
    }
    // 构造对应的GPU block IDs（从already_match_len开始，取copy_block_count个）
    std::vector<int> target_gpu_block_ids(gpu_block_ids.begin(), gpu_block_ids.begin() + match_result.matched_len);

    // 构造对应的内存block IDs（从already_match_len开始，取copy_block_count个）
    std::vector<int> source_memory_block_ids(match_result.block_ids.begin(),
                                             match_result.block_ids.begin() + match_result.matched_len);

    // 将内存block的内容拷贝到GPU block中
    auto ret = copyKVDataForAllRank(source_memory_block_ids, target_gpu_block_ids, CopyDirection::TO_GPU, request_id);

    if (!ret) {
        RTP_LLM_LOG_WARNING("MemoryBlockCache::match copyKVDataForAllRank failed, request_id=%ld", request_id);
        recordMatchMetrics(false, result, timer.done_us(), static_cast<int64_t>(cache_keys.size()));
        return result;
    }

    // 返回增量的匹配结果
    result = {source_memory_block_ids.size(), target_gpu_block_ids, match_result.losses};
    recordMatchMetrics(true, result, timer.done_us(), static_cast<int64_t>(cache_keys.size()));
    return result;
}

void MemoryBlockCache::put(const std::vector<int64_t>& cache_keys,
                           const std::vector<int>&     gpu_block_ids,
                           const std::vector<float>&   losses,
                           bool                        is_resident,
                           int64_t                     request_id) {
    autil::ScopedTime2 timer;

    if (cache_keys.empty() || gpu_block_ids.empty() || cache_keys.size() != gpu_block_ids.size()) {
        RTP_LLM_LOG_WARNING(
            "MemoryBlockCache::put invalid parameters, cache keys size %d, gpu block ids size %d, request_id=%ld",
            cache_keys.size(),
            gpu_block_ids.size(),
            request_id);
        recordPutMetrics(false, timer.done_us(), static_cast<int64_t>(cache_keys.size()), 0);
        return;
    }
    // 拷贝过程加锁, 防止拷贝没完成的block被并发利用
    std::lock_guard<std::mutex> lock(mutex_);

    // 先检查哪些block已经在LRU缓存中存在
    auto   match_result    = block_lru_cache_->match(cache_keys);
    size_t existing_blocks = match_result.matched_len;
    size_t new_blocks      = cache_keys.size() - existing_blocks;

    if (new_blocks == 0) {
        recordPutMetrics(true, timer.done_us(), static_cast<int64_t>(cache_keys.size()), 0);
        return;
    }

    std::vector<int> new_memory_block_ids = allocBlock(new_blocks);
    if (new_memory_block_ids.empty()) {
        RTP_LLM_LOG_WARNING("MemoryBlockCache::put allocBlock failed, request_id=%ld", request_id);
        recordPutMetrics(false, timer.done_us(), static_cast<int64_t>(cache_keys.size()), 0);
        return;
    }

    // 只拷贝新block的数据（从existing_blocks开始）
    std::vector<int> new_gpu_block_ids(gpu_block_ids.begin() + existing_blocks,
                                       gpu_block_ids.begin() + existing_blocks + new_memory_block_ids.size());

    if (!copyKVDataForAllRank(new_memory_block_ids, new_gpu_block_ids, CopyDirection::FROM_GPU, request_id)) {
        RTP_LLM_LOG_ERROR("MemoryBlockCache::put copyKVDataForAllRank failed, request_id=%ld", request_id);
        allocator_->free(new_memory_block_ids);
        recordPutMetrics(false, timer.done_us(), static_cast<int64_t>(cache_keys.size()), 0);
        return;
    }

    // 添加新block的ID
    match_result.block_ids.insert(
        match_result.block_ids.end(), new_memory_block_ids.begin(), new_memory_block_ids.end());
    std::vector<int64_t> new_cache_keys(cache_keys.begin(),
                                        cache_keys.begin() + existing_blocks + new_memory_block_ids.size());

    // 使用BlockLRUCache存储映射关系，传入完整的参数
    // BlockLRUCache内部会处理已存在的block，只存储新block，同时更新已存在block热度
    auto duplicate_block_ids = block_lru_cache_->put(new_cache_keys, match_result.block_ids, losses, is_resident);
    if (!duplicate_block_ids.empty()) {
        allocator_->free(duplicate_block_ids);
    }
    recordPutMetrics(true, timer.done_us(), static_cast<int64_t>(cache_keys.size()), new_memory_block_ids.size());
}

bool MemoryBlockCache::copyKVData(const std::vector<int>& memory_block_indices,
                                  const std::vector<int>& gpu_block_indices,
                                  CopyDirection           direction,
                                  int64_t                 request_id) {
    autil::ScopedTime2 timer;

    // 检查参数有效性
    if (memory_block_indices.size() != gpu_block_indices.size()) {
        RTP_LLM_LOG_ERROR("copyKVData: memory_block_indices size (%zu) != gpu_block_indices size (%zu), request_id=%ld",
                          memory_block_indices.size(),
                          gpu_block_indices.size(),
                          request_id);
        recordCopyMetrics(false, timer.done_us(), direction);
        return false;
    }

    if (memory_block_indices.empty()) {
        RTP_LLM_LOG_WARNING("copyKVData: empty block indices, request_id=%ld", request_id);
        recordCopyMetrics(false, timer.done_us(), direction);
        return true;  // 空操作视为成功
    }

    // 根据拷贝方向确定源和目标allocator
    KVCacheAllocator* source_allocator = nullptr;
    KVCacheAllocator* target_allocator = nullptr;

    if (direction == CopyDirection::FROM_GPU) {
        source_allocator = gpu_kvcache_allocator_;
        target_allocator = allocator_.get();
    } else {  // CopyDirection::TO_GPU
        source_allocator = allocator_.get();
        target_allocator = gpu_kvcache_allocator_;
    }

    std::vector<BufferPtr> src_buffers;
    std::vector<BufferPtr> dst_buffers;

    // 拷贝KV数据
    for (size_t i = 0; i < memory_block_indices.size(); ++i) {
        int source_block_id, target_block_id;

        if (direction == CopyDirection::FROM_GPU) {
            source_block_id = gpu_block_indices[i];
            target_block_id = memory_block_indices[i];
        } else {  // CopyDirection::TO_GPU
            source_block_id = memory_block_indices[i];
            target_block_id = gpu_block_indices[i];
        }

        for (uint32_t layer_id = 0; layer_id < config_.layer_num; layer_id++) {
            // 从源allocator获取KV数据
            auto [success, src_k_buffer, src_v_buffer] =
                source_allocator->getKVBlockValueRef(source_block_id, layer_id);
            if (!success || !src_k_buffer || !src_v_buffer) {
                RTP_LLM_LOG_WARNING("copyKVData: failed to get KV data from source block %d, layer %d, request_id=%ld",
                                    source_block_id,
                                    layer_id,
                                    request_id);
                recordCopyMetrics(false, timer.done_us(), direction);
                return false;
            }
            // 从目标allocator获取KV数据
            auto [success2, dst_k_buffer, dst_v_buffer] =
                target_allocator->getKVBlockValueRef(target_block_id, layer_id);
            if (!success2 || !dst_k_buffer || !dst_v_buffer) {
                RTP_LLM_LOG_WARNING("copyKVData: failed to get KV data from target block %d, layer %d, request_id=%ld",
                                    target_block_id,
                                    layer_id,
                                    request_id);
                recordCopyMetrics(false, timer.done_us(), direction);
                return false;
            }
            src_buffers.push_back(src_k_buffer);
            dst_buffers.push_back(dst_k_buffer);

            src_buffers.push_back(src_v_buffer);
            dst_buffers.push_back(dst_v_buffer);
        }
    }

    device_->noBlockCopy({dst_buffers, src_buffers});

    recordCopyMetrics(true, timer.done_us(), direction);
    if (timer.done_us() / 1000 > kv_cache_config_.memory_block_cache_sync_timeout_ms) {
        RTP_LLM_LOG_INFO(
            "memory block cache done, %s copy timeout, request: %ld, block size: %zu, copy buffer count: %zu, latency: %ld ms",
            direction == CopyDirection::FROM_GPU ? "from GPU" : "to GPU",
            request_id,
            memory_block_indices.size(),
            src_buffers.size(),
            timer.done_us() / 1000);
    }
    return true;
}

std::vector<int> MemoryBlockCache::allocBlock(int need_blocks) {
    // 检查allocator还有多少free block
    size_t free_blocks = allocator_->freeBlockNums();
    if (free_blocks < need_blocks) {
        // 内存不足，需要从LRU cache中pop一些block
        size_t need_evict_blocks = need_blocks - free_blocks;
        auto   popped_blocks     = block_lru_cache_->pop(need_evict_blocks);

        // 将evict的block放回allocator的free list中
        if (!popped_blocks.empty()) {
            allocator_->free(popped_blocks);
        }
    }

    // best effort alloc
    auto block_num = std::min((size_t)need_blocks, allocator_->freeBlockNums());
    if (block_num == 0) {
        return {};
    }

    // 现在尝试分配新block需要的内存
    auto [success, memory_resource] = allocator_->malloc(KVCacheAllocator::SimpleMallocInfo(-1, block_num, false));
    if (!success) {
        RTP_LLM_LOG_ERROR("failed to allocate memory blocks after eviction, need: %zu", need_blocks);
        return {};
    }

    return memory_resource.block_id;
}

bool MemoryBlockCache::copyKVDataForAllRank(const std::vector<int>& memory_block_indices,
                                            const std::vector<int>& gpu_block_indices,
                                            CopyDirection           direction,
                                            int64_t                 request_id) {
    if (parallelism_config_.tp_size <= 1) {
        // 单TP场景，直接调用单TP版本
        auto ret = copyKVData(memory_block_indices, gpu_block_indices, direction, request_id);
        return ret;
    }

    // 根据拷贝方向确定RPC操作类型
    MemoryBlockCacheOp op_type;
    if (direction == CopyDirection::FROM_GPU) {
        op_type = MemoryBlockCacheOp::MEMORY_CACHE_COPY_FROM_GPU;
    } else {  // CopyDirection::TO_GPU
        op_type = MemoryBlockCacheOp::MEMORY_CACHE_COPY_TO_GPU;
    }

    // 使用公共RPC同步方法
    bool success = syncRpcCallForAllRank(gpu_block_indices,
                                         memory_block_indices,
                                         op_type,
                                         kv_cache_config_.memory_block_cache_sync_timeout_ms,
                                         request_id);
    if (success) {
        return true;
    }

    const char* direction_str = (direction == CopyDirection::FROM_GPU) ? "from GPU" : "to GPU";
    RTP_LLM_LOG_WARNING("copy %s for all rank failed", direction_str);
    return false;
}

struct MemoryBlockCacheWorkerRpcContext {
    MemoryBlockCacheWorkerRpcContext() {
        client_context = std::make_shared<grpc::ClientContext>();
    }
    grpc::Status                         status;
    std::shared_ptr<RpcService::Stub>    stub;
    std::shared_ptr<grpc::ClientContext> client_context;
    std::string                          server_addr;
    MemoryBlockCacheRequestPB            request;
    MemoryBlockCacheResponsePB           response;
    grpc::CompletionQueue                completion_queue;
};

bool MemoryBlockCache::syncRpcCallForAllRank(const std::vector<int>& gpu_block_indices,
                                             const std::vector<int>& memory_block_indices,
                                             MemoryBlockCacheOp      op_type,
                                             int                     timeout_ms,
                                             int64_t                 request_id) {
    // 多TP场景，使用gRPC同步所有rank
    const auto& grpc_workers = runtime_config_.worker_grpc_addrs;
    if (grpc_workers.empty() || !rpc_pool_) {
        RTP_LLM_LOG_WARNING("sync RPC call for all rank failed, grpc workers empty or rpc pool/thread pool null");
        return false;
    }

    const int                                     worker_size = static_cast<int>(grpc_workers.size());
    std::vector<MemoryBlockCacheWorkerRpcContext> worker_rpc_contexts(worker_size);

    std::chrono::system_clock::time_point deadline      = std::chrono::system_clock::now() + std::chrono::milliseconds(timeout_ms);

    for (int rank = 0; rank < worker_size; ++rank) {
        const auto& worker_addr    = grpc_workers[rank];
        auto        connect_status = rpc_pool_->getConnection(worker_addr);
        if (!connect_status.ok()) {
            RTP_LLM_LOG_WARNING(
                "sync RPC call failed, get grpc connection failed for rank: %d, addr: %s", rank, worker_addr.c_str());
            return false;
        }
        auto& rpc_context       = worker_rpc_contexts[rank];
        rpc_context.stub        = connect_status.value().stub;
        rpc_context.server_addr = worker_addr;

        for (const auto gpu_block_id : gpu_block_indices) {
            rpc_context.request.add_gpu_block_ids(gpu_block_id);
        }

        for (const auto memory_block_id : memory_block_indices) {
            rpc_context.request.add_memory_block_ids(memory_block_id);
        }
        rpc_context.request.set_op(op_type);
        rpc_context.request.set_request_id(request_id);
        rpc_context.client_context->set_deadline(deadline);
    }

    // 发送请求集中在一起，防止出现部分rank请求已发送但是已经返回.
    for (int rank = 0; rank < worker_size; ++rank) {
        auto& rpc_context = worker_rpc_contexts[rank];
        auto  reader      = rpc_context.stub->AsyncMemoryBlockCache(
            rpc_context.client_context.get(), rpc_context.request, &(rpc_context.completion_queue));
        reader->Finish(&(rpc_context.response), &(rpc_context.status), reinterpret_cast<void*>(rank));
    }

    std::vector<int> success_ranks(worker_size, 0);
    int              finished_count      = 0;
    bool             all_request_success = true;
    while (true) {
        if (finished_count == worker_size) {
            break;
        }

        // GRPC部分RANK超时或是等待CQ失败直接退出进程， 防止出现错误数据.
        const int  once_timeout_ms = 1;
        const auto once_deadline   = std::chrono::system_clock::now() + std::chrono::milliseconds(once_timeout_ms);
        for (uint32_t i = 0; i < worker_rpc_contexts.size(); i++) {
            if (success_ranks[i] == 1) {
                continue;
            }
            auto& rpc_context = worker_rpc_contexts.at(i);

            void*      got_tag     = nullptr;
            bool       ok          = false;
            const auto next_status = rpc_context.completion_queue.AsyncNext(&got_tag, &ok, once_deadline);
            if (next_status == grpc::CompletionQueue::NextStatus::TIMEOUT) {
                continue;
            }
            if (!ok) {
                RTP_LLM_FAIL(
                    "request failed, grpc completion queue failed, status: %d, request: %ld", next_status, request_id);
            }
            ++finished_count;

            const int   rank    = reinterpret_cast<intptr_t>(got_tag);
            const auto& status  = rpc_context.status;
            success_ranks[rank] = 1;

            if (status.error_code() == grpc::StatusCode::DEADLINE_EXCEEDED) {
                RTP_LLM_FAIL("request failed, rank %d failed, error: %d(%s), addr: %s, request: %ld",
                             rank,
                             status.error_code(),
                             status.error_message().c_str(),
                             rpc_context.server_addr.c_str(),
                             request_id);
            }

            // 某个rank请求返回失败，可以继续等待其他rank返回.
            if (!status.ok() || !rpc_context.response.success()) {
                RTP_LLM_LOG_WARNING(
                    "request failed, rank %d failed, error: %d(%s), success: %d, addr: %s, request: %ld",
                    rank,
                    status.error_code(),
                    status.error_message().c_str(),
                    rpc_context.response.success(),
                    rpc_context.server_addr.c_str(),
                    request_id);
                all_request_success = false;
            }
        }
    }
    return all_request_success;
}

void MemoryBlockCache::recordMatchMetrics(bool                     success,
                                          const MemoryMatchResult& result,
                                          int64_t                  latency_us,
                                          int64_t                  input_block_count) {
    if (!metrics_reporter_) {
        return;
    }

    RtpLLMMemoryBlockCacheMatchMetricsCollector collector;
    collector.failed        = !success;
    collector.latency_us    = latency_us;
    collector.input_token   = input_block_count * config_.seq_size_per_block;
    collector.matched_token = static_cast<int64_t>(result.matched_len) * config_.seq_size_per_block;

    metrics_reporter_->report<RtpLLMMemoryBlockCacheMetrics, RtpLLMMemoryBlockCacheMatchMetricsCollector>(nullptr,
                                                                                                          &collector);
}

void MemoryBlockCache::recordPutMetrics(bool    success,
                                        int64_t latency_us,
                                        int64_t input_block_count,
                                        int64_t put_block_count) {
    if (!metrics_reporter_) {
        return;
    }

    RtpLLMMemoryBlockCachePutMetricsCollector collector;
    collector.failed      = !success;
    collector.latency_us  = latency_us;
    collector.input_token = input_block_count * config_.seq_size_per_block;
    collector.put_token   = put_block_count * config_.seq_size_per_block;

    metrics_reporter_->report<RtpLLMMemoryBlockCacheMetrics, RtpLLMMemoryBlockCachePutMetricsCollector>(nullptr,
                                                                                                        &collector);
}

void MemoryBlockCache::recordCopyMetrics(bool success, int64_t latency_us, CopyDirection direction) {
    if (!metrics_reporter_) {
        return;
    }

    RtpLLMMemoryBlockCacheCopyMetricsCollector collector;
    collector.failed     = !success;
    collector.latency_us = latency_us;
    collector.from_gpu   = direction == CopyDirection::FROM_GPU;

    metrics_reporter_->report<RtpLLMMemoryBlockCacheMetrics, RtpLLMMemoryBlockCacheCopyMetricsCollector>(nullptr,
                                                                                                         &collector);
}

void MemoryBlockCache::reportMetricsLoop() {
    while (!stop_) {
        if (metrics_reporter_) {
            RtpLLMMemoryBlockCacheStatusMetricsCollector collector;
            collector.total_block_num = allocator_->totalBlocks();  // block lru cache size == allocator total blocks
            collector.allocated_block_num = block_lru_cache_->size();
            collector.available_block_num = allocator_->freeBlockNums();
            metrics_reporter_->report<RtpLLMMemoryBlockCacheMetrics, RtpLLMMemoryBlockCacheStatusMetricsCollector>(
                nullptr, &collector);
        }
    }
}

}  // namespace rtp_llm
