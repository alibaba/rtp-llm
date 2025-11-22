#pragma once

#include <cassert>
#include <iostream>
#include <string>
#include <vector>
#include <mutex>
#include <unordered_map>
#include <atomic>
#include <thread>
#include <chrono>

#include "rtp_llm/cpp/utils/LRUCache.h"
#include "rtp_llm/cpp/utils/AssertUtils.h"
#include "rtp_llm/cpp/cache/KVCacheAllocator.h"
#include "rtp_llm/cpp/cache/BlockLRUCache.h"
#include "rtp_llm/cpp/cache/CacheConfig.h"
#include "rtp_llm/cpp/devices/DeviceBase.h"
#include "rtp_llm/cpp/model_rpc/RPCPool.h"
#include "rtp_llm/cpp/core/Types.h"
#include "rtp_llm/cpp/metrics/RtpLLMMetrics.h"
#include "rtp_llm/cpp/config/ConfigModules.h"
#include "autil/LockFreeThreadPool.h"

namespace rtp_llm {

struct MemoryMatchResult {
    size_t             matched_len = 0;
    std::vector<int>   block_indices;
    std::vector<float> losses;  // 添加losses字段
};

class MemoryBlockCache {
public:
    MemoryBlockCache(const CacheConfig&                  config,
                     rtp_llm::DeviceBase*                device,
                     KVCacheAllocator*                   gpu_kvcache_allocator,
                     const ParallelismConfig&            parallelism_config,
                     const KVCacheConfig&                kv_cache_config,
                     const RuntimeConfig&                runtime_config,
                     const kmonitor::MetricsReporterPtr& metrics_reporter = nullptr);
    ~MemoryBlockCache();

    // 初始化缓存
    bool init();

    // 缓存大小
    size_t size() const;

    // 缓存容量
    size_t capacity() const;

    // 可用容量
    size_t availableBlockNum() const;

    // 匹配缓存项，只需要match没命中的cache key和对应的blockid
    MemoryMatchResult
    match(const std::vector<int64_t>& cache_keys, const std::vector<int>& gpu_block_ids, int64_t request_id);

    // 将数据放入缓存，分配内存并拷贝数据, 这时需要放全部的cache_keys和对应的block_ids
    void put(const std::vector<int64_t>& cache_keys,
             const std::vector<int>&     gpu_block_ids,
             const std::vector<float>&   losses,
             bool                        is_resident,
             int64_t                     request_id);

    // 拷贝方向枚举
    enum class CopyDirection {
        FROM_GPU,  // 从GPU拷贝到内存
        TO_GPU     // 从内存拷贝到GPU
    };

    // 拷贝KV数据（单TP）
    bool copyKVData(const std::vector<int>& memory_block_indices,
                    const std::vector<int>& gpu_block_indices,
                    CopyDirection           direction,
                    int64_t                 request_id);

private:
    // 拷贝KV数据（多TP同步）
    bool copyKVDataForAllRank(const std::vector<int>& memory_block_indices,
                              const std::vector<int>& gpu_block_indices,
                              CopyDirection           direction,
                              int64_t                 request_id);

    // 公共的RPC同步方法
    bool syncRpcCallForAllRank(const std::vector<int>& gpu_block_indices,
                               const std::vector<int>& memory_block_indices,
                               MemoryBlockCacheOp      op_type,
                               int                     timeout_ms,
                               int64_t                 request_id);

    std::vector<int> allocBlock(int need_blocks);

    // 指标收集方法
    void
    recordMatchMetrics(bool success, const MemoryMatchResult& result, int64_t latency_us, int64_t input_block_count);
    void recordPutMetrics(bool success, int64_t latency_us, int64_t input_block_count, int64_t put_block_count);
    void recordCopyMetrics(bool success, int64_t latency_us, CopyDirection direction);
    void reportMetricsLoop();

private:
    std::unique_ptr<BlockLRUCache> block_lru_cache_;  // 使用新的BlockLRUCache类
    mutable std::mutex             mutex_;

    // KVCacheAllocator用于管理内存分配
    CacheConfig                       config_;
    rtp_llm::DeviceBase*              device_;
    KVCacheAllocator*                 gpu_kvcache_allocator_;
    std::unique_ptr<KVCacheAllocator> allocator_;

    // 多TP同步相关
    std::shared_ptr<RPCPool> rpc_pool_;
    ParallelismConfig        parallelism_config_;
    KVCacheConfig            kv_cache_config_;
    RuntimeConfig            runtime_config_;

    // 指标相关
    bool                                       stop_ = false;
    std::shared_ptr<kmonitor::MetricsReporter> metrics_reporter_;
    std::thread                                metrics_reporter_thread_;
};

}  // namespace rtp_llm
