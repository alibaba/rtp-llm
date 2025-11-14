#pragma once

#include <cassert>
#include <mutex>
#include <set>
#include <vector>
#include <thread>

#include "rtp_llm/cpp/cache/CacheConfig.h"
#include "rtp_llm/cpp/cache/KVCacheResource.h"
#include "rtp_llm/cpp/cache/BlockRefCounter.h"
#include "rtp_llm/cpp/core/Buffer.h"
#include "rtp_llm/cpp/core/Types.h"
#include "rtp_llm/cpp/devices/DeviceBase.h"
#include "rtp_llm/cpp/devices/DeviceFactory.h"
#include "kmonitor/client/MetricsReporter.h"
#include "rtp_llm/cpp/cache/KvCacheInfo.h"

namespace rtp_llm {

struct BlockIdPair {
    int src;
    int dst;
};

// KVCacheAllocator 以KVCacheBlock为粒度分配显存or内存
class KVCacheAllocator {
public:
    struct KVCacheBuffer {
        rtp_llm::BufferPtr k_blocks;
        rtp_llm::BufferPtr v_blocks;
        rtp_llm::BufferPtr k_scale;
        rtp_llm::BufferPtr v_scale;
    };

    struct BlockAddrInfo {
        void* k_addr       = nullptr;
        void* v_addr       = nullptr;
        void* k_scale_addr = nullptr;
        void* v_scale_addr = nullptr;
    };

    struct SimpleMallocInfo {
        SimpleMallocInfo(int64_t request_id, uint32_t block_nums, bool verbose = false):
            request_id(request_id), block_nums(block_nums), verbose(verbose) {}

        int64_t  request_id;
        uint32_t block_nums;
        bool     verbose = false;
    };

public:
    KVCacheAllocator(const CacheConfig&   config,
                     rtp_llm::DeviceBase* device,
                     AllocationType       atype = AllocationType::DEVICE);
    ~KVCacheAllocator();

    bool init();

    size_t               totalBlocks() const;
    size_t               freeBlockNums() const;
    const KVCacheBuffer& kvCacheBuffer() const;

    std::tuple<bool, KVCacheResource> malloc(const SimpleMallocInfo& malloc_info);
    void                              free(const std::vector<KVCacheResource>& resource);
    void                              free(const std::vector<int>& indice);

    virtual bool setKVBlockValue(int block_index, int layer_id, rtp_llm::Buffer& k_buffer, rtp_llm::Buffer& v_buffer);
    virtual bool setKVBlockValue(int block_index, rtp_llm::Buffer& k_buffer, rtp_llm::Buffer& v_buffer);

    std::tuple<bool, rtp_llm::BufferPtr, rtp_llm::BufferPtr> getKVBlockValue(int block_index, int layer_id);
    std::tuple<bool, rtp_llm::BufferPtr, rtp_llm::BufferPtr> getKVBlockValue(int block_index);
    std::tuple<bool, rtp_llm::BufferPtr, rtp_llm::BufferPtr> getKVBlockValueRef(int block_index, int layer_id);

    void blockCopy(int src_block_index, int dest_block_index);
    void blockBatchCopy(const std::vector<BlockIdPair>& copy_mapping);
    void blockBatchCopy(const rtp_llm::Buffer& copy_mapping);
    void blockBatchCopy(const BlockIdPair* copy_mapping_begin, const BlockIdPair* copy_mapping_end);

    BlockAddrInfo convertIndexToAddr(int block_index, int layer_id) const;

    void    regUserMr(size_t model_id);
    int64_t getMrCostTimeMs() const;

    const CacheConfig& cacheConfig() const;

    void incrBlockRefCounter(const std::vector<int>& blocks);
    void decrBlockRefCounter(const std::vector<int>& blocks);

protected:
    void initFreeBlock();
    void initKvCache();
    void initKvCacheNormal();
    void initKvCacheMla();
    void initKVCacheScale();

    void deregUserMr();

    // for test
    const BlockRefCounter& blockRefCounter() const;

private:
    const CacheConfig&   config_;
    int                  seq_size_per_block_;
    std::set<int>        free_blocks_index_;
    KVCacheBuffer        kv_cache_;
    rtp_llm::DeviceBase* device_;
    AllocationType       atype_;

    rtp_llm::BufferPtr cache_aligned_buffer_;
    void*              cache_base_ptr_;

    // 被外部引用的block ref counter
    BlockRefCounter block_ref_counter_;

    bool                         stop_ = false;
    std::thread                  metrics_reporter_thread_;
    kmonitor::MetricsReporterPtr metrics_reporter_;

    bool    kvcache_reg_mr_  = false;
    int64_t mr_cost_time_ms_ = 0;

    mutable std::mutex mutex_;
};

typedef std::shared_ptr<KVCacheAllocator> KVCacheAllocatorPtr;

}  // namespace rtp_llm
