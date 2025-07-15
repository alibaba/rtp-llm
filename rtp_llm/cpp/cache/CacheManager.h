#pragma once

#include <cassert>
#include <mutex>
#include <set>
#include <vector>
#include <thread>

#include "rtp_llm/cpp/cache/BlockCache.h"
#include "rtp_llm/cpp/cache/BlockRefCounter.h"
#include "rtp_llm/cpp/cache/CacheConfig.h"
#include "rtp_llm/cpp/cache/KVCacheResource.h"
#include "rtp_llm/cpp/core/Buffer.h"
#include "rtp_llm/cpp/core/Types.h"
#include "rtp_llm/cpp/devices/DeviceBase.h"
#include "rtp_llm/cpp/devices/DeviceFactory.h"
#include "kmonitor/client/MetricsReporter.h"
#include "rtp_llm/cpp/dataclass/KvCacheInfo.h"

namespace rtp_llm {

class DistKvCache;

class CacheManager {
public:
    struct SeqPosition {
        int index;
        int offset;
    };

    struct KVCacheBuffer {
        rtp_llm::BufferPtr k_blocks;
        rtp_llm::BufferPtr v_blocks;
        rtp_llm::BufferPtr k_scale;
        rtp_llm::BufferPtr v_scale;
    };

    struct MatchInfo {
        size_t             reuse_length = 0;
        std::vector<int>   cache_blocks;
        std::vector<float> loss;
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

    struct AdvancedMallocInfo {
        AdvancedMallocInfo(int64_t                                  request_id,
                           const std::vector<int32_t>&              token_ids,
                           const std::vector<int64_t>&              cache_keys,
                           const std::vector<std::vector<int32_t>>& mm_bounds = {},
                           bool                                     need_loss = false,
                           bool                                     verbose   = false):
            request_id(request_id),
            token_ids(token_ids),
            cache_keys(cache_keys),
            mm_bounds(mm_bounds),
            need_loss(need_loss),
            verbose(verbose) {}

        int64_t                                 request_id;
        const std::vector<int32_t>&             token_ids;
        const std::vector<int64_t>&             cache_keys;
        const std::vector<std::vector<int32_t>> mm_bounds = {};
        bool                                    need_loss = false;
        bool                                    verbose   = false;
    };

    struct FreeInfo {
        FreeInfo(int64_t                     request_id,
                 const std::vector<int32_t>& token_ids,
                 const std::vector<int64_t>& cache_keys,
                 const std::vector<int32_t>& block_indices,
                 const std::vector<float>    loss = {}):
            request_id(request_id),
            token_ids(token_ids),
            cache_keys(cache_keys),
            block_indices(block_indices),
            loss(loss) {}

        int64_t                     request_id;
        const std::vector<int32_t>& token_ids;
        const std::vector<int64_t>& cache_keys;
        const std::vector<int32_t>& block_indices;
        const std::vector<float>    loss;
        bool                        is_resident = false;
    };

public:
    CacheManager(const CacheConfig&                 config,
                 rtp_llm::DeviceBase*               device,
                 bool                               warmup           = false,
                 const kmonitor::MetricsReporterPtr metrics_reporter = nullptr,
                 const GptInitParameter&            params           = GptInitParameter{});
    ~CacheManager();

    const CacheConfig&   cacheConfig() const;
    size_t               freeBlockNums() const;
    size_t               availableBlockNums() const;
    KVCacheInfo          getKVCacheInfo(int64_t latest_version) const;
    uint32_t             maxSeqLen() const;
    const KVCacheBuffer& kvCacheBuffer() const;

    std::tuple<bool, KVCacheResource> malloc(const SimpleMallocInfo& malloc_info);
    MatchInfo                         mallocWithCache(const AdvancedMallocInfo& malloc_info);
    void                              reserveBlocks(int nums);
    void                              incrRefCounter(const std::vector<int>& blocks_index);

    void free(const std::vector<KVCacheResource>& resource);
    void free(const std::vector<int>& indice);
    void freeWithCache(FreeInfo& free_info);
    void insertResidentCache(FreeInfo& free_info);

    virtual void setKVBlockValue(int block_index, int layer_id, rtp_llm::Buffer& k_buffer, rtp_llm::Buffer& v_buffer);
    virtual void setKVBlockValue(int block_index, rtp_llm::Buffer& k_buffer, rtp_llm::Buffer& v_buffer);
    std::tuple<rtp_llm::BufferPtr, rtp_llm::BufferPtr> getKVBlockValue(int block_index, int layer_id);
    std::tuple<rtp_llm::BufferPtr, rtp_llm::BufferPtr> getKVBlockValue(int block_index);
    void                                               blockCopy(int src_block_index, int dest_block_index);

    BlockAddrInfo convertIndexToAddr(int block_index, int layer_id) const;

    void beamSearchKvUpdate(rtp_llm::BufferPtr src_block_offset, rtp_llm::BufferPtr target_block_offset);

    void                                    regUserMr(size_t model_id);

    bool getCacheForRank(const std::vector<int64_t>& cache_keys,
                         const std::vector<int32_t>& block_indices,
                         int64_t                     request_id,
                         const std::map<std::string, std::string>& extra_metas) const;
    bool putCacheForRank(const std::vector<int64_t>& cache_keys,
                         const std::vector<int32_t>& block_indices,
                         int64_t                     request_id,
                         const std::map<std::string, std::string>& extra_metas) const;

protected:
    const BlockCache&                  blockCache() const;
    size_t                             cacheItemNum() const;
    uint32_t                           totalBlocks() const;
    void                               initFreeBlock();
    rtp_llm::BufferPtr                 tryAllocateMaxBuffer();
    void                               allocateAndSync();
    void                               initFakeKVCache();
    void                               initKvCache();
    void                               initKvCacheNormal();
    void                               initKvCacheMla();
    void                               initKVCacheScale();
    size_t                             getKBlockSize() const;
    size_t                             getVBlockSize() const;
    MatchInfo                          matchImpl(const AdvancedMallocInfo& malloc_info);
    void                               deregUserMr();
    std::tuple<bool, std::vector<int>> mallocIndex(const SimpleMallocInfo& malloc_info);
    std::tuple<bool, std::vector<int>> mallocImpl(const SimpleMallocInfo& malloc_info);
    void                               maybeFreeBlockFromCache(int nums);

    void freeImpl(const std::vector<int>& indice);
    void insertIntoCache(FreeInfo& free_info);

    void        copyKvCacheFromSeqIdxs(const std::vector<int>& block_indice_list,
                                       const std::vector<int>& src_index,
                                       const std::vector<int>& target_index);
    SeqPosition getSeqPosition(const std::vector<int>& block_indice_list, int idx);
    void        copyKvCacheFromSeqPosition(const SeqPosition& src_seq_position, const SeqPosition& dst_seq_position);

    const BlockRefCounter& blockRefCounter() const;
    void                   incrBlockRefCounter(const std::vector<int>& blocks);
    void                   incrQueryRefCounter(const std::vector<int>& blocks);
    void                   decrQueryRefCounter(const std::vector<int>& blocks);

    void reportMetricsLoop();

private:
    bool initDistKvCache();
    BlockCache::MatchResult
         matchInDistKvCache(const std::vector<int64_t>& cache_keys, int32_t seq_cache_key_num, int64_t request_id);
    bool putCacheForAllRank(const std::vector<int64_t>& cache_keys,
                            const std::vector<int32_t>& block_indices,
                            int64_t                     request_id) const;

protected:
    CacheConfig          config_;
    int                  seq_size_per_block_;
    std::set<int>        free_blocks_index_;
    BlockRefCounter      block_ref_counter_;
    BlockRefCounter      query_ref_counter_;
    int                  available_blocks_;
    BlockCache           block_cache_;
    KVCacheBuffer        kv_cache_;
    rtp_llm::DeviceBase* device_;

    rtp_llm::BufferPtr cache_aligned_buffer_;
    void*              cache_base_ptr_;

    bool                         stop_ = false;
    std::thread                  metrics_reporter_thread_;
    kmonitor::MetricsReporterPtr metrics_reporter_;

    bool    kvcache_reg_mr_  = false;
    int64_t mr_cost_time_ms_ = 0;

    std::mutex mutex_;

    const GptInitParameter       params_;
    bool                         enable_dist_kvcache_{true};
    bool                         enable_3fs_{true};
    std::unique_ptr<DistKvCache> dist_kvcache_;
};

typedef std::shared_ptr<CacheManager> CacheManagerPtr;

}  // namespace rtp_llm
