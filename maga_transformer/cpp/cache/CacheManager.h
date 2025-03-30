#pragma once

#include <cassert>
#include <mutex>
#include <set>
#include <vector>
#include <thread>

#include "maga_transformer/cpp/cache/BlockCache.h"
#include "maga_transformer/cpp/cache/BlockRefCounter.h"
#include "maga_transformer/cpp/cache/CacheConfig.h"
#include "maga_transformer/cpp/cache/KVCacheResource.h"
#include "src/fastertransformer/core/Buffer.h"
#include "src/fastertransformer/core/Types.h"
#include "src/fastertransformer/devices/DeviceBase.h"
#include "src/fastertransformer/devices/DeviceFactory.h"
#include "kmonitor/client/MetricsReporter.h"

namespace ft = fastertransformer;
namespace rtp_llm {

struct KVCacheInfo {
    size_t available_kv_cache;
    size_t total_kv_cache;
};


class CacheManager {
public:
    struct SeqPosition {
        int index;
        int offset;
    };

    struct KVCacheBuffer {
        ft::BufferPtr k_blocks;
        ft::BufferPtr v_blocks;
        ft::BufferPtr k_scale;
        ft::BufferPtr v_scale;
    };

    struct MatchInfo {
        size_t reuse_length = 0;
        std::vector<int> cache_blocks;
        std::vector<float> loss;
    };

    struct BlockAddrInfo {
        void* k_addr = nullptr;
        void* v_addr = nullptr;
        void* k_scale_addr = nullptr;
        void* v_scale_addr = nullptr;
    };

    struct SimpleMallocInfo {
        SimpleMallocInfo(int64_t request_id, uint32_t block_nums, bool verbose = false)
                        : request_id(request_id), block_nums(block_nums), verbose(verbose) {}

        int64_t request_id;
        uint32_t block_nums;
        bool verbose = false;
    };

    struct AdvancedMallocInfo {
        AdvancedMallocInfo(int64_t request_id,
                        const std::vector<int32_t>& token_ids,
                        const std::vector<int64_t>& cache_keys,
                        const std::vector<std::vector<int32_t>>& mm_bounds = {},
                        bool need_loss = false,
                        bool verbose = false)
                        : request_id(request_id), token_ids(token_ids),
                        cache_keys(cache_keys), mm_bounds(mm_bounds), need_loss(need_loss), verbose(verbose) {}

        int64_t request_id;
        const std::vector<int32_t>& token_ids;
        const std::vector<int64_t>& cache_keys;
        const std::vector<std::vector<int32_t>> mm_bounds = {};
        bool need_loss = false;
        bool verbose = false;
    };

    struct FreeInfo {
        FreeInfo(int64_t request_id,
                 const std::vector<int32_t>& token_ids,
                 const std::vector<int64_t>& cache_keys,
                 const std::vector<int32_t>& block_indices,
                 const std::vector<float> loss = {})
                 : request_id(request_id), token_ids(token_ids),
                   cache_keys(cache_keys), block_indices(block_indices), loss(loss) {}

        int64_t request_id;
        const std::vector<int32_t>& token_ids;
        const std::vector<int64_t>& cache_keys;
        const std::vector<int32_t>& block_indices;
        const std::vector<float> loss;
        bool is_resident = false;
    };

public:
    CacheManager(const CacheConfig& config, ft::DeviceBase* device,
                 const kmonitor::MetricsReporterPtr metrics_reporter = nullptr);
    ~CacheManager();

    const CacheConfig&     cacheConfig() const;
    size_t                 freeBlockNums() const;
    size_t                 availableBlockNums() const;
    KVCacheInfo            getKVCacheInfo() const;
    uint32_t               maxSeqLen() const;
    const KVCacheBuffer&   kvCacheBuffer() const;

    std::tuple<bool, KVCacheResource>   malloc(const SimpleMallocInfo& malloc_info);
    MatchInfo                           mallocWithCache(const AdvancedMallocInfo& malloc_info);
    void                                reserveBlocks(int nums);
    void                                incrRefCounter(const std::vector<int>& blocks_index);

    void free(const std::vector<KVCacheResource>& resource);
    void free(const std::vector<int>& indice);
    void freeWithCache(FreeInfo& free_info);
    void insertResidentCache(FreeInfo& free_info);

    virtual void setKVBlockValue(int block_index, int layer_id, ft::Buffer& k_buffer, ft::Buffer& v_buffer);
    virtual void setKVBlockValue(int block_index, ft::Buffer& k_buffer, ft::Buffer& v_buffer);
    std::tuple<ft::BufferPtr, ft::BufferPtr> getKVBlockValue(int block_index, int layer_id);
    std::tuple<ft::BufferPtr, ft::BufferPtr> getKVBlockValue(int block_index);
    void blockCopy(int src_block_index, int dest_block_index);

    BlockAddrInfo convertIndexToAddr(int block_index, int layer_id) const;

    void beamSearchKvUpdate(ft::BufferPtr src_block_offset,
                            ft::BufferPtr target_block_offset);

    void                                    regUserMr();

protected:
    const BlockCache&                       blockCache() const;
    size_t                                  cacheItemNum() const;
    uint32_t                                totalBlocks() const;
    void                                    initFreeBlock();
    ft::BufferPtr                           tryAllocateMaxBuffer();
    void                                    allocateAndSync();
    void                                    initKvCache();
    void                                    initKvCacheNormal();
    void                                    initKvCacheMla();
    MatchInfo                               matchImpl(const AdvancedMallocInfo& malloc_info);
    void                                    deregUserMr();
    std::tuple<bool, std::vector<int>>      mallocIndex(const SimpleMallocInfo& malloc_info);
    std::tuple<bool, std::vector<int>>      mallocImpl(const SimpleMallocInfo& malloc_info);
    void                                    maybeFreeBlockFromCache(int nums);

    void freeImpl(const std::vector<int>& indice);
    void insertIntoCache(FreeInfo& free_info);

    void copyKvCacheFromSeqIdxs(const std::vector<int>& block_indice_list,
                                const std::vector<int>& src_index, const std::vector<int>& target_index);
    SeqPosition getSeqPosition(const std::vector<int>& block_indice_list, int idx);
    void copyKvCacheFromSeqPosition(const SeqPosition& src_seq_position, const SeqPosition& dst_seq_position);

    const BlockRefCounter& blockRefCounter() const;
    void incrBlockRefCounter(const std::vector<int>& blocks);
    void incrQueryRefCounter(const std::vector<int>& blocks);
    void decrQueryRefCounter(const std::vector<int>& blocks);

    void reportMetricsLoop();

protected:
    CacheConfig     config_;
    int             seq_size_per_block_;
    std::set<int>   free_blocks_index_;
    BlockRefCounter block_ref_counter_;
    BlockRefCounter query_ref_counter_;
    int             available_blocks_;
    BlockCache      block_cache_;
    KVCacheBuffer   kv_cache_;
    ft::DeviceBase* device_;

    ft::BufferPtr   cache_aligned_buffer_;
    void*           cache_base_ptr_;

    bool            stop_ = false;
    std::thread     metrics_reporter_thread_;
    kmonitor::MetricsReporterPtr metrics_reporter_;

    bool kvcache_reg_mr_        = false;
    int64_t mr_cost_time_ms_    = 0;

    std::mutex mutex_;
};

typedef std::shared_ptr<CacheManager> CacheManagerPtr;

}  // namespace rtp_llm
