#pragma once

#include <memory>
#include <mutex>
#include <vector>

#include "rtp_llm/cpp/cache/CacheConfig.h"
#include "rtp_llm/cpp/cache/Types.h"

namespace rtp_llm {

class KVCacheManager;
class DeviceBase;
class BlockPool;

/// Encapsulates the staging-block lifecycle and GPU scatter kernel invocation
/// for CP (Context Parallel) round-robin KV cache transfer during PD separation.
///
/// During prefill with round-robin CP, each rank's KV data is interleaved at
/// token granularity.  On the decode side (no CP), blocks must be contiguous.
/// This helper borrows staging blocks from BlockPool for RDMA receive, then
/// runs a GPU scatter kernel to reassemble tokens into contiguous decode blocks.
class CPCacheScatterHelper {
public:
    struct StagingPlan {
        std::vector<int> staging_block_ids;
        int              vblock_count = 0;
        int              cp_size      = 1;

        struct LayerBlockInfo {
            std::vector<BlockInfo> infos;
        };
        // Per-layer, per staging block: layer_infos[layer_id].infos[v * cp_size + peer]
        std::vector<LayerBlockInfo> layer_infos;

        ~StagingPlan();

        StagingPlan()                              = default;
        StagingPlan(StagingPlan&&)                 = default;
        StagingPlan& operator=(StagingPlan&&)      = default;
        StagingPlan(const StagingPlan&)            = delete;
        StagingPlan& operator=(const StagingPlan&) = delete;

    private:
        friend class CPCacheScatterHelper;
        BlockPool* block_pool_ = nullptr;
    };

    CPCacheScatterHelper(KVCacheManager* cache_manager, DeviceBase* device);
    ~CPCacheScatterHelper();

    CPCacheScatterHelper(const CPCacheScatterHelper&)            = delete;
    CPCacheScatterHelper& operator=(const CPCacheScatterHelper&) = delete;

    /// Phase 1: Allocate staging blocks and resolve per-layer GPU addresses.
    /// Returns a StagingPlan whose layer_infos can be used to build
    /// RequestBlockBuffer entries for RDMA receive.
    /// Throws on allocation failure.
    std::unique_ptr<StagingPlan> prepareStagingPlan(int vblock_count, int cp_size, size_t layer_num);

    /// Phase 2: Run the paged scatter kernel to copy tokens from staging blocks
    /// into contiguous decode blocks, then release staging blocks.
    /// The plan is consumed (moved) and freed after scatter completes.
    void scatterAndRelease(std::unique_ptr<StagingPlan> plan,
                           const GroupBlockIds&         block_ids_by_group,
                           const CacheConfig&           cache_config,
                           size_t                       layer_num,
                           int                          total_tokens);

private:
    void* getOrCreateScatterStream();

    KVCacheManager* cache_manager_;
    DeviceBase*     device_;
    void*           scatter_stream_ = nullptr;
    std::once_flag  scatter_stream_init_;
};

}  // namespace rtp_llm
