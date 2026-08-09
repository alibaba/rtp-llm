#pragma once

#include <atomic>
#include <map>
#include <memory>
#include <string>
#include <vector>

#include "rtp_llm/cpp/cache/block_tree_cache/benchmark/BenchmarkFixture.h"
#include "rtp_llm/cpp/cache/block_tree_cache/benchmark/BenchmarkJsonWriter.h"
#include "rtp_llm/cpp/cache/block_tree_cache/benchmark/ModelProfile.h"
#include "rtp_llm/cpp/cache/block_tree_cache/benchmark/TransferBenchmarkOptions.h"
#include "rtp_llm/cpp/cache/block_tree_cache/transfer/PerRankBlockTransferEngine.h"

namespace rtp_llm::benchmark {

struct BenchmarkDeviceHostCopyStats {
    std::atomic<size_t> staged_sm{0};
    std::atomic<size_t> cuda_batch{0};
    std::atomic<size_t> generic{0};

    void reset() {
        staged_sm.store(0, std::memory_order_relaxed);
        cuda_batch.store(0, std::memory_order_relaxed);
        generic.store(0, std::memory_order_relaxed);
    }
};

class TransferBenchmarkRunner {
public:
    TransferBenchmarkRunner(const ModelProfile&    profile,
                            const TransferOptions& options,
                            uint64_t               seed,
                            const std::string&     output_json_path);

    bool run();

private:
    using MemberDeviceBlocks = std::vector<std::vector<BlockIdxType>>;

    struct TransferSetup {
        std::vector<const GroupInfo*>                 members;
        std::vector<DeviceBlockPoolPtr>               device_pools;
        GroupSetPtr                                   group_set;
        std::shared_ptr<BenchmarkDeviceHostCopyStats> copy_stats;
        std::shared_ptr<PerRankBlockTransferEngine>   engine;
    };

    struct DirectionStats {
        size_t      attempted{0};
        size_t      succeeded{0};
        size_t      failed{0};
        std::string first_error;
        std::string first_failure_type;
    };

    struct BatchResult {
        std::map<std::string, DirectionStats> directions;
        std::vector<bool>                     visited_working_set;

        size_t attempted() const;
        size_t succeeded() const;
        size_t failed() const;
        size_t visitedWorkingSetBlocks() const;
    };

    const ModelProfile& profile_;
    TransferOptions     options_;
    uint64_t            seed_;
    std::string         output_json_path_;
    BenchmarkJsonWriter writer_;

    bool runPurePathTransfer();

    void recordBatchFailure(const std::string& phase, const BatchResult& batch);

    TransferSetup buildTransferSetup(const GroupSetInfo&            gs_info,
                                     size_t                         device_block_count,
                                     const std::string&             pool_prefix,
                                     bool                           need_device,
                                     std::shared_ptr<HostBlockPool> host_pool,
                                     BlockTreeDiskBlockPoolPtr      disk_pool);

    TransferDescriptor createDescriptor(const std::string&               direction,
                                        const MemberDeviceBlocks&        device_blocks,
                                        const std::vector<BlockIdxType>& host_blocks,
                                        const std::vector<BlockIdxType>& disk_blocks,
                                        size_t                           worker_slot_index,
                                        size_t                           working_set_index,
                                        bool                             host_is_working_set);

    BatchResult runTransferBatch(const std::shared_ptr<PerRankBlockTransferEngine>& engine,
                                 const std::vector<std::string>&                    directions,
                                 const MemberDeviceBlocks&                          device_blocks,
                                 const std::vector<BlockIdxType>&                   host_blocks,
                                 const std::vector<BlockIdxType>&                   disk_blocks,
                                 size_t                                             worker_count,
                                 size_t                                             operation_count,
                                 size_t                                             start_coordinate,
                                 size_t                                             working_set_blocks,
                                 bool                                               host_is_working_set);

    std::string createDiskWorkDir();
};

}  // namespace rtp_llm::benchmark
