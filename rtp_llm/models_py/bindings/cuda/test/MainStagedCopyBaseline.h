#pragma once

#include "rtp_llm/cpp/cache/block_tree_cache/transfer/DeviceHostCopyStrategy.h"
#include "rtp_llm/models_py/bindings/NoBlockCopy.h"

namespace rtp_llm::crc_copy_benchmark::main_baseline {

// Test-only snapshot of main e73826e27a6005640bb75ce2a73f235bfa51c55b.
inline constexpr char kSourceCommit[] = "e73826e27a6005640bb75ce2a73f235bfa51c55b";

struct StagedMemoryCopyScratch {
    void*  host_staging    = nullptr;
    size_t host_capacity   = 0;
    void*  device_staging  = nullptr;
    size_t device_capacity = 0;
    void*  device_ptrs     = nullptr;
    void*  device_offsets  = nullptr;
    void*  device_sizes    = nullptr;
    size_t meta_capacity   = 0;
    int    device_index    = -1;
};

void releaseStagedMemoryCopyScratch(StagedMemoryCopyScratch& scratch);

class StagedSmDeviceHostCopyStrategy: public DeviceHostCopyStrategy {
public:
    ~StagedSmDeviceHostCopyStrategy() override;

    StrategyResult tryExecute(const DeviceHostCopyPlan& plan, const DeviceHostCopyOptions& options) override;

private:
    std::mutex                                              scratch_mutex_;
    std::map<int, std::unique_ptr<StagedMemoryCopyScratch>> scratch_by_device_;
};

}  // namespace rtp_llm::crc_copy_benchmark::main_baseline
