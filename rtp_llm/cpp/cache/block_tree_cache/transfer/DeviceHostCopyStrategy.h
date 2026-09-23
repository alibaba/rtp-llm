#pragma once

#include <cstddef>
#include <cstdint>
#include <map>
#include <memory>
#include <mutex>
#include <vector>

#include "rtp_llm/cpp/cache/block_tree_cache/transfer/TransferTypes.h"
#include "rtp_llm/cpp/cache/block_tree_cache/block_pool/IBlockPool.h"

namespace rtp_llm {

struct StagedMemoryCopyScratch;
struct StagedMemoryCopyParams;
enum class StagedMemoryCopyStatus;

// --- Copy Plan types ---

struct DeviceHostCopyTile {
    void*  host_addr{nullptr};
    void*  device_addr{nullptr};
    size_t host_offset{0};
    size_t bytes{0};
    int    device_index{-1};
    size_t member_group_id{0};
    size_t local_layer_index{0};
    size_t origin_index{0};
    size_t buffer_index{0};
    size_t device_buffer_bytes{0};
    size_t layer_offset{0};
};

// One numeric provenance snapshot per descriptor/member pool, shared by its
// layer tiles. No strings, CUDA queries or per-tile heap allocations are needed.
struct DeviceHostCopyOrigin {
    size_t         descriptor_index{0};
    size_t         group_set_id{0};
    size_t         member_group_id{0};
    size_t         topology_group_id{0};
    size_t         path_index{0};
    uintptr_t      node{0};  // identity only; never dereferenced by diagnostics
    Tier           source_tier{Tier::NONE};
    Tier           target_tier{Tier::NONE};
    BlockIdxType   device_block{0};
    BlockIdxType   other_block{0};
    HostBufferView host;
    uintptr_t      host_pool_base{0};  // 0 for a disk staging view, not a host-cache block
    size_t         host_pool_bytes{0};
    size_t         host_pool_stride{0};
    uintptr_t      device_pool_base{0};
    size_t         device_pool_bytes{0};
    size_t         layer_stride{0};
    size_t         kv_bytes{0};
    size_t         scale_bytes{0};
    // Keep pools (not individual blocks) alive until the synchronous copy ends.
    // Comparing these snapshots detects release/reallocation during the copy.
    std::shared_ptr<IBlockPool> device_pool;
    std::shared_ptr<IBlockPool> host_pool;
    BlockDiagnosticSnapshot     device_at_plan;
    BlockDiagnosticSnapshot     host_at_plan;
};

struct DeviceHostCopyPlan {
    bool                              device_to_host{false};
    size_t                            group_set_id{0};
    HostBufferView                    host;
    size_t                            first_descriptor_index{0};
    bool                              mixed_descriptors{false};
    std::vector<DeviceHostCopyTile>   copy_tiles;
    std::vector<DeviceHostCopyOrigin> origins;
};

// Host-only validation and cold-path serialization; end addresses are exclusive.
bool        validateDeviceHostCopyPlan(const DeviceHostCopyPlan& plan);
std::string deviceHostCopyPlanJson(const DeviceHostCopyPlan& plan);

enum class StrategyStatus {
    DONE,
    NOT_APPLICABLE,
    FAILED,
};

struct StrategyResult {
    StrategyStatus status{StrategyStatus::NOT_APPLICABLE};
    TransferStatus copy_status{TransferStatus::OK};

    static StrategyResult done() {
        return {StrategyStatus::DONE, TransferStatus::OK};
    }
    static StrategyResult notApplicable() {
        return {StrategyStatus::NOT_APPLICABLE, TransferStatus::OK};
    }
    static StrategyResult failed(TransferStatus s) {
        return {StrategyStatus::FAILED, s};
    }
};

class DeviceHostCopyStrategy {
public:
    virtual ~DeviceHostCopyStrategy()                                                                       = default;
    virtual StrategyResult tryExecute(const DeviceHostCopyPlan& plan, const DeviceHostCopyOptions& options) = 0;
};

class StagedSmDeviceHostCopyStrategy: public DeviceHostCopyStrategy {
public:
    ~StagedSmDeviceHostCopyStrategy() override;

    StrategyResult tryExecute(const DeviceHostCopyPlan& plan, const DeviceHostCopyOptions& options) override;

protected:
    virtual StagedMemoryCopyStatus executeStagedCopy(const StagedMemoryCopyParams& params,
                                                     StagedMemoryCopyScratch*      scratch);

private:
    std::mutex                                              scratch_mutex_;
    std::map<int, std::unique_ptr<StagedMemoryCopyScratch>> scratch_by_device_;
};

class CudaBatchDeviceHostCopyStrategy: public DeviceHostCopyStrategy {
public:
    StrategyResult tryExecute(const DeviceHostCopyPlan& plan, const DeviceHostCopyOptions& options) override;
};

class GenericMultiCopyDeviceHostCopyStrategy: public DeviceHostCopyStrategy {
public:
    StrategyResult tryExecute(const DeviceHostCopyPlan& plan, const DeviceHostCopyOptions& options) override;
};

}  // namespace rtp_llm
