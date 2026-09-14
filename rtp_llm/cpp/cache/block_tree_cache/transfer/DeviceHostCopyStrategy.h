#pragma once

#include <cstddef>
#include <cstdint>
#include <map>
#include <memory>
#include <mutex>
#include <vector>

#include "rtp_llm/cpp/cache/block_tree_cache/transfer/TransferTypes.h"
#include "rtp_llm/models_py/bindings/NoBlockCopy.h"

namespace rtp_llm {

struct StagedMemoryCopyScratch;

// --- Copy Plan types ---

struct DeviceHostCopyTile {
    void*  host_addr{nullptr};
    void*  device_addr{nullptr};
    size_t host_offset{0};
    size_t bytes{0};
    int    device_index{-1};
    size_t member_group_id{0};
    size_t local_layer_index{0};
    // Missing identity is deliberately non-coalescible for externally built plans.
    size_t descriptor_index{SIZE_MAX};
    size_t layout_index{SIZE_MAX};
    size_t component_index{SIZE_MAX};
};

struct DeviceHostCopyPlan {
    bool                            device_to_host{false};
    size_t                          group_set_id{0};
    HostBufferView                  host;
    std::vector<DeviceHostCopyTile> copy_tiles;
};

enum class StrategyStatus {
    DONE,
    NOT_APPLICABLE,
    FAILED,
};

struct StrategyResult {
    StrategyStatus status{StrategyStatus::NOT_APPLICABLE};
    TransferStatus copy_status{TransferStatus::OK};
    size_t copy_operation_count{0};

    static StrategyResult done(size_t operations = 0) {
        return {StrategyStatus::DONE, TransferStatus::OK, operations};
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
    virtual StrategyResult tryExecute(const DeviceHostCopyPlan&             plan,
                                      const DeviceHostCopyOptions&          options,
                                      const DeviceHostCopyExecutionContext& context)                        = 0;
};

class StagedSmDeviceHostCopyStrategy: public DeviceHostCopyStrategy {
public:
    ~StagedSmDeviceHostCopyStrategy() override;

    StrategyResult tryExecute(const DeviceHostCopyPlan&             plan,
                              const DeviceHostCopyOptions&          options,
                              const DeviceHostCopyExecutionContext& context) override;

private:
    std::mutex                                              scratch_mutex_;
    std::map<int, std::unique_ptr<StagedMemoryCopyScratch>> scratch_by_device_;
};

class Cuda3DBatchDeviceHostCopyStrategy: public DeviceHostCopyStrategy {
public:
    explicit Cuda3DBatchDeviceHostCopyStrategy(bool coalesce_tiles = true): coalesce_tiles_(coalesce_tiles) {}

    StrategyResult tryExecute(const DeviceHostCopyPlan&             plan,
                              const DeviceHostCopyOptions&          options,
                              const DeviceHostCopyExecutionContext& context) override;

private:
    // False is an exact unmerged control using the same API and sub-batch.
    bool coalesce_tiles_;
};

class CudaBatchDeviceHostCopyStrategy: public DeviceHostCopyStrategy {
public:
    StrategyResult tryExecute(const DeviceHostCopyPlan&             plan,
                              const DeviceHostCopyOptions&          options,
                              const DeviceHostCopyExecutionContext& context) override;
};

class GenericMultiCopyDeviceHostCopyStrategy: public DeviceHostCopyStrategy {
public:
    StrategyResult tryExecute(const DeviceHostCopyPlan&             plan,
                              const DeviceHostCopyOptions&          options,
                              const DeviceHostCopyExecutionContext& context) override;
};

}  // namespace rtp_llm
