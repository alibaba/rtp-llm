#pragma once

#include <cstddef>
#include <cstdint>
#include <map>
#include <memory>
#include <mutex>
#include <vector>

#include "rtp_llm/cpp/cache/block_tree_cache/transfer/TransferTypes.h"

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
