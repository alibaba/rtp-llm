#pragma once

#include <cstddef>
#include <cstdint>
#include <map>
#include <memory>
#include <mutex>
#include <vector>

#include "rtp_llm/cpp/cache/block_tree_cache/copy_engine/TransferTypes.h"

namespace rtp_llm {

struct StagedMemoryCopyScratch;

// --- Copy Plan types ---

struct DeviceHostCopyTile {
    void*  host_addr{nullptr};
    void*  device_addr{nullptr};
    size_t host_offset{0};
    size_t bytes{0};
    int    device_index{-1};
    int    component_index{-1};
    int    layer_id{-1};
};

struct HostZeroTile {
    void*  host_addr{nullptr};
    size_t bytes{0};
};

struct HostPayloadView {
    void*  base{nullptr};
    size_t payload_bytes{0};
};

struct DeviceHostCopyPlan {
    bool             device_to_host{false};
    bool             single_device{true};
    int              component_group_id{-1};
    HostPayloadView  host;
    std::vector<DeviceHostCopyTile> copy_tiles;
    std::vector<HostZeroTile>       zero_tiles;
};

// --- Strategy result ---

enum class StrategyStatus {
    DONE,
    NOT_APPLICABLE,
    FAILED,
};

struct StrategyResult {
    StrategyStatus status{StrategyStatus::NOT_APPLICABLE};
    CopyStatus     copy_status{CopyStatus::OK};

    static StrategyResult done() { return {StrategyStatus::DONE, CopyStatus::OK}; }
    static StrategyResult notApplicable() { return {StrategyStatus::NOT_APPLICABLE, CopyStatus::OK}; }
    static StrategyResult failed(CopyStatus s) { return {StrategyStatus::FAILED, s}; }
};

// --- Strategy interface ---

class DeviceHostCopyStrategy {
public:
    virtual ~DeviceHostCopyStrategy() = default;
    virtual StrategyResult tryExecute(const DeviceHostCopyPlan& plan,
                                      const DeviceHostCopyOptions& options) = 0;
};

// --- Concrete strategies ---

class StagedSmDeviceHostCopyStrategy: public DeviceHostCopyStrategy {
public:
    ~StagedSmDeviceHostCopyStrategy() override;

    StrategyResult tryExecute(const DeviceHostCopyPlan& plan,
                              const DeviceHostCopyOptions& options) override;

private:
    std::mutex                                              scratch_mutex_;
    std::map<int, std::unique_ptr<StagedMemoryCopyScratch>> scratch_by_device_;
};

class CudaBatchDeviceHostCopyStrategy: public DeviceHostCopyStrategy {
public:
    StrategyResult tryExecute(const DeviceHostCopyPlan& plan,
                              const DeviceHostCopyOptions& options) override;
};

class GenericMultiCopyDeviceHostCopyStrategy: public DeviceHostCopyStrategy {
public:
    StrategyResult tryExecute(const DeviceHostCopyPlan& plan,
                              const DeviceHostCopyOptions& options) override;
};

}  // namespace rtp_llm
