#pragma once

#include <atomic>
#include <cstdint>
#include <memory>
#include <optional>
#include <utility>

namespace rtp_llm {

// Only locally observed runtime results belong here, never Mooncake/network
// codes or codes inferred from exception text. No accelerator headers/runtime
// are needed to retain or query this state.
enum class DeviceRuntimeDomain : uint8_t {
    UNKNOWN,
    CUDA,
    HIP
};
enum class DeviceRuntimeOperation : uint8_t {
    GET_DEVICE,
    SET_DEVICE,
    RESTORE_DEVICE,
    EVENT_WAIT,
    STREAM_WAIT,
    DEVICE_WAIT,
    STREAM_QUERY,
    PEEK_LAST_ERROR,
};

struct ServingDeviceFault {
    DeviceRuntimeDomain domain{DeviceRuntimeDomain::UNKNOWN};
    int32_t             code{0};
    // Bound resource for context operations; merely the API target/previous
    // device for Set/Restore diagnostics, never evidence of fault attribution.
    int32_t                device{-1};
    DeviceRuntimeOperation operation{DeviceRuntimeOperation::GET_DEVICE};
};

// A sticky code alone is not attribution proof; use the runtime boundary below.
constexpr bool isServingDeviceFatal(DeviceRuntimeDomain domain, int32_t code) noexcept {
    // CUDA runtime / HIP runtime IllegalAddress, Assert, LaunchFailure. The
    // native publisher adapter static_asserts these ABI values against its SDK.
    // Deliberately exclude OOM, invalid configuration, launch-out-of-resources,
    // not-ready, driver/library status domains, and all unrecognized results.
    switch (domain) {
        case DeviceRuntimeDomain::CUDA:
            return code == 700 || code == 710 || code == 719;
        case DeviceRuntimeDomain::HIP:
            return code == 700 || code == 710 || code == 719;
        default:
            return false;
    }
}

constexpr bool isContextBoundRuntimeOperation(DeviceRuntimeOperation operation) noexcept {
    return operation == DeviceRuntimeOperation::EVENT_WAIT || operation == DeviceRuntimeOperation::STREAM_WAIT
           || operation == DeviceRuntimeOperation::DEVICE_WAIT || operation == DeviceRuntimeOperation::STREAM_QUERY;
}

class ServingDeviceHealth final {
public:
    bool healthy() const noexcept {
        return !failed_.load(std::memory_order_acquire);
    }

    bool observeRuntimeResult(ServingDeviceFault fault) noexcept {
        if (fault.device < 0 || !isContextBoundRuntimeOperation(fault.operation)
            || !isServingDeviceFatal(fault.domain, fault.code)) {
            return false;
        }
        // Publish the sticky admission gate before diagnostics. Owner-only
        // allocation failures never write this bit and cannot mask a later
        // device fatal. A single writer publishes immutable first-fault data.
        failed_.store(true, std::memory_order_release);
        if (!first_fault_claimed_.test_and_set(std::memory_order_relaxed)) {
            first_fault_ = fault;
            first_fault_ready_.store(true, std::memory_order_release);
        }
        return true;
    }

    std::optional<ServingDeviceFault> firstFault() const noexcept {
        if (!first_fault_ready_.load(std::memory_order_acquire)) {
            return std::nullopt;
        }
        return first_fault_;
    }

private:
    std::atomic<bool>  failed_{false};
    std::atomic_flag   first_fault_claimed_ = ATOMIC_FLAG_INIT;
    std::atomic<bool>  first_fault_ready_{false};
    ServingDeviceFault first_fault_;
};

using ServingDeviceHealthPtr = std::shared_ptr<ServingDeviceHealth>;

struct ServingRuntimeResult {
    ServingDeviceFault fault;
    bool               operation_performed{false};
};

// Injectable native boundary, also used by CPU tests. GetDevice succeeding is
// a prerequisite, not device-fatal evidence. A handle's owning device AND the
// native current device must match the serving device before health can latch.
// No host-thread last-error slot is read (it may belong to a foreign device).
// The caller handles owner quarantine/diagnostics only AFTER this returns.
template<typename GetDevice, typename Operation>
ServingRuntimeResult runServingDeviceOperation(ServingDeviceHealth*   health,
                                               DeviceRuntimeDomain    domain,
                                               int                    serving_device,
                                               int                    resource_device,
                                               DeviceRuntimeOperation operation,
                                               GetDevice&&            get_device,
                                               Operation&&            invoke) {
    int        current        = -1;
    const auto selection_code = static_cast<int32_t>(std::forward<GetDevice>(get_device)(current));
    if (selection_code != 0) {
        return {{domain, selection_code, -1, DeviceRuntimeOperation::GET_DEVICE}, false};
    }
    if (current < 0 || resource_device < 0 || current != resource_device) {
        return {{domain, 0, resource_device, operation}, false};
    }
    ServingDeviceFault result{
        domain, static_cast<int32_t>(std::forward<Operation>(invoke)()), resource_device, operation};
    if (health && serving_device >= 0 && current == serving_device) {
        health->observeRuntimeResult(result);
    }
    return {result, true};
}

}  // namespace rtp_llm
