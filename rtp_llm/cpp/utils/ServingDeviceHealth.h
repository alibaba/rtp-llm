#pragma once

#include <atomic>
#include <cstdint>
#include <memory>
#include <utility>

namespace rtp_llm {

enum class DeviceRuntimeDomain : uint8_t { UNKNOWN, CUDA, HIP };
enum class DeviceRuntimeOperation : uint8_t { EVENT_WAIT, STREAM_WAIT, DEVICE_WAIT };

struct ServingDeviceFault {
    DeviceRuntimeDomain domain{DeviceRuntimeDomain::UNKNOWN};
    int32_t code{0};
    int device{-1};
    DeviceRuntimeOperation operation{DeviceRuntimeOperation::EVENT_WAIT};
};

// Only native context-bound runtime returns qualify, never exception text,
// driver/library/network status codes, or a thread's last-error slot.
constexpr bool isServingDeviceFatal(DeviceRuntimeDomain domain, int32_t code) noexcept {
    if (domain == DeviceRuntimeDomain::CUDA) {
        // CUDA SDK driver_types.h requires process restart after these errors.
        return code == 700 || code == 702 || code == 710 || code == 714 || code == 715 || code == 716
               || code == 717 || code == 718 || code == 719;
    }
    // Keep HIP's verified set separate; matching CUDA numbers are not evidence
    // of identical HIP failure semantics.
    return domain == DeviceRuntimeDomain::HIP && (code == 700 || code == 710 || code == 719);
}

class ServingDeviceHealth final {
public:
    bool healthy() const noexcept { return !failed_.load(std::memory_order_acquire); }

    // The caller must have verified the native current device and resource
    // owner against this engine's serving device before recording the result.
    bool observeRuntimeResult(ServingDeviceFault fault) noexcept {
        if (fault.device < 0
            || (fault.operation != DeviceRuntimeOperation::EVENT_WAIT
                && fault.operation != DeviceRuntimeOperation::STREAM_WAIT
                && fault.operation != DeviceRuntimeOperation::DEVICE_WAIT)
            || !isServingDeviceFatal(fault.domain, fault.code)) {
            return false;
        }
        failed_.store(true, std::memory_order_release);
        return true;
    }

private:
    std::atomic<bool> failed_{false};
};

using ServingDeviceHealthPtr = std::shared_ptr<ServingDeviceHealth>;

// Selection failure or unknown/foreign resource ownership is not evidence of a
// serving-context fault. Do not execute the operation in an unverified context.
// The owner must retain/quarantine unfenced resources when this returns false.
template<typename GetDevice, typename Operation>
bool runServingDeviceOperation(ServingDeviceHealth* health,
                               DeviceRuntimeDomain domain,
                               int serving_device,
                               int resource_device,
                               DeviceRuntimeOperation operation,
                               GetDevice&& get_device,
                               Operation&& invoke,
                               int32_t& result) {
    int current = -1;
    result = static_cast<int32_t>(std::forward<GetDevice>(get_device)(current));
    if (result != 0 || current < 0 || resource_device < 0 || current != resource_device) {
        return false;
    }
    result = static_cast<int32_t>(std::forward<Operation>(invoke)());
    if (health && serving_device >= 0 && current == serving_device) {
        health->observeRuntimeResult({domain, result, resource_device, operation});
    }
    return true;
}

}  // namespace rtp_llm
