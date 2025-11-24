#pragma once
#include "rtp_llm/cpp/devices/GraphUtils.h"
#include "rtp_llm/cpp/devices/rocm_impl/ROCmDevice.h"
#include <ATen/hip/HIPGraph.h>
#include <hip/hip_runtime.h>
#include <hip/hip_runtime_api.h>

namespace rtp_llm {

// ROCm特定的Stream生命周期管理
class HipGraphStreamLife {
public:
    // ROCmDevice's `stream_` is torch default stream
    HipGraphStreamLife(at::hip::HIPStream capture_stream, rtp_llm::DeviceBase* device):
        origin_stream_(at::hip::getCurrentHIPStream()) {
        rocm_device_ = dynamic_cast<rtp_llm::ROCmDevice*>(device);
        // Set `capture_stream` for capture. All kernels should use this stream while capturing.
        origin_rocm_device_stream_ = rocm_device_->getStream();
        rocm_device_->setStream(capture_stream.stream());
        RTP_LLM_LOG_INFO("Set Hip Stream: capture_stream -> %d, set_stream -> %d, origin_rocm_device_stream_-> %d",
                         reinterpret_cast<int64_t>(capture_stream.stream()),
                         reinterpret_cast<int64_t>(rocm_device_->getStream()),
                         reinterpret_cast<int64_t>(origin_rocm_device_stream_));
        at::hip::setCurrentHIPStream(capture_stream);
    }

    ~HipGraphStreamLife() {
        at::hip::setCurrentHIPStream(origin_stream_);
        rocm_device_->setStream(origin_rocm_device_stream_);
    }

private:
    at::hip::HIPStream   origin_stream_;
    hipStream_t          origin_rocm_device_stream_;
    rtp_llm::ROCmDevice* rocm_device_;
};

// HIP特定的工具函数命名空间
namespace HipGraphUtils {

// 设备同步
inline void deviceSynchronize() {
    hipDeviceSynchronize();
}

// 流管理
inline at::hip::HIPStream getStreamFromPool(bool high_priority = false) {
    return at::hip::getStreamFromPool(high_priority);
}

inline at::hip::HIPStream getCurrentStream() {
    return at::hip::getCurrentHIPStream();
}

// 创建HIP特定的流生命周期管理
template<typename DeviceType>
inline auto createStreamLife(at::hip::HIPStream capture_stream, DeviceType* device) {
    return HipGraphStreamLife(capture_stream, device);
}

}  // namespace HipGraphUtils

}  // namespace rtp_llm