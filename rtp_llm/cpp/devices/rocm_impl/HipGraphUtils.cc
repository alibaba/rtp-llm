#include "rtp_llm/cpp/devices/rocm_impl/HipGraphUtils.h"
#include "rtp_llm/cpp/devices/rocm_impl/ROCmDevice.h"
#include "rtp_llm/cpp/devices/DeviceBase.h"

HipGraphStreamLife::HipGraphStreamLife(at::hip::HIPStream capture_stream, rtp_llm::DeviceBase* device):
    origin_stream_(at::hip::getCurrentHIPStream(at::hip::current_device())) {
    rocm_device_ = dynamic_cast<rtp_llm::ROCmDevice*>(device);
    // Set capture_stream for capture. All kernels should use this stream while capturing.
    origin_rocm_device_stream_ = rocm_device_->getStream();
    rocm_device_->setStream(capture_stream.stream());
    RTP_LLM_LOG_INFO("Set HIP Stream: capture_stream -> %d, set_stream -> %d, origin_rocm_device_stream_-> %d",
                     capture_stream.stream(),
                     reinterpret_cast<int64_t>(rocm_device_->getStream()),
                     origin_rocm_device_stream_);
    at::hip::setCurrentHIPStream(capture_stream);
}

HipGraphStreamLife::~HipGraphStreamLife() {
    at::hip::setCurrentHIPStream(origin_stream_);
    rocm_device_->setStream(origin_rocm_device_stream_);
}