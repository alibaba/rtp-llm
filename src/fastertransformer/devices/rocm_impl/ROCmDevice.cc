#include "src/fastertransformer/devices/rocm_impl/ROCmDevice.h"
#include "src/fastertransformer/devices/rocm_impl/ROCmAllocator.h"
#include "src/fastertransformer/devices/DeviceFactory.h"
#include <cstring>

#include <hip/hip_runtime.h>

namespace fastertransformer {

ROCmDevice::ROCmDevice(const DeviceInitParams& params): DeviceBase(params) {
    allocator_.reset(new Allocator<AllocatorType::ROCM>());
}

ROCmDevice::~ROCmDevice() {
}

DeviceProperties ROCmDevice::getDeviceProperties() {
    DeviceProperties props;
    props.type = DeviceType::Cpu;
    return props;
}

void ROCmDevice::copy(const CopyParams& params) {
    auto& src = params.src;
    auto& dst = params.dst;
    auto size = params.src.sizeBytes();
    memcpy(dst.data(), src.data(), size);
}

RTP_LLM_REGISTER_DEVICE(ROCm);

} // namespace fastertransformer
