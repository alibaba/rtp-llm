#include "src/fastertransformer/devices/arm_impl/ArmDevice.h"
#include "src/fastertransformer/devices/DeviceFactory.h"
#include "src/fastertransformer/core/cpu_allocator.h"
#include <cstring>

namespace fastertransformer {

ArmCpuDevice::ArmCpuDevice(const DeviceInitParams& params): DeviceBase(params) {
    allocator_.reset(new Allocator<AllocatorType::CPU>());
}

ArmCpuDevice::~ArmCpuDevice() {
}

DeviceProperties ArmCpuDevice::getDeviceProperties() {
    DeviceProperties props;
    props.type = DeviceType::Cpu;
    return props;
}

void ArmCpuDevice::copy(const CopyParams& params) {
    auto& src = params.src;
    auto& dst = params.dst;
    auto size = params.src.sizeBytes();
    memcpy(dst.data(), src.data(), size);
}

RTP_LLM_REGISTER_DEVICE(ArmCpu);

} // namespace fastertransformer
