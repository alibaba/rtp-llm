#include "src/fastertransformer/devices/arm_impl/ArmDevice.h"
#include "src/fastertransformer/devices/DeviceFactory.h"
#include "src/fastertransformer/core/allocator.h"
#include "src/fastertransformer/core/cpu_allocator.h"
#include <cstring>

namespace fastertransformer {

ArmCpuDevice::ArmCpuDevice(const DeviceInitParams& params): DeviceBase(params) {
    allocator_.reset(new Allocator<AllocatorType::CPU>());
}

ArmCpuDevice::~ArmCpuDevice() {}

DeviceProperties ArmCpuDevice::getDeviceProperties() {
    DeviceProperties props;
    props.type = DeviceType::ArmCpu;
    return props;
}

arm_compute::DataType ArmCpuDevice::getAclDataType(DataType type) {
    using dt = arm_compute::DataType;
    switch (type) {
        case DataType::TYPE_FP32:
            return dt::F32;
        case DataType::TYPE_BF16:
            return dt::BFLOAT16;
        case DataType::TYPE_FP16:
            return dt::F16;
        case DataType::TYPE_UINT8:
            return dt::U8;
        case DataType::TYPE_UINT16:
            return dt::U16;
        case DataType::TYPE_UINT32:
            return dt::U32;
        case DataType::TYPE_INT8:
            return dt::S8;
        case DataType::TYPE_UINT64:
            return dt::U64;
        default:
            return dt::UNKNOWN;
    }
}

void ArmCpuDevice::copy(const CopyParams& params) {
    auto& src  = params.src;
    auto& dst  = params.dst;
    auto  size = params.src.sizeBytes();
    memcpy(dst.data(), src.data(), size);
}

GroupedGemmOutput ArmCpuDevice::groupedGemm(const GroupedGemmParams& params) {
    throw OpException(OpErrorType::ERROR_UNIMPLEMENTED);
}

void ArmCpuDevice::sampleGreedy(const GreedyParams& params) {
    throw OpException(OpErrorType::ERROR_UNIMPLEMENTED);
}

void ArmCpuDevice::sampleBeamSearch(const BeamSearchParams& params) {
    throw OpException(OpErrorType::ERROR_UNIMPLEMENTED);
}

void ArmCpuDevice::broadcast(const BroadcastParams& params) {
    throw OpException(OpErrorType::ERROR_UNIMPLEMENTED);
}

void ArmCpuDevice::allReduceSum(const AllReduceParams& params) {
    throw OpException(OpErrorType::ERROR_UNIMPLEMENTED);
}

RTP_LLM_REGISTER_DEVICE(ArmCpu);

}  // namespace fastertransformer
