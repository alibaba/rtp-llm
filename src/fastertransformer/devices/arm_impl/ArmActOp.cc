#include "src/fastertransformer/devices/arm_impl/ArmDevice.h"
#include "src/fastertransformer/devices/DeviceFactory.h"
#include "src/fastertransformer/core/allocator.h"
#include "src/fastertransformer/core/cpu_allocator.h"
#include "src/fastertransformer/devices/utils/DebugUtils.h"
#include <cstring>

namespace fastertransformer {

BufferPtr ArmCpuDevice::activation(const ActivationParams& params) {
    const auto& states = params.states;
    size_t      m      = states->shape()[0];
    size_t      n      = states->shape()[1];

    arm_compute::ActivationFunction activationFunction;
    if (params.atype == ActivationType::Silu) {
        activationFunction = arm_compute ::ActivationLayerInfo::ActivationFunction::SWISH;
    } else if (params.atype == ActivationType::Gelu) {
        activationFunction = arm_compute ::ActivationLayerInfo::ActivationFunction::GELU;
    } else if (params.atype == ActivationType::Swiglu) {
        activationFunction = arm_compute ::ActivationLayerInfo::ActivationFunction::SWISH;
    } else {
        throw OpException(OpErrorType::ERROR_UNIMPLEMENTED);
    }

    void* gate      = nullptr;

    arm_compute::DataType   acl_data_type = getAclDataType(states->type());
    arm_compute::TensorInfo data_info     = arm_compute::TensorInfo(arm_compute::TensorShape(n, m), 1, acl_data_type);

    arm_compute::NEActivationLayer act;
    arm_compute::Tensor            src_tensor;
    arm_compute::Tensor            dst_tensor;

    src_tensor.allocator()->init(data_info);
    dst_tensor.allocator()->init(data_info);
    if (params.gate) {
        gate = params.gate.value().get().data();
        src_tensor.allocator()->import_memory(gate);
        dst_tensor.allocator()->import_memory(gate);
    } else {
        src_tensor.allocator()->import_memory(states->data());
        dst_tensor.allocator()->import_memory(states->data());
    }

    act.configure(&src_tensor, &dst_tensor, arm_compute::ActivationLayerInfo(activationFunction, 1.0f));

    act.run();

    src_tensor.allocator()->free();
    dst_tensor.allocator()->free();

    if (params.gate) {
        gate = params.gate.value().get().data();
        printBufferData(params.gate.value().get(), "ffn activation gate");
        if (states->type() == DataType::TYPE_FP16) {
            for (size_t i = 0; i < m; i++) {
                for (size_t j = 0; j < n; j++) {
                    *(__fp16*)(states->dataWithOffset(i * n + j)) *= ((__fp16*)gate)[i * n + j];
                }
            }
        } else if (states->type() == DataType::TYPE_FP32) {
            for (size_t i = 0; i < m; i++) {
                for (size_t j = 0; j < n; j++) {
                    *(float*)(states->dataWithOffset(i * n + j)) *= ((float*)gate)[i * n + j];
                }
            }
        } else {
            throw std::runtime_error("FFN gate data type not supported");
        }
    }
    return states;
}

}  // namespace fastertransformer
