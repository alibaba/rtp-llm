#include "src/fastertransformer/devices/arm_impl/ArmDevice.h"
#include "src/fastertransformer/devices/DeviceFactory.h"
#include "src/fastertransformer/core/allocator.h"
#include "src/fastertransformer/core/cpu_allocator.h"

namespace fastertransformer {

/* Apply mask to input.
    Different heads share the same mask. */
template<typename T, typename T_mask>
void context_mask(BufferPtr input, const Buffer& mask) {
    const int dim0 = input->shape()[0];
    const int dim1 = input->shape()[1];
    const int dim2 = input->shape()[2];
    const int dim3 = input->shape()[3];

    const int N = dim0 * dim1;
    parallel_for(N, [&](int tid) {
        int b = tid / dim1;
        for (int i = 0; i < dim2 * dim3; i++) {
            auto v = input->dataWithOffset(tid * dim2 * dim3 + i);
            auto m = mask.dataWithOffset(b * dim2 * dim3 + i);
            *(T*)v += (1.0f - *(T_mask*)m) * -10000.0f;
        }
    });
}

BufferPtr ArmCpuDevice::softmax(const SoftmaxParams& params) {
    if (params.input == nullptr) {
        throw std::runtime_error("softmax input can not be nullptr");
    }
    auto        type  = params.input->type();
    const auto& input = params.input;
    auto output = allocateBuffer({params.output_t == DataType::TYPE_INVALID ? params.input->type() : params.output_t,
                                  params.input->shape(),
                                  AllocationType::HOST});

    size_t type_size = params.input->typeSize();
    if ((type_size != 4) && (type_size != 2)) {
        throw std::runtime_error("Softmax input type is not supported");
    }

    if (params.mask.has_value()) {
        /* Apply mask. */
        auto mask_type = params.mask.value().get().type();
        if (type == DataType::TYPE_FP32 && mask_type == DataType::TYPE_FP16) {
            context_mask<float, __fp16>(params.input, params.mask.value().get());
        } else if (type == DataType::TYPE_FP32 && mask_type == DataType::TYPE_FP32) {
            context_mask<float, float>(params.input, params.mask.value().get());
        } else if (type == DataType::TYPE_FP16) {
            context_mask<__fp16, __fp16>(params.input, params.mask.value().get());
        } else {
            throw std::runtime_error("Softmax data type is not supported");
        }
    }

    arm_compute::DataType   acl_data_type = getAclDataType(type);
    arm_compute::TensorInfo data_info     = arm_compute::TensorInfo(
        arm_compute::TensorShape(input->shape()[3], input->shape()[2], input->shape()[1], input->shape()[0]),
        1,
        acl_data_type);

    arm_compute::NESoftmaxLayer softmax;
    arm_compute::Tensor         src_tensor;
    arm_compute::Tensor         dst_tensor;

    src_tensor.allocator()->init(data_info);
    dst_tensor.allocator()->init(data_info);
    src_tensor.allocator()->import_memory(input->data());
    dst_tensor.allocator()->import_memory(output->data());
    float beta = params.scale;

    softmax.configure(&src_tensor, &dst_tensor, beta);
    softmax.run();

    src_tensor.allocator()->free();
    dst_tensor.allocator()->free();

    return output;
}

}  // namespace fastertransformer
