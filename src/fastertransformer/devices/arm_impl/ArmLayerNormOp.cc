#include "src/fastertransformer/devices/arm_impl/ArmDevice.h"
#include "src/fastertransformer/devices/DeviceFactory.h"
#include "src/fastertransformer/core/allocator.h"
#include "src/fastertransformer/core/cpu_allocator.h"
#include "src/fastertransformer/devices/utils/DebugUtils.h"
#include <cstring>

namespace fastertransformer {

template<typename T>
void add_residual_bias(void* norm_out, const void* input, void* residual, int m, int n) {
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            ((T*)norm_out)[i * n + j] = ((T*)input)[i * n + j] + ((T*)residual)[i * n + j];
        }
    }
}

template<typename T>
void rmsnorm(
    T* norm_out, T* bias_out, const T* input, const double eps, void* gamma, void* beta, void* residual, int m, int n) {
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            norm_out[i * n + j] = ((T*)residual)[i * n + j];
        }
    }

    if (bias_out != nullptr) {
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                norm_out[i * n + j] += input[i * n + j];
                bias_out[i * n + j] = norm_out[i * n + j];
            }
        }
    }

    for (int i = 0; i < m; i++) {
        float sum = 0.0f;
        for (int j = 0; j < n; j++) {
            sum += norm_out[i * n + j] * norm_out[i * n + j];
        }
        sum /= n;
        sum += eps;
        sum = 1.0f / sqrtf(sum);
        for (int j = 0; j < n; j++) {
            norm_out[i * n + j] *= sum;
            if (gamma != nullptr) {
                norm_out[i * n + j] *= ((T*)gamma)[j];
            }
            if (beta != nullptr) {
                norm_out[i * n + j] += ((T*)beta)[j];
            }
        }
    }
}

LayernormOutput ArmCpuDevice::layernorm(const LayernormParams& params) {
    BufferPtr   input       = params.input;
    BufferPtr   norm_output = input;
    void*       bias_output = params.before_norm_output ? params.before_norm_output->data() : nullptr;
    const auto& weights     = params.norm_weight;
    void*       gamma       = weights ? weights->get().gamma.get()->data() : nullptr;
    void*       beta        = (weights && weights->get().beta) ? weights->get().beta.get()->data() : nullptr;
    const auto  eps         = params.eps;
    void*       residual    = params.residual1 ? params.residual1->get().data() : nullptr;
    const auto  norm_type   = params.norm_type;
    int         m           = input->shape()[0];
    int         n           = input->shape()[1];

    const auto data_type = input->type();

    if (!params.is_inplace && params.qscheme == QScheme::NoQuantize) {
        norm_output = allocateBufferLike(*params.input);
    } else if (params.qscheme == Qint8PerChannelLastAxis) {
        throw OpException(OpErrorType::ERROR_UNIMPLEMENTED);
    }

    if (norm_type == NormType::rmsnorm) {
        if (!weights.has_value()) {
            if (data_type == DataType::TYPE_FP32)
                add_residual_bias<float>((float*)norm_output->data(), (float*)input->data(), residual, m, n);
            else if (data_type == DataType::TYPE_FP16)
                add_residual_bias<__fp16>((__fp16*)norm_output->data(), (__fp16*)input->data(), residual, m, n);
            else {
                throw OpException(OpErrorType::ERROR_UNIMPLEMENTED);
            }
            return LayernormOutput({norm_output, params.before_norm_output});
        }

        if ((residual == nullptr) && !params.bias.has_value()) {
            residual = input->data();
        }
        if (data_type == DataType::TYPE_FP32)
            rmsnorm<float>((float*)norm_output->data(),
                           (float*)bias_output,
                           (float*)input->data(),
                           eps,
                           gamma,
                           beta,
                           residual,
                           m,
                           n);
        else if (data_type == DataType::TYPE_FP16)
            rmsnorm<__fp16>((__fp16*)norm_output->data(),
                            (__fp16*)bias_output,
                            (__fp16*)input->data(),
                            eps,
                            gamma,
                            beta,
                            residual,
                            m,
                            n);
        else {
            throw OpException(OpErrorType::ERROR_UNIMPLEMENTED);
        }
        return LayernormOutput({norm_output, params.before_norm_output});
    }

    arm_compute::DataType   acl_data_type = getAclDataType(data_type);
    arm_compute::TensorInfo data_info     = arm_compute::TensorInfo(arm_compute::TensorShape(n, m), 1, acl_data_type);

    arm_compute::NEMeanStdDevNormalizationLayer msdNorm;
    arm_compute::Tensor                         src_tensor;
    arm_compute::Tensor                         dst_tensor;

    src_tensor.allocator()->init(data_info);
    dst_tensor.allocator()->init(data_info);
    src_tensor.allocator()->import_memory(input->data());
    dst_tensor.allocator()->import_memory(norm_output->data());

    /* Note gamma and beta for scale shift not supported by ACL */
    msdNorm.configure(&src_tensor, &dst_tensor, eps);
    msdNorm.run();

    src_tensor.allocator()->free();
    dst_tensor.allocator()->free();

    return LayernormOutput({norm_output, params.before_norm_output});
}

}  // namespace fastertransformer
