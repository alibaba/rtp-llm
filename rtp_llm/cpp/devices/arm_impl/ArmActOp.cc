#include "rtp_llm/cpp/devices/arm_impl/ArmDevice.h"
#include "rtp_llm/cpp/devices/DeviceFactory.h"
#include "rtp_llm/cpp/core/allocator.h"
#include "rtp_llm/cpp/core/cpu_allocator.h"
#include "rtp_llm/cpp/devices/utils/DebugUtils.h"
#include <cstring>

namespace rtp_llm {

void act_convert_fp16_to_float(const __fp16* input, float* output, int length) {
    int d = 0;
    for (; d <= length - 32; d += 32) {
        // Load 32 fp16 values
        float16x8_t fp16_vec0 = vld1q_f16(&input[d]);
        float16x8_t fp16_vec1 = vld1q_f16(&input[d + 8]);
        float16x8_t fp16_vec2 = vld1q_f16(&input[d + 16]);
        float16x8_t fp16_vec3 = vld1q_f16(&input[d + 24]);

        // Convert to float32
        float32x4_t float_vec0_low  = vcvt_f32_f16(vget_low_f16(fp16_vec0));
        float32x4_t float_vec0_high = vcvt_f32_f16(vget_high_f16(fp16_vec0));
        float32x4_t float_vec1_low  = vcvt_f32_f16(vget_low_f16(fp16_vec1));
        float32x4_t float_vec1_high = vcvt_f32_f16(vget_high_f16(fp16_vec1));
        float32x4_t float_vec2_low  = vcvt_f32_f16(vget_low_f16(fp16_vec2));
        float32x4_t float_vec2_high = vcvt_f32_f16(vget_high_f16(fp16_vec2));
        float32x4_t float_vec3_low  = vcvt_f32_f16(vget_low_f16(fp16_vec3));
        float32x4_t float_vec3_high = vcvt_f32_f16(vget_high_f16(fp16_vec3));

        // Store results
        vst1q_f32(&output[d], float_vec0_low);
        vst1q_f32(&output[d + 4], float_vec0_high);
        vst1q_f32(&output[d + 8], float_vec1_low);
        vst1q_f32(&output[d + 12], float_vec1_high);
        vst1q_f32(&output[d + 16], float_vec2_low);
        vst1q_f32(&output[d + 20], float_vec2_high);
        vst1q_f32(&output[d + 24], float_vec3_low);
        vst1q_f32(&output[d + 28], float_vec3_high);
    }
    for (; d < length; ++d) {
        output[d] = static_cast<float>(input[d]);
    }
}

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
    } else if (params.atype == ActivationType::Identity) {
        activationFunction = arm_compute ::ActivationLayerInfo::ActivationFunction::IDENTITY;
    } else {
        throw OpException(OpErrorType::ERROR_UNIMPLEMENTED);
    }

    void* gate = nullptr;

    void* bias = nullptr;
    if (params.bias) {  // add bias before activation
        bias = params.bias.value().get().data();

        printBufferData(params.bias.value().get(), "ffn activation gate");
        if (states->type() == DataType::TYPE_FP16) {
            for (size_t i = 0; i < m; i++) {
                for (size_t j = 0; j < n; j++) {
                    *(__fp16*)(states->dataWithOffset(i * n + j)) += ((__fp16*)bias)[i * n + j];
                }
            }
        } else if (states->type() == DataType::TYPE_FP32) {
            if (params.bias.value().get().type() == DataType::TYPE_FP32) {
                for (size_t i = 0; i < m; i++) {
                    for (size_t j = 0; j < n; j++) {
                        *(float*)(states->dataWithOffset(i * n + j)) += ((float*)bias)[i * n + j];
                    }
                }
            } else if (params.bias.value().get().type() == DataType::TYPE_FP16) {
                float* bias_converted = new float[n];
                act_convert_fp16_to_float((__fp16*)bias, bias_converted, n);
                for (size_t i = 0; i < m; i++) {
                    for (size_t j = 0; j < n; j++) {
                        *(float*)(states->dataWithOffset(i * n + j)) += ((float*)bias_converted)[j];
                    }
                }
                delete[] bias_converted;
            } else {
                throw std::runtime_error("FFN states and bias data type not supported");
            }
        } else {
            throw std::runtime_error("FFN bias data type not supported");
        }
    }

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

}  // namespace rtp_llm
