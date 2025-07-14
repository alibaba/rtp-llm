#include "rtp_llm/cpp/devices/arm_impl/ArmDevice.h"
#include "rtp_llm/cpp/devices/DeviceFactory.h"
#include "rtp_llm/cpp/core/allocator.h"
#include "rtp_llm/cpp/core/cpu_allocator.h"
#include <algorithm>

namespace rtp_llm {

inline void scale_array(float* data, int n, float scale) {
    int         i         = 0;
    float32x4_t scale_vec = vdupq_n_f32(scale);

    for (; i <= n - 16; i += 16) {
        float32x4_t vec1 = vld1q_f32(data + i);
        float32x4_t vec2 = vld1q_f32(data + i + 4);
        float32x4_t vec3 = vld1q_f32(data + i + 8);
        float32x4_t vec4 = vld1q_f32(data + i + 12);

        vec1 = vmulq_f32(vec1, scale_vec);
        vec2 = vmulq_f32(vec2, scale_vec);
        vec3 = vmulq_f32(vec3, scale_vec);
        vec4 = vmulq_f32(vec4, scale_vec);

        vst1q_f32(data + i, vec1);
        vst1q_f32(data + i + 4, vec2);
        vst1q_f32(data + i + 8, vec3);
        vst1q_f32(data + i + 12, vec4);
    }
    for (; i < n; i++) {
        data[i] *= scale;
    }
}

const std::array<float32x4_t, 8> exp_tab = {{
    vdupq_n_f32(1.f),
    vdupq_n_f32(0.0416598916054f),
    vdupq_n_f32(0.500000596046f),
    vdupq_n_f32(0.0014122662833f),
    vdupq_n_f32(1.00000011921f),
    vdupq_n_f32(0.00833693705499f),
    vdupq_n_f32(0.166665703058f),
    vdupq_n_f32(0.000195780929062f),
}};

inline float32x4_t vtaylor_polyq_f32(float32x4_t x, const std::array<float32x4_t, 8>& coeffs) {
    float32x4_t A   = vmlaq_f32(coeffs[0], coeffs[4], x);
    float32x4_t B   = vmlaq_f32(coeffs[2], coeffs[6], x);
    float32x4_t C   = vmlaq_f32(coeffs[1], coeffs[5], x);
    float32x4_t D   = vmlaq_f32(coeffs[3], coeffs[7], x);
    float32x4_t x2  = vmulq_f32(x, x);
    float32x4_t x4  = vmulq_f32(x2, x2);
    float32x4_t res = vmlaq_f32(vmlaq_f32(A, B, x2), vmlaq_f32(C, D, x2), x4);
    return res;
}

inline float32x4_t vexpq_f32(float32x4_t x) {
    static const float32x4_t CONST_LN2          = vdupq_n_f32(0.6931471805f);  // ln(2)
    static const float32x4_t CONST_INV_LN2      = vdupq_n_f32(1.4426950408f);  // 1/ln(2)
    static const float32x4_t CONST_INF          = vdupq_n_f32(std::numeric_limits<float>::infinity());
    static const float32x4_t CONST_MAX_INPUT    = vdupq_n_f32(88.7f);
    static const float32x4_t CONST_0            = vdupq_n_f32(0.f);
    static const int32x4_t   CONST_NEGATIVE_126 = vdupq_n_s32(-126);

    // Perform range reduction [-log(2),log(2)]
    int32x4_t   m   = vcvtq_s32_f32(vmulq_f32(x, CONST_INV_LN2));
    float32x4_t val = vmlsq_f32(x, vcvtq_f32_s32(m), CONST_LN2);

    // Polynomial Approximation
    float32x4_t poly = vtaylor_polyq_f32(val, exp_tab);

    // Reconstruct
    poly = vreinterpretq_f32_s32(vqaddq_s32(vreinterpretq_s32_f32(poly), vqshlq_n_s32(m, 23)));
    poly = vbslq_f32(vcltq_s32(m, CONST_NEGATIVE_126), CONST_0,
                     poly);  // Handle underflow
    poly = vbslq_f32(vcgtq_f32(x, CONST_MAX_INPUT), CONST_INF,
                     poly);  // Handle overflow

    return poly;
}

float vMax(int n, const float* a) {
    float         max = a[0];
    float32x4x4_t max_v;
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        max_v.val[i] = vdupq_n_f32(max);
    }
    int d = 0;
    for (; d <= n - 16; d += 16) {
        float32x4x4_t regs = vld1q_f32_x4(a + d);
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            max_v.val[i] = vmaxq_f32(max_v.val[i], regs.val[i]);
        }
    }
    for (; d < n; ++d) {
        max = std::max(max, a[d]);
    }
    max_v.val[0] = vmaxq_f32(max_v.val[0], max_v.val[1]);
    max_v.val[2] = vmaxq_f32(max_v.val[2], max_v.val[3]);
    max_v.val[0] = vmaxq_f32(max_v.val[0], max_v.val[2]);

#pragma unroll
    for (int i = 0; i < 4; ++i) {
        max = std::max(max, max_v.val[0][i]);
    }
    return max;
}

void vSoftmax(int n, float* vector) {
    int d = 0;
    // Find Max
    const float       max_val    = vMax(n, vector);
    const float32x4_t max_v      = vdupq_n_f32(max_val);
    float             reduce_sum = 0.0f;
    float32x4_t       reduce_sum_v[4];
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        reduce_sum_v[i] = vdupq_n_f32(0.0f);
    }

    // Sub Max and Exp and ReduceSum
    for (; d <= n - 16; d += 16) {
        float32x4x4_t regs = vld1q_f32_x4(vector + d);
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            regs.val[i]     = vexpq_f32(vsubq_f32(regs.val[i], max_v));
            reduce_sum_v[i] = vaddq_f32(reduce_sum_v[i], regs.val[i]);
        }
        vst1q_f32_x4(vector + d, regs);
    }
    for (; d < n; ++d) {
        float val = vector[d];
        val       = std::exp(val - max_val);
        reduce_sum += val;
        vector[d] = val;
    }
    reduce_sum_v[0] = vaddq_f32(reduce_sum_v[0], reduce_sum_v[1]);
    reduce_sum_v[2] = vaddq_f32(reduce_sum_v[2], reduce_sum_v[3]);
    reduce_sum_v[0] = vaddq_f32(reduce_sum_v[0], reduce_sum_v[2]);
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        reduce_sum += reduce_sum_v[0][i];
    }

    // Div ReduceSum
    const float       reduce_sum_mul   = 1.0f / reduce_sum;
    const float32x4_t reduce_sum_mul_v = vdupq_n_f32(reduce_sum_mul);
    d                                  = 0;
    for (; d <= n - 16; d += 16) {
        float32x4x4_t regs = vld1q_f32_x4(vector + d);
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            regs.val[i] = vmulq_f32(regs.val[i], reduce_sum_mul_v);
        }
        vst1q_f32_x4(vector + d, regs);
    }
    for (; d < n; ++d) {
        vector[d] = vector[d] * reduce_sum_mul;
    }
}
// intrinsic softmax

inline void context_mask_float(float* input, float* mask, int n) {
    int           i = 0;
    float32x4x4_t input_v;
    float32x4x4_t mask_v;
    float32x4_t   ones_v        = vdupq_n_f32(1.0f);
    float32x4_t   coefficient_v = vdupq_n_f32(-10000.0f);
    for (; i <= n - 16; i += 16) {
        input_v = vld1q_f32_x4(input + i);
        mask_v  = vld1q_f32_x4(mask + i);
#pragma unroll
        // for (int i = 0; i < 4; ++i) {
        //     mask_v.val[i]  = vmulq_f32(coefficient_v,vsubq_f32(ones_v , mask_v.val[i]));
        //     input_v.val[i] = vaddq_f32(input_v.val[i],mask_v.val[i]);
        // }
        for (int j = 0; j < 4; ++j) {
            mask_v.val[j]  = vmulq_f32(coefficient_v, vsubq_f32(ones_v, mask_v.val[j]));
            input_v.val[j] = vaddq_f32(input_v.val[j], mask_v.val[j]);
        }
        vst1q_f32_x4(input + i, input_v);
    }
    for (; i < n; i++) {
        input[i] += (1.f - mask[i]) * -10000.0f;
    }
}

// void vSoftmaxMask(int n, float* vector, const __fp16* mask_input, float scale) {
template<typename T>
void vSoftmaxMask(int n, float* vector, const T* mask_input, float scale) {
    static_assert(std::is_same<T, float>::value || std::is_same<T, float16_t>::value,
                  "mask_input must be either float or float16_t");
    // set vector based on mask

    int d = 0;
    for (; d <= n - 16; d += 16) {
        float32x4_t   mask_val  = vdupq_n_f32(-100000.0f);
        float32x4_t   one       = vdupq_n_f32(1.f);
        float32x4x4_t regs      = vld1q_f32_x4(vector + d);
        float32x4_t   scale_vec = vdupq_n_f32(scale);
        //// float32x4x4_t mask_v = vld1q_f32_x4(mask_input + d);
        //// Load FP16 mask and convert to FP32
        // float16x8x2_t mask_v_half = vld1q_f16_x2(mask_input + d);
        // float32x4x4_t mask_v = {
        //   vcvt_f32_f16(vget_low_f16(mask_v_half.val[0])),  // Convert low half of first 8 FP16
        //   vcvt_f32_f16(vget_high_f16(mask_v_half.val[0])), // Convert high half of first 8 FP16
        //   vcvt_f32_f16(vget_low_f16(mask_v_half.val[1])),  // Convert low half of second 8 FP16
        //   vcvt_f32_f16(vget_high_f16(mask_v_half.val[1]))  // Convert high half of second 8 FP16
        // };
        float32x4x4_t mask_v;

        if constexpr (std::is_same<T, float16_t>::value) {
            float16x8x2_t mask_v_half = vld1q_f16_x2(mask_input + d);
            mask_v                    = {
                vcvt_f32_f16(vget_low_f16(mask_v_half.val[0])),   // Convert low half of first 8 FP16
                vcvt_f32_f16(vget_high_f16(mask_v_half.val[0])),  // Convert high half of first 8 FP16
                vcvt_f32_f16(vget_low_f16(mask_v_half.val[1])),  // Convert low half of second 8 FP16
                vcvt_f32_f16(vget_high_f16(mask_v_half.val[1]))  // Convert high half of second 8 FP16
            };
        } else {
            mask_v = vld1q_f32_x4(mask_input + d);
        }

#pragma unroll
        for (int i = 0; i < 4; ++i) {
            mask_v.val[i] = vsubq_f32(one, mask_v.val[i]);
            mask_v.val[i] = vmulq_f32(mask_val, mask_v.val[i]);
            regs.val[i]   = vaddq_f32(mask_v.val[i], regs.val[i]);
            regs.val[i]   = vmulq_f32(scale_vec, regs.val[i]);
        }
        vst1q_f32_x4(vector + d, regs);
    }
    for (; d < n; ++d) {
        vector[d] = (vector[d] + (1.0f - (float)mask_input[d]) * (-100000.f)) * scale;
    }

    // Find Max
    const float       max_val    = vMax(n, vector);
    const float32x4_t max_v      = vdupq_n_f32(max_val);
    float             reduce_sum = 0.0f;
    float32x4_t       reduce_sum_v[4];
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        reduce_sum_v[i] = vdupq_n_f32(0.0f);
    }

    // Sub Max and Exp and ReduceSum
    d = 0;
    for (; d <= n - 16; d += 16) {
        float32x4x4_t regs = vld1q_f32_x4(vector + d);
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            regs.val[i]     = vexpq_f32(vsubq_f32(regs.val[i], max_v));
            reduce_sum_v[i] = vaddq_f32(reduce_sum_v[i], regs.val[i]);
        }
        vst1q_f32_x4(vector + d, regs);
    }
    for (; d < n; ++d) {
        float val = vector[d];
        val       = std::exp(val - max_val);
        reduce_sum += val;
        vector[d] = val;
    }
    reduce_sum_v[0] = vaddq_f32(reduce_sum_v[0], reduce_sum_v[1]);
    reduce_sum_v[2] = vaddq_f32(reduce_sum_v[2], reduce_sum_v[3]);
    reduce_sum_v[0] = vaddq_f32(reduce_sum_v[0], reduce_sum_v[2]);
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        reduce_sum += reduce_sum_v[0][i];
    }

    // Div ReduceSum
    const float       reduce_sum_mul   = 1.0f / (reduce_sum + 1e-12f);
    const float32x4_t reduce_sum_mul_v = vdupq_n_f32(reduce_sum_mul);
    d                                  = 0;
    for (; d <= n - 16; d += 16) {
        float32x4x4_t regs = vld1q_f32_x4(vector + d);
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            regs.val[i] = vmulq_f32(regs.val[i], reduce_sum_mul_v);
        }
        vst1q_f32_x4(vector + d, regs);
    }
    for (; d < n; ++d) {
        vector[d] = vector[d] * reduce_sum_mul;
    }
}

/* Fallback path to support inconsistent input type and mask type */
template<typename T, typename T_mask>
void context_mask(BufferPtr input, const Buffer& mask) {
    const int dim0   = input->shape()[0];
    const int dim1   = input->shape()[1];
    const int dim2   = input->shape()[2];
    const int dim3   = input->shape()[3];
    const int mask_m = mask.shape()[1];
    const int mask_n = mask.shape()[2];

    const int N = dim0 * dim1;
    parallel_for(N, [&](int tid) {
        int b = tid / dim1;
        for (int i = 0; i < dim2 * dim3; i++) {
            auto v = input->dataWithOffset(tid * dim2 * dim3 + i);
            // auto m = mask.dataWithOffset(b * dim2 * dim3 + i);
            auto m = mask.dataWithOffset((b * mask_m + i / dim3) * mask_n + (i % dim3));
            *(T*)v += (1.0f - *(T_mask*)m) * -10000.0f;
        }
    });
}

template<typename MaskType>
void processSoftmaxMask(const SoftmaxParams& params) {
    static_assert(std::is_same<MaskType, float>::value || std::is_same<MaskType, float16_t>::value,
                  "MaskType must be float or float16_t");
    auto input      = params.input;
    auto batch_size = input->shape()[0];
    auto num_heads  = input->shape()[1];
    auto q_length   = input->shape()[2];
    auto k_length   = input->shape()[3];
    auto mask_m     = params.mask.value().get().shape()[1];
    auto mask_n     = params.mask.value().get().shape()[2];
    /* Input has 4 dims and mask has 3 dims. The lowest 2 dims of both have identical value.
     * Mask dim[2] is identical to or bigger than Input dim[3].
     */

    float*    score = reinterpret_cast<float*>(input->data());
    MaskType* mask  = reinterpret_cast<MaskType*>(params.mask.value().get().data());

    parallel_for(batch_size * num_heads, [&](int idx) {
        int m = idx / num_heads;  // Batch index
        int n = idx % num_heads;  // Head index

        parallel_for(q_length, [&](int j) {
            size_t score_offset = m * q_length * num_heads * k_length + n * q_length * k_length + j * k_length;
            // size_t mask_offset = (m * k_length + j) * k_length;
            size_t mask_offset = (m * mask_m + j) * mask_n;

            vSoftmaxMask<MaskType>(k_length, score + score_offset, mask + mask_offset, params.scale);
        });
    });
}

BufferPtr ArmCpuDevice::softmax(const SoftmaxParams& params) {
    if (params.input == nullptr) {
        throw std::runtime_error("softmax input can not be nullptr");
    }
    auto        type       = params.input->type();
    int         numThreads = omp_get_num_threads();
    const auto& input      = params.input;
    auto   output = allocateBuffer({params.output_t == DataType::TYPE_INVALID ? params.input->type() : params.output_t,
                                  params.input->shape(),
                                  AllocationType::HOST});
    size_t type_size = params.input->typeSize();
    if ((type_size != 4) && (type_size != 2)) {
        throw std::runtime_error("Softmax input type is not supported");
    }

    if (params.mask.has_value()) {
        /* Apply mask. */
        auto mask_type = params.mask.value().get().type();
        ////if (type == DataType::TYPE_FP32 && mask_type == DataType::TYPE_FP32) {
        // if (params.mask.value().get().shape()[3] > input->shape()[3]) { // params.mask.shape maybe larger than
        // input.shape in BERT
        //     std::vector<size_t> mask_shape = {1, input->shape()[2], input->shape()[3]};
        //     auto mask = allocateBuffer({mask_type,
        //                           mask_shape,
        //                           AllocationType::HOST});
        //     if (type == DataType::TYPE_FP32 && mask_type == DataType::TYPE_FP32) {
        //         for (int i = 0; i < mask_shape[1]; i++) {
        //             std::memcpy((float*)mask->data() +
        //             i*mask_shape[2],(float*)params.mask.value().get().data()+i*params.mask.value().get().shape()[2],mask_shape[2]
        //             * sizeof(float));
        //         }
#pragma  // omp parallel for num_threads(std::min((int)(input->shape()[0] *input->shape()[1]),(int)numThreads))
         // if((input->shape()[0] *input->shape()[1])>=4) collapse(2)
         //     //for(int i = 0;i<input->shape()[0];i++){
         //     //    for(int j = 0;j<input->shape()[1];j++){
         //     // context_mask_float((float*)input->data()+i*input->shape()[1]*input->shape()[2]*input->shape()[3] +
         //     j*input->shape()[2]*input->shape()[3],
         //     //            (float*)params.mask.value().get().data()+i*input->shape()[2]*input->shape()[3],
         //     //            input->shape()[2]*input->shape()[3]);
         //     for(int i = 0;i<input->shape()[0];i++){
         //         for(int j = 0;j<input->shape()[1];j++){
         //                 context_mask_float((float*)input->data()+i*input->shape()[1]*input->shape()[2]*input->shape()[3]
         //                 + j*input->shape()[2]*input->shape()[3],
         //                 (float*)mask->data()+i*input->shape()[2]*input->shape()[3],
         //                 input->shape()[2]*input->shape()[3]);
         //             }
         //         }
         //     } else if (type == DataType::TYPE_FP32 && mask_type == DataType::TYPE_FP16) {
         //         for (int i = 0; i < mask_shape[1]; i++) {
         //             std::memcpy((__fp16*)mask->data() +
         //             i*mask_shape[2],(__fp16*)params.mask.value().get().data()+i*params.mask.value().get().shape()[2],mask_shape[2]
         //             * sizeof(__fp16));
         //         }
         //         context_mask<float, __fp16>(params.input, *mask);
         //     } else if (type == DataType::TYPE_FP16) {
         //         for (int i = 0; i < mask_shape[1]; i++) {
         //             std::memcpy((__fp16*)mask->data() +
         //             i*mask_shape[2],(__fp16*)params.mask.value().get().data()+i*params.mask.value().get().shape()[2],mask_shape[2]
         //             * sizeof(__fp16));
         //         }
         //         context_mask<__fp16, __fp16>(params.input, *mask);
         //     } else {
         //         throw std::runtime_error("Softmax data type is not supported");
         //     }
         ////} else if (type == DataType::TYPE_FP32 && mask_type == DataType::TYPE_FP16) {
         ////    //context_mask<float, __fp16>(params.input, params.mask.value().get());
         ////    auto batch_size = input->shape()[0];
         ////    auto num_heads = input->shape()[1];
         ////    auto q_length = input->shape()[2];
         ////    auto k_length = input->shape()[3];
         //} else {
         //    if (type == DataType::TYPE_FP32 && mask_type == DataType::TYPE_FP32) {
        // #pragma omp parallel for num_threads(std::min((int)(input->shape()[0] *input->shape()[1]),(int)numThreads))
        // if((input->shape()[0] *input->shape()[1])>=4) collapse(2)
        //         for(int i = 0;i<input->shape()[0];i++){
        //             for(int j = 0;j<input->shape()[1];j++){
        //                     context_mask_float((float*)input->data()+i*input->shape()[1]*input->shape()[2]*input->shape()[3]
        //                     + j*input->shape()[2]*input->shape()[3],
        //                     (float*)params.mask.value().get().data()+i*input->shape()[2]*input->shape()[3],
        //                     input->shape()[2]*input->shape()[3]);
        //             }
        //         }
        //     } else if (type == DataType::TYPE_FP32 && mask_type == DataType::TYPE_FP16) {
        //         auto batch_size = input->shape()[0];
        //         auto num_heads = input->shape()[1];
        //         auto q_length = input->shape()[2];
        //         auto k_length = input->shape()[3];

        //    //float* score = (float*)input->data();
        //    //float16_t* mask = (float16_t*)params.mask.value().get().data();
        //    float* score = (float*)input->data();
        //    float16_t* mask = (float16_t*)params.mask.value().get().data();

        //    //parallel_for(batch_size * num_heads, [&](int idx) {
        //    //    int m = idx / num_heads;       // Batch index
        //    //    int n = idx % num_heads;       // Head index
        //    parallel_for(batch_size * num_heads, [&](int idx) {
        //            int m = idx / num_heads;       // Batch index
        //            int n = idx % num_heads;       // Head index

        //        //parallel_for(q_length, [&](int j) {
        //        //    size_t score_offset = m * q_length * num_heads * k_length + n * q_length * k_length + j *
        //        k_length;
        //        //    size_t mask_offset = (m * k_length + j) * k_length;
        //        parallel_for(q_length, [&](int j) {
        //                size_t score_offset = m * q_length * num_heads * k_length + n * q_length * k_length + j *
        //                k_length; size_t mask_offset = (m * k_length + j) * k_length;

        //            //vSoftmaxMask(k_length, score + score_offset, mask + mask_offset, params.scale);
        //            vSoftmaxMask(k_length, score + score_offset, mask + mask_offset, params.scale);
        //        });
        //    });
        if (type == DataType::TYPE_FP32 && mask_type == DataType::TYPE_FP32) {
            processSoftmaxMask<float>(params);
            return input;
        } else if (type == DataType::TYPE_FP32 && mask_type == DataType::TYPE_FP16) {
            processSoftmaxMask<float16_t>(params);
            return input;
        } else if (type == DataType::TYPE_FP16) {
            context_mask<__fp16, __fp16>(params.input, params.mask.value().get());
        } else {
            throw std::runtime_error("Softmax data type is not supported");
        }
        //}
    }

    if (type == DataType::TYPE_FP32) {
#pragma omp parallel for num_threads(std::min((int)(input->shape()[0] * input->shape()[1]),                            \
                                                  (int)numThreads)) if ((input->shape()[0] * input->shape()[1]) >= 4   \
                                                                            && input->shape()[3] >= 16) collapse(2)
        for (int i = 0; i < input->shape()[0]; i++) {
            for (int j = 0; j < input->shape()[1]; j++) {
                for (int k = 0; k < input->shape()[2]; k++) {
                    scale_array((float*)input->data() + i * input->shape()[1] * input->shape()[2] * input->shape()[3]
                                    + j * input->shape()[2] * input->shape()[3] + k * input->shape()[3],
                                input->shape()[3],
                                (float)params.scale);
                    vSoftmax(input->shape()[3],
                             (float*)input->data() + i * input->shape()[1] * input->shape()[2] * input->shape()[3]
                                 + j * input->shape()[2] * input->shape()[3] + k * input->shape()[3]);
                }
            }
        }
    } else
        throw std::runtime_error("Softmax data type is not supported");
    return std::move(input);
}
}  // namespace rtp_llm
