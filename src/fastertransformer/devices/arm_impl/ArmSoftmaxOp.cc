#include "src/fastertransformer/devices/arm_impl/ArmDevice.h"
#include "src/fastertransformer/devices/DeviceFactory.h"
#include "src/fastertransformer/core/allocator.h"
#include "src/fastertransformer/core/cpu_allocator.h"
#include <algorithm>

namespace fastertransformer {

inline void scale_array(float *data, int n, float scale) {
    int i = 0;
    float32x4_t scale_vec = vdupq_n_f32(scale);
    
    for (; i <= n - 16; i += 16) {
        float32x4_t vec1 = vld1q_f32(data+i);
        float32x4_t vec2 = vld1q_f32(data+i+4);
        float32x4_t vec3 = vld1q_f32(data+i+8);
        float32x4_t vec4 = vld1q_f32(data+i+12);

        vec1 = vmulq_f32(vec1, scale_vec);
        vec2 = vmulq_f32(vec2, scale_vec);
        vec3 = vmulq_f32(vec3, scale_vec);
        vec4 = vmulq_f32(vec4, scale_vec);

        vst1q_f32(data+i, vec1);
        vst1q_f32(data+i+4, vec2);
        vst1q_f32(data+i+8, vec3);
        vst1q_f32(data+i+12, vec4);
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

inline float32x4_t vtaylor_polyq_f32(float32x4_t x,
                                     const std::array<float32x4_t, 8>& coeffs) {
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
    static const float32x4_t CONST_LN2 = vdupq_n_f32(0.6931471805f);  // ln(2)
    static const float32x4_t CONST_INV_LN2 =
            vdupq_n_f32(1.4426950408f);  // 1/ln(2)
    static const float32x4_t CONST_INF =
            vdupq_n_f32(std::numeric_limits<float>::infinity());
    static const float32x4_t CONST_MAX_INPUT = vdupq_n_f32(88.7f);
    static const float32x4_t CONST_0 = vdupq_n_f32(0.f);
    static const int32x4_t CONST_NEGATIVE_126 = vdupq_n_s32(-126);

    // Perform range reduction [-log(2),log(2)]
    int32x4_t m = vcvtq_s32_f32(vmulq_f32(x, CONST_INV_LN2)); 
    float32x4_t val = vmlsq_f32(x, vcvtq_f32_s32(m), CONST_LN2);  

    // Polynomial Approximation
    float32x4_t poly = vtaylor_polyq_f32(val, exp_tab);

    // Reconstruct
    poly = vreinterpretq_f32_s32(
            vqaddq_s32(vreinterpretq_s32_f32(poly), vqshlq_n_s32(m, 23)));
    poly = vbslq_f32(vcltq_s32(m, CONST_NEGATIVE_126), CONST_0,
                     poly);  // Handle underflow
    poly = vbslq_f32(vcgtq_f32(x, CONST_MAX_INPUT), CONST_INF,
                     poly);  // Handle overflow

    return poly;
}


float vMax(int n, const float* a) {
    float max = a[0];
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
    const float max_val = vMax(n, vector);
    const float32x4_t max_v = vdupq_n_f32(max_val);
    float reduce_sum = 0.0f;
    float32x4_t reduce_sum_v[4];
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        reduce_sum_v[i] = vdupq_n_f32(0.0f);
    }

    // Sub Max and Exp and ReduceSum
    for (; d <= n - 16; d += 16) {
        float32x4x4_t regs = vld1q_f32_x4(vector + d);
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            regs.val[i] = vexpq_f32(vsubq_f32(regs.val[i], max_v));
            reduce_sum_v[i] = vaddq_f32(reduce_sum_v[i], regs.val[i]);
        }
        vst1q_f32_x4(vector + d, regs);
    }
    for (; d < n; ++d) {
        float val = vector[d];
        val = std::exp(val - max_val);
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
    const float reduce_sum_mul = 1.0f / reduce_sum;
    const float32x4_t reduce_sum_mul_v = vdupq_n_f32(reduce_sum_mul);
    d = 0;
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
//intrinsic softmax

inline void context_mask_float(float *input, float *mask,int n){
    int i = 0;
    float32x4x4_t input_v;
    float32x4x4_t mask_v;
    float32x4_t ones_v = vdupq_n_f32(1.0f);
    float32x4_t coefficient_v = vdupq_n_f32(-10000.0f);
    for (; i <= n - 16; i += 16) {
        input_v = vld1q_f32_x4(input+i);
        mask_v = vld1q_f32_x4(mask+i);
    #pragma unroll
        for (int i = 0; i < 4; ++i) {
            mask_v.val[i]  = vmulq_f32(coefficient_v,vsubq_f32(ones_v , mask_v.val[i]));
            input_v.val[i] = vaddq_f32(input_v.val[i],mask_v.val[i]);
        }
    vst1q_f32_x4(input + i , input_v);
    }
    for (; i < n; i++) {
        input[i] += (1.f - mask[i]) * -10000.0f;
    }
}

/* Fallback path to support inconsistent input type and mask type */
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
    int numThreads = omp_get_num_threads();
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
        if (type == DataType::TYPE_FP32 && mask_type == DataType::TYPE_FP32) {
#pragma omp parallel for num_threads(std::min((int)(input->shape()[0] *input->shape()[1]),(int)numThreads)) if((input->shape()[0] *input->shape()[1])>=4) collapse(2)
            for(int i = 0;i<input->shape()[0];i++){
                for(int j = 0;j<input->shape()[1];j++){
                        context_mask_float((float*)input->data()+i*input->shape()[1]*input->shape()[2]*input->shape()[3] + j*input->shape()[2]*input->shape()[3],\
                        (float*)params.mask.value().get().data()+i*input->shape()[2]*input->shape()[3],\
                        input->shape()[2]*input->shape()[3]);
                }
            }
        } else if (type == DataType::TYPE_FP32 && mask_type == DataType::TYPE_FP16) {
            context_mask<float, __fp16>(params.input, params.mask.value().get());
        } else if (type == DataType::TYPE_FP16) {
            context_mask<__fp16, __fp16>(params.input, params.mask.value().get());
        } else {
            throw std::runtime_error("Softmax data type is not supported");
        }
    }
    
    if(type == DataType::TYPE_FP32){
#pragma omp parallel for num_threads(std::min((int)(input->shape()[0] *input->shape()[1]),(int)numThreads)) if((input->shape()[0] *input->shape()[1])>=4 && input->shape()[3]>=16) collapse(2)
        for(int i = 0;i<input->shape()[0];i++){
            for(int j = 0;j<input->shape()[1];j++){
                for(int k = 0;k<input->shape()[2];k++){
                    scale_array((float*)input->data()+i*input->shape()[1]*input->shape()[2]*input->shape()[3]+j*input->shape()[2]*input->shape()[3]+k*input->shape()[3],\
                    input->shape()[3],\
                    (float)params.scale);
                    vSoftmax(input->shape()[3],\
                    (float*)input->data()+i*input->shape()[1]*input->shape()[2]*input->shape()[3]+j*input->shape()[2]*input->shape()[3]+k*input->shape()[3]
                    );
                }
            }
        }
    }
    else throw std::runtime_error("Softmax data type is not supported");
    return std::move(input);
}
}  // namespace fastertransformer
