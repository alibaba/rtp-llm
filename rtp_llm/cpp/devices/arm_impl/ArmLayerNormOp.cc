#include "rtp_llm/cpp/devices/arm_impl/ArmDevice.h"
#include "rtp_llm/cpp/devices/DeviceFactory.h"
#include "rtp_llm/cpp/core/allocator.h"
#include "rtp_llm/cpp/core/cpu_allocator.h"
#include "rtp_llm/cpp/devices/utils/DebugUtils.h"
#include <cstring>
#include <arm_neon.h>
#include <algorithm>  //std::all_of

namespace rtp_llm {

template<typename T>
void add_residual_bias(void* norm_out, const void* input, void* residual, int m, int n) {
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            ((T*)norm_out)[i * n + j] = ((T*)input)[i * n + j] + ((T*)residual)[i * n + j];
        }
    }
}

void add_residual_bias_float(float* norm_out, float* input, float* residual, float* bias, int n) {
    int d = 0;
    for (; d <= n - 16; d += 16) {
        float32x4x4_t regs = vld1q_f32_x4(input + d);
        float32x4x4_t regs_residual_bias;
        if (residual) {
            regs_residual_bias = vld1q_f32_x4(residual + d);
#pragma unroll
            for (int i = 0; i < 4; ++i) {
                regs.val[i] = vaddq_f32(regs.val[i], regs_residual_bias.val[i]);
            }
        }
        if (bias) {
            regs_residual_bias = vld1q_f32_x4(bias + d);
#pragma unroll
            for (int i = 0; i < 4; ++i) {
                regs.val[i] = vaddq_f32(regs.val[i], regs_residual_bias.val[i]);
            }
        }
        vst1q_f32_x4(norm_out + d, regs);
    }
    for (; d < n; ++d) {
        float val = input[d];
        if (residual)
            val += residual[d];
        if (bias)
            val += bias[d];
        norm_out[d] = val;
    }
}

void add_residual_bias_fp16(__fp16* norm_out, __fp16* input, __fp16* residual, __fp16* bias, int n) {
    int d = 0;
    for (; d <= n - 32; d += 32) {
        float16x8x4_t regs = vld1q_f16_x4(input + d);
        float16x8x4_t regs_residual_bias;
        if (residual) {
            regs_residual_bias = vld1q_f16_x4(residual + d);
#pragma unroll
            for (int i = 0; i < 4; ++i) {
                regs.val[i] = vaddq_f16(regs.val[i], regs_residual_bias.val[i]);
            }
        }
        if (bias) {
            regs_residual_bias = vld1q_f16_x4(bias + d);
#pragma unroll
            for (int i = 0; i < 4; ++i) {
                regs.val[i] = vaddq_f16(regs.val[i], regs_residual_bias.val[i]);
            }
        }
        vst1q_f16_x4(norm_out + d, regs);
    }
    for (; d < n; ++d) {
        float val = input[d];
        if (residual)
            val += residual[d];
        if (bias)
            val += bias[d];
        norm_out[d] = val;
    }
}

void convert_fp16_to_float(const __fp16* input, float* output, int length) {
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

void convert_float_to_fp16(const float* input, __fp16* output, int length) {
    int d = 0;
    for (; d <= length - 32; d += 32) {
        float32x4x4_t vec_float_low  = vld1q_f32_x4(input + d);
        float32x4x4_t vec_float_high = vld1q_f32_x4(input + d + 16);

        float16x4_t vec_fp16_low1  = vcvt_f16_f32(vec_float_low.val[0]);
        float16x4_t vec_fp16_high1 = vcvt_f16_f32(vec_float_low.val[1]);
        float16x4_t vec_fp16_low2  = vcvt_f16_f32(vec_float_low.val[2]);
        float16x4_t vec_fp16_high2 = vcvt_f16_f32(vec_float_low.val[3]);
        float16x4_t vec_fp16_low3  = vcvt_f16_f32(vec_float_high.val[0]);
        float16x4_t vec_fp16_high3 = vcvt_f16_f32(vec_float_high.val[1]);
        float16x4_t vec_fp16_low4  = vcvt_f16_f32(vec_float_high.val[2]);
        float16x4_t vec_fp16_high4 = vcvt_f16_f32(vec_float_high.val[3]);

        float16x8_t result_low1  = vcombine_f16(vec_fp16_low1, vec_fp16_high1);
        float16x8_t result_high1 = vcombine_f16(vec_fp16_low2, vec_fp16_high2);
        float16x8_t result_low2  = vcombine_f16(vec_fp16_low3, vec_fp16_high3);
        float16x8_t result_high2 = vcombine_f16(vec_fp16_low4, vec_fp16_high4);

        vst1q_f16(output + d, result_low1);
        vst1q_f16(output + d + 8, result_high1);
        vst1q_f16(output + d + 16, result_low2);
        vst1q_f16(output + d + 24, result_high2);
    }
    for (; d < length; ++d) {
        output[d] = static_cast<__fp16>(input[d]);
    }
}
void RMSNorm_isoutput(int          n,
                      float*       before_norm_output,
                      float*       input,
                      float*       norm_out,
                      const float* gamma,
                      const float* beta,
                      float*       residual,
                      float*       bias,
                      const double eps) {
    float32x4_t square_sum_v[4];
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        square_sum_v[i] = vdupq_n_f32(0.0f);
    }

    int d = 0;
    for (; d <= n - 16; d += 16) {
        float32x4x4_t regs = vld1q_f32_x4(input + d);
        float32x4x4_t regs_residual_bias;
        if (residual) {
            regs_residual_bias = vld1q_f32_x4(residual + d);
#pragma unroll
            for (int i = 0; i < 4; ++i) {
                regs.val[i] = vaddq_f32(regs.val[i], regs_residual_bias.val[i]);
            }
        }
        if (bias) {
            regs_residual_bias = vld1q_f32_x4(bias + d);
#pragma unroll
            for (int i = 0; i < 4; ++i) {
                regs.val[i] = vaddq_f32(regs.val[i], regs_residual_bias.val[i]);
            }
        }
        if (before_norm_output && before_norm_output != norm_out)
            vst1q_f32_x4(before_norm_output + d, regs);
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            // add_bias_residual
            square_sum_v[i] = vaddq_f32(square_sum_v[i], vmulq_f32(regs.val[i], regs.val[i]));
        }
    }

    float32_t square_sum = 0.0f;

    square_sum_v[0] = vaddq_f32(square_sum_v[0], square_sum_v[1]);
    square_sum_v[2] = vaddq_f32(square_sum_v[2], square_sum_v[3]);
    square_sum_v[0] = vaddq_f32(square_sum_v[0], square_sum_v[2]);
    for (; d < n; ++d) {
        float val = input[d];
        if (residual)
            val += residual[d];
        if (bias)
            val += bias[d];
        if (before_norm_output && before_norm_output != norm_out)
            before_norm_output[d] = val;
        square_sum += val * val;
    }
    //
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        square_sum += square_sum_v[0][i];
    }

    float rms         = square_sum / n;
    rms               = 1.0f / std::sqrt(rms + eps);
    float32x4_t rms_v = vdupq_n_f32(rms);
    // normalization
    d = 0;
    float32x4x4_t input_v;
    float32x4x4_t gamma_v;
    float32x4x4_t beta_v;
    for (; d <= n - 16; d += 16) {
        float32x4x4_t Residual;
        float32x4x4_t Bias;
        input_v = vld1q_f32_x4(input + d);
        gamma_v = vld1q_f32_x4(gamma + d);
        if (residual)
            Residual = vld1q_f32_x4(residual + d);
        if (bias)
            Bias = vld1q_f32_x4(bias + d);
        if (beta)
            beta_v = vld1q_f32_x4(beta + d);
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            if (residual)
                input_v.val[i] = vaddq_f32(input_v.val[i], Residual.val[i]);
            if (bias)
                input_v.val[i] = vaddq_f32(input_v.val[i], Bias.val[i]);
            // input_v.val[i] = vmulq_f32(input_v.val[i],scale_v);
            input_v.val[i] = vmulq_f32(input_v.val[i], rms_v);
            input_v.val[i] = vmulq_f32(input_v.val[i], gamma_v.val[i]);
            if (beta)
                input_v.val[i] = vaddq_f32(input_v.val[i], beta_v.val[i]);
        }
        vst1q_f32_x4(norm_out + d, input_v);
    }
    float input_residual_bias;
    for (; d < n; ++d) {
        input_residual_bias = input[d];
        if (residual)
            input_residual_bias += residual[d];
        if (bias)
            input_residual_bias += bias[d];
        norm_out[d] = input_residual_bias * rms * gamma[d];
        if (beta)
            norm_out[d] = norm_out[d] + beta[d];
    }
}

void RMSNorm_Nogamma_isoutput(int          n,
                              float*       before_norm_output,
                              const float* input,
                              float*       norm_out,
                              const float* beta,
                              float*       residual,
                              float*       bias,
                              const double eps) {
    float32x4_t square_sum_v[4];
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        square_sum_v[i] = vdupq_n_f32(0.0f);
    }

    int d = 0;
    for (; d <= n - 16; d += 16) {
        float32x4x4_t regs = vld1q_f32_x4(input + d);
        float32x4x4_t regs_residual_bias;
        if (residual) {
            regs_residual_bias = vld1q_f32_x4(residual + d);
#pragma unroll
            for (int i = 0; i < 4; ++i) {
                regs.val[i] = vaddq_f32(regs.val[i], regs_residual_bias.val[i]);
            }
        }
        if (bias) {
            regs_residual_bias = vld1q_f32_x4(bias + d);
#pragma unroll
            for (int i = 0; i < 4; ++i) {
                regs.val[i] = vaddq_f32(regs.val[i], regs_residual_bias.val[i]);
            }
        }
        if (before_norm_output)
            vst1q_f32_x4(before_norm_output + d, regs);
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            square_sum_v[i] = vaddq_f32(square_sum_v[i], vmulq_f32(regs.val[i], regs.val[i]));
        }
    }
    float32_t square_sum = 0.0f;
    for (; d < n; ++d) {
        float val = input[d];
        if (residual)
            val += residual[d];
        if (bias)
            val += bias[d];
        if (before_norm_output)
            before_norm_output[d] = val;
        square_sum += val * val;
    }

    square_sum_v[0] = vaddq_f32(square_sum_v[0], square_sum_v[1]);
    square_sum_v[2] = vaddq_f32(square_sum_v[2], square_sum_v[3]);
    square_sum_v[0] = vaddq_f32(square_sum_v[0], square_sum_v[2]);
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        square_sum += square_sum_v[0][i];
    }

    float rms         = square_sum / n;
    rms               = 1.0f / std::sqrt(rms + eps);
    float32x4_t rms_v = vdupq_n_f32(rms);

    // normalization
    d = 0;
    float32x4x4_t input_v;
    float32x4x4_t beta_v;
    for (; d <= n - 16; d += 16) {
        float32x4x4_t Residual;
        float32x4x4_t Bias;
        input_v = vld1q_f32_x4(input + d);
        if (residual)
            Residual = vld1q_f32_x4(residual + d);
        if (bias)
            Bias = vld1q_f32_x4(bias + d);
        if (beta)
            beta_v = vld1q_f32_x4(beta + d);
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            if (residual)
                input_v.val[i] = vaddq_f32(input_v.val[i], Residual.val[i]);
            if (bias)
                input_v.val[i] = vaddq_f32(input_v.val[i], Bias.val[i]);
            input_v.val[i] = vmulq_f32(input_v.val[i], rms_v);
            if (beta)
                input_v.val[i] = vaddq_f32(input_v.val[i], beta_v.val[i]);
        }
        vst1q_f32_x4(norm_out + d, input_v);
    }
    float input_residual_bias;
    for (; d < n; ++d) {
        input_residual_bias = input[d];
        if (residual)
            input_residual_bias += residual[d];
        if (bias)
            input_residual_bias += bias[d];
        norm_out[d] = input_residual_bias * rms;
        if (beta)
            norm_out[d] = norm_out[d] + beta[d];
    }
}

void RMSNorm(int n, const float* input, float* norm_out, const float* gamma, const float* beta, const double eps) {
    float32x4_t square_sum_v[4];
    // float32x4_t scale_v = vdupq_n_f32(2.0f);
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        square_sum_v[i] = vdupq_n_f32(0.0f);
    }

    int d = 0;
    for (; d <= n - 16; d += 16) {
        float32x4x4_t regs = vld1q_f32_x4(input + d);
#pragma unroll
        for (int i = 0; i < 4; ++i) {

            // regs.val[i] = vmulq_f32(regs.val[i],scale_v);
            square_sum_v[i] = vaddq_f32(square_sum_v[i], vmulq_f32(regs.val[i], regs.val[i]));
        }
    }
    float32_t square_sum = 0.0f;
    square_sum_v[0]      = vaddq_f32(square_sum_v[0], square_sum_v[1]);
    square_sum_v[2]      = vaddq_f32(square_sum_v[2], square_sum_v[3]);
    square_sum_v[0]      = vaddq_f32(square_sum_v[0], square_sum_v[2]);
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        square_sum += square_sum_v[0][i];
    }

    for (; d < n; ++d) {
        square_sum += input[d] * input[d];
    }

    float rms         = square_sum / n;
    rms               = 1.0f / std::sqrt(rms + eps);
    float32x4_t rms_v = vdupq_n_f32(rms);

    // normalization
    d = 0;
    float32x4x4_t input_v;
    float32x4x4_t gamma_v;
    float32x4x4_t beta_v;
    for (; d <= n - 16; d += 16) {
        input_v = vld1q_f32_x4(input + d);
        gamma_v = vld1q_f32_x4(gamma + d);
        if (beta)
            beta_v = vld1q_f32_x4(beta + d);
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            // input_v.val[i] = vmulq_f32(input_v.val[i],scale_v);
            input_v.val[i] = vmulq_f32(input_v.val[i], rms_v);
            input_v.val[i] = vmulq_f32(input_v.val[i], gamma_v.val[i]);
            if (beta)
                input_v.val[i] = vaddq_f32(input_v.val[i], beta_v.val[i]);
        }
        vst1q_f32_x4(norm_out + d, input_v);
    }
    for (; d < n; ++d) {
        norm_out[d] = input[d] * rms * gamma[d];
        if (beta)
            norm_out[d] = norm_out[d] + beta[d];
    }
}

void RMSNorm_Nogamma(int n, const float* input, float* norm_out, const float* beta, const double eps) {

    float32x4_t square_sum_v[4];
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        square_sum_v[i] = vdupq_n_f32(0.0f);
    }

    int d = 0;
    for (; d <= n - 16; d += 16) {
        float32x4x4_t regs = vld1q_f32_x4(input + d);
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            square_sum_v[i] = vaddq_f32(square_sum_v[i], vmulq_f32(regs.val[i], regs.val[i]));
        }
    }
    float32_t square_sum = 0.0f;
    for (; d < n; ++d) {
        float val = input[d];
        square_sum += val * val;
    }

    square_sum_v[0] = vaddq_f32(square_sum_v[0], square_sum_v[1]);
    square_sum_v[2] = vaddq_f32(square_sum_v[2], square_sum_v[3]);
    square_sum_v[0] = vaddq_f32(square_sum_v[0], square_sum_v[2]);
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        square_sum += square_sum_v[0][i];
    }

    float rms         = square_sum / n;
    rms               = 1.0f / std::sqrt(rms + eps);
    float32x4_t rms_v = vdupq_n_f32(rms);

    // normalization
    d = 0;
    float32x4x4_t beta_v;
    for (; d <= n - 16; d += 16) {
        float32x4x4_t input_v = vld1q_f32_x4(input + d);
        if (beta)
            beta_v = vld1q_f32_x4(beta + d);
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            input_v.val[i] = vmulq_f32(input_v.val[i], rms_v);
            if (beta)
                input_v.val[i] = vaddq_f32(input_v.val[i], beta_v.val[i]);
        }
        vst1q_f32_x4(norm_out + d, input_v);
    }
    for (; d < n; ++d) {
        norm_out[d] = input[d] * rms;
        if (beta)
            norm_out[d] = norm_out[d] + beta[d];
    }
}

void layerNorm(int n, const float* input, float* norm_out, const float* gamma, const float* beta, const double eps) {
    float32x4_t sum_v[4];
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        sum_v[i] = vdupq_n_f32(0.0f);
    }
    float32x4_t square_sum_v[4];
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        square_sum_v[i] = vdupq_n_f32(0.0f);
    }

    int d = 0;
    for (; d <= n - 16; d += 16) {
        float32x4x4_t regs = vld1q_f32_x4(input + d);
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            sum_v[i]        = vaddq_f32(sum_v[i], regs.val[i]);
            square_sum_v[i] = vaddq_f32(square_sum_v[i], vmulq_f32(regs.val[i], regs.val[i]));
        }
    }
    float32_t sum        = 0.0f;
    float32_t square_sum = 0.0f;
    for (; d < n; ++d) {
        float val = input[d];
        sum += val;
        square_sum += val * val;
    }
    sum_v[0] = vaddq_f32(sum_v[0], sum_v[1]);
    sum_v[2] = vaddq_f32(sum_v[2], sum_v[3]);
    sum_v[0] = vaddq_f32(sum_v[0], sum_v[2]);
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        sum += sum_v[0][i];
    }

    square_sum_v[0] = vaddq_f32(square_sum_v[0], square_sum_v[1]);
    square_sum_v[2] = vaddq_f32(square_sum_v[2], square_sum_v[3]);
    square_sum_v[0] = vaddq_f32(square_sum_v[0], square_sum_v[2]);
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        square_sum += square_sum_v[0][i];
    }

    float mean             = sum / n;
    float variance         = square_sum / n;
    variance               = 1.0f / std::sqrt(variance - mean * mean + eps);
    float32x4_t mean_v     = vdupq_n_f32(mean);
    float32x4_t variance_v = vdupq_n_f32(variance);

    // normalization
    d = 0;
    float32x4x4_t gamma_v;
    float32x4x4_t beta_v;
    for (; d <= n - 16; d += 16) {
        float32x4x4_t input_v = vld1q_f32_x4(input + d);
        if (gamma)
            gamma_v = vld1q_f32_x4(gamma + d);
        if (beta)
            beta_v = vld1q_f32_x4(beta + d);
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            input_v.val[i] = vsubq_f32(input_v.val[i], mean_v);
            if (gamma)
                input_v.val[i] = vmulq_f32(input_v.val[i], gamma_v.val[i]);
            input_v.val[i] = vmulq_f32(input_v.val[i], variance_v);
            if (beta)
                input_v.val[i] = vaddq_f32(input_v.val[i], beta_v.val[i]);
        }
        vst1q_f32_x4(norm_out + d, input_v);
    }
    for (; d < n; ++d) {
        if (gamma && beta)
            norm_out[d] = (input[d] - mean) * variance * gamma[d] + beta[d];  // with gamma and beta
        else if (gamma && !beta)
            norm_out[d] = (input[d] - mean) * gamma[d] * variance;
        else if (!gamma && !beta)
            norm_out[d] = (input[d] - mean) * variance;
        else
            norm_out[d] = (input[d] - mean) * gamma[d] * variance + beta[d];
    }
}

void layerNorm_Nogamma(int n, const float* input, float* norm_out, const float* beta, const double eps) {
    float32x4_t sum_v[4];
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        sum_v[i] = vdupq_n_f32(0.0f);
    }
    float32x4_t square_sum_v[4];
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        square_sum_v[i] = vdupq_n_f32(0.0f);
    }
    // #pragma omp parallel for
    int d = 0;
    for (; d <= n - 16; d += 16) {
        float32x4x4_t regs = vld1q_f32_x4(input + d);
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            sum_v[i]        = vaddq_f32(sum_v[i], regs.val[i]);
            square_sum_v[i] = vaddq_f32(square_sum_v[i], vmulq_f32(regs.val[i], regs.val[i]));
        }
    }
    float32_t sum        = 0.0f;
    float32_t square_sum = 0.0f;
    for (; d < n; ++d) {
        float val = input[d];
        sum += val;
        square_sum += val * val;
    }
    sum_v[0] = vaddq_f32(sum_v[0], sum_v[1]);
    sum_v[2] = vaddq_f32(sum_v[2], sum_v[3]);
    sum_v[0] = vaddq_f32(sum_v[0], sum_v[2]);
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        sum += sum_v[0][i];
    }

    square_sum_v[0] = vaddq_f32(square_sum_v[0], square_sum_v[1]);
    square_sum_v[2] = vaddq_f32(square_sum_v[2], square_sum_v[3]);
    square_sum_v[0] = vaddq_f32(square_sum_v[0], square_sum_v[2]);
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        square_sum += square_sum_v[0][i];
    }

    float mean             = sum / n;
    float variance         = square_sum / n;
    variance               = 1.0f / std::sqrt(variance - mean * mean + eps);
    float32x4_t mean_v     = vdupq_n_f32(mean);
    float32x4_t variance_v = vdupq_n_f32(variance);

    // normalization
    d = 0;
    float32x4x4_t beta_v;
    for (; d <= n - 16; d += 16) {
        float32x4x4_t input_v = vld1q_f32_x4(input + d);
        if (beta)
            beta_v = vld1q_f32_x4(beta + d);
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            input_v.val[i] = vsubq_f32(input_v.val[i], mean_v);
            input_v.val[i] = vmulq_f32(input_v.val[i], variance_v);
            if (beta)
                input_v.val[i] = vaddq_f32(input_v.val[i], beta_v.val[i]);
        }
        vst1q_f32_x4(norm_out + d, input_v);
    }
    for (; d < n; ++d) {
        if (beta)
            norm_out[d] = (input[d] - mean) * variance + beta[d];
        else
            norm_out[d] = (input[d] - mean) * variance;
    }
}

void layerNorm_isoutput(int          n,
                        const float* input,
                        float*       norm_out,
                        const float* gamma,
                        const float* beta,
                        float*       residual,
                        float*       bias,
                        const double eps) {
    float32x4_t sum_v[4];
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        sum_v[i] = vdupq_n_f32(0.0f);
    }
    float32x4_t square_sum_v[4];
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        square_sum_v[i] = vdupq_n_f32(0.0f);
    }

    int d = 0;
    for (; d <= n - 16; d += 16) {
        float32x4x4_t regs = vld1q_f32_x4(input + d);
        float32x4x4_t regs_residual_bias;

        if (residual) {
            regs_residual_bias = vld1q_f32_x4(residual + d);
#pragma unroll
            for (int i = 0; i < 4; ++i) {
                regs.val[i] = vaddq_f32(regs.val[i], regs_residual_bias.val[i]);
            }
        }
        if (bias) {
            regs_residual_bias = vld1q_f32_x4(bias + d);
#pragma unroll
            for (int i = 0; i < 4; ++i) {
                regs.val[i] = vaddq_f32(regs.val[i], regs_residual_bias.val[i]);
            }
        }
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            sum_v[i]        = vaddq_f32(sum_v[i], regs.val[i]);
            square_sum_v[i] = vaddq_f32(square_sum_v[i], vmulq_f32(regs.val[i], regs.val[i]));
        }
    }
    float32_t sum        = 0.0f;
    float32_t square_sum = 0.0f;
    for (; d < n; ++d) {
        float val = input[d];
        if (residual)
            val += residual[d];
        if (bias)
            val += bias[d];
        sum += val;
        square_sum += val * val;
    }
    sum_v[0] = vaddq_f32(sum_v[0], sum_v[1]);
    sum_v[2] = vaddq_f32(sum_v[2], sum_v[3]);
    sum_v[0] = vaddq_f32(sum_v[0], sum_v[2]);
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        sum += sum_v[0][i];
    }

    square_sum_v[0] = vaddq_f32(square_sum_v[0], square_sum_v[1]);
    square_sum_v[2] = vaddq_f32(square_sum_v[2], square_sum_v[3]);
    square_sum_v[0] = vaddq_f32(square_sum_v[0], square_sum_v[2]);
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        square_sum += square_sum_v[0][i];
    }

    float mean             = sum / n;
    float variance         = square_sum / n;
    variance               = 1.0f / std::sqrt(variance - mean * mean + eps);
    float32x4_t mean_v     = vdupq_n_f32(mean);
    float32x4_t variance_v = vdupq_n_f32(variance);

    // normalization
    d = 0;
    float32x4x4_t gamma_v;
    float32x4x4_t beta_v;
    for (; d <= n - 16; d += 16) {
        float32x4x4_t input_v = vld1q_f32_x4(input + d);
        float32x4x4_t Residual;
        float32x4x4_t Bias;
        if (residual)
            Residual = vld1q_f32_x4(residual + d);
        if (bias)
            Bias = vld1q_f32_x4(bias + d);
        gamma_v = vld1q_f32_x4(gamma + d);
        if (beta)
            beta_v = vld1q_f32_x4(beta + d);
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            input_v.val[i] = vaddq_f32(input_v.val[i], Residual.val[i]);
            input_v.val[i] = vaddq_f32(input_v.val[i], Bias.val[i]);
            input_v.val[i] = vsubq_f32(input_v.val[i], mean_v);
            input_v.val[i] = vmulq_f32(input_v.val[i], gamma_v.val[i]);
            input_v.val[i] = vmulq_f32(input_v.val[i], variance_v);
            if (beta)
                input_v.val[i] = vaddq_f32(input_v.val[i], beta_v.val[i]);
        }
        vst1q_f32_x4(norm_out + d, input_v);
    }
    float input_residual_bias;
    for (; d < n; ++d) {
        input_residual_bias = input[d];
        if (residual)
            input_residual_bias += residual[d];
        if (bias)
            input_residual_bias += bias[d];
        norm_out[d] = (input_residual_bias - mean) * gamma[d] * variance;
        if (beta)
            norm_out[d] += beta[d];
    }
}

void layerNorm_Nogamma_isoutput(
    int n, const float* input, float* norm_out, const float* beta, float* residual, float* bias, const double eps) {
    float32x4_t sum_v[4];
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        sum_v[i] = vdupq_n_f32(0.0f);
    }
    float32x4_t square_sum_v[4];
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        square_sum_v[i] = vdupq_n_f32(0.0f);
    }

    int d = 0;
    for (; d <= n - 16; d += 16) {
        float32x4x4_t regs = vld1q_f32_x4(input + d);
        float32x4x4_t regs_residual_bias;

        if (residual) {
            regs_residual_bias = vld1q_f32_x4(residual + d);
#pragma unroll
            for (int i = 0; i < 4; ++i) {
                regs.val[i] = vaddq_f32(regs.val[i], regs_residual_bias.val[i]);
            }
        }
        if (bias) {
            regs_residual_bias = vld1q_f32_x4(bias + d);
#pragma unroll
            for (int i = 0; i < 4; ++i) {
                regs.val[i] = vaddq_f32(regs.val[i], regs_residual_bias.val[i]);
            }
        }

#pragma unroll
        for (int i = 0; i < 4; ++i) {
            sum_v[i]        = vaddq_f32(sum_v[i], regs.val[i]);
            square_sum_v[i] = vaddq_f32(square_sum_v[i], vmulq_f32(regs.val[i], regs.val[i]));
        }
    }
    float32_t sum        = 0.0f;
    float32_t square_sum = 0.0f;
    for (; d < n; ++d) {
        float val = input[d];
        if (residual)
            val += residual[d];
        if (bias)
            val += bias[d];
        sum += val;
        square_sum += val * val;
    }
    sum_v[0] = vaddq_f32(sum_v[0], sum_v[1]);
    sum_v[2] = vaddq_f32(sum_v[2], sum_v[3]);
    sum_v[0] = vaddq_f32(sum_v[0], sum_v[2]);
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        sum += sum_v[0][i];
    }

    square_sum_v[0] = vaddq_f32(square_sum_v[0], square_sum_v[1]);
    square_sum_v[2] = vaddq_f32(square_sum_v[2], square_sum_v[3]);
    square_sum_v[0] = vaddq_f32(square_sum_v[0], square_sum_v[2]);
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        square_sum += square_sum_v[0][i];
    }

    float mean             = sum / n;
    float variance         = square_sum / n;
    variance               = 1.0f / std::sqrt(variance - mean * mean + eps);
    float32x4_t mean_v     = vdupq_n_f32(mean);
    float32x4_t variance_v = vdupq_n_f32(variance);

    // normalization
    d = 0;
    float32x4x4_t beta_v;
    for (; d <= n - 16; d += 16) {
        float32x4x4_t input_v = vld1q_f32_x4(input + d);
        float32x4x4_t Residual;
        float32x4x4_t Bias;
        if (residual)
            Residual = vld1q_f32_x4(residual + d);
        if (bias)
            Bias = vld1q_f32_x4(bias + d);
        if (beta)
            beta_v = vld1q_f32_x4(beta + d);
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            input_v.val[i] = vaddq_f32(input_v.val[i], Residual.val[i]);
            input_v.val[i] = vaddq_f32(input_v.val[i], Bias.val[i]);
            input_v.val[i] = vsubq_f32(input_v.val[i], mean_v);
            input_v.val[i] = vmulq_f32(input_v.val[i], variance_v);
            if (beta)
                input_v.val[i] = vaddq_f32(input_v.val[i], beta_v.val[i]);
        }
        vst1q_f32_x4(norm_out + d, input_v);
    }
    float input_residual_bias;
    for (; d < n; ++d) {
        input_residual_bias = input[d];
        if (residual)
            input_residual_bias += residual[d];
        if (bias)
            input_residual_bias += bias[d];
        norm_out[d] = (input_residual_bias - mean) * variance;
        if (beta)
            norm_out[d] += beta[d];
    }
}

void layerNorm_isoutput_unnormedout(int          n,
                                    const float* input,
                                    float*       norm_out,
                                    const float* gamma,
                                    const float* beta,
                                    float*       residual,
                                    float*       bias,
                                    float*       before_norm_output,
                                    const double eps) {
    float32x4_t sum_v[4];
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        sum_v[i] = vdupq_n_f32(0.0f);
    }
    float32x4_t square_sum_v[4];
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        square_sum_v[i] = vdupq_n_f32(0.0f);
    }

    int d = 0;
    for (; d <= n - 16; d += 16) {
        float32x4x4_t regs = vld1q_f32_x4(input + d);
        float32x4x4_t regs_residual_bias;

        if (residual) {
            regs_residual_bias = vld1q_f32_x4(residual + d);
#pragma unroll
            for (int i = 0; i < 4; ++i) {
                regs.val[i] = vaddq_f32(regs.val[i], regs_residual_bias.val[i]);
            }
        }
        if (bias) {
            regs_residual_bias = vld1q_f32_x4(bias + d);
#pragma unroll
            for (int i = 0; i < 4; ++i) {
                regs.val[i] = vaddq_f32(regs.val[i], regs_residual_bias.val[i]);
            }
        }
        vst1q_f32_x4(before_norm_output + d, regs);
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            sum_v[i]        = vaddq_f32(sum_v[i], regs.val[i]);
            square_sum_v[i] = vaddq_f32(square_sum_v[i], vmulq_f32(regs.val[i], regs.val[i]));
        }
    }
    float32_t sum        = 0.0f;
    float32_t square_sum = 0.0f;
    for (; d < n; ++d) {
        float val = input[d];
        if (residual)
            val += residual[d];
        if (bias)
            val += bias[d];
        before_norm_output[d] = val;
        sum += val;
        square_sum += val * val;
    }
    sum_v[0] = vaddq_f32(sum_v[0], sum_v[1]);
    sum_v[2] = vaddq_f32(sum_v[2], sum_v[3]);
    sum_v[0] = vaddq_f32(sum_v[0], sum_v[2]);
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        sum += sum_v[0][i];
    }

    square_sum_v[0] = vaddq_f32(square_sum_v[0], square_sum_v[1]);
    square_sum_v[2] = vaddq_f32(square_sum_v[2], square_sum_v[3]);
    square_sum_v[0] = vaddq_f32(square_sum_v[0], square_sum_v[2]);
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        square_sum += square_sum_v[0][i];
    }

    float mean             = sum / n;
    float variance         = square_sum / n;
    variance               = 1.0f / std::sqrt(variance - mean * mean + eps);
    float32x4_t mean_v     = vdupq_n_f32(mean);
    float32x4_t variance_v = vdupq_n_f32(variance);

    // normalization
    d = 0;
    float32x4x4_t gamma_v;
    float32x4x4_t beta_v;
    for (; d <= n - 16; d += 16) {
        float32x4x4_t input_v = vld1q_f32_x4(input + d);
        float32x4x4_t Residual;
        float32x4x4_t Bias;
        if (residual)
            Residual = vld1q_f32_x4(residual + d);
        if (bias)
            Bias = vld1q_f32_x4(bias + d);
        gamma_v = vld1q_f32_x4(gamma + d);
        if (beta)
            beta_v = vld1q_f32_x4(beta + d);
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            input_v.val[i] = vaddq_f32(input_v.val[i], Residual.val[i]);
            input_v.val[i] = vaddq_f32(input_v.val[i], Bias.val[i]);
            input_v.val[i] = vsubq_f32(input_v.val[i], mean_v);
            input_v.val[i] = vmulq_f32(input_v.val[i], gamma_v.val[i]);
            input_v.val[i] = vmulq_f32(input_v.val[i], variance_v);
            if (beta)
                input_v.val[i] = vaddq_f32(input_v.val[i], beta_v.val[i]);
        }
        vst1q_f32_x4(norm_out + d, input_v);
    }
    float input_residual_bias;
    for (; d < n; ++d) {
        input_residual_bias = input[d];
        if (residual)
            input_residual_bias += residual[d];
        if (bias)
            input_residual_bias += bias[d];
        norm_out[d] = (input_residual_bias - mean) * gamma[d] * variance;
        if (beta)
            norm_out[d] += beta[d];
    }
}

void layerNorm_Nogamma_isoutput_unnormedout(int          n,
                                            const float* input,
                                            float*       norm_out,
                                            const float* beta,
                                            float*       residual,
                                            float*       bias,
                                            float*       before_norm_output,
                                            const double eps) {
    float32x4_t sum_v[4];
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        sum_v[i] = vdupq_n_f32(0.0f);
    }
    float32x4_t square_sum_v[4];
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        square_sum_v[i] = vdupq_n_f32(0.0f);
    }

    int d = 0;
    for (; d <= n - 16; d += 16) {
        float32x4x4_t regs = vld1q_f32_x4(input + d);
        float32x4x4_t regs_residual_bias;

        if (residual) {
            regs_residual_bias = vld1q_f32_x4(residual + d);
#pragma unroll
            for (int i = 0; i < 4; ++i) {
                regs.val[i] = vaddq_f32(regs.val[i], regs_residual_bias.val[i]);
            }
        }
        if (bias) {
            regs_residual_bias = vld1q_f32_x4(bias + d);
#pragma unroll
            for (int i = 0; i < 4; ++i) {
                regs.val[i] = vaddq_f32(regs.val[i], regs_residual_bias.val[i]);
            }
        }
        vst1q_f32_x4(before_norm_output + d, regs);
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            sum_v[i]        = vaddq_f32(sum_v[i], regs.val[i]);
            square_sum_v[i] = vaddq_f32(square_sum_v[i], vmulq_f32(regs.val[i], regs.val[i]));
        }
    }
    float32_t sum        = 0.0f;
    float32_t square_sum = 0.0f;
    for (; d < n; ++d) {
        float val = input[d];
        if (residual)
            val += residual[d];
        if (bias)
            val += bias[d];
        before_norm_output[d] = val;
        sum += val;
        square_sum += val * val;
    }
    sum_v[0] = vaddq_f32(sum_v[0], sum_v[1]);
    sum_v[2] = vaddq_f32(sum_v[2], sum_v[3]);
    sum_v[0] = vaddq_f32(sum_v[0], sum_v[2]);
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        sum += sum_v[0][i];
    }

    square_sum_v[0] = vaddq_f32(square_sum_v[0], square_sum_v[1]);
    square_sum_v[2] = vaddq_f32(square_sum_v[2], square_sum_v[3]);
    square_sum_v[0] = vaddq_f32(square_sum_v[0], square_sum_v[2]);
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        square_sum += square_sum_v[0][i];
    }

    float mean             = sum / n;
    float variance         = square_sum / n;
    variance               = 1.0f / std::sqrt(variance - mean * mean + eps);
    float32x4_t mean_v     = vdupq_n_f32(mean);
    float32x4_t variance_v = vdupq_n_f32(variance);

    // normalization
    d = 0;
    float32x4x4_t beta_v;
    for (; d <= n - 16; d += 16) {
        float32x4x4_t input_v = vld1q_f32_x4(input + d);
        float32x4x4_t Residual;
        float32x4x4_t Bias;
        if (residual)
            Residual = vld1q_f32_x4(residual + d);
        if (bias)
            Bias = vld1q_f32_x4(bias + d);
        if (beta)
            beta_v = vld1q_f32_x4(beta + d);
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            input_v.val[i] = vaddq_f32(input_v.val[i], Residual.val[i]);
            input_v.val[i] = vaddq_f32(input_v.val[i], Bias.val[i]);
            input_v.val[i] = vsubq_f32(input_v.val[i], mean_v);
            input_v.val[i] = vmulq_f32(input_v.val[i], variance_v);
            if (beta)
                input_v.val[i] = vaddq_f32(input_v.val[i], beta_v.val[i]);
        }
        vst1q_f32_x4(norm_out + d, input_v);
    }
    float input_residual_bias;
    for (; d < n; ++d) {
        input_residual_bias = input[d];
        if (residual)
            input_residual_bias += residual[d];
        if (bias)
            input_residual_bias += bias[d];
        norm_out[d] = (input_residual_bias - mean) * variance;
        if (beta)
            norm_out[d] += beta[d];
    }
}

// FP16 will introduce unacceptable cumulative errors.
LayernormOutput ArmCpuDevice::layernorm(const LayernormParams& params) {
    BufferPtr   input       = params.input;
    BufferPtr   norm_output = input;
    const auto& weights     = params.norm_weight;  // before_norm_output is using for pre-norm,currently not implemented
    void*       gamma       = weights ? weights->get().gamma.get()->data() : nullptr;  //
    void*       beta        = (weights && weights->get().beta) ? weights->get().beta.get()->data() : nullptr;
    const auto  eps         = params.eps;

    void* before_norm_output = params.before_norm_output ? params.before_norm_output->data() : nullptr;
    void* residual           = params.residual1 ? params.residual1->get().data() : nullptr;
    void* bias               = params.bias.has_value() ? params.bias->get().data() : nullptr;
    bool  is_output          = (params.residual1.has_value() || params.bias.has_value());
    int   numThreads         = omp_get_num_threads();
    ;
    const auto norm_type = params.norm_type;
    int        m         = input->shape()[0];
    int        n         = input->shape()[1];
    const auto data_type = input->type();
    if (!params.is_inplace && params.qscheme == QScheme::NoQuantize) {
        norm_output = allocateBufferLike(*params.input);
    } else if (params.qscheme == Qint8PerToken) {
        throw OpException(OpErrorType::ERROR_UNIMPLEMENTED);
    }

    int convert_gamma = 0;
    int convert_beta  = 0;
    int convert_bias  = 0;
    if (data_type == DataType::TYPE_FP32) {
        if (gamma) {
            if (weights->get().gamma.get()->type() == DataType::TYPE_FP16) {
                convert_gamma = 1;
            }
        }
        if (beta) {
            if (weights->get().beta.get()->type() == DataType::TYPE_FP16) {
                convert_beta = 1;
            }
        }
        if (bias) {
            if (params.bias->get().type() == DataType::TYPE_FP16) {
                convert_bias = 1;
            }
        }
    }
    // for BERT
    // before_norm_output       params.return_norm_output        bias/residual exist
    // .  .  F
    // layernorm(input)->normed_output
    // F  .  T
    // layernorm(input+bias+residual)->normed_output
    // T  T  T
    // layernorm(input+bias+residual)->before_norm_output
    // layernorm(input+bias+residual)->normed_output
    // T  F  T
    // (input+bias+residual)->before_norm_output
    // layernorm(input+bias+residual)->normed_output
    if (norm_type == NormType::layernorm && (convert_gamma || convert_beta || convert_bias)) {
        float* gamma_converted = new float[n];
        if (gamma) {
            if (convert_gamma) {
                convert_fp16_to_float((__fp16*)gamma, gamma_converted, n);
            } else {
                for (int d = 0; d < n; ++d) {
                    gamma_converted[d] = static_cast<float>(((float*)gamma)[d]);
                }
            }
        }
        if (!is_output) {  //. .  F
            if (!gamma || std::all_of((float*)gamma_converted, (float*)gamma_converted + n, [](float value) {
                    return value == 1.0f;
                })) {
#pragma omp parallel for num_threads(std::min(m, numThreads)) if (m >= 2)
                for (int i = 0; i < m; i++) {
                    if (convert_beta) {
                        float* beta_converted = new float[n];
                        convert_fp16_to_float((__fp16*)beta, beta_converted, n);
                        layerNorm_Nogamma(
                            n, (float*)input->data() + i * n, (float*)norm_output->data() + i * n, beta_converted, eps);
                        delete[] beta_converted;
                    } else {
                        layerNorm_Nogamma(
                            n, (float*)input->data() + i * n, (float*)norm_output->data() + i * n, (float*)beta, eps);
                    }
                }
                return LayernormOutput({norm_output, params.before_norm_output});
            }  //(gamma =1,1......)OR (no gamma)
            else {
#pragma omp parallel for num_threads(std::min(m, numThreads)) if (m >= 2)
                for (int i = 0; i < m; i++) {
                    if (convert_beta) {
                        float* beta_converted = new float[n];
                        convert_fp16_to_float((__fp16*)beta, beta_converted, n);
                        layerNorm(n,
                                  (float*)input->data() + i * n,
                                  (float*)norm_output->data() + i * n,
                                  gamma_converted,
                                  beta_converted,
                                  eps);
                        delete[] beta_converted;
                    } else {
                        layerNorm(n,
                                  (float*)input->data() + i * n,
                                  (float*)norm_output->data() + i * n,
                                  gamma_converted,
                                  (float*)beta,
                                  eps);
                    }
                }
                return LayernormOutput({norm_output, params.before_norm_output});
            }
        } else if (!before_norm_output) {  // add bias residual   //F . T
            if (!gamma || std::all_of((float*)gamma_converted, (float*)gamma_converted + n, [](float value) {
                    return value == 1.0f;
                })) {
#pragma omp parallel for num_threads(std::min(m, numThreads)) if (m >= 2)
                for (int i = 0; i < m; i++) {
                    if (convert_beta && convert_bias) {
                        float* beta_converted = new float[n];
                        float* bias_converted = new float[n];
                        convert_fp16_to_float((__fp16*)beta, beta_converted, n);
                        convert_fp16_to_float((__fp16*)bias, bias_converted, n);
                        layerNorm_Nogamma_isoutput(n,
                                                   (float*)input->data() + i * n,
                                                   (float*)norm_output->data() + i * n,
                                                   beta_converted,
                                                   (residual != nullptr) ? (float*)residual + i * n : (float*)residual,
                                                   bias_converted,
                                                   eps);
                        delete[] beta_converted;
                        delete[] bias_converted;
                    } else {
                        layerNorm_Nogamma_isoutput(n,
                                                   (float*)input->data() + i * n,
                                                   (float*)norm_output->data() + i * n,
                                                   (float*)beta,
                                                   (residual != nullptr) ? (float*)residual + i * n : (float*)residual,
                                                   (float*)bias,
                                                   eps);
                    }
                }
                return LayernormOutput({norm_output, params.before_norm_output});
            }  //(gamma =1,1......)OR (no gamma)
            else {
#pragma omp parallel for num_threads(std::min(m, numThreads)) if (m >= 2)
                for (int i = 0; i < m; i++) {
                    if (convert_beta && convert_bias) {
                        float* beta_converted = new float[n];
                        float* bias_converted = new float[n];
                        convert_fp16_to_float((__fp16*)beta, beta_converted, n);
                        convert_fp16_to_float((__fp16*)bias, bias_converted, n);
                        layerNorm_isoutput(n,
                                           (float*)input->data() + i * n,
                                           (float*)norm_output->data() + i * n,
                                           gamma_converted,
                                           beta_converted,
                                           (residual != nullptr) ? ((float*)residual + i * n) : nullptr,
                                           bias_converted,
                                           eps);
                        delete[] beta_converted;
                        delete[] bias_converted;
                    } else {
                        layerNorm_isoutput(n,
                                           (float*)input->data() + i * n,
                                           (float*)norm_output->data() + i * n,
                                           (float*)gamma,
                                           (float*)beta,
                                           (residual != nullptr) ? ((float*)residual + i * n) : nullptr,
                                           (float*)bias,
                                           eps);
                    }
                }
                return LayernormOutput({norm_output, params.before_norm_output});
            }
        } else if (params.return_normed_output) {  // T  T  T
            if (!gamma || std::all_of((float*)gamma_converted, (float*)gamma_converted + n, [](float value) {
                    return value == 1.0f;
                })) {
#pragma omp parallel for num_threads(std::min(m, numThreads)) if (m >= 2)
                for (int i = 0; i < m; i++) {
                    if (convert_beta && convert_bias) {
                        float* beta_converted = new float[n];
                        float* bias_converted = new float[n];
                        convert_fp16_to_float((__fp16*)beta, beta_converted, n);
                        convert_fp16_to_float((__fp16*)bias, bias_converted, n);
                        layerNorm_Nogamma_isoutput(n,
                                                   (float*)input->data() + i * n,
                                                   (float*)norm_output->data() + i * n,
                                                   (float*)beta,
                                                   (residual != nullptr) ? ((float*)residual + i * n) :
                                                                           (float*)residual,
                                                   (float*)bias,
                                                   eps);
                        if (before_norm_output != norm_output->data())
                            std::memcpy((float*)before_norm_output + i * n,
                                        (float*)norm_output->data() + i * n,
                                        n * sizeof(float));
                        delete[] beta_converted;
                        delete[] bias_converted;
                    } else {
                        layerNorm_Nogamma_isoutput(n,
                                                   (float*)input->data() + i * n,
                                                   (float*)norm_output->data() + i * n,
                                                   (float*)beta,
                                                   (residual != nullptr) ? ((float*)residual + i * n) :
                                                                           (float*)residual,
                                                   (float*)bias,
                                                   eps);
                        if (before_norm_output != norm_output->data())
                            std::memcpy((float*)before_norm_output + i * n,
                                        (float*)norm_output->data() + i * n,
                                        n * sizeof(float));
                    }
                }
                return LayernormOutput({norm_output, params.before_norm_output});
            }  // gamma =1,1......  No gamma
            else {
#pragma omp parallel for num_threads(std::min(m, numThreads)) if (m >= 2)
                for (int i = 0; i < m; i++) {
                    if (convert_beta && convert_bias) {
                        float* beta_converted = new float[n];
                        float* bias_converted = new float[n];
                        convert_fp16_to_float((__fp16*)beta, beta_converted, n);
                        convert_fp16_to_float((__fp16*)bias, bias_converted, n);
                        layerNorm_isoutput(n,
                                           (float*)input->data() + i * n,
                                           (float*)norm_output->data() + i * n,
                                           gamma_converted,
                                           beta_converted,
                                           (residual != nullptr) ? ((float*)residual + i * n) : (float*)residual,
                                           bias_converted,
                                           eps);
                        if (before_norm_output != norm_output->data())
                            std::memcpy((float*)before_norm_output + i * n,
                                        (float*)norm_output->data() + i * n,
                                        n * sizeof(float));
                        delete[] beta_converted;
                        delete[] bias_converted;
                    } else {
                        layerNorm_isoutput(n,
                                           (float*)input->data() + i * n,
                                           (float*)norm_output->data() + i * n,
                                           (float*)gamma,
                                           (float*)beta,
                                           (residual != nullptr) ? ((float*)residual + i * n) : (float*)residual,
                                           (float*)bias,
                                           eps);
                        if (before_norm_output != norm_output->data())
                            std::memcpy((float*)before_norm_output + i * n,
                                        (float*)norm_output->data() + i * n,
                                        n * sizeof(float));
                    }
                }
                return LayernormOutput({norm_output, params.before_norm_output});
            }
        } else {  // T F T
            if (!gamma || std::all_of((float*)gamma_converted, (float*)gamma_converted + n, [](float value) {
                    return value == 1.0f;
                })) {
#pragma omp parallel for num_threads(std::min(m, numThreads)) if (m >= 2)
                for (int i = 0; i < m; i++) {
                    if (convert_beta && convert_bias) {
                        float* beta_converted = new float[n];
                        float* bias_converted = new float[n];
                        convert_fp16_to_float((__fp16*)beta, beta_converted, n);
                        convert_fp16_to_float((__fp16*)bias, bias_converted, n);
                        layerNorm_Nogamma_isoutput_unnormedout(n,
                                                               (float*)input->data() + i * n,
                                                               (float*)norm_output->data() + i * n,
                                                               beta_converted,
                                                               (residual != nullptr) ? ((float*)residual + i * n) :
                                                                                       (float*)residual,
                                                               bias_converted,
                                                               ((float*)before_norm_output + i * n),
                                                               eps);
                        delete[] beta_converted;
                        delete[] bias_converted;
                    } else {
                        layerNorm_Nogamma_isoutput_unnormedout(n,
                                                               (float*)input->data() + i * n,
                                                               (float*)norm_output->data() + i * n,
                                                               (float*)beta,
                                                               (residual != nullptr) ? ((float*)residual + i * n) :
                                                                                       (float*)residual,
                                                               (float*)bias,
                                                               ((float*)before_norm_output + i * n),
                                                               eps);
                    }
                }
                return LayernormOutput({norm_output, params.before_norm_output});
            }  // gamma =1,1......  No gamma
            else {
#pragma omp parallel for num_threads(std::min(m, numThreads)) if (m >= 2)
                for (int i = 0; i < m; i++) {
                    if (convert_beta && convert_bias) {
                        float* beta_converted = new float[n];
                        float* bias_converted = new float[n];
                        convert_fp16_to_float((__fp16*)beta, beta_converted, n);
                        convert_fp16_to_float((__fp16*)bias, bias_converted, n);
                        layerNorm_isoutput_unnormedout(n,
                                                       (float*)input->data() + i * n,
                                                       (float*)norm_output->data() + i * n,
                                                       gamma_converted,
                                                       beta_converted,
                                                       (residual != nullptr) ? ((float*)residual + i * n) :
                                                                               (float*)residual,
                                                       bias_converted,
                                                       ((float*)before_norm_output + i * n),
                                                       eps);
                        delete[] beta_converted;
                        delete[] bias_converted;
                    } else {
                        layerNorm_isoutput_unnormedout(n,
                                                       (float*)input->data() + i * n,
                                                       (float*)norm_output->data() + i * n,
                                                       (float*)gamma,
                                                       (float*)beta,
                                                       (residual != nullptr) ? ((float*)residual + i * n) :
                                                                               (float*)residual,
                                                       (float*)bias,
                                                       ((float*)before_norm_output + i * n),
                                                       eps);
                    }
                }
                return LayernormOutput({norm_output, params.before_norm_output});
            }
        }
    }

    // Due to the cumulative errors caused by using fp16 precision calculations, the fp16 input is first converted to
    // fp32 before using the fp32 kernel.
    if (norm_type == NormType::rmsnorm) {
        if (!weights.has_value()) {  // In this case, norm_output = input+residual
            if (data_type == DataType::TYPE_FP32) {
#pragma omp parallel for num_threads(std::min(m, numThreads)) if (m >= 2)
                for (int i = 0; i < m; i++) {
                    add_residual_bias_float((float*)norm_output->data() + i * n,
                                            (float*)input->data() + i * n,
                                            (bool)residual ? (float*)residual + i * n : nullptr,
                                            (float*)bias,
                                            n);
                }
            } else if (data_type == DataType::TYPE_FP16) {
#pragma omp parallel for num_threads(std::min(m, numThreads)) if (m >= 2)
                for (int i = 0; i < m; i++) {
                    add_residual_bias_fp16((__fp16*)norm_output->data() + i * n,
                                           (__fp16*)input->data() + i * n,
                                           (bool)residual ? (__fp16*)residual + i * n : nullptr,
                                           (__fp16*)bias,
                                           n);
                }
            } else {
                throw OpException(OpErrorType::ERROR_UNIMPLEMENTED);
            }
            return LayernormOutput({norm_output, params.before_norm_output});
        }

        if (data_type == DataType::TYPE_FP32 || data_type == DataType::TYPE_FP16) {  //
            if (!is_output
                && (!before_norm_output
                    || before_norm_output != norm_output->data())) {  // without before_norm_output  is_output false
                if ((data_type == DataType::TYPE_FP32
                     && (!gamma
                         || std::all_of((float*)gamma, (float*)gamma + n, [](float value) { return value == 1.0f; })))
                    || (data_type == DataType::TYPE_FP16
                        && (!gamma || std::all_of((__fp16*)gamma, (__fp16*)gamma + n, [](__fp16 value) {
                               return value == 1.0;
                           })))) {
#pragma omp parallel for num_threads(std::min(m, numThreads)) if (m >= 2)
                    for (int i = 0; i < m; i++) {
                        if (data_type == DataType::TYPE_FP16) {  // convert_float_to_fp16
                            float* input_converted  = new float[n];
                            float* output_converted = new float[n];
                            float* beta_converted   = new float[n];
                            convert_fp16_to_float((__fp16*)input->data() + i * n, input_converted, n);
                            convert_fp16_to_float((__fp16*)norm_output->data() + i * n, output_converted, n);
                            if (beta)
                                convert_fp16_to_float((__fp16*)beta, beta_converted, n);
                            RMSNorm_Nogamma(
                                n, input_converted, output_converted, beta != nullptr ? beta_converted : nullptr, eps);
                            convert_float_to_fp16(output_converted, (__fp16*)norm_output->data() + i * n, n);
                            delete[] input_converted;
                            delete[] output_converted;
                            delete[] beta_converted;
                        } else if (data_type == DataType::TYPE_FP32) {
                            RMSNorm_Nogamma(n,
                                            (float*)input->data() + i * n,
                                            (float*)norm_output->data() + i * n,
                                            beta != nullptr ? (float*)beta : nullptr,
                                            eps);
                        } else
                            throw OpException(OpErrorType::ERROR_UNIMPLEMENTED);
                    }
                    return LayernormOutput({norm_output, params.before_norm_output});
                } else {
#pragma omp parallel for num_threads(std::min(m, numThreads)) if (m >= 2)
                    for (int i = 0; i < m; i++) {
                        if (data_type == DataType::TYPE_FP16) {  // convert_float_to_fp16
                            float* input_converted  = new float[n];
                            float* output_converted = new float[n];
                            float* beta_converted   = new float[n];
                            float* gamma_converted  = new float[n];
                            convert_fp16_to_float((__fp16*)input->data() + i * n, input_converted, n);
                            convert_fp16_to_float((__fp16*)norm_output->data() + i * n, output_converted, n);
                            if (beta)
                                convert_fp16_to_float((__fp16*)beta, beta_converted, n);
                            convert_fp16_to_float((__fp16*)gamma, gamma_converted, n);
                            RMSNorm(n,
                                    input_converted,
                                    output_converted,
                                    gamma_converted,
                                    beta != nullptr ? beta_converted : nullptr,
                                    eps);
                            convert_float_to_fp16(output_converted, (__fp16*)norm_output->data() + i * n, n);
                            delete[] input_converted;
                            delete[] output_converted;
                            delete[] gamma_converted;
                            delete[] beta_converted;
                        } else if (data_type == DataType::TYPE_FP32) {  // beta!= nullptr ? (float*)beta : nullptr
                            RMSNorm(n,
                                    (float*)input->data() + i * n,
                                    (float*)norm_output->data() + i * n,
                                    (float*)gamma,
                                    beta != nullptr ? (float*)beta : nullptr,
                                    eps);
                        } else
                            throw OpException(OpErrorType::ERROR_UNIMPLEMENTED);
                    }
                    return LayernormOutput({norm_output, params.before_norm_output});
                }
            } else {
                if ((data_type == DataType::TYPE_FP32
                     && (!gamma
                         || std::all_of((float*)gamma, (float*)gamma + n, [](float value) { return value == 1.0f; })))
                    || (data_type == DataType::TYPE_FP16
                        && (!gamma || std::all_of((__fp16*)gamma, (__fp16*)gamma + n, [](__fp16 value) {
                               return value == 1.0;
                           })))) {
#pragma omp parallel for num_threads(std::min(m, numThreads)) if (m >= 2)
                    for (int i = 0; i < m; i++) {
                        if (data_type == DataType::TYPE_FP16) {
                            float* input_converted              = new float[n];
                            float* output_converted             = new float[n];
                            float* beta_converted               = new float[n];
                            float* bias_converted               = new float[n];
                            float* before_norm_output_converted = new float[n];
                            float* residual_converted           = new float[n];
                            convert_fp16_to_float((__fp16*)input->data() + i * n, input_converted, n);
                            convert_fp16_to_float((__fp16*)norm_output->data() + i * n, output_converted, n);
                            if (beta)
                                convert_fp16_to_float((__fp16*)beta, beta_converted, n);
                            if (before_norm_output && before_norm_output != norm_output->data())
                                convert_fp16_to_float(
                                    (__fp16*)before_norm_output + i * n, before_norm_output_converted, n);
                            if (residual)
                                convert_fp16_to_float((__fp16*)residual + i * n, residual_converted, n);
                            if (bias)
                                convert_fp16_to_float((__fp16*)bias + i * n, bias_converted, n);
                            RMSNorm_Nogamma_isoutput(n,
                                                     (before_norm_output && before_norm_output != norm_output->data()) ?
                                                         before_norm_output_converted :
                                                         nullptr,
                                                     input_converted,
                                                     output_converted,
                                                     beta != nullptr ? beta_converted : nullptr,
                                                     (residual != nullptr) ? residual_converted : nullptr,
                                                     (bias != nullptr) ? bias_converted : nullptr,
                                                     eps);
                            convert_float_to_fp16(output_converted, (__fp16*)norm_output->data() + i * n, n);
                            convert_float_to_fp16(before_norm_output_converted, (__fp16*)before_norm_output + i * n, n);
                            delete[] input_converted;
                            delete[] output_converted;
                            delete[] beta_converted;
                            delete[] bias_converted;
                            delete[] before_norm_output_converted;
                            delete[] residual_converted;
                        } else {
                            RMSNorm_Nogamma_isoutput(n,
                                                     (before_norm_output && before_norm_output != norm_output->data()) ?
                                                         (float*)before_norm_output + i * n :
                                                         nullptr,
                                                     (float*)input->data() + i * n,
                                                     (float*)norm_output->data() + i * n,
                                                     (float*)beta,
                                                     (residual != nullptr) ? (float*)residual + i * n : nullptr,
                                                     (bias != nullptr) ? (float*)bias : nullptr,
                                                     eps);
                        }
                    }
                    return LayernormOutput({norm_output, params.before_norm_output});
                } else {
#pragma omp parallel for num_threads(std::min(m, numThreads)) if (m >= 2)
                    for (int i = 0; i < m; i++) {
                        if (data_type == DataType::TYPE_FP16) {
                            float* input_converted              = new float[n];
                            float* output_converted             = new float[n];
                            float* gamma_converted              = new float[n];
                            float* beta_converted               = new float[n];
                            float* bias_converted               = new float[n];
                            float* before_norm_output_converted = new float[n];
                            float* residual_converted           = new float[n];
                            convert_fp16_to_float((__fp16*)input->data() + i * n, input_converted, n);
                            convert_fp16_to_float((__fp16*)gamma, gamma_converted, n);
                            if (beta)
                                convert_fp16_to_float((__fp16*)beta, beta_converted, n);
                            if (before_norm_output && before_norm_output != norm_output->data())
                                convert_fp16_to_float(
                                    (__fp16*)before_norm_output + i * n, before_norm_output_converted, n);
                            if (residual)
                                convert_fp16_to_float((__fp16*)residual + i * n, residual_converted, n);
                            if (bias)
                                convert_fp16_to_float((__fp16*)bias, bias_converted, n);

                            RMSNorm_isoutput(n,
                                             (before_norm_output && before_norm_output != norm_output->data()) ?
                                                 before_norm_output_converted :
                                                 nullptr,
                                             input_converted,
                                             output_converted,
                                             gamma_converted,
                                             (beta != nullptr) ? beta_converted : nullptr,
                                             (residual != nullptr) ? residual_converted : nullptr,
                                             (bias != nullptr) ? bias_converted : nullptr,
                                             eps);
                            convert_float_to_fp16(output_converted, (__fp16*)norm_output->data() + i * n, n);
                            if (before_norm_output && before_norm_output != norm_output->data())
                                convert_float_to_fp16(
                                    before_norm_output_converted, (__fp16*)before_norm_output + i * n, n);
                            delete[] input_converted;
                            delete[] output_converted;
                            delete[] gamma_converted;
                            delete[] beta_converted;
                            delete[] bias_converted;
                            delete[] before_norm_output_converted;
                            delete[] residual_converted;
                        } else {
                            float* before_norm_output_converted = new float[n];
                            RMSNorm_isoutput(n,
                                             (before_norm_output && before_norm_output != norm_output->data()) ?
                                                 before_norm_output_converted :
                                                 nullptr,
                                             (float*)input->data() + i * n,
                                             (float*)norm_output->data() + i * n,
                                             (float*)gamma,
                                             (beta != nullptr) ? (float*)beta : nullptr,
                                             (residual != nullptr) ? (float*)residual + i * n : (float*)residual,
                                             (bias != nullptr) ? (float*)bias : nullptr,
                                             eps);
                            if (before_norm_output && before_norm_output != norm_output->data())
                                std::memcpy((float*)before_norm_output + i * n,
                                            before_norm_output_converted,
                                            n * sizeof(float));
                            delete[] before_norm_output_converted;
                        }
                    }
                    return LayernormOutput({norm_output, params.before_norm_output});
                }
            }
        } else
            throw OpException(OpErrorType::ERROR_UNIMPLEMENTED);
    }
    // **********************************************

    // before_norm_output       params.return_norm_output        bias/residual exist
    // .  .  F
    // layernorm(input)->normed_output
    // F  .  T
    // layernorm(input+bias+residual)->normed_output
    // T  T  T
    // layernorm(input+bias+residual)->before_norm_output
    // layernorm(input+bias+residual)->normed_output
    // T  F  T
    // (input+bias+residual)->before_norm_output
    // layernorm(input+bias+residual)->normed_output

    // **********************************************
    else if (norm_type == NormType::layernorm && data_type == DataType::TYPE_FP32) {
        if (!is_output) {  //. .  F
            if (!gamma || std::all_of((float*)gamma, (float*)gamma + n, [](float value) { return value == 1.0f; })) {

#pragma omp parallel for num_threads(std::min(m, numThreads)) if (m >= 2)
                for (int i = 0; i < m; i++) {
                    layerNorm_Nogamma(
                        n, (float*)input->data() + i * n, (float*)norm_output->data() + i * n, (float*)beta, eps);
                }

                return LayernormOutput({norm_output, params.before_norm_output});
            }  //(gamma =1,1......)OR (no gamma)
            else {
#pragma omp parallel for num_threads(std::min(m, numThreads)) if (m >= 2)
                for (int i = 0; i < m; i++) {
                    layerNorm(n,
                              (float*)input->data() + i * n,
                              (float*)norm_output->data() + i * n,
                              (float*)gamma,
                              (float*)beta,
                              eps);
                }
                return LayernormOutput({norm_output, params.before_norm_output});
            }
        } else if (!before_norm_output) {  // add bias residual   //F . T
            if (!gamma || std::all_of((float*)gamma, (float*)gamma + n, [](float value) { return value == 1.0f; })) {
#pragma omp parallel for num_threads(std::min(m, numThreads)) if (m >= 2)
                for (int i = 0; i < m; i++) {
                    layerNorm_Nogamma_isoutput(n,
                                               (float*)input->data() + i * n,
                                               (float*)norm_output->data() + i * n,
                                               (float*)beta,
                                               (residual != nullptr) ? (float*)residual + i * n : (float*)residual,
                                               (float*)bias,
                                               eps);
                }
                return LayernormOutput({norm_output, params.before_norm_output});
            }  //(gamma =1,1......)OR (no gamma)
            else {
#pragma omp parallel for num_threads(std::min(m, numThreads)) if (m >= 2)
                for (int i = 0; i < m; i++) {
                    layerNorm_isoutput(n,
                                       (float*)input->data() + i * n,
                                       (float*)norm_output->data() + i * n,
                                       (float*)gamma,
                                       (float*)beta,
                                       (residual != nullptr) ? ((float*)residual + i * n) : nullptr,
                                       (float*)bias,
                                       eps);
                }
                return LayernormOutput({norm_output, params.before_norm_output});
            }
        } else if (params.return_normed_output) {  // T  T  T
            if (!gamma || std::all_of((float*)gamma, (float*)gamma + n, [](float value) { return value == 1.0f; })) {
#pragma omp parallel for num_threads(std::min(m, numThreads)) if (m >= 2)
                for (int i = 0; i < m; i++) {
                    layerNorm_Nogamma_isoutput(n,
                                               (float*)input->data() + i * n,
                                               (float*)norm_output->data() + i * n,
                                               (float*)beta,
                                               (residual != nullptr) ? ((float*)residual + i * n) : (float*)residual,
                                               (float*)bias,
                                               eps);
                    if (before_norm_output != norm_output->data())
                        std::memcpy(
                            (float*)before_norm_output + i * n, (float*)norm_output->data() + i * n, n * sizeof(float));
                }
                return LayernormOutput({norm_output, params.before_norm_output});
            }  // gamma =1,1......  No gamma
            else {
#pragma omp parallel for num_threads(std::min(m, numThreads)) if (m >= 2)
                for (int i = 0; i < m; i++) {
                    layerNorm_isoutput(n,
                                       (float*)input->data() + i * n,
                                       (float*)norm_output->data() + i * n,
                                       (float*)gamma,
                                       (float*)beta,
                                       (residual != nullptr) ? ((float*)residual + i * n) : (float*)residual,
                                       (float*)bias,
                                       eps);
                    if (before_norm_output != norm_output->data())
                        std::memcpy(
                            (float*)before_norm_output + i * n, (float*)norm_output->data() + i * n, n * sizeof(float));
                }
                return LayernormOutput({norm_output, params.before_norm_output});
            }
        } else {  // T F T
            if (!gamma || std::all_of((float*)gamma, (float*)gamma + n, [](float value) { return value == 1.0f; })) {
#pragma omp parallel for num_threads(std::min(m, numThreads)) if (m >= 2)
                for (int i = 0; i < m; i++) {
                    layerNorm_Nogamma_isoutput_unnormedout(n,
                                                           (float*)input->data() + i * n,
                                                           (float*)norm_output->data() + i * n,
                                                           (float*)beta,
                                                           (residual != nullptr) ? ((float*)residual + i * n) :
                                                                                   (float*)residual,
                                                           (float*)bias,
                                                           ((float*)before_norm_output + i * n),
                                                           eps);
                }
                return LayernormOutput({norm_output, params.before_norm_output});
            }  // gamma =1,1......  No gamma
            else {
#pragma omp parallel for num_threads(std::min(m, numThreads)) if (m >= 2)
                for (int i = 0; i < m; i++) {
                    layerNorm_isoutput_unnormedout(n,
                                                   (float*)input->data() + i * n,
                                                   (float*)norm_output->data() + i * n,
                                                   (float*)gamma,
                                                   (float*)beta,
                                                   (residual != nullptr) ? ((float*)residual + i * n) :
                                                                           (float*)residual,
                                                   (float*)bias,
                                                   ((float*)before_norm_output + i * n),
                                                   eps);
                }
                return LayernormOutput({norm_output, params.before_norm_output});
            }
        }
    } else
        throw OpException(OpErrorType::ERROR_UNIMPLEMENTED);
}

}  // namespace rtp_llm
