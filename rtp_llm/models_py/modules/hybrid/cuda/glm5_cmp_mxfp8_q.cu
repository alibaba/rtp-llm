#include <torch/extension.h>

#include <rtp_kernel/glm5/norm_quant.cuh>

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cstdint>

namespace rtp_llm {
namespace {

constexpr int   kHeads           = 32;
constexpr int   kHeadDim         = 128;
constexpr int   kRopeDim         = 64;
constexpr int   kBlockRows       = 16;
constexpr int   kThreads         = kBlockRows * 16;
constexpr float kHadamardScale   = 0.08838834764831845f;
constexpr float kHeadWeightScale = 1.0f / 64.0f;

using rtp_kernel::glm5::detail::Fp8x8;
using rtp_kernel::glm5::detail::Bf16x8;
using rtp_kernel::glm5::detail::fp32x2_to_bf16x2;
using rtp_kernel::glm5::detail::fp32x2_to_e4m3x2;
using rtp_kernel::glm5::detail::ue8m0_exponent;
using rtp_kernel::glm5::detail::ue8m0_inv_scale;
using rtp_kernel::glm5::detail::unpack_bf16x8;
using rtp_kernel::glm5::detail::pack_bf16x8;

__device__ __forceinline__ void hadamard8(float (&values)[8]) {
#pragma unroll
    for (int stride = 1; stride < 8; stride <<= 1) {
#pragma unroll
        for (int base = 0; base < 8; base += 2 * stride) {
#pragma unroll
            for (int i = 0; i < stride; ++i) {
                const float left          = values[base + i];
                const float right         = values[base + i + stride];
                values[base + i]          = left + right;
                values[base + i + stride] = left - right;
            }
        }
    }
}

__device__ __forceinline__ Bf16x8 load_bf16x8(const __nv_bfloat16* projected, int row, int head, int sublane) {
    const int offset = (row * kHeads + head) * kHeadDim + sublane * 8;
    return *reinterpret_cast<const Bf16x8*>(projected + offset);
}

__device__ __forceinline__ void transform_and_store(Bf16x8              input,
                                                    const float*        cos_sin,
                                                    const std::int32_t* positions,
                                                    const float*        gate_raw,
                                                    std::uint8_t*       q_fp8,
                                                    std::uint8_t*       q_sf,
                                                    __nv_bfloat16*      head_weights,
                                                    int                 row,
                                                    int                 head,
                                                    int                 sublane,
                                                    bool                valid) {
    float values[8];
    unpack_bf16x8(input, values);

    if (valid && sublane < kRopeDim / 8) {
        const int    position = positions[row];
        const float* cos      = cos_sin + position * kRopeDim + sublane * 4;
        const float* sin      = cos + kRopeDim / 2;
#pragma unroll
        for (int i = 0; i < 8; i += 2) {
            const float x0 = values[i];
            const float x1 = values[i + 1];
            values[i]      = fmaf(-x1, sin[i / 2], x0 * cos[i / 2]);
            values[i + 1]  = fmaf(x0, sin[i / 2], x1 * cos[i / 2]);
        }
        // Match the model's BF16 output before the Hadamard transform.
        input = pack_bf16x8(values);
        unpack_bf16x8(input, values);
    }

    hadamard8(values);
#pragma unroll
    for (int mask = 1; mask < 16; mask <<= 1) {
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            const float partner = __shfl_xor_sync(0xffffffffu, values[i], mask, 16);
            values[i]           = (sublane & mask) != 0 ? partner - values[i] : values[i] + partner;
        }
    }

    // The unfused producer rounds Hadamard output to BF16 before MXFP8 quantization.
#pragma unroll
    for (float& value : values) {
        value *= kHadamardScale;
    }
    input = pack_bf16x8(values);
    unpack_bf16x8(input, values);
    float local_absmax = 0.0f;
#pragma unroll
    for (int i = 0; i < 8; ++i) {
        local_absmax = fmaxf(local_absmax, fabsf(values[i]));
    }
    for (int mask = 1; mask < 4; mask <<= 1) {
        local_absmax = fmaxf(local_absmax, __shfl_xor_sync(0xffffffffu, local_absmax, mask, 4));
    }
    const std::uint32_t exponent  = ue8m0_exponent(local_absmax);
    const float         inv_scale = ue8m0_inv_scale(exponent);
    Fp8x8               quantized;
#pragma unroll
    for (int i = 0; i < 8; i += 2) {
        quantized.pairs[i / 2] = fp32x2_to_e4m3x2(values[i] * inv_scale, values[i + 1] * inv_scale);
    }

    if (valid) {
        const int token_head                                                   = row * kHeads + head;
        *reinterpret_cast<Fp8x8*>(q_fp8 + token_head * kHeadDim + sublane * 8) = quantized;
        if ((sublane & 3) == 0) {
            q_sf[token_head * 4 + sublane / 4] = exponent;
        }
        if (sublane == 0) {
            head_weights[token_head] = __float2bfloat16_rn(gate_raw[token_head] * kHeadWeightScale);
        }
    }
}

template<bool kPdl>
__global__ void indexer_q_rope_quant_kernel(const __nv_bfloat16* projected,
                                            const float*         cos_sin,
                                            const std::int32_t*  positions,
                                            const float*         gate_raw,
                                            std::uint8_t*        q_fp8,
                                            std::uint8_t*        q_sf,
                                            __nv_bfloat16*       head_weights,
                                            int                  rows) {
    if constexpr (kPdl) {
        if (threadIdx.x == 0) {
            cudaTriggerProgrammaticLaunchCompletion();
        }
        cudaGridDependencySynchronize();
    }
    const int  thread    = static_cast<int>(threadIdx.x);
    const int  sublane   = thread & 15;
    const int  local_row = thread >> 4;
    const int  task      = static_cast<int>(blockIdx.x);
    const int  head      = task % kHeads;
    const int  row       = (task / kHeads) * kBlockRows + local_row;
    const bool valid     = row < rows;
    Bf16x8     input{};
    if (valid) {
        input = load_bf16x8(projected, row, head, sublane);
    }
    transform_and_store(input, cos_sin, positions, gate_raw, q_fp8, q_sf, head_weights, row, head, sublane, valid);
}

void check_tensor(const torch::Tensor& tensor, at::ScalarType dtype, const torch::Device& device, const char* name) {
    TORCH_CHECK(tensor.is_cuda(), name, " must be CUDA");
    TORCH_CHECK(tensor.scalar_type() == dtype, name, " has unexpected dtype");
    TORCH_CHECK(tensor.device() == device, name, " must be on ", device);
}

}  // namespace

void glm5CmpMxfp8IndexerQ(const torch::Tensor& projected,
                          const torch::Tensor& cos_sin,
                          const torch::Tensor& positions,
                          const torch::Tensor& gate_raw,
                          const torch::Tensor& q_fp8,
                          const torch::Tensor& q_sf,
                          const torch::Tensor& head_weights,
                          std::uintptr_t       notify_event,
                          bool                 pdl) {
    const auto device = projected.device();
    check_tensor(projected, at::kBFloat16, device, "projected");
    check_tensor(cos_sin, at::kFloat, device, "cos_sin");
    check_tensor(positions, at::kInt, device, "positions");
    check_tensor(gate_raw, at::kFloat, device, "gate_raw");
    check_tensor(q_fp8, at::kFloat8_e4m3fn, device, "q_fp8");
    check_tensor(q_sf, at::kInt, device, "q_sf");
    check_tensor(head_weights, at::kBFloat16, device, "head_weights");
    TORCH_CHECK(projected.dim() == 2 && projected.size(0) > 0 && projected.size(1) == kHeads * kHeadDim
                    && projected.is_contiguous(),
                "projected must be contiguous BF16 [M,4096]");
    const int rows = static_cast<int>(projected.size(0));
    TORCH_CHECK(cos_sin.dim() == 2 && cos_sin.size(1) == kRopeDim && cos_sin.is_contiguous(),
                "cos_sin must be contiguous FP32 [max_position,64]");
    TORCH_CHECK(positions.sizes() == torch::IntArrayRef({rows}) && positions.is_contiguous(),
                "positions must be contiguous INT32 [M]");
    TORCH_CHECK(gate_raw.sizes() == torch::IntArrayRef({rows, kHeads}) && gate_raw.is_contiguous(),
                "gate_raw must be contiguous FP32 [M,32]");
    TORCH_CHECK(q_fp8.sizes() == torch::IntArrayRef({rows, kHeads, kHeadDim}) && q_fp8.is_contiguous(),
                "q_fp8 must be contiguous FP8 [M,32,128]");
    TORCH_CHECK(q_sf.sizes() == torch::IntArrayRef({rows, kHeads}) && q_sf.is_contiguous(),
                "q_sf must be contiguous INT32 [M,32]");
    TORCH_CHECK(head_weights.sizes() == torch::IntArrayRef({rows, kHeads}) && head_weights.is_contiguous(),
                "head_weights must be contiguous BF16 [M,32]");

    const c10::cuda::CUDAGuard guard(device);
    const cudaStream_t         stream = at::cuda::getCurrentCUDAStream();
    const int                  tasks  = kHeads * ((rows + kBlockRows - 1) / kBlockRows);
    const auto                 kernel = pdl ? indexer_q_rope_quant_kernel<true> : indexer_q_rope_quant_kernel<false>;
    const __nv_bfloat16*       projected_ptr    = reinterpret_cast<const __nv_bfloat16*>(projected.data_ptr());
    const float*               cos_sin_ptr      = cos_sin.data_ptr<float>();
    const std::int32_t*        positions_ptr    = positions.data_ptr<std::int32_t>();
    const float*               gate_raw_ptr     = gate_raw.data_ptr<float>();
    std::uint8_t*              q_fp8_ptr        = reinterpret_cast<std::uint8_t*>(q_fp8.data_ptr());
    auto*                      q_sf_ptr         = reinterpret_cast<std::uint8_t*>(q_sf.data_ptr());
    auto*                      head_weights_ptr = reinterpret_cast<__nv_bfloat16*>(head_weights.data_ptr());
    void*                      args[]           = {&projected_ptr,
                                                   &cos_sin_ptr,
                                                   &positions_ptr,
                                                   &gate_raw_ptr,
                                                   &q_fp8_ptr,
                                                   &q_sf_ptr,
                                                   &head_weights_ptr,
                                                   const_cast<int*>(&rows)};

    cudaLaunchAttribute attributes[2]{};
    cudaLaunchConfig_t  config{};
    config.gridDim  = dim3(tasks);
    config.blockDim = dim3(kThreads);
    config.stream   = stream;
    if (pdl) {
        auto& attribute                                      = attributes[config.numAttrs++];
        attribute.id                                         = cudaLaunchAttributeProgrammaticStreamSerialization;
        attribute.val.programmaticStreamSerializationAllowed = 1;
    }
    if (pdl && notify_event != 0) {
        auto& attribute                                     = attributes[config.numAttrs++];
        attribute.id                                        = cudaLaunchAttributeProgrammaticEvent;
        attribute.val.programmaticEvent.event               = reinterpret_cast<cudaEvent_t>(notify_event);
        attribute.val.programmaticEvent.flags               = 0;
        attribute.val.programmaticEvent.triggerAtBlockStart = 0;
    }
    config.attrs = attributes;
    C10_CUDA_CHECK(cudaLaunchKernelExC(&config, reinterpret_cast<const void*>(kernel), args));
    if (!pdl && notify_event != 0) {
        C10_CUDA_CHECK(cudaEventRecord(reinterpret_cast<cudaEvent_t>(notify_event), stream));
    }
}

}  // namespace rtp_llm
