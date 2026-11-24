#include "per_token_group_quant_8bit_v3.h"

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <climits>

namespace rtp_llm {
namespace {

// ATen's overlap check returns TooHard for padded column-major scales.
// Conservatively include the padding in S's occupied byte interval.
bool overlaps(const at::Tensor& a, int64_t a_bytes, const at::Tensor& b, int64_t b_bytes) {
    const auto a_begin = reinterpret_cast<uintptr_t>(a.data_ptr());
    const auto b_begin = reinterpret_cast<uintptr_t>(b.data_ptr());
    return a_begin < b_begin + b_bytes && b_begin < a_begin + a_bytes;
}

// For finite BF16 amax = (1 + mantissa/128) * 2^(exponent-127),
// ceil(log2(amax/448)) = exponent-127-8 + (mantissa > 96), since
// 448 = 1.75 * 2^8. Subnormals lie below the legacy scale floor.
// ceil(log2(1e-10)) + 127 = 94. Integer amax avoids rounding the input
// reduction and gives the exact legacy scale across all finite BF16 values.
__device__ __forceinline__ unsigned scale_code(unsigned abs_bits) {
    return static_cast<unsigned>(max(94, static_cast<int>(abs_bits >> 7) - 8 + ((abs_bits & 127) > 96)));
}

template<int BLOCK_THREADS, int K = 0>
__global__ void per_token_group_quant_v3_kernel(const __nv_bfloat16* __restrict__ input,
                                                uint8_t* __restrict__ output,
                                                uint32_t* __restrict__ scales,
                                                int     groups,
                                                int     groups_per_row,
                                                int64_t scale_stride) {
    constexpr int THREADS_PER_GROUP = 2;
    const int     group  = blockIdx.x * (BLOCK_THREADS / THREADS_PER_GROUP) + threadIdx.x / THREADS_PER_GROUP;
    const int     lane   = threadIdx.x & 1;
    const int64_t offset = static_cast<int64_t>(group) * 32 + lane * 16;
    union {
        uint4          words[2];
        __nv_bfloat162 pairs[8];
    } data;
    unsigned ab = 0;
#pragma unroll
    for (int i = 0; i < 2; ++i) {
        data.words[i] = make_uint4(0, 0, 0, 0);
        if (group < groups) {
            data.words[i] = reinterpret_cast<const uint4*>(input + offset)[i];
        }
        unsigned a = __vmaxu2(data.words[i].x & 0x7fff7fff, data.words[i].y & 0x7fff7fff);
        unsigned b = __vmaxu2(data.words[i].z & 0x7fff7fff, data.words[i].w & 0x7fff7fff);
        ab         = __vmaxu2(ab, __vmaxu2(a, b));
    }
    unsigned amax            = max(ab & 0xffff, ab >> 16);
    amax                     = max(amax, __shfl_xor_sync(0xffffffff, amax, 1));
    const unsigned code      = scale_code(amax);
    const float    inv_scale = __uint_as_float((254 - code) << 23);

    // All lanes participate, including padded groups in the final CTA.
    unsigned packed = code;
    packed |= __shfl_down_sync(0xffffffff, code, 2) << 8;
    packed |= __shfl_down_sync(0xffffffff, code, 4) << 16;
    packed |= __shfl_down_sync(0xffffffff, code, 6) << 24;
    if (group >= groups) {
        return;
    }

    union {
        uint4                words;
        __nv_fp8x2_storage_t pairs[8];
    } result;
#pragma unroll
    for (int i = 0; i < 8; ++i) {
        float2 value = __bfloat1622float2(data.pairs[i]);
        value.x *= inv_scale;
        value.y *= inv_scale;
        result.pairs[i] = __nv_cvt_float2_to_fp8x2(value, __NV_SATFINITE, __NV_E4M3);
    }
    *reinterpret_cast<uint4*>(output + offset) = result.words;
    if ((group & 3) == 0 && lane == 0) {
        const int row                     = group / (K ? K / 32 : groups_per_row);
        const int pack                    = (group % (K ? K / 32 : groups_per_row)) / 4;
        scales[pack * scale_stride + row] = packed;
    }
}

}  // namespace

void per_token_group_quant_fp8_v3(const at::Tensor& input, at::Tensor& output_q, at::Tensor& output_s) {
    TORCH_CHECK(input.is_cuda() && input.device() == output_q.device() && input.device() == output_s.device(),
                "v3 requires tensors on the same CUDA device");
    TORCH_CHECK(input.scalar_type() == at::kBFloat16, "v3 requires BF16 input");
    TORCH_CHECK(output_q.scalar_type() == at::ScalarType::Float8_e4m3fn, "v3 requires E4M3 output");
    TORCH_CHECK(output_s.scalar_type() == at::kInt, "v3 requires int32 packed scales");
    TORCH_CHECK(input.dim() == 2 && input.size(1) > 0 && input.size(1) % 128 == 0,
                "v3 requires [M,K] with K a positive multiple of 128");
    TORCH_CHECK(input.is_contiguous() && output_q.is_contiguous(), "v3 input and Q must be contiguous");
    TORCH_CHECK(output_q.sizes() == input.sizes(), "v3 Q shape mismatch");
    const int64_t m = input.size(0), k = input.size(1);
    TORCH_CHECK(output_s.dim() == 2 && output_s.size(0) == m && output_s.size(1) == k / 128, "v3 scale shape mismatch");
    if (m == 0) {
        return;
    }
    TORCH_CHECK(output_s.stride(0) == 1 && output_s.stride(1) == ((m + 3) / 4) * 4,
                "v3 requires column-major TMA-aligned scale stride");
    TORCH_CHECK(reinterpret_cast<uintptr_t>(input.data_ptr()) % 16 == 0
                    && reinterpret_cast<uintptr_t>(output_q.data_ptr()) % 16 == 0,
                "v3 requires 16-byte aligned input/Q pointers");
    TORCH_CHECK(input.numel() / 32 <= INT_MAX, "v3 shape exceeds launch limits");
    const int64_t input_bytes = input.numel() * input.element_size();
    const int64_t q_bytes     = output_q.numel() * output_q.element_size();
    const int64_t s_bytes     = ((k / 128 - 1) * output_s.stride(1) + m) * output_s.element_size();
    TORCH_CHECK(!overlaps(input, input_bytes, output_q, q_bytes) && !overlaps(input, input_bytes, output_s, s_bytes)
                    && !overlaps(output_q, q_bytes, output_s, s_bytes),
                "v3 input, Q and scales must not overlap (including scale padding)");
    const c10::cuda::CUDAGuard guard(input.device());
    const int                  groups = static_cast<int>(input.numel() / 32);
    auto                       launch = [&](auto block_size, auto columns) {
        constexpr int threads = decltype(block_size)::value;
        constexpr int width   = decltype(columns)::value;
        const int     blocks = static_cast<int>((static_cast<int64_t>(groups) * 2 + threads - 1) / threads);
        per_token_group_quant_v3_kernel<threads, width><<<blocks, threads, 0, at::cuda::getCurrentCUDAStream()>>>(
            reinterpret_cast<const __nv_bfloat16*>(input.data_ptr()),
            reinterpret_cast<uint8_t*>(output_q.data_ptr()),
            reinterpret_cast<uint32_t*>(output_s.data_ptr()),
            groups,
            static_cast<int>(k / 32),
            output_s.stride(1));
    };
    auto dispatch = [&](auto columns) {
        if (input.numel() <= 512 * 1024) {
            launch(std::integral_constant<int, 32>{}, columns);
        } else {
            launch(std::integral_constant<int, 128>{}, columns);
        }
    };
    switch (k) {
        case 1280:
            dispatch(std::integral_constant<int, 1280>{});
            break;
        case 2304:
            dispatch(std::integral_constant<int, 2304>{});
            break;
        case 4096:
            dispatch(std::integral_constant<int, 4096>{});
            break;
        case 5120:
            dispatch(std::integral_constant<int, 5120>{});
            break;
        case 6144:
            dispatch(std::integral_constant<int, 6144>{});
            break;
        case 8192:
            dispatch(std::integral_constant<int, 8192>{});
            break;
        case 15360:
            dispatch(std::integral_constant<int, 15360>{});
            break;
        default:
            dispatch(std::integral_constant<int, 0>{});
            break;
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

}  // namespace rtp_llm
