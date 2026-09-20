#include "rtp_llm/models_py/bindings/cuda/kernels/fast_bf16_int8.h"

#ifndef USE_PPU
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_bf16.h>
#include <type_traits>

namespace rtp_llm {
namespace {

constexpr int kThreads = 64;
// Each thread processes one uint4 of INT8 codes and the corresponding BF16 values.
constexpr int kElementsPerThread = 16;
constexpr int kVectorAlignment   = alignof(uint4);
static_assert(kElementsPerThread * sizeof(int8_t) == sizeof(uint4), "Each thread must process one INT8 uint4");

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
constexpr int kElementsPerPair      = 2;
constexpr int kPairsPerThread       = kElementsPerThread / kElementsPerPair;
constexpr int kBf16VectorsPerThread = kElementsPerThread * sizeof(__nv_bfloat16) / sizeof(uint4);
// Clear both BF16 sign bits; 0x4300 is the BF16 encoding of 128.
constexpr uint32_t kAbsMask = 0x7fff7fff;
constexpr uint32_t kBias128 = 0x43004300;

// Broadcast one 16-bit encoding to both lanes of a 32-bit register.
__device__ __forceinline__ uint32_t duplicate(uint16_t bits) {
    return uint32_t(bits) | (uint32_t(bits) << 16);
}

// Quantize two BF16 lanes using a broadcast BF16 reciprocal; return INT8 codes in bytes 0 and 2.
__device__ __forceinline__ uint32_t quantPair(uint32_t value, uint32_t inverse) {
    uint32_t biased;
    // BF16 spacing in [128,256) is 1: adding 128 rounds to integers stored in the seven mantissa bits.
    // Clamp to 255 (0x437f per lane) so magnitude 128 cannot wrap to zero when masking the mantissa.
    asm("fma.rn.bf16x2 %0, %1, %2, %3;" : "=r"(biased) : "r"(value & kAbsMask), "r"(inverse), "r"(kBias128));
    asm("min.bf16x2 %0, %0, %1;" : "+r"(biased) : "r"(0x437f437f));
    const uint32_t sign = (value >> 15) & 0x00010001;
    // Each sign lane is 0 or 1; (sign << 8) - sign is 0x00 or 0xff.
    // XOR followed by +sign converts the magnitude to signed INT8 two's complement.
    return ((biased & 0x007f007f) ^ ((sign << 8) - sign)) + sign;
}

// Decode INT8 bytes 0 and 2 to BF16x2 using a shared BF16 scale.
// UseBiasFma requires a positive normal scale with finite 128*scale.
template<bool UseBiasFma>
__device__ __forceinline__ uint32_t dequantPair(uint32_t codes, uint16_t scale) {
    // Undo two's complement independently in each byte; -128 produces unsigned magnitude 128.
    const uint32_t sign      = (codes >> 7) & 0x00010001;
    const uint32_t magnitude = ((codes ^ ((sign << 8) - sign)) + sign) & 0x00ff00ff;
    // OR with 0x4300 encodes 128+magnitude exactly; magnitude 128 gives 0x4380 (256).
    const uint32_t biased = magnitude | kBias128;
    const uint32_t scales = duplicate(scale);
    uint32_t       value;
    if constexpr (UseBiasFma) {
        // BF16 has seven mantissa bits: adding 7<<7 raises the exponent by 7 (multiply by 128).
        // Set the sign bit to form -128*scale; the FMA cancels the bias before rounding.
        const uint32_t offset = duplicate(uint16_t((scale + 0x0380) | 0x8000));
        asm("fma.rn.bf16x2 %0, %1, %2, %3;" : "=r"(value) : "r"(biased), "r"(scales), "r"(offset));
    } else {
        asm("sub.bf16x2 %0, %1, %2;" : "=r"(value) : "r"(biased), "r"(kBias128));
        asm("mul.bf16x2 %0, %0, %1;" : "+r"(value) : "r"(scales));
    }
    // Move each INT8 sign bit (bit 7) to the BF16 sign position (bit 15).
    return value ^ ((codes & 0x00800080) << 8);
}

// Decode one thread's INT8 tile using one scale; first initializes acc[kPairsPerThread], otherwise add to it.
template<bool UseBiasFma>
__device__ __forceinline__ void accumulateTile(uint4 codes, uint16_t scale, uint32_t* acc, bool first) {
    const uint32_t words[4] = {codes.x, codes.y, codes.z, codes.w};
#pragma unroll
    for (int i = 0; i < kPairsPerThread; ++i) {
        // 0x4140 selects bytes 0,1; 0x4342 selects bytes 2,3. Bytes from the zero operand pad each to 16 bits.
        uint32_t value = dequantPair<UseBiasFma>(__byte_perm(words[i / 2], 0, i % 2 ? 0x4342 : 0x4140), scale);
        if (!first) {
            asm("add.bf16x2 %0, %1, %0;" : "+r"(value) : "r"(acc[i]));
        }
        acc[i] = value;
    }
}
#endif

// Quantize `groups` contiguous groups of GroupSize BF16 values into INT8 output and one BF16 scale per group.
// GroupSize/kElementsPerThread threads cooperate on absmax and each computes its own BF16 scale/reciprocal.
template<int GroupSize>
__global__ void quantizeKernel(const uint4* input, uint4* output, uint16_t* scales, int64_t groups) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
    constexpr int group_threads = GroupSize / kElementsPerThread;
    const int64_t tile          = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
    const int64_t group         = tile / group_threads;
    const int     lane          = threadIdx.x % group_threads;
    if (group >= groups) {
        return;
    }
    uint4    values[kBf16VectorsPerThread];
    uint32_t maximum = 0;
#pragma unroll
    for (int i = 0; i < kBf16VectorsPerThread; ++i) {
        values[i]               = input[kBf16VectorsPerThread * tile + i];
        const uint32_t words[4] = {values[i].x, values[i].y, values[i].z, values[i].w};
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            asm("max.bf16x2 %0, %0, %1;" : "+r"(maximum) : "r"(words[j] & kAbsMask));
        }
    }
    // Nonnegative BF16 encodings preserve numeric order, allowing integer max within and between threads.
    uint32_t abs_bits = max(maximum & 0xffff, maximum >> 16);
    // Blocks and tails contain complete groups. The mask names only the live lanes of this group.
    const unsigned mask = ((1u << group_threads) - 1) << ((threadIdx.x % 32) - lane);
#pragma unroll
    for (int offset = group_threads / 2; offset > 0; offset /= 2) {
        abs_bits = max(abs_bits, __shfl_xor_sync(mask, abs_bits, offset, group_threads));
    }
    // A BF16 encoding shifted left by 16 represents the same value in FP32.
    const float         absmax  = fmaxf(__uint_as_float(abs_bits << 16), 1.e-10f);
    const __nv_bfloat16 scale   = __float2bfloat16_rn(absmax / 127.f);
    const uint32_t      inverse = duplicate(__bfloat16_as_ushort(__float2bfloat16_rn(1.f / __bfloat162float(scale))));
    if (lane == 0) {
        scales[group] = __bfloat16_as_ushort(scale);
    }
    const uint4 a = values[0];
    const uint4 b = values[1];
    // Selector 0x6420 packs bytes 0,2 of each pair result into four consecutive INT8 bytes.
    output[tile] = make_uint4(__byte_perm(quantPair(a.x, inverse), quantPair(a.y, inverse), 0x6420),
                              __byte_perm(quantPair(a.z, inverse), quantPair(a.w, inverse), 0x6420),
                              __byte_perm(quantPair(b.x, inverse), quantPair(b.y, inverse), 0x6420),
                              __byte_perm(quantPair(b.z, inverse), quantPair(b.w, inverse), 0x6420));
#else
    asm volatile("trap;");
#endif
}

// Decode and sum `inputs` sources in order; each thread writes one of `tiles` kElementsPerThread-element tiles.
// GroupSize is the number of codes per scale; source strides are in elements of their respective arrays.
// StaticInputs selects a fixed source count, or uses `inputs` when zero. Each source is contiguous [M,K].
template<int GroupSize, int StaticInputs>
__global__ void reduceKernel(const int8_t*   codes,
                             const uint16_t* scales,
                             uint4*          output,
                             int64_t         tiles,
                             int64_t         inputs,
                             int64_t         code_stride,
                             int64_t         scale_stride) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
    const int64_t tile = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (tile >= tiles) {
        return;
    }
    uint32_t      acc[kPairsPerThread];
    const int64_t count = StaticInputs ? StaticInputs : inputs;
#pragma unroll
    for (int64_t source = 0; source < count; ++source) {
        const uint4    q     = reinterpret_cast<const uint4*>(codes + source * code_stride)[tile];
        const uint16_t scale = scales[source * scale_stride + tile / (GroupSize / kElementsPerThread)];
        // 0x7f80 masks the exponent. A positive normal scale <=0x7bff remains finite when multiplied by 128.
        if ((scale & 0x7f80) != 0 && scale <= 0x7bff) {
            accumulateTile<true>(q, scale, acc, source == 0);
        } else {
            accumulateTile<false>(q, scale, acc, source == 0);
        }
    }
    output[kBf16VectorsPerThread * tile]     = make_uint4(acc[0], acc[1], acc[2], acc[3]);
    output[kBf16VectorsPerThread * tile + 1] = make_uint4(acc[4], acc[5], acc[6], acc[7]);
#else
    asm volatile("trap;");
#endif
}

// Invoke function with the compile-time group size selected by group_size.
template<typename Function>
void dispatchGroup(int64_t group_size, Function function) {
    switch (group_size) {
        case 16:
            return function(std::integral_constant<int, 16>{});
        case 32:
            return function(std::integral_constant<int, 32>{});
        case 64:
            return function(std::integral_constant<int, 64>{});
        case 128:
            return function(std::integral_constant<int, 128>{});
        default:
            TORCH_CHECK(false, "group_size must be 16, 32, 64 or 128");
    }
}

// Check devices, dtypes, group_size and alignment; x is the BF16 input or output, q the INT8 codes, s the scales.
// Value finiteness and non-overlapping storage are caller preconditions, not checked here.
void checkTensors(const at::Tensor& x, const at::Tensor& q, const at::Tensor& s, int64_t group_size) {
    TORCH_CHECK(group_size == 16 || group_size == 32 || group_size == 64 || group_size == 128,
                "Unsupported group_size");
    TORCH_CHECK(x.is_cuda() && q.is_cuda() && s.is_cuda(), "CUDA tensors required");
    TORCH_CHECK(x.device() == q.device() && x.device() == s.device(), "Devices must match");
    TORCH_CHECK(at::cuda::getDeviceProperties(x.get_device())->major >= 9, "Fast BF16/INT8 requires SM90+");
    TORCH_CHECK(x.scalar_type() == at::kBFloat16 && q.scalar_type() == at::kChar && s.scalar_type() == at::kBFloat16,
                "Expected BF16 data, INT8 codes, BF16 scales");
    TORCH_CHECK(x.dim() == 2 && x.is_contiguous() && x.size(1) % group_size == 0,
                "Expected contiguous BF16 [M,K], K divisible by group_size");
    TORCH_CHECK(reinterpret_cast<uintptr_t>(x.data_ptr()) % kVectorAlignment == 0
                    && reinterpret_cast<uintptr_t>(q.data_ptr()) % kVectorAlignment == 0,
                "16-byte alignment required");
}

// Launch reduction of q[R,M,K] with scales[R,M,K/GroupSize] into out[M,K]; Inputs=0 uses runtime R.
template<int GroupSize, int Inputs>
void launchReduce(const at::Tensor& q, const at::Tensor& scales, at::Tensor& out) {
    const int64_t tiles = out.numel() / kElementsPerThread;
    reduceKernel<GroupSize, Inputs>
        <<<(tiles + kThreads - 1) / kThreads, kThreads, 0, at::cuda::getCurrentCUDAStream()>>>(
            static_cast<const int8_t*>(q.data_ptr()),
            static_cast<const uint16_t*>(scales.data_ptr()),
            static_cast<uint4*>(out.data_ptr()),
            tiles,
            q.size(0),
            q.stride(0),
            scales.stride(0));
}

}  // namespace

// Quantize BF16 x[M,K] into INT8 q[M,K] and BF16 scales[M,K/group_size] on the current CUDA stream.
void fastBf16Int8Quantize(at::Tensor x, at::Tensor q, at::Tensor scales, int64_t group_size) {
    checkTensors(x, q, scales, group_size);
    TORCH_CHECK(q.sizes() == x.sizes() && q.is_contiguous(), "Expected contiguous INT8 [M,K]");
    TORCH_CHECK(scales.dim() == 2 && scales.size(0) == x.size(0) && scales.size(1) == x.size(1) / group_size
                    && scales.is_contiguous(),
                "Expected contiguous BF16 scales [M,K/group_size]");
    const c10::cuda::CUDAGuard guard(x.device());
    const int64_t              groups = x.numel() / group_size;
    if (groups == 0) {
        return;
    }
    const int64_t tiles = x.numel() / kElementsPerThread;
    dispatchGroup(group_size, [&](auto group) {
        quantizeKernel<decltype(group)::value>
            <<<(tiles + kThreads - 1) / kThreads, kThreads, 0, at::cuda::getCurrentCUDAStream()>>>(
                static_cast<const uint4*>(x.data_ptr()),
                static_cast<uint4*>(q.data_ptr()),
                static_cast<uint16_t*>(scales.data_ptr()),
                groups);
    });
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

// Validate q[R,M,K], scales[R,M,K/group_size] and out[M,K], then dispatch by group size and source count.
void fastBf16Int8DequantizeReduce(at::Tensor q, at::Tensor scales, at::Tensor out, int64_t group_size) {
    checkTensors(out, q, scales, group_size);
    TORCH_CHECK(q.dim() == 3 && q.size(0) > 0 && q.size(1) == out.size(0) && q.size(2) == out.size(1),
                "Expected INT8 [R,M,K], R > 0");
    TORCH_CHECK(scales.dim() == 3 && scales.size(0) == q.size(0) && scales.size(1) == out.size(0)
                    && scales.size(2) == out.size(1) / group_size,
                "Expected BF16 scales [R,M,K/group_size]");
    TORCH_CHECK(q.select(0, 0).is_contiguous() && scales.select(0, 0).is_contiguous(),
                "Each source must be contiguous");
    TORCH_CHECK(q.size(0) == 1
                    || (q.stride(0) >= out.numel() && q.stride(0) % kVectorAlignment == 0
                        && scales.stride(0) >= out.numel() / group_size),
                "Invalid source strides");
    const c10::cuda::CUDAGuard guard(out.device());
    if (out.numel() == 0) {
        return;
    }
    dispatchGroup(group_size, [&](auto group) {
        constexpr int size = decltype(group)::value;
        switch (q.size(0)) {
            case 1:
                launchReduce<size, 1>(q, scales, out);
                break;
            case 2:
                launchReduce<size, 2>(q, scales, out);
                break;
            case 4:
                launchReduce<size, 4>(q, scales, out);
                break;
            case 8:
                launchReduce<size, 8>(q, scales, out);
                break;
            default:
                launchReduce<size, 0>(q, scales, out);
                break;
        }
    });
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

}  // namespace rtp_llm

#endif  // !USE_PPU
