#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <pybind11/pybind11.h>
#include <cstdint>
#include <stdexcept>
namespace rtp_llm {
union alignas(16) MtpBf16x8 {
    uint4         bits;
    __nv_bfloat16 values[8];
    __device__    MtpBf16x8() {}
};
// One 8-element vector per thread matches the original FlashInfer reduction.
__global__ void mtp_frontend(const __nv_bfloat16* embedding,
                             const __nv_bfloat16* hidden,
                             const __nv_bfloat16* ew,
                             const __nv_bfloat16* hw,
                             const int32_t*       ids,
                             const int32_t*       positions,
                             __nv_bfloat16*       output,
                             __nv_bfloat16*       residual,
                             int32_t              d,
                             float                eps) {
    int32_t              row = blockIdx.x, side = blockIdx.y;
    int32_t              tx = threadIdx.x, ty = threadIdx.y;
    int32_t              base       = (tx + 32 * ty) * 8;
    int64_t              row_offset = static_cast<int64_t>(row) * d;
    const __nv_bfloat16* x          = side ? hidden + row_offset : embedding + static_cast<int64_t>(ids[row]) * d;
    const __nv_bfloat16* w          = side ? hw : ew;
    MtpBf16x8            input;
    input.bits = !side && positions[row] == 0 ? make_uint4(0, 0, 0, 0) : *reinterpret_cast<const uint4*>(x + base);
    float sum  = 0.f;
#pragma unroll
    for (int j = 0; j < 8; ++j) {
        float v = __bfloat162float(input.values[j]);
        sum += v * v;
    }
#pragma unroll
    for (int off = 16; off > 0; off /= 2)
        sum += __shfl_xor_sync(0xffffffff, sum, off);
    __shared__ float shared[32];
    shared[ty] = sum;
    __syncthreads();
    if (ty == 0) {
        sum = tx < blockDim.y ? shared[tx] : 0.f;
#pragma unroll
        for (int off = 16; off > 0; off /= 2)
            sum += __shfl_xor_sync(0xffffffff, sum, off);
        shared[0] = sum;
    }
    __syncthreads();
    float     inv = rsqrtf(shared[0] / float(d) + eps);
    MtpBf16x8 weight, result;
    weight.bits = *reinterpret_cast<const uint4*>(w + base);
#pragma unroll
    for (int j = 0; j < 8; ++j) {
        float v = __bfloat162float(input.values[j]);
        // Preserve FlashInfer's runtime weight_bias=+0 addition, including -0.
        float gamma      = __fadd_rn(0.f, __bfloat162float(weight.values[j]));
        result.values[j] = __float2bfloat16(v * inv * gamma);
    }
    *reinterpret_cast<uint4*>(output + row_offset * 2 + side * d + base) = result.bits;
    if (side == 0)
        *reinterpret_cast<uint4*>(residual + row_offset + base) = make_uint4(0, 0, 0, 0);
}
void launch(uintptr_t e,
            uintptr_t h,
            uintptr_t ew,
            uintptr_t hw,
            uintptr_t ids,
            uintptr_t positions,
            uintptr_t out,
            uintptr_t residual,
            int       m,
            int       d,
            float     eps,
            uintptr_t stream) {
    if (d != 6144 || ((e | h | ew | hw | out | residual) & 15))
        throw std::invalid_argument("MTP frontend requires H6144 and aligned BF16 rows");
    mtp_frontend<<<dim3(m, 2), dim3(32, 24), 0, (cudaStream_t)stream>>>((const __nv_bfloat16*)e,
                                                                        (const __nv_bfloat16*)h,
                                                                        (const __nv_bfloat16*)ew,
                                                                        (const __nv_bfloat16*)hw,
                                                                        (const int32_t*)ids,
                                                                        (const int32_t*)positions,
                                                                        (__nv_bfloat16*)out,
                                                                        (__nv_bfloat16*)residual,
                                                                        d,
                                                                        eps);
    auto err = cudaGetLastError();
    if (err != cudaSuccess)
        throw std::runtime_error(cudaGetErrorString(err));
}
}  // namespace rtp_llm
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("launch", &rtp_llm::launch);
}
