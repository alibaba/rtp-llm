#pragma once
#include <cuda_bf16.h>
#include <rtp_llm/models_py/bindings/cuda/kernels/engram/common/post_args.h>
namespace rtp_llm::engram {
CUTLASS_DEVICE void front_post_tiled_worker(const FrontArguments& a, uint32_t m, uint32_t worker) {
    const uint32_t lane = threadIdx.x & 31;
    for (uint32_t task = uint32_t(blockIdx.x) * 2 + worker; task < m * 4; task += uint32_t(gridDim.x) * 2) {
        const uint32_t row = task / 4, tile = task % 4;
        float          p[4], c[4][4];
#pragma unroll
        for (int r = 0; r < 4; ++r) {
            p[r] = a.post[row * 4 + r];
#pragma unroll
            for (int s = 0; s < 4; ++s)
                c[r][s] = a.comb[row * 16 + s * 4 + r];
        }
        for (uint32_t j = tile * 1280 + lane * 8; j < (tile + 1) * 1280; j += 256) {
            union Vec {
                uint4         word;
                __nv_bfloat16 values[8];
            } xv, rv[4], out;
            xv.word = *reinterpret_cast<const uint4*>(static_cast<const __nv_bfloat16*>(a.x) + row * 5120 + j);
#pragma unroll
            for (int s = 0; s < 4; ++s)
                rv[s].word = *reinterpret_cast<const uint4*>(static_cast<const __nv_bfloat16*>(a.residual)
                                                             + (row * 4 + s) * 5120 + j);
#pragma unroll
            for (int r = 0; r < 4; ++r) {
#pragma unroll
                for (int i = 0; i < 8; ++i) {
                    float h = __fmul_rn(c[r][0], __bfloat162float(rv[0].values[i]));
                    h       = __fmaf_rn(p[r], __bfloat162float(xv.values[i]), h);
#pragma unroll
                    for (int s = 1; s < 4; ++s)
                        h = __fmaf_rn(c[r][s], __bfloat162float(rv[s].values[i]), h);
                    out.values[i] = __float2bfloat16_rn(h);
                }
                *reinterpret_cast<uint4*>(static_cast<__nv_bfloat16*>(a.hpost) + (row * 4 + r) * 5120 + j) = out.word;
            }
        }
    }
}
}  // namespace rtp_llm::engram
