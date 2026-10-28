#pragma once
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <rtp_llm/models_py/bindings/cuda/kernels/engram/common/utils.cuh>
namespace rtp_llm::engram {
__device__ __forceinline__ float native_full_div(float a, float b) {
    float v;
    asm("div.full.f32 %0, %1, %2;" : "=f"(v) : "f"(a), "f"(b));
    return v;
}
__device__ __forceinline__ float native_rsqrt(float a) {
    float v;
    asm("rsqrt.approx.ftz.f32 %0, %1;" : "=f"(v) : "f"(a));
    return v;
}
__global__ __launch_bounds__(128) void engram_gate_update_kernel(const nv_bfloat16* h,
                                                                 const nv_bfloat16* kv,
                                                                 const nv_bfloat16* qw,
                                                                 const nv_bfloat16* kw,
                                                                 const bool*        mask,
                                                                 nv_bfloat16*       updated,
                                                                 float              eps) {
    cudaGridDependencySynchronize();
    const int tid = threadIdx.x, lane = tid % 32, warp = tid / 32, m = blockIdx.x, r = blockIdx.y;
    float     sh = 0, sk = 0, sd = 0;
    // The scalar accumulation order matches native; memory is transferred as
    // the native contiguous eight-BF16 vector owned by each lane.
    uint4 held_h[5];
#pragma unroll
    for (int chunk = 0; chunk < 5; ++chunk) {
        const int j        = tid * 8 + chunk * 1024;
        held_h[chunk]      = *reinterpret_cast<const uint4*>(h + (m * 4 + r) * 5120 + j);
        const uint4     kb = *reinterpret_cast<const uint4*>(kv + m * 25600 + r * 5120 + j);
        const uint4     qb = *reinterpret_cast<const uint4*>(qw + r * 5120 + j);
        const uint4     wb = *reinterpret_cast<const uint4*>(kw + r * 5120 + j);
        const uint32_t* hp = reinterpret_cast<const uint32_t*>(&held_h[chunk]);
        const uint32_t* kp = reinterpret_cast<const uint32_t*>(&kb);
        const uint32_t* qp = reinterpret_cast<const uint32_t*>(&qb);
        const uint32_t* wp = reinterpret_cast<const uint32_t*>(&wb);
#pragma unroll
        for (int pair = 0; pair < 4; ++pair) {
            const float2 av  = __bfloat1622float2(*reinterpret_cast<const nv_bfloat162*>(hp + pair));
            const float2 kvv = __bfloat1622float2(*reinterpret_cast<const nv_bfloat162*>(kp + pair));
            const float2 qv  = __bfloat1622float2(*reinterpret_cast<const nv_bfloat162*>(qp + pair));
            const float2 wv  = __bfloat1622float2(*reinterpret_cast<const nv_bfloat162*>(wp + pair));
            sh               = __fadd_rn(sh, __fmul_rn(av.x, av.x));
            sk               = __fadd_rn(sk, __fmul_rn(kvv.x, kvv.x));
            sd               = __fadd_rn(sd, __fmul_rn(__fmul_rn(__fmul_rn(av.x, qv.x), wv.x), kvv.x));
            sh               = __fadd_rn(sh, __fmul_rn(av.y, av.y));
            sk               = __fadd_rn(sk, __fmul_rn(kvv.y, kvv.y));
            sd               = __fadd_rn(sd, __fmul_rn(__fmul_rn(__fmul_rn(av.y, qv.y), wv.y), kvv.y));
        }
    }
#pragma unroll
    for (int d = 16; d; d /= 2) {
        sh = __fadd_rn(sh, __shfl_xor_sync(0xffffffff, sh, d));
        sk = __fadd_rn(sk, __shfl_xor_sync(0xffffffff, sk, d));
        sd = __fadd_rn(sd, __shfl_xor_sync(0xffffffff, sd, d));
    }
    __shared__ float stats[3][4];
    __shared__ float gate_value;
    if (lane == 0) {
        stats[0][warp] = sh;
        stats[1][warp] = sk;
        stats[2][warp] = sd;
    }
    __syncthreads();
    if (tid == 0) {
        // Production reduction uses butterfly 2 then 1 across four warps.
        sh              = __fadd_rn(__fadd_rn(stats[0][0], stats[0][2]), __fadd_rn(stats[0][1], stats[0][3]));
        sk              = __fadd_rn(__fadd_rn(stats[1][0], stats[1][2]), __fadd_rn(stats[1][1], stats[1][3]));
        sd              = __fadd_rn(__fadd_rn(stats[2][0], stats[2][2]), __fadd_rn(stats[2][1], stats[2][3]));
        const float rh  = native_rsqrt(__fadd_rn(native_full_div(sh, 5120.f), eps));
        const float rk  = native_rsqrt(__fadd_rn(native_full_div(sk, 5120.f), eps));
        const float dot = __fmul_rn(__fmul_rn(sd, __fmul_rn(rh, rk)), 0.013975424859373685f);
        float       inp = fmaxf(fabsf(dot), 1.e-6f), sq;
        asm("sqrt.approx.ftz.f32 %0, %1;" : "=f"(sq) : "f"(inp));
        inp                  = copysignf(sq, dot);
        const float exponent = __fmul_rn(__fsub_rn(0.f, inp), 1.4426950408889634f);
        float       expv;
        asm("ex2.approx.f32 %0, %1;" : "=f"(expv) : "f"(exponent));
        float g    = native_full_div(1.f, __fadd_rn(expv, 1.f));
        gate_value = (mask && !mask[m]) ? 0.f : g;
    }
    __syncthreads();
#pragma unroll
    for (int chunk = 0; chunk < 5; ++chunk) {
        const int       j  = tid * 8 + chunk * 1024;
        const uint4     vb = *reinterpret_cast<const uint4*>(kv + m * 25600 + 20480 + j);
        const uint32_t* vp = reinterpret_cast<const uint32_t*>(&vb);
        const uint32_t* hp = reinterpret_cast<const uint32_t*>(&held_h[chunk]);
        uint4           output;
        uint32_t*       op = reinterpret_cast<uint32_t*>(&output);
#pragma unroll
        for (int pair = 0; pair < 4; ++pair) {
            const float2       hv = __bfloat1622float2(*reinterpret_cast<const nv_bfloat162*>(hp + pair));
            const float2       vv = __bfloat1622float2(*reinterpret_cast<const nv_bfloat162*>(vp + pair));
            const float        g  = gate_value;
            const nv_bfloat162 v  = __float22bfloat162_rn(__ffma2_rn({g, g}, vv, hv));
            op[pair]              = *reinterpret_cast<const uint32_t*>(&v);
        }
        *reinterpret_cast<uint4*>(updated + (m * 4 + r) * 5120 + j) = output;
    }
}
}  // namespace rtp_llm::engram
