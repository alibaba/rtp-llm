#pragma once
#include <cuda/atomic>
#include <rtp_llm/models_py/bindings/cuda/kernels/engram/impls/hc_body.cuh>
#include <rtp_llm/models_py/bindings/cuda/kernels/engram/epilogue/norm_fp8.cuh>

namespace rtp_llm::engram {
namespace pre_detail {
using layout::mega_mhc::MixArgs;
using layout::mega_mhc::NormArgs;
using Atomic = cuda::atomic_ref<unsigned long long, cuda::thread_scope_device>;

CUTLASS_DEVICE float xor_sum(float v, int first, int last) {
    for (int offset = first; offset >= last; offset >>= 1)
        v += __shfl_xor_sync(0xffffffff, v, offset);
    return v;
}

template<int Splits>
CUTLASS_DEVICE void mix_row(int                 row,
                            int                 m,
                            const float*        partial,
                            const float*        sqr,
                            const MixArgs&      args,
                            unsigned long long* ready,
                            unsigned long long  epoch) {
    const int  lane     = threadIdx.x & 31;
    const auto terminal = (epoch << 32) + Splits;
    while (Atomic(ready[row / 64]).load(cuda::memory_order_acquire) != terminal) {}
    float norm = 0.f, hc = 0.f;
    // The native epilogue serially sums every split, rather than warp-reducing
    // the norm across splits. Preserve its FP32 addition order.
    // Fetch independent partials ahead of their ordered accumulations. Only
    // memory scheduling changes; every FP32 add retains native split order.
    constexpr int Prefetch = 8;
#pragma unroll 1
    for (int base = 0; base < Splits; base += Prefetch) {
        float ns[Prefetch], hs[Prefetch];
#pragma unroll
        for (int j = 0; j < Prefetch; ++j) {
            const int s = base + j;
            ns[j]       = s < Splits ? sqr[s * m + row] : 0.f;
            hs[j]       = (s < Splits && lane < 24) ? partial[(s * m + row) * 24 + lane] : 0.f;
        }
#pragma unroll
        for (int j = 0; j < Prefetch; ++j) {
            if (base + j < Splits) {
                norm += ns[j];
                hc += hs[j];
            }
        }
    }
    norm = rsqrtf(norm / 20480.f + args.hc_norm_eps);
    hc   = __fmul_rn(hc, norm);
    if (lane < 8) {
        const float affine  = fmaf(hc, args.scales[lane < 4 ? 0 : 1], args.bases[lane]);
        const float sigmoid = 1.f / (1.f + expf(-affine));
        if (lane < 4)
            args.new_prev_mix[row * 4 + lane] = sigmoid + args.hc_pre_eps;
        else
            args.new_post_mix[row * 4 + lane - 4] = sigmoid * args.hc_post_scale;
    }
    // Use all 32 lanes as two copies of the native 4x4 Sinkhorn matrix.
    const int idx = lane & 15;
    float     cm  = __shfl_sync(0xffffffff, hc, idx + 8);
    cm            = fmaf(cm, args.scales[2], args.bases[idx + 8]);
    float maximum = fmaxf(cm, __shfl_xor_sync(0xffffffff, cm, 2));
    maximum       = fmaxf(maximum, __shfl_xor_sync(0xffffffff, maximum, 1));
    cm            = expf(cm - maximum);
    cm            = cm / xor_sum(cm, 2, 1) + args.sinkhorn_eps;
    cm            = cm / (xor_sum(cm, 8, 4) + args.sinkhorn_eps);
    for (int i = 1; i < args.num_sinkhorn_iters; ++i) {
        cm = cm / (xor_sum(cm, 2, 1) + args.sinkhorn_eps);
        cm = cm / (xor_sum(cm, 8, 4) + args.sinkhorn_eps);
    }
    if (lane < 16)
        args.new_comb_res_mix[row * 16 + lane] = cm;
}

CUTLASS_DEVICE void
readout_norm_row(int row, const nv_bfloat16* h, const float* previous, const NormArgs& args, float* warp_sums) {
    using namespace epilogue::mega_mhc;
    const int local = threadIdx.x - 256;
    const int lane = local & 31, warp = local / 32;
    uint4     retained[5];
    float     coeff[4];
#pragma unroll
    for (int r = 0; r < 4; ++r)
        coeff[r] = previous[row * 4 + r];
// Emulate the native RMSNorm's 640 logical threads (20 warps), 8 BF16
// elements per thread. Five logical threads share one physical thread.
#pragma unroll
    for (int group = 0; group < 5; ++group) {
        const int col      = (local + group * 128) * 8;
        float2    value[4] = {};
#pragma unroll
        for (int r = 0; r < 4; ++r) {
            uint4      pack  = *reinterpret_cast<const uint4*>(h + (row * 4 + r) * 5120 + col);
            const auto pairs = reinterpret_cast<const nv_bfloat162*>(&pack);
#pragma unroll
            for (int j = 0; j < 4; ++j)
                value[j] = __ffma2_rn(__bfloat1622float2(pairs[j]), {coeff[r], coeff[r]}, value[j]);
        }
        auto rounded = reinterpret_cast<nv_bfloat162*>(&retained[group]);
#pragma unroll
        for (int j = 0; j < 4; ++j)
            rounded[j] = __float22bfloat162_rn(value[j]);
        const auto scalar = reinterpret_cast<const nv_bfloat16*>(&retained[group]);
        float      sum    = 0.f;
#pragma unroll
        for (int j = 0; j < 8; ++j)
            sum = fmaf(float(scalar[j]), float(scalar[j]), sum);
        sum = xor_sum(sum, 16, 1);
        if (lane == 0)
            warp_sums[group * 4 + warp] = sum;
    }
    cutlass::arch::NamedBarrier::sync(128, 3);
    if (warp == 0) {
        float sum = lane < 20 ? warp_sums[lane] : 0.f;
        sum       = xor_sum(sum, 16, 1);
        if (lane == 0) {
            // FlashInfer compiles RMSNorm with --use_fast_math and uses the
            // explicit rsqrt.approx.ftz instruction from flashinfer/math.cuh.
            const float variance = __fdividef(sum, 5120.f) + args.eps;
            float       inv;
            asm volatile("rsqrt.approx.ftz.f32 %0, %1;" : "=f"(inv) : "f"(variance));
            warp_sums[0] = inv;
        }
    }
    cutlass::arch::NamedBarrier::sync(128, 3);
    const float inv = warp_sums[0];
#pragma unroll
    for (int group = 0; group < 5; ++group) {
        const int  col     = (local + group * 128) * 8;
        uint4      weight  = *reinterpret_cast<const uint4*>(args.weight + col);
        auto       values  = reinterpret_cast<nv_bfloat162*>(&retained[group]);
        const auto weights = reinterpret_cast<const nv_bfloat162*>(&weight);
#pragma unroll
        for (int j = 0; j < 4; ++j)
            values[j] = normalize_bf16_pair(values[j], weights[j], inv);
        const auto offset                               = static_cast<uint64_t>(row) * 5120 + col;
        *reinterpret_cast<uint4*>(args.y_bf16 + offset) = retained[group];
        const auto amax                                 = math::warp_reduce_max<4>(
            math::get_packed_bf16_amax(reinterpret_cast<const nv_bfloat162(&)[4]>(retained[group])));
        const auto sf = store_fp8_and_pack_sf(args, offset, retained[group], math::get_ue8m0_sf_exp(amax));
        if (lane % 16 == 0)
            args.y_primary_sf[row * args.y_primary_sf_stride_token + (col / 128) * args.y_primary_sf_stride_word] = sf;
    }
    cutlass::arch::NamedBarrier::sync(128, 3);
}
}  // namespace pre_detail

// All 148 CTAs fit concurrently (one CTA/SM). Each HC producer runs once;
// row owners independently consume previous-pre readout or completed HC splits.
// No gate/update arithmetic or post allocation is moved across the K2 boundary.
template<int Splits, int Sms = 148>
CUTLASS_GLOBAL void __launch_bounds__(512, 1)
    engram_pre_kernel(uint32_t                m,
                      const __grid_constant__ cute::TmaDescriptor map_a,
                      const __grid_constant__ cute::TmaDescriptor map_b,
                      const __grid_constant__ cute::TmaDescriptor map_d,
                      const nv_bfloat16*                          h,
                      const float*                                previous,
                      float*                                      partial,
                      float*                                      sqr,
                      unsigned long long*                         protocol,
                      const __grid_constant__ layout::mega_mhc::MixArgs mix,
                      const __grid_constant__ layout::mega_mhc::NormArgs norm) {
    using namespace pre_detail;
    extern __shared__ __align__(1024) uint8_t smem_buffer[];
    float*                                    norm_scratch = reinterpret_cast<float*>(smem_buffer + 205312);
    __shared__ unsigned long long             epoch;
    cudaGridDependencySynchronize();
    auto ready = protocol + Sms;
    if (threadIdx.x == 0) {
        epoch = protocol[blockIdx.x] + 1;
        if (blockIdx.x < (m + 63) / 64)
            Atomic(ready[blockIdx.x]).store(epoch << 32, cuda::memory_order_release);
    }
    __syncthreads();
    if (threadIdx.x < 256) {
        if (blockIdx.x < ((m + 63) / 64) * Splits) {
            hc_body<24, 20480, 64, 32, 64, Splits, 128, 12, 128, 128>(m, map_a, map_b, map_d, sqr);
            // Join both producer groups and make every sqr/store visible before
            // the last-split release. A function return is not a TMA-store fence.
            __threadfence();
            cutlass::arch::NamedBarrier::sync(256, 1);
            if (threadIdx.x == 0) {
                auto& state = ready[blockIdx.x / Splits];
                while ((Atomic(state).load(cuda::memory_order_acquire) >> 32) != epoch) {}
                Atomic(state).fetch_add(1, cuda::memory_order_acq_rel);
            }
        }
    } else if (threadIdx.x < 384) {
        for (int row = blockIdx.x; row < m; row += Sms)
            readout_norm_row(row, h, previous, norm, norm_scratch);
    } else {
        const int warp = (threadIdx.x - 384) / 32;
        for (int row = blockIdx.x * 4 + warp; row < m; row += Sms * 4)
            mix_row<Splits>(row, m, partial, sqr, mix, ready, epoch);
    }
    __syncthreads();
    if (threadIdx.x == 0)
        protocol[blockIdx.x] = epoch;
}
}  // namespace rtp_llm::engram
