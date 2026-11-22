// SPDX-License-Identifier: Apache-2.0
// Fused TP2 P2P AllReduce + 4-branch Hyper-Connection Residual Injection (+ Grouped RMSNorm + FP8 AMAX).
// Uses per-CTA 2-slot ping-pong IPC buffers and a single system-scope release-acquire barrier
// while overlapping local HBM preloads (gamma, residual branches, gates) with NVLink P2P transfer.
#include "rtp_llm/models_py/bindings/cuda/kernels/tp2_ar_hc_fused.h"
#include <algorithm>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <cstdint>
#include <cuda_runtime.h>

namespace rtp_llm {
namespace tp2_hc_ipc {

constexpr int    kMaxBlocks     = 64;
constexpr int    kMaxRows       = 512;
constexpr int    kMaxIters      = kMaxRows / kMaxBlocks;  // 8
constexpr int    kHidden        = 2560;
constexpr int    kBranches      = 4;
constexpr int    kThreads       = 320;  // 320 * 8 BF16 = 2560 columns
constexpr int    kWarps         = kThreads / 32;  // 10
constexpr size_t kRowU4         = kHidden / 8;    // 320 uint4s
constexpr size_t kSlotStrideU4  = 2 * kRowU4;     // 640 uint4s (routed + shared)

struct Tp2HcSignal {
    alignas(128) uint32_t cta_gen[kMaxBlocks][32];
    alignas(128) uint32_t peer_ready[2][kMaxBlocks][32];
};

struct __align__(16) RankBuffers {
    uint4* ptrs[2];
};

struct __align__(16) RankSignals {
    Tp2HcSignal* sigs[2];
};

#define DINLINE __device__ __forceinline__

static DINLINE void st_flag_release(uint32_t* addr, uint32_t val) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 700
    asm volatile("st.release.sys.global.u32 [%1], %0;" ::"r"(val), "l"(addr) : "memory");
#else
    asm volatile("membar.sys; st.volatile.global.u32 [%1], %0;" ::"r"(val), "l"(addr) : "memory");
#endif
}

static DINLINE uint32_t ld_flag_acquire(uint32_t* addr) {
    uint32_t val;
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 700
    asm volatile("ld.acquire.sys.global.u32 %0, [%1];" : "=r"(val) : "l"(addr) : "memory");
#else
    asm volatile("ld.volatile.global.u32 %0, [%1]; membar.gl;" : "=r"(val) : "l"(addr) : "memory");
#endif
    return val;
}

static DINLINE void st_u4_volatile(uint4* addr, uint4 val) {
    asm volatile("st.volatile.global.v4.u32 [%0], {%1, %2, %3, %4};"
                 ::"l"(addr), "r"(val.x), "r"(val.y), "r"(val.z), "r"(val.w)
                 : "memory");
}

static DINLINE uint4 ld_u4_volatile(const uint4* addr) {
    uint4 val;
    asm volatile("ld.volatile.global.v4.u32 {%0, %1, %2, %3}, [%4];"
                 : "=r"(val.x), "=r"(val.y), "=r"(val.z), "=r"(val.w)
                 : "l"(addr)
                 : "memory");
    return val;
}

static DINLINE uint16_t f32_to_bf16_bits(float x) {
    uint16_t out;
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    asm("cvt.rn.bf16.f32 %0, %1;" : "=h"(out) : "f"(x));
#else
    uint32_t u = __float_as_uint(x);
    if ((u & 0x7fffffffU) > 0x7f800000U) {
        out = static_cast<uint16_t>((u >> 16) | 0x0040U);
    } else {
        out = static_cast<uint16_t>((u + (0x7fffU + ((u >> 16) & 1U))) >> 16);
    }
#endif
    return out;
}

static DINLINE float bf16_bits_to_f32(uint16_t b) {
    return __uint_as_float(static_cast<uint32_t>(b) << 16);
}

static DINLINE uint32_t pack_bf16x2(float lo, float hi) {
    uint32_t out;
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
    asm("cvt.rn.bf16x2.f32 %0, %2, %1;" : "=r"(out) : "f"(lo), "f"(hi));
#else
    out = static_cast<uint32_t>(f32_to_bf16_bits(lo)) | (static_cast<uint32_t>(f32_to_bf16_bits(hi)) << 16);
#endif
    return out;
}

static DINLINE void unpack_u4_bf16(uint4 v, float out[8]) {
    out[0] = __uint_as_float(v.x << 16);
    out[1] = __uint_as_float(v.x & 0xffff0000U);
    out[2] = __uint_as_float(v.y << 16);
    out[3] = __uint_as_float(v.y & 0xffff0000U);
    out[4] = __uint_as_float(v.z << 16);
    out[5] = __uint_as_float(v.z & 0xffff0000U);
    out[6] = __uint_as_float(v.w << 16);
    out[7] = __uint_as_float(v.w & 0xffff0000U);
}

static DINLINE uint4 pack_u4_bf16(const float in[8]) {
    return make_uint4(
        pack_bf16x2(in[0], in[1]),
        pack_bf16x2(in[2], in[3]),
        pack_bf16x2(in[4], in[5]),
        pack_bf16x2(in[6], in[7]));
}

static DINLINE size_t ipc_slot_offset(int slot, int bid, int iter) {
    return ((static_cast<size_t>(slot) * kMaxBlocks + bid) * kMaxIters + iter) * kSlotStrideU4;
}

template<bool kPackedDual>
static DINLINE void reduce_tp2_u4(uint4 lr, uint4 pr, uint4 ls, uint4 ps, float sub[8]) {
    float lrf[8], prf[8];
    unpack_u4_bf16(lr, lrf);
    unpack_u4_bf16(pr, prf);
    if constexpr (!kPackedDual) {
#pragma unroll
        for (int k = 0; k < 8; ++k) {
            sub[k] = bf16_bits_to_f32(f32_to_bf16_bits(__fadd_rn(lrf[k], prf[k])));
        }
    } else {
        float lsf[8], psf[8];
        unpack_u4_bf16(ls, lsf);
        unpack_u4_bf16(ps, psf);
#pragma unroll
        for (int k = 0; k < 8; ++k) {
            float r_sum = bf16_bits_to_f32(f32_to_bf16_bits(__fadd_rn(lrf[k], prf[k])));
            float s_sum = bf16_bits_to_f32(f32_to_bf16_bits(__fadd_rn(lsf[k], psf[k])));
            sub[k]      = bf16_bits_to_f32(f32_to_bf16_bits(__fadd_rn(r_sum, s_sum)));
        }
    }
}

template<bool kEnabled>
static DINLINE void pdl_wait() {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
    if constexpr (kEnabled) {
        cudaGridDependencySynchronize();
    }
#endif
}

template<bool kEnabled>
static DINLINE void pdl_trigger() {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
    if constexpr (kEnabled) {
        cudaTriggerProgrammaticLaunchCompletion();
    }
#endif
}

template<bool kPackedDual, bool kCollectAmax, bool kEnablePdl>
__global__ void __launch_bounds__(kThreads, 2)
tp2ArHcInjectNormKernel(const uint4* __restrict__    hyper_x,
                        const uint4* __restrict__    local_sub,
                        const uint64_t* __restrict__ gates_packed,
                        const uint4* __restrict__    gamma,
                        uint4* __restrict__          residual_out,
                        uint4* __restrict__          norm_out,
                        float* __restrict__          amax_out,
                        RankBuffers                  buffers,
                        RankSignals                  signals,
                        int                          rank,
                        int                          rows,
                        float                        eps) {
    __shared__ uint32_t sh_gen;
    __shared__ float    sh_warp[kWarps];
    __shared__ float    sh_scalar;

    const int bid  = blockIdx.x;
    const int tid  = threadIdx.x;
    const int lane = tid & 31;
    const int warp = tid >> 5;

    Tp2HcSignal* self_sig = signals.sigs[rank];
    Tp2HcSignal* peer_sig = signals.sigs[1 - rank];
    if (tid == 0) {
        sh_gen = (self_sig->cta_gen[bid][0] += 1U);
    }

    // Preload gamma (independent of preceding sublayer GEMM) during PDL overlap window.
    uint4 g_vec[kBranches];
#pragma unroll
    for (int g = 0; g < kBranches; ++g) {
        g_vec[g] = gamma[g * kRowU4 + tid];
    }

    const bool single_iter = (rows <= gridDim.x);
    uint4      x_first[kBranches];
    uint64_t   gate_first = 0;
    if (single_iter) {
#pragma unroll
        for (int g = 0; g < kBranches; ++g) {
            x_first[g] = hyper_x[(static_cast<size_t>(bid) * kBranches + g) * kRowU4 + tid];
        }
        gate_first = gates_packed[bid];
    }

    __syncthreads();
    const uint32_t gen  = sh_gen;
    const int      slot = static_cast<int>(gen & 1U);

    uint4*       self_buf = buffers.ptrs[rank];
    const uint4* peer_buf = buffers.ptrs[1 - rank];

    pdl_wait<kEnablePdl>();

    uint4 lr_reg{}, ls_reg{};
    int   iter = 0;
    for (int r = bid; r < rows; r += gridDim.x, ++iter) {
        const size_t base_off = ipc_slot_offset(slot, bid, iter);
        const uint4  lr       = local_sub[static_cast<size_t>(r) * kRowU4 + tid];
        st_u4_volatile(self_buf + base_off + tid, lr);
        if (iter == 0) {
            lr_reg = lr;
        }
        if constexpr (kPackedDual) {
            const uint4 ls = local_sub[static_cast<size_t>(rows + r) * kRowU4 + tid];
            st_u4_volatile(self_buf + base_off + kRowU4 + tid, ls);
            if (iter == 0) {
                ls_reg = ls;
            }
        }
    }

    __syncthreads();
    if (tid == 0) {
        st_flag_release(&peer_sig->peer_ready[slot][bid][0], gen);
    }

    // If multi-iter, preload iter 0's X and gates while NVLink P2P transfer is in flight.
    if (!single_iter) {
#pragma unroll
        for (int g = 0; g < kBranches; ++g) {
            x_first[g] = hyper_x[(static_cast<size_t>(bid) * kBranches + g) * kRowU4 + tid];
        }
        gate_first = gates_packed[bid];
    }

    if (tid == 0) {
        while (ld_flag_acquire(&self_sig->peer_ready[slot][bid][0]) != gen) {}
    }
    __syncthreads();

    float gamma_f[kBranches][8];
#pragma unroll
    for (int g = 0; g < kBranches; ++g) {
        unpack_u4_bf16(g_vec[g], gamma_f[g]);
    }

    iter = 0;
    for (int r = bid; r < rows; r += gridDim.x, ++iter) {
        const bool   last_iter = (r + gridDim.x >= rows);
        const size_t base_off  = ipc_slot_offset(slot, bid, iter);
        const uint4  lr        = (iter == 0) ? lr_reg : ld_u4_volatile(self_buf + base_off + tid);
        const uint4  pr        = ld_u4_volatile(peer_buf + base_off + tid);
        uint4        ls{}, ps{};
        if constexpr (kPackedDual) {
            ls = (iter == 0) ? ls_reg : ld_u4_volatile(self_buf + base_off + kRowU4 + tid);
            ps = ld_u4_volatile(peer_buf + base_off + kRowU4 + tid);
        }

        uint4    x_cur[kBranches];
        uint64_t gate_cur;
        if (iter == 0) {
#pragma unroll
            for (int g = 0; g < kBranches; ++g) {
                x_cur[g] = x_first[g];
            }
            gate_cur = gate_first;
        } else {
#pragma unroll
            for (int g = 0; g < kBranches; ++g) {
                x_cur[g] = hyper_x[(static_cast<size_t>(r) * kBranches + g) * kRowU4 + tid];
            }
            gate_cur = gates_packed[r];
        }

        float sub[8];
        reduce_tp2_u4<kPackedDual>(lr, pr, ls, ps, sub);

#pragma unroll
        for (int g = 0; g < kBranches; ++g) {
            const uint16_t gate_bf16 = static_cast<uint16_t>((gate_cur >> (g * 16)) & 0xffffU);
            const float    gate_f    = bf16_bits_to_f32(gate_bf16);

            float xf[8];
            unpack_u4_bf16(x_cur[g], xf);

            float res[8];
            float local_sq = 0.0f;
#pragma unroll
            for (int k = 0; k < 8; ++k) {
                float inj      = bf16_bits_to_f32(f32_to_bf16_bits(__fmul_rn(sub[k], gate_f)));
                uint16_t rbits = f32_to_bf16_bits(__fadd_rn(xf[k], inj));
                float rf       = bf16_bits_to_f32(rbits);
                res[k]         = rf;
                local_sq       = __fmaf_rn(rf, rf, local_sq);
            }

            const size_t out_idx  = (static_cast<size_t>(r) * kBranches + g) * kRowU4 + tid;
            residual_out[out_idx] = pack_u4_bf16(res);

#pragma unroll
            for (int offset = 16; offset > 0; offset >>= 1) {
                local_sq += __shfl_xor_sync(0xffffffffU, local_sq, offset);
            }
            if (lane == 0) {
                sh_warp[warp] = local_sq;
            }
            __syncthreads();
            if (warp == 0) {
                float v = (lane < kWarps) ? sh_warp[lane] : 0.0f;
#pragma unroll
                for (int offset = 16; offset > 0; offset >>= 1) {
                    v += __shfl_xor_sync(0xffffffffU, v, offset);
                }
                if (lane == 0) {
                    sh_scalar = rsqrtf(v * (1.0f / 2560.0f) + eps);
                }
            }
            __syncthreads();
            const float inv_rms = sh_scalar;

            float norm_f[8];
            float local_amax = 0.0f;
#pragma unroll
            for (int k = 0; k < 8; ++k) {
                float nf  = (res[k] * inv_rms) * (1.0f + gamma_f[g][k]);
                norm_f[k] = nf;
                if constexpr (kCollectAmax) {
                    float av   = fabsf(bf16_bits_to_f32(f32_to_bf16_bits(nf)));
                    av         = (av == av) ? av : 0.0f;
                    local_amax = fmaxf(local_amax, av);
                }
            }

            if (last_iter && g == kBranches - 1 && !kCollectAmax) {
                pdl_trigger<kEnablePdl>();
            }
            norm_out[out_idx] = pack_u4_bf16(norm_f);

            if constexpr (kCollectAmax) {
#pragma unroll
                for (int offset = 16; offset > 0; offset >>= 1) {
                    local_amax = fmaxf(local_amax, __shfl_xor_sync(0xffffffffU, local_amax, offset));
                }
                if (lane == 0) {
                    sh_warp[warp] = local_amax;
                }
                __syncthreads();
                if (warp == 0) {
                    float v = (lane < kWarps) ? sh_warp[lane] : 0.0f;
#pragma unroll
                    for (int offset = 16; offset > 0; offset >>= 1) {
                        v = fmaxf(v, __shfl_xor_sync(0xffffffffU, v, offset));
                    }
                    if (lane == 0) {
                        if (last_iter && g == kBranches - 1) {
                            pdl_trigger<kEnablePdl>();
                        }
                        amax_out[r * kBranches + g] = v;
                    }
                }
                __syncthreads();
            }
        }
    }
}

template<bool kPackedDual, bool kEnablePdl>
__global__ void __launch_bounds__(kThreads, 2)
tp2ArHcInjectOnlyKernel(const uint4* __restrict__    hyper_x,
                        const uint4* __restrict__    local_sub,
                        const uint64_t* __restrict__ gates_packed,
                        uint4* __restrict__          residual_out,
                        RankBuffers                  buffers,
                        RankSignals                  signals,
                        int                          rank,
                        int                          rows) {
    __shared__ uint32_t sh_gen;

    const int bid = blockIdx.x;
    const int tid = threadIdx.x;

    Tp2HcSignal* self_sig = signals.sigs[rank];
    Tp2HcSignal* peer_sig = signals.sigs[1 - rank];
    if (tid == 0) {
        sh_gen = (self_sig->cta_gen[bid][0] += 1U);
    }

    const bool single_iter = (rows <= gridDim.x);
    uint4      x_first[kBranches];
    uint64_t   gate_first = 0;
    if (single_iter) {
#pragma unroll
        for (int g = 0; g < kBranches; ++g) {
            x_first[g] = hyper_x[(static_cast<size_t>(bid) * kBranches + g) * kRowU4 + tid];
        }
        gate_first = gates_packed[bid];
    }

    __syncthreads();
    const uint32_t gen  = sh_gen;
    const int      slot = static_cast<int>(gen & 1U);

    uint4*       self_buf = buffers.ptrs[rank];
    const uint4* peer_buf = buffers.ptrs[1 - rank];

    pdl_wait<kEnablePdl>();

    uint4 lr_reg{}, ls_reg{};
    int   iter = 0;
    for (int r = bid; r < rows; r += gridDim.x, ++iter) {
        const size_t base_off = ipc_slot_offset(slot, bid, iter);
        const uint4  lr       = local_sub[static_cast<size_t>(r) * kRowU4 + tid];
        st_u4_volatile(self_buf + base_off + tid, lr);
        if (iter == 0) {
            lr_reg = lr;
        }
        if constexpr (kPackedDual) {
            const uint4 ls = local_sub[static_cast<size_t>(rows + r) * kRowU4 + tid];
            st_u4_volatile(self_buf + base_off + kRowU4 + tid, ls);
            if (iter == 0) {
                ls_reg = ls;
            }
        }
    }

    __syncthreads();
    if (tid == 0) {
        st_flag_release(&peer_sig->peer_ready[slot][bid][0], gen);
    }

    if (!single_iter) {
#pragma unroll
        for (int g = 0; g < kBranches; ++g) {
            x_first[g] = hyper_x[(static_cast<size_t>(bid) * kBranches + g) * kRowU4 + tid];
        }
        gate_first = gates_packed[bid];
    }

    if (tid == 0) {
        while (ld_flag_acquire(&self_sig->peer_ready[slot][bid][0]) != gen) {}
    }
    __syncthreads();

    iter = 0;
    for (int r = bid; r < rows; r += gridDim.x, ++iter) {
        const bool   last_iter = (r + gridDim.x >= rows);
        const size_t base_off  = ipc_slot_offset(slot, bid, iter);
        const uint4  lr        = (iter == 0) ? lr_reg : ld_u4_volatile(self_buf + base_off + tid);
        const uint4  pr        = ld_u4_volatile(peer_buf + base_off + tid);
        uint4        ls{}, ps{};
        if constexpr (kPackedDual) {
            ls = (iter == 0) ? ls_reg : ld_u4_volatile(self_buf + base_off + kRowU4 + tid);
            ps = ld_u4_volatile(peer_buf + base_off + kRowU4 + tid);
        }

        uint4    x_cur[kBranches];
        uint64_t gate_cur;
        if (iter == 0) {
#pragma unroll
            for (int g = 0; g < kBranches; ++g) {
                x_cur[g] = x_first[g];
            }
            gate_cur = gate_first;
        } else {
#pragma unroll
            for (int g = 0; g < kBranches; ++g) {
                x_cur[g] = hyper_x[(static_cast<size_t>(r) * kBranches + g) * kRowU4 + tid];
            }
            gate_cur = gates_packed[r];
        }

        float sub[8];
        reduce_tp2_u4<kPackedDual>(lr, pr, ls, ps, sub);

        if (last_iter) {
            pdl_trigger<kEnablePdl>();
        }

#pragma unroll
        for (int g = 0; g < kBranches; ++g) {
            const uint16_t gate_bf16 = static_cast<uint16_t>((gate_cur >> (g * 16)) & 0xffffU);
            const float    gate_f    = bf16_bits_to_f32(gate_bf16);

            float xf[8];
            unpack_u4_bf16(x_cur[g], xf);

            float res[8];
#pragma unroll
            for (int k = 0; k < 8; ++k) {
                float inj = bf16_bits_to_f32(f32_to_bf16_bits(__fmul_rn(sub[k], gate_f)));
                res[k]    = __fadd_rn(xf[k], inj);
            }
            const size_t out_idx  = (static_cast<size_t>(r) * kBranches + g) * kRowU4 + tid;
            residual_out[out_idx] = pack_u4_bf16(res);
        }
    }
}

template<typename KernelFunc, typename... Args>
static void launch_maybe_pdl(KernelFunc kernel, dim3 grid, dim3 block, cudaStream_t stream, bool pdl, Args... args) {
    if (pdl) {
        cudaLaunchAttribute attr[1];
        attr[0].id                                         = cudaLaunchAttributeProgrammaticStreamSerialization;
        attr[0].val.programmaticStreamSerializationAllowed = 1;
        cudaLaunchConfig_t cfg{};
        cfg.gridDim          = grid;
        cfg.blockDim         = block;
        cfg.dynamicSmemBytes = 0;
        cfg.stream           = stream;
        cfg.attrs            = attr;
        cfg.numAttrs         = 1;
        C10_CUDA_CHECK(cudaLaunchKernelEx(&cfg, kernel, args...));
    } else {
        kernel<<<grid, block, 0, stream>>>(args...);
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }
}

#undef DINLINE
}  // namespace tp2_hc_ipc

int64_t tp2ArHcMetaSize() {
    return sizeof(tp2_hc_ipc::Tp2HcSignal);
}

int64_t tp2ArHcBufferSize() {
    return static_cast<int64_t>(2) * tp2_hc_ipc::kMaxBlocks * tp2_hc_ipc::kMaxIters
           * tp2_hc_ipc::kSlotStrideU4 * sizeof(uint4);
}

std::tuple<torch::Tensor, torch::Tensor, c10::optional<torch::Tensor>>
tp2ArHcInjectNorm(const torch::Tensor&        hyper_x,
                  const torch::Tensor&        local_sub,
                  const torch::Tensor&        gates,
                  const torch::Tensor&        gamma,
                  const std::vector<int64_t>& buffers,
                  const std::vector<int64_t>& signals,
                  int64_t                     rank,
                  double                      eps,
                  bool                        packed_dual,
                  bool                        collect_amax,
                  bool                        enable_pdl) {
    TORCH_CHECK(hyper_x.is_cuda() && local_sub.is_cuda() && gates.is_cuda() && gamma.is_cuda(),
                "tp2ArHcInjectNorm inputs must be CUDA tensors");
    TORCH_CHECK(hyper_x.scalar_type() == torch::kBFloat16 && local_sub.scalar_type() == torch::kBFloat16
                    && gates.scalar_type() == torch::kBFloat16 && gamma.scalar_type() == torch::kBFloat16,
                "tp2ArHcInjectNorm inputs must be BF16");
    TORCH_CHECK(hyper_x.is_contiguous() && local_sub.is_contiguous() && gates.is_contiguous() && gamma.is_contiguous(),
                "tp2ArHcInjectNorm inputs must be contiguous");
    const int64_t rows = hyper_x.numel() / (tp2_hc_ipc::kBranches * tp2_hc_ipc::kHidden);
    TORCH_CHECK(rows >= 1 && rows <= tp2_hc_ipc::kMaxRows
                    && hyper_x.size(-1) == tp2_hc_ipc::kBranches * tp2_hc_ipc::kHidden,
                "tp2ArHcInjectNorm hyper_x shape is outside [1..512, 10240]");
    const int64_t expected_sub_rows = packed_dual ? (2 * rows) : rows;
    TORCH_CHECK(local_sub.dim() == 2 && local_sub.size(0) == expected_sub_rows
                    && local_sub.size(1) == tp2_hc_ipc::kHidden,
                "tp2ArHcInjectNorm local_sub shape mismatch");
    TORCH_CHECK(gates.numel() == rows * tp2_hc_ipc::kBranches && gates.size(-1) == tp2_hc_ipc::kBranches,
                "tp2ArHcInjectNorm gates shape mismatch");
    TORCH_CHECK(gamma.dim() == 1 && gamma.size(0) == tp2_hc_ipc::kBranches * tp2_hc_ipc::kHidden,
                "tp2ArHcInjectNorm gamma shape mismatch");
    TORCH_CHECK(rank >= 0 && rank < 2 && buffers.size() == 2 && signals.size() == 2,
                "tp2ArHcInjectNorm requires 2 rank pointers");

    c10::cuda::CUDAGuard guard(hyper_x.device());
    auto residual = torch::empty_like(hyper_x);
    auto normed   = torch::empty_like(hyper_x);
    c10::optional<torch::Tensor> amax_opt;
    float* amax_ptr = nullptr;
    if (collect_amax) {
        auto amax = torch::empty({rows * tp2_hc_ipc::kBranches}, hyper_x.options().dtype(torch::kFloat32));
        amax_ptr  = amax.data_ptr<float>();
        amax_opt  = std::move(amax);
    }

    tp2_hc_ipc::RankBuffers rank_bufs{};
    tp2_hc_ipc::RankSignals rank_sigs{};
    for (int i = 0; i < 2; ++i) {
        rank_bufs.ptrs[i] = reinterpret_cast<uint4*>(buffers[i]);
        rank_sigs.sigs[i] = reinterpret_cast<tp2_hc_ipc::Tp2HcSignal*>(signals[i]);
    }

    const dim3 grid(static_cast<unsigned>(std::min<int64_t>(rows, tp2_hc_ipc::kMaxBlocks)));
    const dim3 block(tp2_hc_ipc::kThreads);
    auto       stream = c10::cuda::getCurrentCUDAStream(hyper_x.get_device()).stream();

    const auto* x_ptr = reinterpret_cast<const uint4*>(hyper_x.data_ptr());
    const auto* s_ptr = reinterpret_cast<const uint4*>(local_sub.data_ptr());
    const auto* g_ptr = reinterpret_cast<const uint64_t*>(gates.data_ptr());
    const auto* w_ptr = reinterpret_cast<const uint4*>(gamma.data_ptr());
    auto*       r_ptr = reinterpret_cast<uint4*>(residual.data_ptr());
    auto*       n_ptr = reinterpret_cast<uint4*>(normed.data_ptr());
    const int   r_i32 = static_cast<int>(rank);
    const int   n_i32 = static_cast<int>(rows);
    const float eps_f = static_cast<float>(eps);

#define DISPATCH_NORM(PACKED, COLLECT, PDL)                                                                            \
    tp2_hc_ipc::launch_maybe_pdl(tp2_hc_ipc::tp2ArHcInjectNormKernel<PACKED, COLLECT, PDL>,                            \
                                 grid,                                                                                 \
                                 block,                                                                                \
                                 stream,                                                                               \
                                 PDL,                                                                                  \
                                 x_ptr,                                                                                \
                                 s_ptr,                                                                                \
                                 g_ptr,                                                                                \
                                 w_ptr,                                                                                \
                                 r_ptr,                                                                                \
                                 n_ptr,                                                                                \
                                 amax_ptr,                                                                             \
                                 rank_bufs,                                                                            \
                                 rank_sigs,                                                                            \
                                 r_i32,                                                                                \
                                 n_i32,                                                                                \
                                 eps_f)

    if (packed_dual) {
        if (collect_amax) {
            if (enable_pdl) DISPATCH_NORM(true, true, true);
            else            DISPATCH_NORM(true, true, false);
        } else {
            if (enable_pdl) DISPATCH_NORM(true, false, true);
            else            DISPATCH_NORM(true, false, false);
        }
    } else {
        if (collect_amax) {
            if (enable_pdl) DISPATCH_NORM(false, true, true);
            else            DISPATCH_NORM(false, true, false);
        } else {
            if (enable_pdl) DISPATCH_NORM(false, false, true);
            else            DISPATCH_NORM(false, false, false);
        }
    }
#undef DISPATCH_NORM

    return {std::move(residual), std::move(normed), std::move(amax_opt)};
}

torch::Tensor tp2ArHcInjectOnly(const torch::Tensor&        hyper_x,
                                const torch::Tensor&        local_sub,
                                const torch::Tensor&        gates,
                                const std::vector<int64_t>& buffers,
                                const std::vector<int64_t>& signals,
                                int64_t                     rank,
                                bool                        packed_dual,
                                bool                        enable_pdl) {
    TORCH_CHECK(hyper_x.is_cuda() && local_sub.is_cuda() && gates.is_cuda(),
                "tp2ArHcInjectOnly inputs must be CUDA tensors");
    TORCH_CHECK(hyper_x.scalar_type() == torch::kBFloat16 && local_sub.scalar_type() == torch::kBFloat16
                    && gates.scalar_type() == torch::kBFloat16,
                "tp2ArHcInjectOnly inputs must be BF16");
    TORCH_CHECK(hyper_x.is_contiguous() && local_sub.is_contiguous() && gates.is_contiguous(),
                "tp2ArHcInjectOnly inputs must be contiguous");
    const int64_t rows = hyper_x.numel() / (tp2_hc_ipc::kBranches * tp2_hc_ipc::kHidden);
    TORCH_CHECK(rows >= 1 && rows <= tp2_hc_ipc::kMaxRows
                    && hyper_x.size(-1) == tp2_hc_ipc::kBranches * tp2_hc_ipc::kHidden,
                "tp2ArHcInjectOnly hyper_x shape is outside [1..512, 10240]");
    const int64_t expected_sub_rows = packed_dual ? (2 * rows) : rows;
    TORCH_CHECK(local_sub.dim() == 2 && local_sub.size(0) == expected_sub_rows
                    && local_sub.size(1) == tp2_hc_ipc::kHidden,
                "tp2ArHcInjectOnly local_sub shape mismatch");
    TORCH_CHECK(gates.numel() == rows * tp2_hc_ipc::kBranches && gates.size(-1) == tp2_hc_ipc::kBranches,
                "tp2ArHcInjectOnly gates shape mismatch");
    TORCH_CHECK(rank >= 0 && rank < 2 && buffers.size() == 2 && signals.size() == 2,
                "tp2ArHcInjectOnly requires 2 rank pointers");

    c10::cuda::CUDAGuard guard(hyper_x.device());
    auto residual = torch::empty_like(hyper_x);

    tp2_hc_ipc::RankBuffers rank_bufs{};
    tp2_hc_ipc::RankSignals rank_sigs{};
    for (int i = 0; i < 2; ++i) {
        rank_bufs.ptrs[i] = reinterpret_cast<uint4*>(buffers[i]);
        rank_sigs.sigs[i] = reinterpret_cast<tp2_hc_ipc::Tp2HcSignal*>(signals[i]);
    }

    const dim3 grid(static_cast<unsigned>(std::min<int64_t>(rows, tp2_hc_ipc::kMaxBlocks)));
    const dim3 block(tp2_hc_ipc::kThreads);
    auto       stream = c10::cuda::getCurrentCUDAStream(hyper_x.get_device()).stream();

    const auto* x_ptr = reinterpret_cast<const uint4*>(hyper_x.data_ptr());
    const auto* s_ptr = reinterpret_cast<const uint4*>(local_sub.data_ptr());
    const auto* g_ptr = reinterpret_cast<const uint64_t*>(gates.data_ptr());
    auto*       r_ptr = reinterpret_cast<uint4*>(residual.data_ptr());
    const int   r_i32 = static_cast<int>(rank);
    const int   n_i32 = static_cast<int>(rows);

    if (packed_dual) {
        if (enable_pdl) {
            tp2_hc_ipc::launch_maybe_pdl(tp2_hc_ipc::tp2ArHcInjectOnlyKernel<true, true>,
                                         grid, block, stream, true,
                                         x_ptr, s_ptr, g_ptr, r_ptr, rank_bufs, rank_sigs, r_i32, n_i32);
        } else {
            tp2_hc_ipc::launch_maybe_pdl(tp2_hc_ipc::tp2ArHcInjectOnlyKernel<true, false>,
                                         grid, block, stream, false,
                                         x_ptr, s_ptr, g_ptr, r_ptr, rank_bufs, rank_sigs, r_i32, n_i32);
        }
    } else {
        if (enable_pdl) {
            tp2_hc_ipc::launch_maybe_pdl(tp2_hc_ipc::tp2ArHcInjectOnlyKernel<false, true>,
                                         grid, block, stream, true,
                                         x_ptr, s_ptr, g_ptr, r_ptr, rank_bufs, rank_sigs, r_i32, n_i32);
        } else {
            tp2_hc_ipc::launch_maybe_pdl(tp2_hc_ipc::tp2ArHcInjectOnlyKernel<false, false>,
                                         grid, block, stream, false,
                                         x_ptr, s_ptr, g_ptr, r_ptr, rank_bufs, rank_sigs, r_i32, n_i32);
        }
    }

    return residual;
}

}  // namespace rtp_llm
