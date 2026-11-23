// SPDX-License-Identifier: Apache-2.0
// Synchronization adapted from FlashInfer's vLLM custom_all_reduce.cuh:
// https://github.com/vllm-project/vllm/blob/v0.8.2/csrc/custom_all_reduce.cuh
// Preserve its per-CTA alternating counter protocol and both barriers. Inputs
// are copied to dedicated IPC staging; peer reads never alias the output.
#include "rtp_llm/models_py/bindings/cuda/kernels/head_ipc_gather.h"
#include <algorithm>
#include <cfloat>
#include <cstdlib>
#include <cuda.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAStream.h>
#include <cstdint>

namespace rtp_llm {
namespace qwen_head_ipc {
constexpr int                 kMaxBlocks         = 64;
constexpr int                 kMaxRows           = 256;
constexpr int                 kMaxArgmaxIters    = kMaxRows / kMaxBlocks;  // 4
constexpr CUpointer_attribute rangeStartAddrAttr = CU_POINTER_ATTRIBUTE_RANGE_START_ADDR;

// Counter may overflow, but it's fine since unsigned int overflow is
// well-defined behavior.
using FlagType = uint32_t;
struct Signal {
    alignas(128) FlagType self_counter[kMaxBlocks][8];
    // Two sets of peer counters are needed for two syncs. The reason is that
    // it's possible for peer GPU block to arrive at the second sync point while
    // the current GPU block haven't passed the first sync point. Thus, peer GPU
    // may write counter+1 while current GPU is busy waiting for counter. We use
    // alternating counter array to avoid this possibility.
    alignas(128) FlagType peer_counter[2][kMaxBlocks][8];
};

struct ArgmaxSignal {
    alignas(128) FlagType cta_gen[kMaxBlocks][32];
    alignas(128) FlagType local_ready[2][kMaxBlocks][32];
    alignas(128) FlagType peer_ready[2][kMaxBlocks][32];
};

struct __align__(8) ArgmaxPair {
    float   val;
    int32_t idx;
};

struct __align__(16) RankData {
    void* ptrs[8];
};

struct __align__(16) RankSignals {
    Signal* signals[8];
};

struct __align__(16) ArgmaxRankBuffers {
    uint64_t* ptrs[2];
};

struct __align__(16) ArgmaxRankSignals {
    ArgmaxSignal* sigs[2];
};

#define DINLINE __device__ __forceinline__
static DINLINE void st_flag_release(FlagType* flag_addr, FlagType flag) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 700
    asm volatile("st.release.sys.global.u32 [%1], %0;" ::"r"(flag), "l"(flag_addr) : "memory");
#else
    asm volatile("membar.sys; st.volatile.global.u32 [%1], %0;" ::"r"(flag), "l"(flag_addr) : "memory");
#endif
}

static DINLINE FlagType ld_flag_acquire(FlagType* flag_addr) {
    FlagType flag;
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 700
    asm volatile("ld.acquire.sys.global.u32 %0, [%1];" : "=r"(flag) : "l"(flag_addr) : "memory");
#else
    asm volatile("ld.volatile.global.u32 %0, [%1]; membar.gl;" : "=r"(flag) : "l"(flag_addr) : "memory");
#endif
    return flag;
}

static DINLINE void st_flag_volatile(FlagType* flag_addr, FlagType flag) {
    asm volatile("st.volatile.global.u32 [%1], %0;" ::"r"(flag), "l"(flag_addr) : "memory");
}

static DINLINE FlagType ld_flag_volatile(FlagType* flag_addr) {
    FlagType flag;
    asm volatile("ld.volatile.global.u32 %0, [%1];" : "=r"(flag) : "l"(flag_addr) : "memory");
    return flag;
}

static DINLINE void st_u64_volatile(uint64_t* addr, uint64_t val) {
    asm volatile("st.volatile.global.u64 [%1], %0;" ::"l"(val), "l"(addr) : "memory");
}

static DINLINE uint64_t ld_u64_volatile(const uint64_t* addr) {
    uint64_t val;
    asm volatile("ld.volatile.global.u64 %0, [%1];" : "=l"(val) : "l"(addr) : "memory");
    return val;
}

static DINLINE uint64_t pack_pair(float val, int32_t idx) {
    return static_cast<uint64_t>(__float_as_uint(val)) | (static_cast<uint64_t>(static_cast<uint32_t>(idx)) << 32);
}

static DINLINE void unpack_pair(uint64_t raw, float& val, int32_t& idx) {
    val = __uint_as_float(static_cast<uint32_t>(raw & 0xffffffffULL));
    idx = static_cast<int32_t>(static_cast<uint32_t>(raw >> 32));
}

static DINLINE bool argmax_better(float va, int32_t ia, float vb, int32_t ib) {
    const bool na = isnan(va);
    const bool nb = isnan(vb);
    if (na != nb) {
        return na;
    }
    if (na || va == vb) {
        return ia < ib;
    }
    return va > vb;
}

// is_start: whether this is the very first synchronization barrier.
// need_fence: whether a memory fence is needed. If true, a release-acquire
// semantic is used to enforce memory access order before and after this
// barrier.
template<int ngpus, bool is_start, bool need_fence = false>
DINLINE void multi_gpu_barrier(const RankSignals& sg, Signal* self_sg, int rank) {
    if constexpr (!is_start)
        __syncthreads();
    static_assert(!(is_start && need_fence));  // Start barrier shouldn't need fence.
    if (threadIdx.x < ngpus) {
        auto val              = self_sg->self_counter[blockIdx.x][threadIdx.x] += 1;
        auto peer_counter_ptr = &sg.signals[threadIdx.x]->peer_counter[val % 2][blockIdx.x][rank];
        auto self_counter_ptr = &self_sg->peer_counter[val % 2][blockIdx.x][threadIdx.x];
        if constexpr (need_fence) {
            st_flag_release(peer_counter_ptr, val);
            while (ld_flag_acquire(self_counter_ptr) != val)
                ;
        } else {
            st_flag_volatile(peer_counter_ptr, val);
            while (ld_flag_volatile(self_counter_ptr) != val)
                ;
        }
    }
    if constexpr (is_start || need_fence)
        __syncthreads();
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

static bool sm90_pdl_enabled(int dev) {
    const char* env = std::getenv("RTP_LLM_DISABLE_PDL");
    if (env != nullptr && env[0] == '1' && env[1] == '\0') {
        return false;
    }
    static int cached[16] = {0};
    if (dev >= 0 && dev < 16) {
        int v = cached[dev];
        if (v != 0) {
            return v > 0;
        }
        int major = 0, minor = 0;
        if (cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, dev) == cudaSuccess
            && cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, dev) == cudaSuccess) {
            const bool ok = (major == 9 && minor == 0);
            cached[dev]   = ok ? 1 : -1;
            return ok;
        }
    }
    return false;
}

#undef DINLINE
}  // namespace qwen_head_ipc

__global__ void __launch_bounds__(512, 2) headIpcGatherFp32(qwen_head_ipc::RankData    data,
                                                            qwen_head_ipc::RankSignals signals,
                                                            qwen_head_ipc::Signal*     self,
                                                            uint4*                     output,
                                                            int                        rank,
                                                            int                        packs,
                                                            int                        columns) {
    qwen_head_ipc::multi_gpu_barrier<2, true>(signals, self, rank);
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < packs; i += gridDim.x * blockDim.x) {
        const int row = i / columns;
        const int col = i % columns;
#pragma unroll
        for (int peer = 0; peer < 2; ++peer) {
            const uint4 value = static_cast<const uint4*>(data.ptrs[peer])[i];
            const int   dst   = ((row * 2 + peer) * columns + col) * 2;
            output[dst]       = make_uint4(value.x << 16, value.x & 0xffff0000u, value.y << 16, value.y & 0xffff0000u);
            output[dst + 1]   = make_uint4(value.z << 16, value.z & 0xffff0000u, value.w << 16, value.w & 0xffff0000u);
        }
    }
    qwen_head_ipc::multi_gpu_barrier<2, false>(signals, self, rank);
}

template<bool kEnablePdl>
__global__ void __launch_bounds__(512, 2)
headIpcArgmaxKernel(const uint4* __restrict__        local_u4,
                    qwen_head_ipc::ArgmaxRankBuffers buffers,
                    qwen_head_ipc::ArgmaxRankSignals signals,
                    int64_t* __restrict__            out_tokens,
                    int                              rank,
                    int                              rows,
                    int                              cols_u4,
                    int                              valid_vocab_size,
                    int                              splits) {
    __shared__ uint32_t sh_gen;
    __shared__ float    sh_val[16];
    __shared__ int32_t  sh_idx[16];

    const int bid  = blockIdx.x;
    const int tid  = threadIdx.x;
    const int lane = tid & 31;
    const int warp = tid >> 5;

    auto* self_sig = signals.sigs[rank];
    auto* peer_sig = signals.sigs[1 - rank];
    if (tid == 0) {
        sh_gen = (self_sig->cta_gen[bid][0] += 1U);
    }
    __syncthreads();
    const uint32_t gen  = sh_gen;
    const int      slot = static_cast<int>(gen & 1U);

    uint64_t*       self_buf = buffers.ptrs[rank];
    const uint64_t* peer_buf = buffers.ptrs[1 - rank];

    const int base_global_col = rank * (cols_u4 * 8);

    if (splits > 1) {
        const int active_blocks = rows * splits;
        if (bid >= active_blocks) {
            if (tid == 0) {
                qwen_head_ipc::pdl_trigger<kEnablePdl>();
            }
            return;
        }
        const int    r       = bid / splits;
        const int    s       = bid % splits;
        const int    c_start = (cols_u4 * s) / splits;
        const int    c_end   = (cols_u4 * (s + 1)) / splits;
        const uint4* row_ptr = local_u4 + static_cast<size_t>(r) * cols_u4;

        qwen_head_ipc::pdl_wait<kEnablePdl>();

        float   best_val = -INFINITY;
        int32_t best_idx = 2147483647;
        for (int c = c_start + tid; c < c_end; c += 512) {
            const uint4 v    = row_ptr[c];
            const int   gcol = base_global_col + c * 8;
            float       vals[8];
            vals[0] = __uint_as_float(v.x << 16);
            vals[1] = __uint_as_float(v.x & 0xffff0000U);
            vals[2] = __uint_as_float(v.y << 16);
            vals[3] = __uint_as_float(v.y & 0xffff0000U);
            vals[4] = __uint_as_float(v.z << 16);
            vals[5] = __uint_as_float(v.z & 0xffff0000U);
            vals[6] = __uint_as_float(v.w << 16);
            vals[7] = __uint_as_float(v.w & 0xffff0000U);
#pragma unroll
            for (int k = 0; k < 8; ++k) {
                const int idx = gcol + k;
                if (idx < valid_vocab_size && qwen_head_ipc::argmax_better(vals[k], idx, best_val, best_idx)) {
                    best_val = vals[k];
                    best_idx = idx;
                }
            }
        }

#pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1) {
            const float   ov = __shfl_xor_sync(0xffffffffU, best_val, offset);
            const int32_t oi = __shfl_xor_sync(0xffffffffU, best_idx, offset);
            if (qwen_head_ipc::argmax_better(ov, oi, best_val, best_idx)) {
                best_val = ov;
                best_idx = oi;
            }
        }
        if (lane == 0) {
            sh_val[warp] = best_val;
            sh_idx[warp] = best_idx;
        }
        __syncthreads();

        if (warp == 0) {
            float   wv = (lane < 16) ? sh_val[lane] : -INFINITY;
            int32_t wi = (lane < 16) ? sh_idx[lane] : 2147483647;
#pragma unroll
            for (int offset = 8; offset > 0; offset >>= 1) {
                const float   ov = __shfl_xor_sync(0xffffffffU, wv, offset);
                const int32_t oi = __shfl_xor_sync(0xffffffffU, wi, offset);
                if (qwen_head_ipc::argmax_better(ov, oi, wv, wi)) {
                    wv = ov;
                    wi = oi;
                }
            }
            if (lane == 0) {
                const uint64_t packed = qwen_head_ipc::pack_pair(wv, wi);
                const size_t   off =
                    (static_cast<size_t>(slot) * qwen_head_ipc::kMaxBlocks + bid) * qwen_head_ipc::kMaxArgmaxIters;
                qwen_head_ipc::st_u64_volatile(self_buf + off, packed);
                qwen_head_ipc::st_flag_release(&self_sig->local_ready[slot][bid][0], gen);
                qwen_head_ipc::st_flag_release(&peer_sig->peer_ready[slot][bid][0], gen);
            }
            if (s != 0) {
                if (lane == 0) {
                    qwen_head_ipc::pdl_trigger<kEnablePdl>();
                }
                return;
            }
            if (lane < splits) {
                const int target_bid = bid + lane;
                while (qwen_head_ipc::ld_flag_acquire(&self_sig->local_ready[slot][target_bid][0]) != gen) {}
                while (qwen_head_ipc::ld_flag_acquire(&self_sig->peer_ready[slot][target_bid][0]) != gen) {}
            }
            __syncwarp(0xffffffffU);

            float   cv = -INFINITY;
            int32_t ci = 2147483647;
            if (lane < 2 * splits) {
                const bool     is_peer    = (lane >= splits);
                const int      target_bid = bid + (is_peer ? (lane - splits) : lane);
                const size_t   off =
                    (static_cast<size_t>(slot) * qwen_head_ipc::kMaxBlocks + target_bid)
                    * qwen_head_ipc::kMaxArgmaxIters;
                const uint64_t raw = qwen_head_ipc::ld_u64_volatile((is_peer ? peer_buf : self_buf) + off);
                qwen_head_ipc::unpack_pair(raw, cv, ci);
            }
#pragma unroll
            for (int offset = 16; offset > 0; offset >>= 1) {
                const float   ov = __shfl_xor_sync(0xffffffffU, cv, offset);
                const int32_t oi = __shfl_xor_sync(0xffffffffU, ci, offset);
                if (qwen_head_ipc::argmax_better(ov, oi, cv, ci)) {
                    cv = ov;
                    ci = oi;
                }
            }
            if (lane == 0) {
                qwen_head_ipc::pdl_trigger<kEnablePdl>();
                out_tokens[r] = static_cast<int64_t>(ci);
            }
        }
        return;
    }

    if (bid >= rows) {
        if (tid == 0) {
            qwen_head_ipc::pdl_trigger<kEnablePdl>();
        }
        return;
    }

    qwen_head_ipc::pdl_wait<kEnablePdl>();
    uint64_t local_pairs[qwen_head_ipc::kMaxArgmaxIters];

    int iter = 0;
    for (int r = bid; r < rows; r += qwen_head_ipc::kMaxBlocks, ++iter) {
        const uint4* row_ptr  = local_u4 + static_cast<size_t>(r) * cols_u4;
        float        best_val = -INFINITY;
        int32_t      best_idx = 2147483647;

        for (int c = tid; c < cols_u4; c += 512) {
            const uint4 v    = row_ptr[c];
            const int   gcol = base_global_col + c * 8;
            float       vals[8];
            vals[0] = __uint_as_float(v.x << 16);
            vals[1] = __uint_as_float(v.x & 0xffff0000U);
            vals[2] = __uint_as_float(v.y << 16);
            vals[3] = __uint_as_float(v.y & 0xffff0000U);
            vals[4] = __uint_as_float(v.z << 16);
            vals[5] = __uint_as_float(v.z & 0xffff0000U);
            vals[6] = __uint_as_float(v.w << 16);
            vals[7] = __uint_as_float(v.w & 0xffff0000U);
#pragma unroll
            for (int k = 0; k < 8; ++k) {
                const int idx = gcol + k;
                if (idx < valid_vocab_size && qwen_head_ipc::argmax_better(vals[k], idx, best_val, best_idx)) {
                    best_val = vals[k];
                    best_idx = idx;
                }
            }
        }

#pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1) {
            const float   ov = __shfl_xor_sync(0xffffffffU, best_val, offset);
            const int32_t oi = __shfl_xor_sync(0xffffffffU, best_idx, offset);
            if (qwen_head_ipc::argmax_better(ov, oi, best_val, best_idx)) {
                best_val = ov;
                best_idx = oi;
            }
        }
        if (lane == 0) {
            sh_val[warp] = best_val;
            sh_idx[warp] = best_idx;
        }
        __syncthreads();
        if (warp == 0) {
            float   wv = (lane < 16) ? sh_val[lane] : -INFINITY;
            int32_t wi = (lane < 16) ? sh_idx[lane] : 2147483647;
#pragma unroll
            for (int offset = 8; offset > 0; offset >>= 1) {
                const float   ov = __shfl_xor_sync(0xffffffffU, wv, offset);
                const int32_t oi = __shfl_xor_sync(0xffffffffU, wi, offset);
                if (qwen_head_ipc::argmax_better(ov, oi, wv, wi)) {
                    wv = ov;
                    wi = oi;
                }
            }
            if (lane == 0) {
                const uint64_t packed = qwen_head_ipc::pack_pair(wv, wi);
                local_pairs[iter]     = packed;
                const size_t off =
                    (static_cast<size_t>(slot) * qwen_head_ipc::kMaxBlocks + bid) * qwen_head_ipc::kMaxArgmaxIters
                    + iter;
                qwen_head_ipc::st_u64_volatile(self_buf + off, packed);
            }
        }
        __syncthreads();
    }

    if (tid == 0) {
        qwen_head_ipc::st_flag_release(&peer_sig->peer_ready[slot][bid][0], gen);
        while (qwen_head_ipc::ld_flag_acquire(&self_sig->peer_ready[slot][bid][0]) != gen) {}

        qwen_head_ipc::pdl_trigger<kEnablePdl>();
        int out_iter = 0;
        for (int r = bid; r < rows; r += qwen_head_ipc::kMaxBlocks, ++out_iter) {
            const size_t off =
                (static_cast<size_t>(slot) * qwen_head_ipc::kMaxBlocks + bid) * qwen_head_ipc::kMaxArgmaxIters
                + out_iter;
            const uint64_t self_raw = local_pairs[out_iter];
            const uint64_t peer_raw = qwen_head_ipc::ld_u64_volatile(peer_buf + off);
            float          sv, pv;
            int32_t        si, pi;
            qwen_head_ipc::unpack_pair(self_raw, sv, si);
            qwen_head_ipc::unpack_pair(peer_raw, pv, pi);
            out_tokens[r] = static_cast<int64_t>(qwen_head_ipc::argmax_better(sv, si, pv, pi) ? si : pi);
        }
    }
}

int64_t headIpcMetaSize() {
    return sizeof(qwen_head_ipc::Signal);
}

int64_t headIpcArgmaxMetaSize() {
    return sizeof(qwen_head_ipc::ArgmaxSignal);
}

int64_t headIpcArgmaxBufferSize() {
    return static_cast<int64_t>(2) * qwen_head_ipc::kMaxBlocks * qwen_head_ipc::kMaxArgmaxIters * sizeof(uint64_t);
}

torch::Tensor headIpcGather(const torch::Tensor&        input,
                            const std::vector<int64_t>& buffers,
                            const std::vector<int64_t>& signals,
                            int64_t                     rank) {
    TORCH_CHECK(input.is_cuda() && input.scalar_type() == torch::kBFloat16 && input.dim() == 2 && input.is_contiguous(),
                "Head IPC input must be contiguous CUDA BF16 matrix");
    TORCH_CHECK(input.size(0) >= 1 && input.size(0) <= qwen_head_ipc::kMaxRows && input.size(1) == 124160,
                "Head IPC input is outside the measured vocabulary/row domain");
    TORCH_CHECK(rank >= 0 && rank < 2 && buffers.size() == 2 && signals.size() == 2,
                "Head IPC requires exactly two rank pointers");
    for (int i = 0; i < 2; ++i)
        TORCH_CHECK(buffers[i] > 0 && buffers[i] % 16 == 0 && signals[i] > 0 && signals[i] % 128 == 0,
                    "Head IPC pointers must be non-null and aligned");
    c10::cuda::CUDAGuard guard(input.device());
    auto output = torch::empty({input.size(0), 2 * input.size(1)}, input.options().dtype(torch::kFloat32));
    auto stream = c10::cuda::getCurrentCUDAStream(input.get_device()).stream();
    qwen_head_ipc::RankData    data{};
    qwen_head_ipc::RankSignals peer_signals{};
    for (int i = 0; i < 2; ++i) {
        data.ptrs[i]            = reinterpret_cast<void*>(buffers[i]);
        peer_signals.signals[i] = reinterpret_cast<qwen_head_ipc::Signal*>(signals[i]);
    }
    C10_CUDA_CHECK(cudaMemcpyAsync(
        data.ptrs[rank], input.data_ptr(), input.numel() * sizeof(uint16_t), cudaMemcpyDeviceToDevice, stream));
    const int blocks = std::min<int>(qwen_head_ipc::kMaxBlocks, std::max<int>(16, static_cast<int>(input.size(0) * 2)));
    headIpcGatherFp32<<<blocks, 512, 0, stream>>>(data,
                                                  peer_signals,
                                                  peer_signals.signals[rank],
                                                  reinterpret_cast<uint4*>(output.data_ptr<float>()),
                                                  rank,
                                                  input.numel() / 8,
                                                  input.size(1) / 8);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return output;
}

torch::Tensor headIpcArgmax(const torch::Tensor&        input,
                            const std::vector<int64_t>& buffers,
                            const std::vector<int64_t>& signals,
                            int64_t                     rank,
                            int64_t                     valid_vocab_size) {
    TORCH_CHECK(input.is_cuda() && input.scalar_type() == torch::kBFloat16 && input.dim() == 2 && input.is_contiguous(),
                "Head IPC argmax input must be contiguous CUDA BF16 matrix");
    TORCH_CHECK(input.size(0) >= 1 && input.size(0) <= qwen_head_ipc::kMaxRows && input.size(1) == 124160,
                "Head IPC argmax input is outside [1..256, 124160]");
    TORCH_CHECK(rank >= 0 && rank < 2 && buffers.size() == 2 && signals.size() == 2,
                "Head IPC argmax requires two rank pointers");
    for (int i = 0; i < 2; ++i)
        TORCH_CHECK(buffers[i] > 0 && buffers[i] % 16 == 0 && signals[i] > 0 && signals[i] % 128 == 0,
                    "Head IPC argmax pointers must be non-null and aligned");

    const int64_t full_vocab = 2 * input.size(1);
    const int64_t eff_vocab  = (valid_vocab_size > 0 && valid_vocab_size <= full_vocab) ? valid_vocab_size : full_vocab;

    c10::cuda::CUDAGuard guard(input.device());
    auto output = torch::empty({input.size(0), 1}, input.options().dtype(torch::kInt64));
    auto stream = c10::cuda::getCurrentCUDAStream(input.get_device()).stream();

    qwen_head_ipc::ArgmaxRankBuffers rank_bufs{};
    qwen_head_ipc::ArgmaxRankSignals rank_sigs{};
    for (int i = 0; i < 2; ++i) {
        rank_bufs.ptrs[i] = reinterpret_cast<uint64_t*>(buffers[i]);
        rank_sigs.sigs[i] = reinterpret_cast<qwen_head_ipc::ArgmaxSignal*>(signals[i]);
    }

    const int rows = static_cast<int>(input.size(0));
    int       splits = 1;
    if (rows <= 32) {
        while (splits < 16 && rows * (splits * 2) <= qwen_head_ipc::kMaxBlocks) {
            splits *= 2;
        }
    }
    const dim3        grid(qwen_head_ipc::kMaxBlocks);
    const dim3        block(512);
    const auto*       in_ptr   = reinterpret_cast<const uint4*>(input.data_ptr());
    int64_t*          out_ptr  = output.data_ptr<int64_t>();
    const int         rank_i32 = static_cast<int>(rank);
    const int         cols_u4  = static_cast<int>(input.size(1) / 8);
    const int         eff_i32  = static_cast<int>(eff_vocab);
    const bool        use_pdl  = qwen_head_ipc::sm90_pdl_enabled(input.get_device());
    if (use_pdl) {
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
        C10_CUDA_CHECK(cudaLaunchKernelEx(&cfg,
                                          headIpcArgmaxKernel<true>,
                                          in_ptr,
                                          rank_bufs,
                                          rank_sigs,
                                          out_ptr,
                                          rank_i32,
                                          rows,
                                          cols_u4,
                                          eff_i32,
                                          splits));
    } else {
        headIpcArgmaxKernel<false><<<grid, block, 0, stream>>>(
            in_ptr, rank_bufs, rank_sigs, out_ptr, rank_i32, rows, cols_u4, eff_i32, splits);
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }
    return output;
}

void headIpcClose(const std::vector<int64_t>& pointers, int64_t rank) {
    TORCH_CHECK(pointers.size() == 2 && rank >= 0 && rank < 2, "Head IPC close requires two rank pointers");
    C10_CUDA_CHECK(cudaIpcCloseMemHandle(reinterpret_cast<void*>(pointers[1 - rank])));
    C10_CUDA_CHECK(cudaFree(reinterpret_cast<void*>(pointers[rank])));
}
}  // namespace rtp_llm
