// SPDX-License-Identifier: Apache-2.0
// Synchronization adapted from FlashInfer's vLLM custom_all_reduce.cuh:
// https://github.com/vllm-project/vllm/blob/v0.8.2/csrc/custom_all_reduce.cuh
// Preserve its per-CTA alternating counter protocol and both barriers. Inputs
// are copied to dedicated IPC staging; peer reads never alias the output.
#include "rtp_llm/models_py/bindings/cuda/kernels/head_ipc_gather.h"
#include <cuda.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAStream.h>
#include <cstdint>

namespace rtp_llm {
namespace qwen_head_ipc {
constexpr int                 kMaxBlocks         = 36;
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

struct __align__(16) RankData {
    void* ptrs[8];
};

struct __align__(16) RankSignals {
    Signal* signals[8];
};

#define DINLINE __device__ __forceinline__
static DINLINE void st_flag_release(FlagType* flag_addr, FlagType flag) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 700
    asm volatile("st.release.sys.global.u32 [%1], %0;" ::"r"(flag), "l"(flag_addr));
#else
    asm volatile("membar.sys; st.volatile.global.u32 [%1], %0;" ::"r"(flag), "l"(flag_addr));
#endif
}

static DINLINE FlagType ld_flag_acquire(FlagType* flag_addr) {
    FlagType flag;
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 700
    asm volatile("ld.acquire.sys.global.u32 %0, [%1];" : "=r"(flag) : "l"(flag_addr));
#else
    asm volatile("ld.volatile.global.u32 %0, [%1]; membar.gl;" : "=r"(flag) : "l"(flag_addr));
#endif
    return flag;
}

static DINLINE void st_flag_volatile(FlagType* flag_addr, FlagType flag) {
    asm volatile("st.volatile.global.u32 [%1], %0;" ::"r"(flag), "l"(flag_addr));
}

static DINLINE FlagType ld_flag_volatile(FlagType* flag_addr) {
    FlagType flag;
    asm volatile("ld.volatile.global.u32 %0, [%1];" : "=r"(flag) : "l"(flag_addr));
    return flag;
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
        // Increment the counter. Technically we only need one counter, but we use
        // multiple per block to eliminate the need to share the counter via smem.
        auto val = self_sg->self_counter[blockIdx.x][threadIdx.x] += 1;
        // Write the expected counter value to peer and wait for correct value from
        // peer.
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

#undef DINLINE
}  // namespace qwen_head_ipc
__global__ void __launch_bounds__(512, 1) headIpcGatherFp32(qwen_head_ipc::RankData    data,
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

int64_t headIpcMetaSize() {
    return sizeof(qwen_head_ipc::Signal);
}

torch::Tensor headIpcGather(const torch::Tensor&        input,
                            const std::vector<int64_t>& buffers,
                            const std::vector<int64_t>& signals,
                            int64_t                     rank) {
    TORCH_CHECK(input.is_cuda() && input.scalar_type() == torch::kBFloat16 && input.dim() == 2 && input.is_contiguous(),
                "Head IPC input must be contiguous CUDA BF16 matrix");
    TORCH_CHECK(input.size(0) >= 1 && input.size(0) <= 32 && input.size(1) == 124160,
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
    headIpcGatherFp32<<<16, 512, 0, stream>>>(data,
                                              peer_signals,
                                              peer_signals.signals[rank],
                                              reinterpret_cast<uint4*>(output.data_ptr<float>()),
                                              rank,
                                              input.numel() / 8,
                                              input.size(1) / 8);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return output;
}

void headIpcClose(const std::vector<int64_t>& pointers, int64_t rank) {
    TORCH_CHECK(pointers.size() == 2 && rank >= 0 && rank < 2, "Head IPC close requires two rank pointers");
    C10_CUDA_CHECK(cudaIpcCloseMemHandle(reinterpret_cast<void*>(pointers[1 - rank])));
    C10_CUDA_CHECK(cudaFree(reinterpret_cast<void*>(pointers[rank])));
}
}  // namespace rtp_llm
