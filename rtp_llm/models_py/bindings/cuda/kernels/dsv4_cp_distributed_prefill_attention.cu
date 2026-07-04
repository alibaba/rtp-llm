#include "rtp_llm/models_py/bindings/cuda/kernels/dsv4_cp_distributed_prefill_attention.h"
#include "rtp_llm/models_py/bindings/cuda/kernels/dsv4_top_k_per_row_prefill.h"
#include "rtp_llm/models_py/bindings/cuda/kernels/user_buffer/user_buffers.h"

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cuda_fp16.h>
#include <cuda/atomic>
#include <cuda_runtime.h>
#include <torch/all.h>
#include <torch/csrc/autograd/python_variable.h>

#include <cmath>
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <optional>
#include <atomic>
#include <vector>

namespace torch_ext {
namespace py = pybind11;

#ifndef USE_ROCM
namespace {

constexpr int kMaxCompressedTopK = 4096;
constexpr int kMaxAttentionKeys = 2048;
constexpr int kProtocolBytes = 64;
constexpr int kMaxUserBufferRegions = 16;
constexpr int kSwaHeadDim = 512;
constexpr int kSwaNopeDim = 448;
constexpr int kSwaQuantBlock = 64;
constexpr int kSwaNopeTiles = 7;
constexpr int kSwaScaleBytes = 8;
constexpr int kSwaTokenDataBytes = 576;
constexpr int kSwaEntryBytes = 584;
constexpr int kIndexerEntryBytes = 132;
constexpr int kIndexerHeadDim = 128;
constexpr float kSwaFp8Max = 448.0f;
constexpr int kAttentionSignalPadChannelBase = 224;
constexpr int kMegaSideEffectReleaseChannel = 14;
constexpr int kMegaKernelDoneChannel = 15;
constexpr int64_t kSymmMemBarrierTimeoutCycles = 60000000000ll;
constexpr float kInvalidCompressorScore = -3.4028234663852886e38F;
constexpr int64_t kMegaPayloadAlignBytes = 256;
constexpr int64_t kMegaLocalPhaseOffset = 16;
constexpr int64_t kMegaLocalPhaseComplementOffset = 24;
constexpr int kMegaGridBarrierLogicalSlots = 24;
constexpr int kMegaGridBarrierEpochStride = 4;
constexpr int kMegaGridBarrierSlots = kMegaGridBarrierLogicalSlots * kMegaGridBarrierEpochStride;
constexpr int64_t kMegaGridCounterBytes =
    ((kMegaGridBarrierSlots * static_cast<int64_t>(sizeof(unsigned int)) + 63) / 64) * 64;
constexpr int64_t kMegaGridEpochBytes =
    ((kMegaGridBarrierSlots * static_cast<int64_t>(sizeof(unsigned long long)) + 63) / 64) * 64;
constexpr int64_t kMegaGridSyncBytes = kMegaGridCounterBytes + kMegaGridEpochBytes;
constexpr int kMegaNonCoopGridWaveMultiplier = 16;
constexpr int kMegaAttentionKeyTile = 4;
// Larger grouped MQA key tiles amortize packed FP8 row setup and block-wide
// synchronizations across more keys. Tile=16 still fits Blackwell opt-in shared
// memory at one resident CTA/SM while halving the long-key attention loop count
// versus tile=8.
constexpr int kMegaAttentionGroupedKeyTile = 16;
constexpr int kMegaAttentionHeadsPerCta = 8;
constexpr int kMegaAttentionWarpSize = 32;
constexpr int kMegaAttentionDimsPerWarp = kSwaHeadDim / kMegaAttentionWarpSize;
constexpr int kMegaAttentionGroupedScaleSlots = kMegaAttentionGroupedKeyTile * kSwaScaleBytes;
static_assert(kMegaAttentionGroupedKeyTile <= 16,
              "grouped attention tile needs dynamic/shared-memory retuning above 16 keys");
constexpr int kMegaSplitKKeysPerBlock = 64;
constexpr int kMegaSplitKMinKeys = 512;
constexpr int kMegaSplitKBlocksPerWave = 7;
constexpr int kMegaSplitKGroupSize = kMegaSplitKBlocksPerWave + 1;
constexpr int kMegaSplitKGroupBarrierSlots = 4;
constexpr int64_t kMegaSplitKRecordBytes =
    (((static_cast<int64_t>(kMegaAttentionHeadsPerCta) * (kSwaHeadDim + 2) * static_cast<int64_t>(sizeof(float)))
      + kMegaPayloadAlignBytes - 1)
     / kMegaPayloadAlignBytes)
    * kMegaPayloadAlignBytes;
constexpr int kMegaBlockThreads = 256;
static_assert((kMegaBlockThreads & (kMegaBlockThreads - 1)) == 0,
              "mega attention reductions require a power-of-two CTA size");
static_assert((kMegaBlockThreads % kMegaAttentionWarpSize) == 0,
              "mega attention CTA size must be warp-aligned");
constexpr int kMegaIndexerCandidateHeadLanes = 2;
constexpr int kMegaIndexerCandidateGroups =
    kMegaBlockThreads / (64 * kMegaIndexerCandidateHeadLanes);
constexpr int kMegaIndexerCandidatesPerGroup = 4;
constexpr int kMegaIndexerCandidateTile = kMegaIndexerCandidateGroups * kMegaIndexerCandidatesPerGroup;
constexpr int kMegaIndexerHeads = 64;
constexpr int kMegaIndexerWarpsPerCandidate = kMegaIndexerHeads / kMegaAttentionWarpSize;
constexpr int kMegaIndexerScorerFloatSlots =
    kMegaIndexerCandidateTile * (kMegaIndexerHeads + kMegaIndexerWarpsPerCandidate);
constexpr int kMegaIndexerMetadataBytes =
    ((kMegaIndexerCandidateTile * static_cast<int>(sizeof(const uint8_t*))
      + kMegaIndexerCandidateTile * static_cast<int>(sizeof(float)) + 7)
     / 8)
    * 8;
constexpr int kMegaReductionFloatSlots = kMegaBlockThreads * 2;
constexpr int kMegaReduceBufSlots =
    kMegaIndexerScorerFloatSlots > kMegaReductionFloatSlots ? kMegaIndexerScorerFloatSlots :
                                                              kMegaReductionFloatSlots;
constexpr int kMegaReduceResultSlots =
    kMegaIndexerCandidateTile > kMegaAttentionKeyTile ? kMegaIndexerCandidateTile : kMegaAttentionKeyTile;
static_assert(kMegaIndexerCandidateGroups * 64 * kMegaIndexerCandidateHeadLanes == kMegaBlockThreads,
              "CSA matrix scorer maps one CTA exactly across group/head/lane work");
static_assert(kMegaIndexerCandidateTile
                      + (kMegaBlockThreads / kMegaAttentionWarpSize) * kMegaIndexerCandidateTile
                  <= kMegaBlockThreads,
              "CSA worst-slot tile scratch must fit in reduce_indices_second without aliasing the tile list");
constexpr unsigned int kMegaCompressedDoneFlag = 0x80000000u;
constexpr unsigned int kMegaCompressedCountMask = 0x7fffffffu;
constexpr int kWarpDotAttentionKeyThreshold = 128;

void validateTensor(const torch::Tensor& t, const char* name, c10::ScalarType dtype, int64_t dim);
void validateFloatPayloadTensor(const torch::Tensor& t, const char* name, int64_t dim);

bool dsv4CpAttentionDebugSyncEnabled() {
    const char* raw = std::getenv("DSV4_CP_ATTENTION_DEBUG_SYNC");
    return raw != nullptr && raw[0] != '\0' && raw[0] != '0';
}

bool dsv4CpAttentionReturnAfter(const char* stage) {
    const char* raw = std::getenv("DSV4_CP_ATTENTION_RETURN_AFTER_STAGE");
    return raw != nullptr && std::strcmp(raw, stage) == 0;
}

bool dsv4CpAttentionFlashMlaEnabled() {
    const char* raw = std::getenv("DSV4_CP_ATTENTION_FLASHMLA");
    return raw == nullptr || raw[0] == '\0' || raw[0] != '0';
}

bool dsv4CpAttentionMegaKernelEnabled() {
    const char* raw = std::getenv("DSV4_CP_ATTENTION_MEGA_KERNEL");
    return raw == nullptr || raw[0] == '\0' || raw[0] != '0';
}

bool dsv4CpAttentionMegaGridSideEffectsEnabled() {
    const char* raw = std::getenv("DSV4_CP_ATTENTION_MEGA_GRID_SIDE_EFFECTS");
    return raw != nullptr && raw[0] != '\0' && raw[0] != '0';
}

bool dsv4CpAttentionMegaSplitKEnabled() {
    const char* raw = std::getenv("DSV4_CP_ATTENTION_MEGA_SPLITK");
    return raw != nullptr && raw[0] != '\0' && raw[0] != '0';
}

bool dsv4CpAttentionMegaSplitKDisabled() {
    const char* raw = std::getenv("DSV4_CP_ATTENTION_MEGA_SPLITK");
    return raw != nullptr && raw[0] == '0';
}

__host__ __device__ __forceinline__ int64_t alignUpInt64(int64_t value, int64_t alignment) {
    return ((value + alignment - 1) / alignment) * alignment;
}

torch::Tensor makeZeroAttentionOutput(const torch::Tensor& q, int64_t rows, int64_t heads, int64_t head_dim) {
    return torch::zeros({rows, heads, head_dim}, q.options());
}

py::object tensorToPyObject(const torch::Tensor& tensor) {
    return py::reinterpret_steal<py::object>(THPVariable_Wrap(tensor));
}

void debugStage(bool enabled, int64_t cp_rank, const char* stage, const char* point, cudaStream_t stream, bool sync) {
    if (!enabled) {
        return;
    }
    std::fprintf(stderr, "[DSV4 CP Attention Cuda] rank=%ld %s %s\n", static_cast<long>(cp_rank), point, stage);
    std::fflush(stderr);
    if (sync) {
        const cudaError_t err = cudaStreamSynchronize(stream);
        TORCH_CHECK(err == cudaSuccess,
                    "dsv4_cp_distributed_prefill_attention debug sync failed after ",
                    stage,
                    ": ",
                    cudaGetErrorString(err));
        std::fprintf(stderr, "[DSV4 CP Attention Cuda] rank=%ld synced %s\n", static_cast<long>(cp_rank), stage);
        std::fflush(stderr);
    }
}

template<typename T>
__device__ __forceinline__ float to_float_device(T v) {
    return static_cast<float>(v);
}

template<typename T>
__device__ __forceinline__ T from_float_device(float v) {
    return static_cast<T>(v);
}

template<typename scalar_t>
__device__ __forceinline__ float q_at(const scalar_t* q, int64_t row, int h, int d, int H, int D) {
    return to_float_device(q[(row * H + h) * D + d]);
}

__device__ __forceinline__ int min_int(int a, int b) {
    return a < b ? a : b;
}

__device__ __forceinline__ int max_int(int a, int b) {
    return a > b ? a : b;
}

__device__ __forceinline__ float warp_sum(float v) {
    for (int offset = 16; offset > 0; offset >>= 1) {
        v += __shfl_down_sync(0xffffffffu, v, offset);
    }
    return v;
}

__device__ __forceinline__ float dsv4MegaWarpReduceSum(float value) {
    value += __shfl_down_sync(0xffffffffu, value, 16);
    value += __shfl_down_sync(0xffffffffu, value, 8);
    value += __shfl_down_sync(0xffffffffu, value, 4);
    value += __shfl_down_sync(0xffffffffu, value, 2);
    value += __shfl_down_sync(0xffffffffu, value, 1);
    return value;
}

__device__ __forceinline__ float dsv4MegaWarpReduceMax(float value) {
    value = fmaxf(value, __shfl_down_sync(0xffffffffu, value, 16));
    value = fmaxf(value, __shfl_down_sync(0xffffffffu, value, 8));
    value = fmaxf(value, __shfl_down_sync(0xffffffffu, value, 4));
    value = fmaxf(value, __shfl_down_sync(0xffffffffu, value, 2));
    value = fmaxf(value, __shfl_down_sync(0xffffffffu, value, 1));
    return value;
}

template<int Lanes>
__device__ __forceinline__ float dsv4MegaHeadLaneReduceSum(float value) {
    static_assert(Lanes == 1 || Lanes == 2 || Lanes == 4 || Lanes == 8 || Lanes == 16 || Lanes == 32,
                  "head-lane reduction requires a power-of-two lane count up to one warp");
#pragma unroll
    for (int offset = Lanes >> 1; offset > 0; offset >>= 1) {
        value += __shfl_xor_sync(0xffffffffu, value, offset);
    }
    return value;
}

__device__ __forceinline__ float dsv4MegaBlockReduceSum(float value, float* reduce_buf, float* result) {
    constexpr int kWarpSize = 32;
    const int     tid       = static_cast<int>(threadIdx.x);
    const int     lane      = tid & (kWarpSize - 1);
    const int     warp_id   = tid / kWarpSize;

    value = dsv4MegaWarpReduceSum(value);
    if (lane == 0) {
        reduce_buf[warp_id] = value;
    }
    __syncthreads();

    float block_sum = 0.0f;
    if (warp_id == 0) {
        const int num_warps = (static_cast<int>(blockDim.x) + kWarpSize - 1) / kWarpSize;
        block_sum = lane < num_warps ? reduce_buf[lane] : 0.0f;
        block_sum = dsv4MegaWarpReduceSum(block_sum);
    }
    if (tid == 0) {
        *result = block_sum;
    }
    __syncthreads();
    return *result;
}

__device__ __forceinline__ void dsv4MegaBlockReduceSumTile(float (&values)[kMegaAttentionKeyTile],
                                                           float* reduce_buf,
                                                           float* results) {
    constexpr int kWarpSize = kMegaAttentionWarpSize;
    const int     tid       = static_cast<int>(threadIdx.x);
    const int     lane      = tid & (kWarpSize - 1);
    const int     warp_id   = tid / kWarpSize;
    const int     num_warps = (static_cast<int>(blockDim.x) + kWarpSize - 1) / kWarpSize;

#pragma unroll
    for (int i = 0; i < kMegaAttentionKeyTile; ++i) {
        values[i] = dsv4MegaWarpReduceSum(values[i]);
        if (lane == 0) {
            reduce_buf[i * num_warps + warp_id] = values[i];
        }
    }
    __syncthreads();

    if (warp_id == 0) {
#pragma unroll
        for (int i = 0; i < kMegaAttentionKeyTile; ++i) {
            float block_sum = lane < num_warps ? reduce_buf[i * num_warps + lane] : 0.0f;
            block_sum       = dsv4MegaWarpReduceSum(block_sum);
            if (lane == 0) {
                results[i] = block_sum;
            }
        }
    }
    __syncthreads();
}

__device__ __forceinline__ float dsv4MegaBlockReduceMax(float value, float* reduce_buf, float* result) {
    constexpr int kWarpSize = 32;
    const int     tid       = static_cast<int>(threadIdx.x);
    const int     lane      = tid & (kWarpSize - 1);
    const int     warp_id   = tid / kWarpSize;

    value = dsv4MegaWarpReduceMax(value);
    if (lane == 0) {
        reduce_buf[warp_id] = value;
    }
    __syncthreads();

    float block_max = 0.0f;
    if (warp_id == 0) {
        const int num_warps = (static_cast<int>(blockDim.x) + kWarpSize - 1) / kWarpSize;
        block_max = lane < num_warps ? reduce_buf[lane] : 0.0f;
        block_max = dsv4MegaWarpReduceMax(block_max);
    }
    if (tid == 0) {
        *result = block_max;
    }
    __syncthreads();
    return *result;
}

__device__ __forceinline__ void dsv4MegaGridBarrier(uint8_t* scratch_base, int barrier_id, unsigned long long launch_epoch) {
    if (gridDim.x <= 1) {
        __syncthreads();
        return;
    }
    if (scratch_base == nullptr) {
        __syncthreads();
        return;
    }
    if (barrier_id < 0 || barrier_id >= kMegaGridBarrierLogicalSlots) {
        if (threadIdx.x == 0) {
            printf("DSV4 mega grid barrier id out of range: block=%d barrier=%d\n",
                   static_cast<int>(blockIdx.x),
                   barrier_id);
        }
        asm("trap;");
    }
    auto* counters = reinterpret_cast<unsigned int*>(scratch_base);
    auto* epochs = reinterpret_cast<unsigned long long*>(scratch_base + kMegaGridCounterBytes);
    const int physical_barrier_id =
        barrier_id * kMegaGridBarrierEpochStride
        + static_cast<int>(launch_epoch & static_cast<unsigned long long>(kMegaGridBarrierEpochStride - 1));
    const unsigned long long epoch = launch_epoch ^ (static_cast<unsigned long long>(barrier_id + 1) << 48);
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        counters[physical_barrier_id] = 0;
        cuda::atomic_ref<unsigned long long, cuda::thread_scope_device> epoch_ref(epochs[physical_barrier_id]);
        epoch_ref.store(epoch, cuda::std::memory_order_release);
    }
    __threadfence();
    __syncthreads();

    cuda::atomic_ref<unsigned long long, cuda::thread_scope_device> epoch_ref(epochs[physical_barrier_id]);
    const unsigned long long start_epoch = clock64();
    while (epoch_ref.load(cuda::std::memory_order_acquire) != epoch) {
        if (clock64() - start_epoch > static_cast<unsigned long long>(kSymmMemBarrierTimeoutCycles)) {
            if (threadIdx.x == 0) {
                printf("DSV4 mega grid barrier epoch timeout: block=%d barrier=%d\n",
                       static_cast<int>(blockIdx.x),
                       barrier_id);
            }
            asm("trap;");
        }
    }
    __threadfence();
    __syncthreads();

    if (threadIdx.x == 0) {
        cuda::atomic_ref<unsigned int, cuda::thread_scope_device> counter_ref(counters[physical_barrier_id]);
        const unsigned int ticket = counter_ref.fetch_add(1, cuda::std::memory_order_acq_rel);
        if (ticket + 1 == static_cast<unsigned int>(gridDim.x)) {
            counters[physical_barrier_id] = 0;
            epoch_ref.store(epoch + 1, cuda::std::memory_order_release);
        }
    }
    __threadfence();
    __syncthreads();
    const unsigned long long start_release = clock64();
    while (epoch_ref.load(cuda::std::memory_order_acquire) != epoch + 1) {
        if (clock64() - start_release > static_cast<unsigned long long>(kSymmMemBarrierTimeoutCycles)) {
            if (threadIdx.x == 0) {
                printf("DSV4 mega grid barrier release timeout: block=%d barrier=%d\n",
                       static_cast<int>(blockIdx.x),
                       barrier_id);
            }
            asm("trap;");
        }
    }
    __threadfence();
    __syncthreads();
}

__device__ __forceinline__ void dsv4MegaSplitKGroupBarrier(uint8_t* scratch_base,
                                                           int physical_group,
                                                           int physical_group_count,
                                                           int barrier_id,
                                                           unsigned long long launch_epoch) {
    if (scratch_base == nullptr || physical_group_count <= 0) {
        __syncthreads();
        return;
    }
    if (barrier_id < 0 || barrier_id >= kMegaSplitKGroupBarrierSlots) {
        if (threadIdx.x == 0) {
            printf("DSV4 split-K group barrier id out of range: block=%d group=%d barrier=%d\n",
                   static_cast<int>(blockIdx.x),
                   physical_group,
                   barrier_id);
        }
        asm("trap;");
    }
    const int role = static_cast<int>(blockIdx.x) - physical_group * kMegaSplitKGroupSize;
    const int slot = physical_group * kMegaSplitKGroupBarrierSlots + barrier_id;
    const int64_t slot_count =
        static_cast<int64_t>(physical_group_count) * static_cast<int64_t>(kMegaSplitKGroupBarrierSlots);
    const int64_t counter_bytes =
        ((slot_count * static_cast<int64_t>(sizeof(unsigned int)) + 63) / 64) * 64;
    auto* counters = reinterpret_cast<unsigned int*>(scratch_base);
    auto* epochs = reinterpret_cast<unsigned long long*>(scratch_base + counter_bytes);
    const unsigned long long epoch = launch_epoch ^ (static_cast<unsigned long long>(barrier_id + 1) << 44)
                                     ^ (static_cast<unsigned long long>(physical_group + 1) << 32);
    if (role == 0 && threadIdx.x == 0) {
        counters[slot] = 0;
        cuda::atomic_ref<unsigned long long, cuda::thread_scope_device> epoch_ref(epochs[slot]);
        epoch_ref.store(epoch, cuda::std::memory_order_release);
    }
    __threadfence();
    __syncthreads();

    cuda::atomic_ref<unsigned long long, cuda::thread_scope_device> epoch_ref(epochs[slot]);
    const unsigned long long start_epoch = clock64();
    while (epoch_ref.load(cuda::std::memory_order_acquire) != epoch) {
        if (clock64() - start_epoch > static_cast<unsigned long long>(kSymmMemBarrierTimeoutCycles)) {
            if (threadIdx.x == 0) {
                printf("DSV4 split-K group barrier epoch timeout: block=%d group=%d barrier=%d\n",
                       static_cast<int>(blockIdx.x),
                       physical_group,
                       barrier_id);
            }
            asm("trap;");
        }
    }
    __threadfence();
    __syncthreads();

    if (threadIdx.x == 0) {
        cuda::atomic_ref<unsigned int, cuda::thread_scope_device> counter_ref(counters[slot]);
        const unsigned int ticket = counter_ref.fetch_add(1, cuda::std::memory_order_acq_rel);
        if (ticket + 1 == static_cast<unsigned int>(kMegaSplitKGroupSize)) {
            counters[slot] = 0;
            epoch_ref.store(epoch + 1, cuda::std::memory_order_release);
        }
    }
    __threadfence();
    __syncthreads();
    const unsigned long long start_release = clock64();
    while (epoch_ref.load(cuda::std::memory_order_acquire) != epoch + 1) {
        if (clock64() - start_release > static_cast<unsigned long long>(kSymmMemBarrierTimeoutCycles)) {
            if (threadIdx.x == 0) {
                printf("DSV4 split-K group barrier release timeout: block=%d group=%d barrier=%d\n",
                       static_cast<int>(blockIdx.x),
                       physical_group,
                       barrier_id);
            }
            asm("trap;");
        }
    }
    __threadfence();
    __syncthreads();
}

__device__ __forceinline__ bool dsv4MegaTopKWorse(float lhs_score, int lhs_idx, float rhs_score, int rhs_idx) {
    return lhs_score < rhs_score || (lhs_score == rhs_score && lhs_idx > rhs_idx);
}

__device__ __forceinline__ bool dsv4MegaTopKBetter(float lhs_score, int lhs_idx, float rhs_score, int rhs_idx) {
    return lhs_score > rhs_score || (lhs_score == rhs_score && lhs_idx < rhs_idx);
}

__device__ __forceinline__ void dsv4MegaInsertWorstCandidateList(int slot,
                                                                 float score,
                                                                 int idx,
                                                                 int* __restrict__ slots,
                                                                 float* __restrict__ scores,
                                                                 int* __restrict__ indices,
                                                                 int limit) {
    if (slot < 0 || limit <= 0) {
        return;
    }
    int insert_pos = -1;
#pragma unroll
    for (int i = 0; i < kMegaIndexerCandidateTile; ++i) {
        if (i >= limit) {
            break;
        }
        if (slots[i] < 0 || dsv4MegaTopKWorse(score, idx, scores[i], indices[i])) {
            insert_pos = i;
            break;
        }
    }
    if (insert_pos < 0) {
        return;
    }
#pragma unroll
    for (int i = kMegaIndexerCandidateTile - 1; i > 0; --i) {
        if (i < limit && i > insert_pos) {
            slots[i] = slots[i - 1];
            scores[i] = scores[i - 1];
            indices[i] = indices[i - 1];
        }
    }
    slots[insert_pos] = slot;
    scores[insert_pos] = score;
    indices[insert_pos] = idx;
}

__device__ __forceinline__ void dsv4MegaInsertWorstCandidate(int slot,
                                                             float score,
                                                             int idx,
                                                             int& worst_slot,
                                                             float& worst_score,
                                                             int& worst_idx,
                                                             int& second_slot,
                                                             float& second_score,
                                                             int& second_idx) {
    if (slot < 0) {
        return;
    }
    if (worst_slot < 0 || dsv4MegaTopKWorse(score, idx, worst_score, worst_idx)) {
        second_slot = worst_slot;
        second_score = worst_score;
        second_idx = worst_idx;
        worst_slot = slot;
        worst_score = score;
        worst_idx = idx;
    } else if (second_slot < 0 || dsv4MegaTopKWorse(score, idx, second_score, second_idx)) {
        second_slot = slot;
        second_score = score;
        second_idx = idx;
    }
}

__device__ __forceinline__ void dsv4MegaFindTwoWorstTopKSlots(const int* __restrict__ scratch_indices,
                                                              const float* __restrict__ scratch_scores,
                                                              int k_eff,
                                                              int* __restrict__ reduce_indices,
                                                              int* __restrict__ reduce_indices_second,
                                                              float* __restrict__ reduce_scores,
                                                              int& worst_slot_out,
                                                              int& second_slot_out) {
    const int tid = static_cast<int>(threadIdx.x);
    int worst_slot = -1;
    float worst_score = 3.4028234663852886e38F;
    int worst_idx = -2147483647;
    int second_slot = -1;
    float second_score = 3.4028234663852886e38F;
    int second_idx = -2147483647;
    for (int i = tid; i < k_eff; i += static_cast<int>(blockDim.x)) {
        dsv4MegaInsertWorstCandidate(i,
                                     scratch_scores[i],
                                     scratch_indices[i],
                                     worst_slot,
                                     worst_score,
                                     worst_idx,
                                     second_slot,
                                     second_score,
                                     second_idx);
    }
    reduce_indices[tid] = worst_slot;
    reduce_indices_second[tid] = second_slot;
    reduce_scores[tid] = worst_score;
    reduce_scores[static_cast<int>(blockDim.x) + tid] = second_score;
    __syncthreads();

    for (int stride = static_cast<int>(blockDim.x) / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            worst_slot = reduce_indices[tid];
            worst_score = reduce_scores[tid];
            worst_idx = worst_slot >= 0 ? scratch_indices[worst_slot] : -2147483647;
            second_slot = reduce_indices_second[tid];
            second_score = reduce_scores[static_cast<int>(blockDim.x) + tid];
            second_idx = second_slot >= 0 ? scratch_indices[second_slot] : -2147483647;

            const int rhs_worst_slot = reduce_indices[tid + stride];
            if (rhs_worst_slot >= 0) {
                dsv4MegaInsertWorstCandidate(rhs_worst_slot,
                                             reduce_scores[tid + stride],
                                             scratch_indices[rhs_worst_slot],
                                             worst_slot,
                                             worst_score,
                                             worst_idx,
                                             second_slot,
                                             second_score,
                                             second_idx);
            }
            const int rhs_second_slot = reduce_indices_second[tid + stride];
            if (rhs_second_slot >= 0) {
                dsv4MegaInsertWorstCandidate(rhs_second_slot,
                                             reduce_scores[static_cast<int>(blockDim.x) + tid + stride],
                                             scratch_indices[rhs_second_slot],
                                             worst_slot,
                                             worst_score,
                                             worst_idx,
                                             second_slot,
                                             second_score,
                                             second_idx);
            }
            reduce_indices[tid] = worst_slot;
            reduce_indices_second[tid] = second_slot;
            reduce_scores[tid] = worst_score;
            reduce_scores[static_cast<int>(blockDim.x) + tid] = second_score;
        }
        __syncthreads();
    }
    worst_slot_out = reduce_indices[0];
    second_slot_out = reduce_indices_second[0];
}

__device__ __forceinline__ void dsv4MegaFindWorstTopKSlotsTile(const int* __restrict__ scratch_indices,
                                                               const float* __restrict__ scratch_scores,
                                                               int k_eff,
                                                               int requested_slots,
                                                               int* __restrict__ warp_slots,
                                                               int* __restrict__ worst_slots,
                                                               float* __restrict__ warp_scores) {
    const int tid = static_cast<int>(threadIdx.x);
    const int lane = tid & (kMegaAttentionWarpSize - 1);
    const int warp_id = tid / kMegaAttentionWarpSize;
    const int num_warps = (static_cast<int>(blockDim.x) + kMegaAttentionWarpSize - 1) / kMegaAttentionWarpSize;
    const int limit = min_int(min_int(requested_slots, kMegaIndexerCandidateTile), k_eff);
    int slots[kMegaIndexerCandidateTile];
    float scores[kMegaIndexerCandidateTile];
    int indices[kMegaIndexerCandidateTile];
#pragma unroll
    for (int i = 0; i < kMegaIndexerCandidateTile; ++i) {
        slots[i] = -1;
        scores[i] = 3.4028234663852886e38F;
        indices[i] = -2147483647;
    }
    for (int i = tid; i < k_eff; i += static_cast<int>(blockDim.x)) {
        const int idx = scratch_indices[i];
        dsv4MegaInsertWorstCandidateList(i, scratch_scores[i], idx, slots, scores, indices, limit);
    }
#pragma unroll
    for (int offset = kMegaAttentionWarpSize / 2; offset > 0; offset >>= 1) {
#pragma unroll
        for (int i = 0; i < kMegaIndexerCandidateTile; ++i) {
            const int rhs_slot = __shfl_down_sync(0xffffffffu, slots[i], offset);
            const float rhs_score = __shfl_down_sync(0xffffffffu, scores[i], offset);
            const int rhs_idx = __shfl_down_sync(0xffffffffu, indices[i], offset);
            if (lane + offset < kMegaAttentionWarpSize) {
                dsv4MegaInsertWorstCandidateList(rhs_slot, rhs_score, rhs_idx, slots, scores, indices, limit);
            }
        }
    }
    if (lane == 0) {
#pragma unroll
        for (int i = 0; i < kMegaIndexerCandidateTile; ++i) {
            const int out = warp_id * kMegaIndexerCandidateTile + i;
            if (i < limit) {
                warp_slots[out] = slots[i];
                warp_scores[out] = scores[i];
            }
        }
    }
    __syncthreads();
    if (tid == 0) {
        int merged_slots[kMegaIndexerCandidateTile];
        float merged_scores[kMegaIndexerCandidateTile];
        int merged_indices[kMegaIndexerCandidateTile];
#pragma unroll
        for (int i = 0; i < kMegaIndexerCandidateTile; ++i) {
            merged_slots[i] = -1;
            merged_scores[i] = 3.4028234663852886e38F;
            merged_indices[i] = -2147483647;
        }
        for (int w = 0; w < num_warps; ++w) {
#pragma unroll
            for (int i = 0; i < kMegaIndexerCandidateTile; ++i) {
                if (i < limit) {
                    const int slot = warp_slots[w * kMegaIndexerCandidateTile + i];
                    const int idx = slot >= 0 ? scratch_indices[slot] : -2147483647;
                    dsv4MegaInsertWorstCandidateList(
                        slot, warp_scores[w * kMegaIndexerCandidateTile + i], idx, merged_slots, merged_scores, merged_indices, limit);
                }
            }
        }
#pragma unroll
        for (int i = 0; i < kMegaIndexerCandidateTile; ++i) {
            if (i < limit) {
                worst_slots[i] = merged_slots[i];
            }
        }
    }
    __syncthreads();
}

template<typename T>
__device__ __forceinline__ float scalar_load_float(const T* ptr, int64_t idx) {
    return to_float_device(ptr[idx]);
}

__device__ __forceinline__ float bf16_round_float(float x) {
    __nv_bfloat16 y = __float2bfloat16(x);
    return __bfloat162float(y);
}

__device__ __forceinline__ uint16_t float_to_bf16_bits(float x) {
    __nv_bfloat16 y = __float2bfloat16(x);
    return *reinterpret_cast<uint16_t*>(&y);
}

__device__ __forceinline__ uint16_t load_u16_unaligned(const void* ptr) {
    const auto* bytes = reinterpret_cast<const uint8_t*>(ptr);
    return static_cast<uint16_t>(bytes[0]) | (static_cast<uint16_t>(bytes[1]) << 8);
}

__device__ __forceinline__ void store_u16_unaligned(void* ptr, uint16_t value) {
    auto* bytes = reinterpret_cast<uint8_t*>(ptr);
    bytes[0] = static_cast<uint8_t>(value & 0xffu);
    bytes[1] = static_cast<uint8_t>((value >> 8) & 0xffu);
}

__device__ __forceinline__ float load_f32_unaligned(const void* ptr) {
    const auto* bytes = reinterpret_cast<const uint8_t*>(ptr);
    const uint32_t bits = static_cast<uint32_t>(bytes[0]) | (static_cast<uint32_t>(bytes[1]) << 8)
                          | (static_cast<uint32_t>(bytes[2]) << 16) | (static_cast<uint32_t>(bytes[3]) << 24);
    return __uint_as_float(bits);
}

__device__ __forceinline__ void store_f32_unaligned(void* ptr, float value) {
    auto* bytes = reinterpret_cast<uint8_t*>(ptr);
    const uint32_t bits = __float_as_uint(value);
    bytes[0] = static_cast<uint8_t>(bits & 0xffu);
    bytes[1] = static_cast<uint8_t>((bits >> 8) & 0xffu);
    bytes[2] = static_cast<uint8_t>((bits >> 16) & 0xffu);
    bytes[3] = static_cast<uint8_t>((bits >> 24) & 0xffu);
}

__device__ __forceinline__ float fp8_e4m3_to_float(uint8_t x, float scale) {
    __half_raw raw = __nv_cvt_fp8_to_halfraw(x, __NV_E4M3);
    __half h = __ushort_as_half(raw.x);
    return __half2float(h) * scale;
}

__device__ __forceinline__ int dsv4MegaSequentialCompressedCount(int compress_ratio,
                                                                 int compressed_topk,
                                                                 int q_pos,
                                                                 int kv_len,
                                                                 int req,
                                                                 int LI,
                                                                 const uint8_t* csa_indexer_k_cache,
                                                                 const float* csa_indexer_weights,
                                                                 const int64_t* csa_indexer_cu_lens,
                                                                 int csa_indexer_total_len,
                                                                 const uint8_t* csa_indexer_k_pool,
                                                                 const int32_t* csa_indexer_block_table,
                                                                 const int32_t* csa_indexer_seq_lens,
                                                                 const int32_t* attention_cmp_seq_lens) {
    if (compress_ratio <= 0) {
        return 0;
    }
    const int visible_boundaries = max_int(0, kv_len / compress_ratio);
    if (compressed_topk <= 0 || visible_boundaries <= 0) {
        return 0;
    }
    if (compress_ratio == 128) {
        int valid = min_int((q_pos + 1) / compress_ratio, compressed_topk);
        if (attention_cmp_seq_lens != nullptr) {
            valid = min_int(valid, static_cast<int>(attention_cmp_seq_lens[req]));
        }
        return min_int(min_int(valid, kMaxCompressedTopK), visible_boundaries);
    }
    if (compress_ratio != 4) {
        return -1;
    }

    const bool use_fp8_indexer_pool =
        csa_indexer_k_pool != nullptr && csa_indexer_block_table != nullptr && csa_indexer_weights != nullptr;
    const bool use_fp8_indexer = use_fp8_indexer_pool
                                 || (csa_indexer_k_cache != nullptr && csa_indexer_weights != nullptr
                                     && csa_indexer_cu_lens != nullptr);
    int req_k_len = LI;
    if (use_fp8_indexer_pool) {
        req_k_len = csa_indexer_seq_lens == nullptr ? LI : static_cast<int>(csa_indexer_seq_lens[req]);
    } else if (use_fp8_indexer) {
        req_k_len = static_cast<int>(csa_indexer_cu_lens[req + 1] - csa_indexer_cu_lens[req]);
        req_k_len = min_int(req_k_len, csa_indexer_total_len);
    }
    const int valid = min_int((q_pos + 1) / compress_ratio, req_k_len);
    const int k_eff = min_int(min_int(valid, compressed_topk), kMaxCompressedTopK);
    if (k_eff < valid) {
        return -1;
    }
    return min_int(valid, visible_boundaries);
}

__device__ __forceinline__ float bf16_bits_to_float(uint16_t bits) {
    __nv_bfloat16 bf = *reinterpret_cast<__nv_bfloat16*>(&bits);
    return __bfloat162float(bf);
}

template<typename scalar_t>
__device__ __forceinline__ float symm_fresh_kv_at(const uint8_t* const* __restrict__ buffer_ptrs,
                                                  int64_t per_rank_buffer_bytes,
                                                  int64_t payload_offset,
                                                  int L_local,
                                                  int KH,
                                                  int D,
                                                  int64_t gathered_row,
                                                  int h,
                                                  int d) {
    if (L_local <= 0) {
        return 0.0f;
    }
    const int owner = static_cast<int>(gathered_row / L_local);
    const int local_row = static_cast<int>(gathered_row - static_cast<int64_t>(owner) * L_local);
    if (owner < 0 || owner >= 8 || local_row < 0 || local_row >= L_local) {
        return 0.0f;
    }
    const int kh = KH == 1 ? 0 : h;
    const int64_t elem_offset = (static_cast<int64_t>(local_row) * KH + kh) * D + d;
    const int64_t byte_offset = static_cast<int64_t>(owner) * per_rank_buffer_bytes + payload_offset
                                + elem_offset * static_cast<int64_t>(sizeof(scalar_t));
    const scalar_t* ptr = reinterpret_cast<const scalar_t*>(buffer_ptrs[owner] + byte_offset);
    return to_float_device(*ptr);
}

template<typename scalar_t>
__device__ __forceinline__ float symm_4d_at(const uint8_t* const* __restrict__ buffer_ptrs,
                                            int64_t per_rank_buffer_bytes,
                                            int64_t payload_offset,
                                            int L_local,
                                            int IH,
                                            int ID,
                                            int req,
                                            int pos,
                                            int h,
                                            int d) {
    if (L_local <= 0) {
        return 0.0f;
    }
    const int owner = pos / L_local;
    const int local_pos = pos - owner * L_local;
    if (owner < 0 || owner >= 8 || local_pos < 0 || local_pos >= L_local) {
        return 0.0f;
    }
    const int64_t elem_offset = ((static_cast<int64_t>(req) * L_local + local_pos) * IH + h) * ID + d;
    const int64_t byte_offset = static_cast<int64_t>(owner) * per_rank_buffer_bytes + payload_offset
                                + elem_offset * static_cast<int64_t>(sizeof(scalar_t));
    const scalar_t* ptr = reinterpret_cast<const scalar_t*>(buffer_ptrs[owner] + byte_offset);
    return to_float_device(*ptr);
}

__device__ __forceinline__ float model1_cache_at(const uint8_t* cache,
                                                  const int64_t* cu_lens,
                                                  int req,
                                                  int compressed_idx,
                                                  int d) {
    const int64_t row_idx = cu_lens[req] + compressed_idx;
    const uint8_t* row = cache + row_idx * kSwaEntryBytes;
    if (d < kSwaNopeDim) {
        const uint8_t encoded = row[kSwaTokenDataBytes + d / kSwaQuantBlock];
        const float scale = exp2f(static_cast<float>(encoded) - 127.0f);
        return fp8_e4m3_to_float(row[d], scale);
    }
    const int rope_d = d - kSwaNopeDim;
    if (rope_d < kSwaHeadDim - kSwaNopeDim) {
        const uint16_t bits = load_u16_unaligned(row + kSwaNopeDim + rope_d * static_cast<int>(sizeof(uint16_t)));
        return bf16_bits_to_float(bits);
    }
    return 0.0f;
}

__device__ __forceinline__ float model1_pool_slot_at(const uint8_t* pool,
                                                     int64_t block_stride,
                                                     int block_size,
                                                     int64_t slot,
                                                     int d) {
    if (slot < 0) {
        return 0.0f;
    }
    const int64_t block_idx = slot / block_size;
    const int64_t pos_in_block = slot % block_size;
    const uint8_t* block = pool + block_idx * block_stride;
    const uint8_t* token_data = block + pos_in_block * kSwaTokenDataBytes;
    const uint8_t* token_scale = block + static_cast<int64_t>(block_size) * kSwaTokenDataBytes
                                 + pos_in_block * kSwaScaleBytes;
    if (d < kSwaNopeDim) {
        const uint8_t encoded = token_scale[d / kSwaQuantBlock];
        const float scale = exp2f(static_cast<float>(encoded) - 127.0f);
        return fp8_e4m3_to_float(token_data[d], scale);
    }
    const int rope_d = d - kSwaNopeDim;
    if (rope_d < kSwaHeadDim - kSwaNopeDim) {
        const uint16_t bits =
            load_u16_unaligned(token_data + kSwaNopeDim + rope_d * static_cast<int>(sizeof(uint16_t)));
        return bf16_bits_to_float(bits);
    }
    return 0.0f;
}

__device__ __forceinline__ float model1_cmp_pool_at(const uint8_t* pool,
                                                    const int32_t* block_table,
                                                    const int32_t* seq_lens,
                                                    int req,
                                                    int compressed_idx,
                                                    int d,
                                                    int block_size,
                                                    int64_t block_stride,
                                                    int64_t block_table_stride) {
    if (seq_lens != nullptr && compressed_idx >= seq_lens[req]) {
        return 0.0f;
    }
    const int64_t block_row = compressed_idx / block_size;
    const int64_t in_block = compressed_idx % block_size;
    const int32_t block_id = block_table[static_cast<int64_t>(req) * block_table_stride + block_row];
    if (block_id < 0) {
        return 0.0f;
    }
    return model1_pool_slot_at(pool, block_stride, block_size, static_cast<int64_t>(block_id) * block_size + in_block, d);
}

__device__ __forceinline__ bool model1_cmp_pool_ptrs(const uint8_t* pool,
                                                     const int32_t* block_table,
                                                     const int32_t* seq_lens,
                                                     int req,
                                                     int compressed_idx,
                                                     int block_size,
                                                     int64_t block_stride,
                                                     int64_t block_table_stride,
                                                     const uint8_t** data_ptr,
                                                     const uint8_t** scale_ptr) {
    if (seq_lens != nullptr && compressed_idx >= seq_lens[req]) {
        return false;
    }
    const int64_t block_row = compressed_idx / block_size;
    const int64_t in_block = compressed_idx % block_size;
    const int32_t block_id = block_table[static_cast<int64_t>(req) * block_table_stride + block_row];
    if (block_id < 0) {
        return false;
    }
    const uint8_t* block = pool + static_cast<int64_t>(block_id) * block_stride;
    *data_ptr = block + in_block * kSwaTokenDataBytes;
    *scale_ptr = block + static_cast<int64_t>(block_size) * kSwaTokenDataBytes + in_block * kSwaScaleBytes;
    return true;
}

__device__ __forceinline__ bool model1_cache_ptrs(const uint8_t* cache,
                                                  const int64_t* cu_lens,
                                                  int req,
                                                  int compressed_idx,
                                                  const uint8_t** data_ptr,
                                                  const uint8_t** scale_ptr) {
    const uint8_t* row = cache + (cu_lens[req] + compressed_idx) * kSwaEntryBytes;
    *data_ptr = row;
    *scale_ptr = row + kSwaTokenDataBytes;
    return true;
}

__device__ __forceinline__ bool model1_swa_pool_ptrs(const uint8_t* pool,
                                                     const int64_t* slot_mapping,
                                                     const int32_t* gather_lens,
                                                     int req,
                                                     int key_idx,
                                                     int prefix_len,
                                                     int block_size,
                                                     int64_t block_stride,
                                                     int64_t slot_mapping_stride,
                                                     const uint8_t** data_ptr,
                                                     const uint8_t** scale_ptr) {
    const int gather_len = gather_lens == nullptr ? prefix_len : gather_lens[req];
    const int start = prefix_len - gather_len;
    const int col = key_idx - start;
    if (col < 0 || col >= gather_len) {
        return false;
    }
    const int64_t slot = slot_mapping[static_cast<int64_t>(req) * slot_mapping_stride + col];
    if (slot < 0) {
        return false;
    }
    const int64_t block_idx = slot / block_size;
    const int64_t pos_in_block = slot % block_size;
    const uint8_t* block = pool + block_idx * block_stride;
    *data_ptr = block + pos_in_block * kSwaTokenDataBytes;
    *scale_ptr = block + static_cast<int64_t>(block_size) * kSwaTokenDataBytes + pos_in_block * kSwaScaleBytes;
    return true;
}

__device__ __forceinline__ float model1_swa_pool_at(const uint8_t* pool,
                                                    const int64_t* slot_mapping,
                                                    const int32_t* gather_lens,
                                                    int req,
                                                    int key_idx,
                                                    int prefix_len,
                                                    int d,
                                                    int block_size,
                                                    int64_t block_stride,
                                                    int64_t slot_mapping_stride) {
    const int gather_len = gather_lens == nullptr ? prefix_len : gather_lens[req];
    const int start = prefix_len - gather_len;
    const int col = key_idx - start;
    if (col < 0 || col >= gather_len) {
        return 0.0f;
    }
    const int64_t slot = slot_mapping[static_cast<int64_t>(req) * slot_mapping_stride + col];
    return model1_pool_slot_at(pool, block_stride, block_size, slot, d);
}

template<typename scalar_t>
__device__ __forceinline__ float kv_at(const scalar_t* kv, int req, int pos, int h, int d, int L, int KH, int D) {
    const int kh = KH == 1 ? 0 : h;
    return to_float_device(kv[((static_cast<int64_t>(req) * L + pos) * KH + kh) * D + d]);
}

template<typename scalar_t>
__device__ __forceinline__ float kv_flat_at(const scalar_t* kv, int64_t row, int h, int d, int KH, int D) {
    const int kh = KH == 1 ? 0 : h;
    return to_float_device(kv[(row * KH + kh) * D + d]);
}

template<typename scalar_t>
__device__ __forceinline__ float attention_key_at(const scalar_t* kv,
                                                  const uint8_t* cmp_cache,
                                                  const int64_t* cmp_cu_lens,
                                                  const uint8_t* cmp_pool,
                                                  const int32_t* cmp_block_table,
                                                  const int32_t* cmp_seq_lens,
                                                  int cmp_pool_block_size,
                                                  int64_t cmp_pool_block_stride,
                                                  int64_t cmp_block_table_stride,
                                                  const uint8_t* swa_cache,
                                                  const int64_t* swa_cu_lens,
                                                  const uint8_t* swa_pool,
                                                  const int64_t* swa_slot_mapping,
                                                  const int32_t* swa_gather_lens,
                                                  int swa_pool_block_size,
                                                  int64_t swa_pool_block_stride,
                                                  int64_t swa_slot_mapping_stride,
                                                  int key_is_compressed,
                                                  int req,
                                                  int key_idx,
                                                  int prefix_len,
                                                  int h,
                                                  int d,
                                                  int L,
                                                  int KH,
                                                  int D,
                                                  const int64_t* kv_unpad_restore,
                                                  const int64_t* kv_cu_lens,
                                                  const uint8_t* const* __restrict__ symm_buffer_ptrs,
                                                  int use_symm_direct_fresh,
                                                  int64_t per_rank_buffer_bytes,
                                                  int64_t fresh_payload_offset,
                                                  int symm_l_local) {
    if (key_is_compressed && cmp_pool != nullptr && cmp_block_table != nullptr) {
        return model1_cmp_pool_at(cmp_pool,
                                  cmp_block_table,
                                  cmp_seq_lens,
                                  req,
                                  key_idx,
                                  d,
                                  cmp_pool_block_size,
                                  cmp_pool_block_stride,
                                  cmp_block_table_stride);
    }
    if (key_is_compressed && cmp_cache != nullptr && cmp_cu_lens != nullptr) {
        return model1_cache_at(cmp_cache, cmp_cu_lens, req, key_idx, d);
    }
    if (!key_is_compressed && swa_pool != nullptr && swa_slot_mapping != nullptr && key_idx < prefix_len) {
        return model1_swa_pool_at(swa_pool,
                                  swa_slot_mapping,
                                  swa_gather_lens,
                                  req,
                                  key_idx,
                                  prefix_len,
                                  d,
                                  swa_pool_block_size,
                                  swa_pool_block_stride,
                                  swa_slot_mapping_stride);
    }
    if (!key_is_compressed && swa_cache != nullptr && swa_cu_lens != nullptr && key_idx < prefix_len) {
        return model1_cache_at(swa_cache, swa_cu_lens, req, key_idx, d);
    }
    if (kv_unpad_restore != nullptr && kv_cu_lens != nullptr) {
        const int fresh_pos = key_idx - prefix_len;
        if (fresh_pos < 0) {
            return 0.0f;
        }
        const int64_t global_row = kv_cu_lens[req] + fresh_pos;
        const int64_t gathered_row = kv_unpad_restore[global_row];
        if (gathered_row < 0 || gathered_row >= L) {
            return 0.0f;
        }
        if (use_symm_direct_fresh && symm_buffer_ptrs != nullptr) {
            return symm_fresh_kv_at<scalar_t>(
                symm_buffer_ptrs, per_rank_buffer_bytes, fresh_payload_offset, symm_l_local, KH, D, gathered_row, h, d);
        }
        return kv_flat_at(kv, gathered_row, h, d, KH, D);
    }
    return kv_at(kv, req, key_idx, h, d, L, KH, D);
}

template<typename scalar_t>
__device__ __forceinline__ void dsv4MegaLoadAttentionKeyTile(const scalar_t* __restrict__ kv,
                                                             const uint8_t* __restrict__ cmp_cache,
                                                             const int64_t* __restrict__ cmp_cu_lens,
                                                             const uint8_t* __restrict__ cmp_pool,
                                                             const int32_t* __restrict__ cmp_block_table,
                                                             const int32_t* __restrict__ cmp_seq_lens,
                                                             int cmp_pool_block_size,
                                                             int64_t cmp_pool_block_stride,
                                                             int64_t cmp_block_table_stride,
                                                             const uint8_t* __restrict__ swa_cache,
                                                             const int64_t* __restrict__ swa_cu_lens,
                                                             const uint8_t* __restrict__ swa_pool,
                                                             const int64_t* __restrict__ swa_slot_mapping,
                                                             const int32_t* __restrict__ swa_gather_lens,
                                                             int swa_pool_block_size,
                                                             int64_t swa_pool_block_stride,
                                                             int64_t swa_slot_mapping_stride,
                                                             const int* __restrict__ key_is_compressed,
                                                             const int* __restrict__ key_pos,
                                                             int key_count,
                                                             int req,
                                                             int prefix_len,
                                                             int D,
                                                             int L,
                                                             int KH,
                                                             const int64_t* __restrict__ kv_unpad_restore,
                                                             const int64_t* __restrict__ kv_cu_lens,
                                                             const uint8_t* const* __restrict__ symm_buffer_ptrs,
                                                             int use_symm_direct_fresh,
                                                             int64_t per_rank_buffer_bytes,
                                                             int64_t fresh_payload_offset,
                                                             int symm_l_local,
                                                             float* __restrict__ key_tile_shared) {
    const int tid = static_cast<int>(threadIdx.x);
    __shared__ float packed_scales[kMegaAttentionGroupedScaleSlots];
    __shared__ const uint8_t* packed_data_rows[kMegaAttentionGroupedKeyTile];
    __shared__ const uint8_t* packed_scale_rows[kMegaAttentionGroupedKeyTile];
    if (tid < kMegaAttentionGroupedKeyTile) {
        const int key_i = tid;
        const uint8_t* data_ptr = nullptr;
        const uint8_t* scale_ptr = nullptr;
        if (key_i < key_count) {
            if (key_is_compressed[key_i] && cmp_pool != nullptr && cmp_block_table != nullptr) {
                (void)model1_cmp_pool_ptrs(cmp_pool,
                                           cmp_block_table,
                                           cmp_seq_lens,
                                           req,
                                           key_pos[key_i],
                                           cmp_pool_block_size,
                                           cmp_pool_block_stride,
                                           cmp_block_table_stride,
                                           &data_ptr,
                                           &scale_ptr);
            } else if (key_is_compressed[key_i] && cmp_cache != nullptr && cmp_cu_lens != nullptr) {
                (void)model1_cache_ptrs(cmp_cache, cmp_cu_lens, req, key_pos[key_i], &data_ptr, &scale_ptr);
            } else if (!key_is_compressed[key_i] && swa_pool != nullptr && swa_slot_mapping != nullptr
                       && key_pos[key_i] < prefix_len) {
                (void)model1_swa_pool_ptrs(swa_pool,
                                           swa_slot_mapping,
                                           swa_gather_lens,
                                           req,
                                           key_pos[key_i],
                                           prefix_len,
                                           swa_pool_block_size,
                                           swa_pool_block_stride,
                                           swa_slot_mapping_stride,
                                           &data_ptr,
                                           &scale_ptr);
            } else if (!key_is_compressed[key_i] && swa_cache != nullptr && swa_cu_lens != nullptr
                       && key_pos[key_i] < prefix_len) {
                (void)model1_cache_ptrs(swa_cache, swa_cu_lens, req, key_pos[key_i], &data_ptr, &scale_ptr);
            }
        }
        packed_data_rows[key_i] = data_ptr;
        packed_scale_rows[key_i] = scale_ptr;
    }
    __syncthreads();
    for (int scale_idx = tid; scale_idx < key_count * kSwaScaleBytes; scale_idx += static_cast<int>(blockDim.x)) {
        const int key_i = scale_idx / kSwaScaleBytes;
        const int scale_i = scale_idx - key_i * kSwaScaleBytes;
        const uint8_t* scale_ptr = packed_scale_rows[key_i];
        packed_scales[scale_idx] =
            scale_ptr == nullptr ? 1.0f : exp2f(static_cast<float>(scale_ptr[scale_i]) - 127.0f);
    }
    __syncthreads();

    if (D == kSwaHeadDim) {
        const int nope_groups = kSwaNopeDim / kSwaQuantBlock;
        const int fp8_total = key_count * nope_groups * kSwaQuantBlock;
        for (int idx = tid; idx < fp8_total; idx += static_cast<int>(blockDim.x)) {
            const int lane = idx % kSwaQuantBlock;
            const int group_linear = idx / kSwaQuantBlock;
            const int scale_i = group_linear % nope_groups;
            const int key_i = group_linear / nope_groups;
            const int d = scale_i * kSwaQuantBlock + lane;
            const uint8_t* data_ptr = packed_data_rows[key_i];
            if (data_ptr != nullptr) {
                key_tile_shared[static_cast<int64_t>(key_i) * kSwaHeadDim + d] =
                    fp8_e4m3_to_float(data_ptr[d], packed_scales[key_i * kSwaScaleBytes + scale_i]);
            } else {
                key_tile_shared[static_cast<int64_t>(key_i) * kSwaHeadDim + d] =
                    attention_key_at(kv,
                                     cmp_cache,
                                     cmp_cu_lens,
                                     cmp_pool,
                                     cmp_block_table,
                                     cmp_seq_lens,
                                     cmp_pool_block_size,
                                     cmp_pool_block_stride,
                                     cmp_block_table_stride,
                                     swa_cache,
                                     swa_cu_lens,
                                     swa_pool,
                                     swa_slot_mapping,
                                     swa_gather_lens,
                                     swa_pool_block_size,
                                     swa_pool_block_stride,
                                     swa_slot_mapping_stride,
                                     key_is_compressed[key_i],
                                     req,
                                     key_pos[key_i],
                                     prefix_len,
                                     0,
                                     d,
                                     L,
                                     KH,
                                     D,
                                     kv_unpad_restore,
                                     kv_cu_lens,
                                     symm_buffer_ptrs,
                                     use_symm_direct_fresh,
                                     per_rank_buffer_bytes,
                                     fresh_payload_offset,
                                     symm_l_local);
            }
        }
        const int rope_total = key_count * (kSwaHeadDim - kSwaNopeDim);
        for (int idx = tid; idx < rope_total; idx += static_cast<int>(blockDim.x)) {
            const int key_i = idx / (kSwaHeadDim - kSwaNopeDim);
            const int rope_d = idx - key_i * (kSwaHeadDim - kSwaNopeDim);
            const int d = kSwaNopeDim + rope_d;
            const uint8_t* data_ptr = packed_data_rows[key_i];
            if (data_ptr != nullptr) {
                key_tile_shared[static_cast<int64_t>(key_i) * kSwaHeadDim + d] =
                    bf16_bits_to_float(load_u16_unaligned(data_ptr + kSwaNopeDim
                                                          + rope_d * static_cast<int>(sizeof(uint16_t))));
            } else {
                key_tile_shared[static_cast<int64_t>(key_i) * kSwaHeadDim + d] =
                    attention_key_at(kv,
                                     cmp_cache,
                                     cmp_cu_lens,
                                     cmp_pool,
                                     cmp_block_table,
                                     cmp_seq_lens,
                                     cmp_pool_block_size,
                                     cmp_pool_block_stride,
                                     cmp_block_table_stride,
                                     swa_cache,
                                     swa_cu_lens,
                                     swa_pool,
                                     swa_slot_mapping,
                                     swa_gather_lens,
                                     swa_pool_block_size,
                                     swa_pool_block_stride,
                                     swa_slot_mapping_stride,
                                     key_is_compressed[key_i],
                                     req,
                                     key_pos[key_i],
                                     prefix_len,
                                     0,
                                     d,
                                     L,
                                     KH,
                                     D,
                                     kv_unpad_restore,
                                     kv_cu_lens,
                                     symm_buffer_ptrs,
                                     use_symm_direct_fresh,
                                     per_rank_buffer_bytes,
                                     fresh_payload_offset,
                                     symm_l_local);
            }
        }
    } else {
        const int total = key_count * D;
        for (int idx = tid; idx < total; idx += static_cast<int>(blockDim.x)) {
            const int key_i = idx / D;
            const int d = idx - key_i * D;
            key_tile_shared[static_cast<int64_t>(key_i) * kSwaHeadDim + d] =
                attention_key_at(kv,
                                 cmp_cache,
                                 cmp_cu_lens,
                                 cmp_pool,
                                 cmp_block_table,
                                 cmp_seq_lens,
                                 cmp_pool_block_size,
                                 cmp_pool_block_stride,
                                 cmp_block_table_stride,
                                 swa_cache,
                                 swa_cu_lens,
                                 swa_pool,
                                 swa_slot_mapping,
                                 swa_gather_lens,
                                 swa_pool_block_size,
                                 swa_pool_block_stride,
                                 swa_slot_mapping_stride,
                                 key_is_compressed[key_i],
                                 req,
                                 key_pos[key_i],
                                 prefix_len,
                                 0,
                                 d,
                                 L,
                                 KH,
                                 D,
                                 kv_unpad_restore,
                                 kv_cu_lens,
                                 symm_buffer_ptrs,
                                 use_symm_direct_fresh,
                                 per_rank_buffer_bytes,
                                 fresh_payload_offset,
                                 symm_l_local);
        }
    }
    __syncthreads();
}

__device__ __forceinline__ void dsv4MegaConsumeLoadedKeyTileGrouped(
    const float* __restrict__ key_tile_shared,
    const float (&q_values)[kMegaAttentionDimsPerWarp],
    int key_count,
    float scale,
    float& online_max,
    float& online_denom,
    float (&acc)[kMegaAttentionDimsPerWarp]) {
    const int lane = static_cast<int>(threadIdx.x) & (kMegaAttentionWarpSize - 1);
#pragma unroll
    for (int key_i = 0; key_i < kMegaAttentionGroupedKeyTile; ++key_i) {
        if (key_i >= key_count) {
            continue;
        }
        const float* key_base = key_tile_shared + static_cast<int64_t>(key_i) * kSwaHeadDim;
        float dot_part = 0.0f;
#pragma unroll
        for (int chunk = 0; chunk < kMegaAttentionDimsPerWarp; ++chunk) {
            const int d = lane + chunk * kMegaAttentionWarpSize;
            dot_part += q_values[chunk] * key_base[d];
        }
        const float online_logit = dsv4MegaWarpReduceSum(dot_part) * scale;
        float new_max = 0.0f;
        float old_scale = 0.0f;
        float key_scale = 0.0f;
        float new_denom = 0.0f;
        if (lane == 0) {
            new_max = fmaxf(online_max, online_logit);
            old_scale = expf(online_max - new_max);
            key_scale = expf(online_logit - new_max);
            new_denom = online_denom * old_scale + key_scale;
        }
        new_max = __shfl_sync(0xffffffffu, new_max, 0);
        old_scale = __shfl_sync(0xffffffffu, old_scale, 0);
        key_scale = __shfl_sync(0xffffffffu, key_scale, 0);
        new_denom = __shfl_sync(0xffffffffu, new_denom, 0);
        online_max = new_max;
        online_denom = new_denom;
#pragma unroll
        for (int chunk = 0; chunk < kMegaAttentionDimsPerWarp; ++chunk) {
            const int d = lane + chunk * kMegaAttentionWarpSize;
            acc[chunk] = acc[chunk] * old_scale + key_scale * key_base[d];
        }
    }
}

__device__ __forceinline__ float* dsv4MegaSplitKRecord(uint8_t* splitk_base, int record_slot) {
    return reinterpret_cast<float*>(splitk_base + static_cast<int64_t>(record_slot) * kMegaSplitKRecordBytes);
}

__device__ __forceinline__ float* dsv4MegaSplitKRecordMax(float* record) {
    return record;
}

__device__ __forceinline__ float* dsv4MegaSplitKRecordDenom(float* record) {
    return record + kMegaAttentionHeadsPerCta;
}

__device__ __forceinline__ float* dsv4MegaSplitKRecordAcc(float* record) {
    return record + 2 * kMegaAttentionHeadsPerCta;
}

struct MegaSplitKTiling {
    int keys_per_block;
    int key_blocks;
};

__device__ __forceinline__ int dsv4MegaCeilDivInt(int x, int y) {
    return (x + y - 1) / y;
}

__device__ __forceinline__ int dsv4MegaAlignUpInt(int x, int align) {
    return dsv4MegaCeilDivInt(x, align) * align;
}

__device__ __forceinline__ int dsv4MegaSplitKBaseKeysPerBlock(int total_keys, int requested_keys_per_block) {
    if (total_keys >= 2048) {
        return max_int(requested_keys_per_block, 256);
    }
    if (total_keys >= 1024) {
        return max_int(requested_keys_per_block, 128);
    }
    return max_int(requested_keys_per_block, kMegaSplitKKeysPerBlock);
}

__device__ __forceinline__ MegaSplitKTiling dsv4MegaSplitKActualTiling(int total_keys, int requested_keys_per_block) {
    const int keys = max_int(total_keys, 1);
    int keys_per_block =
        max_int(dsv4MegaSplitKBaseKeysPerBlock(keys, requested_keys_per_block), kMegaAttentionGroupedKeyTile);
    int key_blocks = dsv4MegaCeilDivInt(keys, keys_per_block);
    if (key_blocks == kMegaSplitKGroupSize) {
        keys_per_block = max_int(keys_per_block, dsv4MegaCeilDivInt(keys, kMegaSplitKBlocksPerWave));
        keys_per_block =
            dsv4MegaAlignUpInt(keys_per_block, kMegaAttentionGroupedKeyTile);
        key_blocks = dsv4MegaCeilDivInt(keys, keys_per_block);
    }
    return {keys_per_block, max_int(key_blocks, 1)};
}

__device__ __forceinline__ int dsv4MegaCombinedKeyPos(int combined_idx,
                                                      int compressed_count,
                                                      int compress_ratio,
                                                      int swa_start,
                                                      const int* __restrict__ row_compressed_indices,
                                                      int sequential_compressed_count,
                                                      int use_compressed_cache,
                                                      int& key_is_compressed) {
    if (combined_idx < compressed_count) {
        const int cmp_idx =
            sequential_compressed_count >= 0 ? combined_idx : row_compressed_indices[combined_idx];
        key_is_compressed = use_compressed_cache ? 1 : 0;
        return use_compressed_cache ? cmp_idx : (cmp_idx + 1) * compress_ratio - 1;
    }
    key_is_compressed = 0;
    return swa_start + (combined_idx - compressed_count);
}

__device__ __forceinline__ void dsv4MegaInitializeSplitKRecord(float* record, int include_sink) {
    const int tid = static_cast<int>(threadIdx.x);
    for (int h = tid; h < kMegaAttentionHeadsPerCta; h += static_cast<int>(blockDim.x)) {
        dsv4MegaSplitKRecordMax(record)[h] = include_sink ? 0.0f : kInvalidCompressorScore;
        dsv4MegaSplitKRecordDenom(record)[h] = include_sink ? 1.0f : 0.0f;
    }
    const int total_acc = kMegaAttentionHeadsPerCta * kSwaHeadDim;
    float* acc = dsv4MegaSplitKRecordAcc(record);
    for (int i = tid; i < total_acc; i += static_cast<int>(blockDim.x)) {
        acc[i] = 0.0f;
    }
    __syncthreads();
}

__device__ __forceinline__ void dsv4MegaInitializeSplitKMergeRecord(float* record,
                                                                    const float* __restrict__ attn_sink,
                                                                    int h_base) {
    const int tid = static_cast<int>(threadIdx.x);
    for (int h = tid; h < kMegaAttentionHeadsPerCta; h += static_cast<int>(blockDim.x)) {
        dsv4MegaSplitKRecordMax(record)[h] = attn_sink[h_base + h];
        dsv4MegaSplitKRecordDenom(record)[h] = 1.0f;
    }
    const int total_acc = kMegaAttentionHeadsPerCta * kSwaHeadDim;
    float* acc = dsv4MegaSplitKRecordAcc(record);
    for (int i = tid; i < total_acc; i += static_cast<int>(blockDim.x)) {
        acc[i] = 0.0f;
    }
    __syncthreads();
}

__device__ __forceinline__ void dsv4MegaMergeSplitKWave(float* dst,
                                                        uint8_t* splitk_base,
                                                        int physical_group,
                                                        int wave_blocks) {
    __shared__ float wave_src_scale[kMegaSplitKBlocksPerWave][kMegaAttentionHeadsPerCta];
    __shared__ float wave_partial_m[kMegaSplitKBlocksPerWave][kMegaAttentionHeadsPerCta];
    __shared__ float wave_dst_scale[kMegaAttentionHeadsPerCta];
    __shared__ float wave_new_m[kMegaAttentionHeadsPerCta];
    __shared__ float wave_new_l[kMegaAttentionHeadsPerCta];
    const int tid = static_cast<int>(threadIdx.x);
    if (tid < kMegaAttentionHeadsPerCta) {
        const int h = tid;
        const float dst_m = dsv4MegaSplitKRecordMax(dst)[h];
        const float dst_l = dsv4MegaSplitKRecordDenom(dst)[h];
        float new_m = dst_m;
#pragma unroll
        for (int partial_i = 1; partial_i <= kMegaSplitKBlocksPerWave; ++partial_i) {
            float src_m = kInvalidCompressorScore;
            float src_l = 0.0f;
            if (partial_i <= wave_blocks) {
                const float* partial_record =
                    dsv4MegaSplitKRecord(splitk_base, physical_group * kMegaSplitKGroupSize + partial_i);
                src_m = partial_record[h];
                src_l = partial_record[kMegaAttentionHeadsPerCta + h];
                if (src_l > 0.0f) {
                    new_m = fmaxf(new_m, src_m);
                }
            }
            wave_partial_m[partial_i - 1][h] = src_m;
            wave_src_scale[partial_i - 1][h] = src_l;
        }
        const float dst_scale = dst_l > 0.0f ? expf(dst_m - new_m) : 0.0f;
        float new_l = dst_l * dst_scale;
        wave_dst_scale[h] = dst_scale;
#pragma unroll
        for (int partial_i = 1; partial_i <= kMegaSplitKBlocksPerWave; ++partial_i) {
            float src_scale = 0.0f;
            const float src_l = wave_src_scale[partial_i - 1][h];
            if (src_l > 0.0f) {
                src_scale = expf(wave_partial_m[partial_i - 1][h] - new_m);
                new_l += src_l * src_scale;
            }
            wave_src_scale[partial_i - 1][h] = src_scale;
        }
        wave_new_m[h] = new_m;
        wave_new_l[h] = new_l;
    }
    __syncthreads();

    float* dst_acc = dsv4MegaSplitKRecordAcc(dst);
    const int total_acc = kMegaAttentionHeadsPerCta * kSwaHeadDim;
    for (int i = tid; i < total_acc; i += static_cast<int>(blockDim.x)) {
        const int h = i / kSwaHeadDim;
        float value = dst_acc[i] * wave_dst_scale[h];
#pragma unroll
        for (int partial_i = 1; partial_i <= kMegaSplitKBlocksPerWave; ++partial_i) {
            if (partial_i <= wave_blocks) {
                const float* partial_record =
                    dsv4MegaSplitKRecord(splitk_base, physical_group * kMegaSplitKGroupSize + partial_i);
                const float* partial_acc = partial_record + 2 * kMegaAttentionHeadsPerCta;
                value += partial_acc[i] * wave_src_scale[partial_i - 1][h];
            }
        }
        dst_acc[i] = value;
    }
    __syncthreads();
    if (tid < kMegaAttentionHeadsPerCta) {
        dsv4MegaSplitKRecordMax(dst)[tid] = wave_new_m[tid];
        dsv4MegaSplitKRecordDenom(dst)[tid] = wave_new_l[tid];
    }
    __syncthreads();
}

template<typename scalar_t>
__device__ __forceinline__ void dsv4MegaComputeSplitKPartial(const scalar_t* __restrict__ q,
                                                             const scalar_t* __restrict__ kv,
                                                             const uint8_t* __restrict__ cmp_cache,
                                                             const int64_t* __restrict__ cmp_cu_lens,
                                                             const uint8_t* __restrict__ cmp_pool,
                                                             const int32_t* __restrict__ cmp_block_table,
                                                             const int32_t* __restrict__ cmp_seq_lens,
                                                             int cmp_pool_block_size,
                                                             int64_t cmp_pool_block_stride,
                                                             int64_t cmp_block_table_stride,
                                                             const uint8_t* __restrict__ swa_cache,
                                                             const int64_t* __restrict__ swa_cu_lens,
                                                             const uint8_t* __restrict__ swa_pool,
                                                             const int64_t* __restrict__ swa_slot_mapping,
                                                             const int32_t* __restrict__ swa_gather_lens,
                                                             int swa_pool_block_size,
                                                             int64_t swa_pool_block_stride,
                                                             int64_t swa_slot_mapping_stride,
                                                             const int* __restrict__ row_compressed_indices,
                                                             int sequential_compressed_count,
                                                             const int64_t* __restrict__ kv_unpad_restore,
                                                             const int64_t* __restrict__ kv_cu_lens,
                                                             const uint8_t* const* __restrict__ symm_buffer_ptrs,
                                                             int use_symm_direct_fresh,
                                                             int64_t per_rank_buffer_bytes,
                                                             int64_t fresh_payload_offset,
                                                             int symm_l_local,
                                                             int64_t row,
                                                             int req,
                                                             int q_pos,
                                                             int prefix_len,
                                                             int h_base,
                                                             int H,
                                                             int D,
                                                             int L,
                                                             int KH,
                                                             int compress_ratio,
                                                             int compressed_count,
                                                             int total_keys,
                                                             int key_block_start,
                                                             int key_block_end,
                                                             int use_compressed_cache,
                                                             int swa_start,
                                                             float scale,
                                                             float* __restrict__ key_tile_shared,
                                                             float* record) {
    const int tid = static_cast<int>(threadIdx.x);
    const int warp_id = tid / kMegaAttentionWarpSize;
    const int lane_id = tid & (kMegaAttentionWarpSize - 1);

    // Active partials fully overwrite max/denom/acc; empty partials must still
    // publish a clean zero record because the merge path may read one block.
    if (key_block_start >= total_keys) {
        dsv4MegaInitializeSplitKRecord(record, 0);
        return;
    }
    if (warp_id >= kMegaAttentionHeadsPerCta) {
        return;
    }
    const int head = h_base + warp_id;
    float q_values[kMegaAttentionDimsPerWarp];
    float acc[kMegaAttentionDimsPerWarp];
#pragma unroll
    for (int chunk = 0; chunk < kMegaAttentionDimsPerWarp; ++chunk) {
        const int d = lane_id + chunk * kMegaAttentionWarpSize;
        q_values[chunk] = q_at(q, row, head, d, H, D);
        acc[chunk] = 0.0f;
    }
    float online_max = kInvalidCompressorScore;
    float online_denom = 0.0f;
    const int end = min_int(key_block_end, total_keys);
    int tile_key_pos[kMegaAttentionGroupedKeyTile];
    int tile_key_is_compressed[kMegaAttentionGroupedKeyTile];
    for (int key_idx = key_block_start; key_idx < end;) {
        int tile_count = 0;
#pragma unroll
        for (int key_i = 0; key_i < kMegaAttentionGroupedKeyTile; ++key_i) {
            if (key_idx + key_i < end) {
                int key_is_compressed = 0;
                tile_key_pos[key_i] = dsv4MegaCombinedKeyPos(key_idx + key_i,
                                                             compressed_count,
                                                             compress_ratio,
                                                             swa_start,
                                                             row_compressed_indices,
                                                             sequential_compressed_count,
                                                             use_compressed_cache,
                                                             key_is_compressed);
                tile_key_is_compressed[key_i] = key_is_compressed;
                ++tile_count;
            }
        }
        dsv4MegaLoadAttentionKeyTile<scalar_t>(kv,
                                               cmp_cache,
                                               cmp_cu_lens,
                                               cmp_pool,
                                               cmp_block_table,
                                               cmp_seq_lens,
                                               cmp_pool_block_size,
                                               cmp_pool_block_stride,
                                               cmp_block_table_stride,
                                               swa_cache,
                                               swa_cu_lens,
                                               swa_pool,
                                               swa_slot_mapping,
                                               swa_gather_lens,
                                               swa_pool_block_size,
                                               swa_pool_block_stride,
                                               swa_slot_mapping_stride,
                                               tile_key_is_compressed,
                                               tile_key_pos,
                                               tile_count,
                                               req,
                                               prefix_len,
                                               D,
                                               L,
                                               KH,
                                               kv_unpad_restore,
                                               kv_cu_lens,
                                               symm_buffer_ptrs,
                                               use_symm_direct_fresh,
                                               per_rank_buffer_bytes,
                                               fresh_payload_offset,
                                               symm_l_local,
                                               key_tile_shared);
        dsv4MegaConsumeLoadedKeyTileGrouped(key_tile_shared, q_values, tile_count, scale, online_max, online_denom, acc);
        __syncthreads();
        key_idx += tile_count;
    }
    if (lane_id == 0) {
        record[warp_id] = online_max;
        record[kMegaAttentionHeadsPerCta + warp_id] = online_denom;
    }
#pragma unroll
    for (int chunk = 0; chunk < kMegaAttentionDimsPerWarp; ++chunk) {
        const int d = lane_id + chunk * kMegaAttentionWarpSize;
        record[2 * kMegaAttentionHeadsPerCta + warp_id * kSwaHeadDim + d] = acc[chunk];
    }
    __syncthreads();
}

template<typename scalar_t>
__device__ __forceinline__ void dsv4MegaAttentionLogitsForKeyTile(const float* __restrict__ q_shared,
                                                                  const scalar_t* __restrict__ kv,
                                                                  const uint8_t* __restrict__ cmp_cache,
                                                                  const int64_t* __restrict__ cmp_cu_lens,
                                                                  const uint8_t* __restrict__ cmp_pool,
                                                                  const int32_t* __restrict__ cmp_block_table,
                                                                  const int32_t* __restrict__ cmp_seq_lens,
                                                                  int cmp_pool_block_size,
                                                                  int64_t cmp_pool_block_stride,
                                                                  int64_t cmp_block_table_stride,
                                                                  const uint8_t* __restrict__ swa_cache,
                                                                  const int64_t* __restrict__ swa_cu_lens,
                                                                  const uint8_t* __restrict__ swa_pool,
                                                                  const int64_t* __restrict__ swa_slot_mapping,
                                                                  const int32_t* __restrict__ swa_gather_lens,
                                                                  int swa_pool_block_size,
                                                                  int64_t swa_pool_block_stride,
                                                                  int64_t swa_slot_mapping_stride,
                                                                  const int* __restrict__ key_is_compressed,
                                                                  const int* __restrict__ key_pos,
                                                                  int key_count,
                                                                  int req,
                                                                  int prefix_len,
                                                                  int h,
                                                                  int D,
                                                                  int L,
                                                                  int KH,
                                                                  const int64_t* __restrict__ kv_unpad_restore,
                                                                  const int64_t* __restrict__ kv_cu_lens,
                                                                  const uint8_t* const* __restrict__ symm_buffer_ptrs,
                                                                  int use_symm_direct_fresh,
                                                                  int64_t per_rank_buffer_bytes,
                                                                  int64_t fresh_payload_offset,
                                                                  int symm_l_local,
                                                                  float scale,
                                                                  float* reduce_buf,
                                                                  float* dot_shared,
                                                                  float* logits,
                                                                  float* values0,
                                                                  float* values1) {
    const int tid = static_cast<int>(threadIdx.x);
    float dot_part[kMegaAttentionKeyTile];
#pragma unroll
    for (int i = 0; i < kMegaAttentionKeyTile; ++i) {
        dot_part[i] = 0.0f;
        if (tid == 0) {
            logits[i] = kInvalidCompressorScore;
        }
    }

    float local_value0[kMegaAttentionKeyTile];
    float local_value1[kMegaAttentionKeyTile];
#pragma unroll
    for (int i = 0; i < kMegaAttentionKeyTile; ++i) {
        local_value0[i] = 0.0f;
        local_value1[i] = 0.0f;
    }

    for (int d = tid; d < D; d += static_cast<int>(blockDim.x)) {
        const float q_value = q_shared[d];
#pragma unroll
        for (int i = 0; i < kMegaAttentionKeyTile; ++i) {
            if (i < key_count) {
                const float key_value = attention_key_at(kv,
                                                         cmp_cache,
                                                         cmp_cu_lens,
                                                         cmp_pool,
                                                         cmp_block_table,
                                                         cmp_seq_lens,
                                                         cmp_pool_block_size,
                                                         cmp_pool_block_stride,
                                                         cmp_block_table_stride,
                                                         swa_cache,
                                                         swa_cu_lens,
                                                         swa_pool,
                                                         swa_slot_mapping,
                                                         swa_gather_lens,
                                                         swa_pool_block_size,
                                                         swa_pool_block_stride,
                                                         swa_slot_mapping_stride,
                                                         key_is_compressed[i],
                                                         req,
                                                         key_pos[i],
                                                         prefix_len,
                                                         h,
                                                         d,
                                                         L,
                                                         KH,
                                                         D,
                                                         kv_unpad_restore,
                                                         kv_cu_lens,
                                                         symm_buffer_ptrs,
                                                         use_symm_direct_fresh,
                                                         per_rank_buffer_bytes,
                                                         fresh_payload_offset,
                                                         symm_l_local);
                if (d == tid) {
                    local_value0[i] = key_value;
                } else {
                    local_value1[i] = key_value;
                }
                dot_part[i] += q_value * key_value;
            }
        }
    }

#pragma unroll
    for (int i = 0; i < kMegaAttentionKeyTile; ++i) {
        if (i < key_count) {
            values0[i] = local_value0[i];
            values1[i] = local_value1[i];
        }
    }
    dsv4MegaBlockReduceSumTile(dot_part, reduce_buf, dot_shared);
    if (tid == 0) {
#pragma unroll
        for (int i = 0; i < kMegaAttentionKeyTile; ++i) {
            if (i < key_count) {
                logits[i] = dot_shared[i] * scale;
            }
        }
    }
    __syncthreads();
}

template<typename scalar_t>
__device__ __forceinline__ float indexer_q_at(const scalar_t* q, int64_t row, int h, int d, int IH, int ID) {
    return to_float_device(q[(row * IH + h) * ID + d]);
}

template<typename scalar_t>
__device__ __forceinline__ float
indexer_k_at(const scalar_t* k,
             int req,
             int pos,
             int h,
             int d,
             int LI,
             int IH,
             int ID,
             const uint8_t* const* __restrict__ symm_buffer_ptrs,
             int use_symm_direct_indexer,
             int64_t per_rank_buffer_bytes,
             int64_t payload_offset,
             int symm_l_local) {
    if (use_symm_direct_indexer && symm_buffer_ptrs != nullptr) {
        return symm_4d_at<scalar_t>(
            symm_buffer_ptrs, per_rank_buffer_bytes, payload_offset, symm_l_local, IH, ID, req, pos, h, d);
    }
    return to_float_device(k[((static_cast<int64_t>(req) * LI + pos) * IH + h) * ID + d]);
}

__device__ __forceinline__ float indexer_pool_k_at(const uint8_t* pool,
                                                   const int32_t* block_table,
                                                   int req,
                                                   int pos,
                                                   int d,
                                                   int block_size,
                                                   int64_t block_stride,
                                                   int64_t block_table_stride) {
    const int64_t block_row = pos / block_size;
    const int64_t in_block = pos % block_size;
    const int32_t block_id = block_table[static_cast<int64_t>(req) * block_table_stride + block_row];
    if (block_id < 0) {
        return 0.0f;
    }
    const uint8_t* block = pool + static_cast<int64_t>(block_id) * block_stride;
    const uint8_t* token_ptr = block + in_block * kIndexerHeadDim;
    const uint8_t* scale_ptr = block + static_cast<int64_t>(block_size) * kIndexerHeadDim + in_block * 4;
    const float scale = load_f32_unaligned(scale_ptr);
    return fp8_e4m3_to_float(token_ptr[d], scale);
}

template<typename scalar_t>
__device__ float indexer_score(const scalar_t* indexer_q,
                               const scalar_t* indexer_k,
                               int64_t row,
                               int req,
                               int pos,
                               int IH,
                               int ID,
                               int LI,
                               const uint8_t* const* __restrict__ symm_buffer_ptrs,
                               int use_symm_direct_indexer,
                               int64_t per_rank_buffer_bytes,
                               int64_t payload_offset,
                               int symm_l_local) {
    float score = 0.0f;
    for (int h = 0; h < IH; ++h) {
        for (int d = 0; d < ID; ++d) {
            score += indexer_q_at(indexer_q, row, h, d, IH, ID)
                     * indexer_k_at(indexer_k,
                                    req,
                                    pos,
                                    h,
                                    d,
                                    LI,
                                    IH,
                                    ID,
                                    symm_buffer_ptrs,
                                    use_symm_direct_indexer,
                                    per_rank_buffer_bytes,
                                    payload_offset,
                                    symm_l_local);
        }
    }
    return score;
}

template<typename scalar_t>
__device__ float indexer_score_fp8_pool(const scalar_t* indexer_q,
                                        const uint8_t* fp8_pool,
                                        const int32_t* block_table,
                                        const float* weights,
                                        int64_t row,
                                        int req,
                                        int pos,
                                        int IH,
                                        int ID,
                                        int block_size,
                                        int64_t block_stride,
                                        int64_t block_table_stride) {
    float score = 0.0f;
    for (int h = 0; h < IH; ++h) {
        float dot = 0.0f;
        for (int d = 0; d < ID; ++d) {
            dot += indexer_q_at(indexer_q, row, h, d, IH, ID)
                   * indexer_pool_k_at(fp8_pool, block_table, req, pos, d, block_size, block_stride, block_table_stride);
        }
        if (dot > 0.0f) {
            score += dot * weights[row * IH + h];
        }
    }
    return score;
}

template<typename scalar_t>
__device__ float indexer_score_fp8_cache(const scalar_t* indexer_q,
                                         const uint8_t* fp8_cache,
                                         const float* weights,
                                         const int64_t* cu_lens,
                                         int64_t row,
                                         int req,
                                         int pos,
                                         int IH,
                                         int ID) {
    const int64_t base = cu_lens[req] + pos;
    const uint8_t* k_row = fp8_cache + base * kIndexerEntryBytes;
    const float k_scale = load_f32_unaligned(k_row + kIndexerHeadDim);
    float score = 0.0f;
    for (int h = 0; h < IH; ++h) {
        float dot = 0.0f;
        for (int d = 0; d < ID; ++d) {
            dot += indexer_q_at(indexer_q, row, h, d, IH, ID) * fp8_e4m3_to_float(k_row[d], k_scale);
        }
        if (dot > 0.0f) {
            score += dot * weights[row * IH + h];
        }
    }
    return score;
}

template<typename scalar_t>
__device__ __forceinline__ float dsv4MegaIndexerScoreParallel(const scalar_t* __restrict__ indexer_q,
                                                              const scalar_t* __restrict__ indexer_k,
                                                              const uint8_t* __restrict__ csa_indexer_k_cache,
                                                              const float* __restrict__ csa_indexer_weights,
                                                              const int64_t* __restrict__ csa_indexer_cu_lens,
                                                              const uint8_t* __restrict__ csa_indexer_k_pool,
                                                              const int32_t* __restrict__ csa_indexer_block_table,
                                                              int64_t row,
                                                              int req,
                                                              int c,
                                                              int IH,
                                                              int ID,
                                                              int LI,
                                                              int csa_indexer_pool_block_size,
                                                              int64_t csa_indexer_pool_block_stride,
                                                              int64_t csa_indexer_block_table_stride,
                                                              const uint8_t* const* __restrict__ symm_buffer_ptrs,
                                                              int use_symm_direct_indexer,
                                                              int64_t symm_per_rank_buffer_bytes,
                                                              int64_t symm_indexer_payload_offset,
                                                              int symm_indexer_l_local,
                                                              float* reduce_buf,
                                                              float* result_buf) {
    const int tid = static_cast<int>(threadIdx.x);
    float     score_part = 0.0f;
    const bool use_fp8_indexer_pool = csa_indexer_k_pool != nullptr && csa_indexer_block_table != nullptr
                                      && csa_indexer_weights != nullptr;
    const bool use_fp8_indexer_cache = !use_fp8_indexer_pool && csa_indexer_k_cache != nullptr
                                       && csa_indexer_weights != nullptr && csa_indexer_cu_lens != nullptr;

    if (ID == kIndexerHeadDim && IH > 0 && IH <= 64 && static_cast<int>(blockDim.x) >= IH * 4) {
        float* head_scores = reduce_buf + 128;
        for (int h = tid; h < IH; h += static_cast<int>(blockDim.x)) {
            head_scores[h] = 0.0f;
        }
        __syncthreads();

        const int head = tid >> 2;
        const int lane_in_head = tid & 3;
        const unsigned int active_head_mask = __ballot_sync(0xffffffffu, head < IH);
        float head_dot = 0.0f;
        if (head < IH) {
            if (use_fp8_indexer_pool) {
                for (int d = lane_in_head; d < ID; d += 4) {
                    head_dot += indexer_q_at(indexer_q, row, head, d, IH, ID)
                                * indexer_pool_k_at(csa_indexer_k_pool,
                                                    csa_indexer_block_table,
                                                    req,
                                                    c,
                                                    d,
                                                    csa_indexer_pool_block_size,
                                                    csa_indexer_pool_block_stride,
                                                    csa_indexer_block_table_stride);
                }
            } else if (use_fp8_indexer_cache) {
                const int64_t base = csa_indexer_cu_lens[req] + c;
                const uint8_t* k_row = csa_indexer_k_cache + base * kIndexerEntryBytes;
                const float k_scale = load_f32_unaligned(k_row + kIndexerHeadDim);
                for (int d = lane_in_head; d < ID; d += 4) {
                    head_dot += indexer_q_at(indexer_q, row, head, d, IH, ID) * fp8_e4m3_to_float(k_row[d], k_scale);
                }
            } else {
                for (int d = lane_in_head; d < ID; d += 4) {
                    head_dot += indexer_q_at(indexer_q, row, head, d, IH, ID)
                                * indexer_k_at(indexer_k,
                                               req,
                                               c,
                                               head,
                                               d,
                                               LI,
                                               IH,
                                               ID,
                                               symm_buffer_ptrs,
                                               use_symm_direct_indexer,
                                               symm_per_rank_buffer_bytes,
                                               symm_indexer_payload_offset,
                                               symm_indexer_l_local);
                }
            }
            head_dot += __shfl_xor_sync(active_head_mask, head_dot, 1);
            head_dot += __shfl_xor_sync(active_head_mask, head_dot, 2);
            if (lane_in_head == 0) {
                if (use_fp8_indexer_pool || use_fp8_indexer_cache) {
                    head_scores[head] = head_dot > 0.0f ? head_dot * csa_indexer_weights[row * IH + head] : 0.0f;
                } else {
                    head_scores[head] = head_dot;
                }
            }
        }
        __syncthreads();

        float score_part = 0.0f;
        for (int h = tid; h < IH; h += static_cast<int>(blockDim.x)) {
            score_part += head_scores[h];
        }
        return dsv4MegaBlockReduceSum(score_part, reduce_buf, result_buf);
    }

    if (use_fp8_indexer_pool) {
        for (int h = tid; h < IH; h += static_cast<int>(blockDim.x)) {
            float dot = 0.0f;
            for (int d = 0; d < ID; ++d) {
                dot += indexer_q_at(indexer_q, row, h, d, IH, ID)
                       * indexer_pool_k_at(csa_indexer_k_pool,
                                           csa_indexer_block_table,
                                           req,
                                           c,
                                           d,
                                           csa_indexer_pool_block_size,
                                           csa_indexer_pool_block_stride,
                                           csa_indexer_block_table_stride);
            }
            if (dot > 0.0f) {
                score_part += dot * csa_indexer_weights[row * IH + h];
            }
        }
    } else if (use_fp8_indexer_cache) {
        const int64_t base = csa_indexer_cu_lens[req] + c;
        const uint8_t* k_row = csa_indexer_k_cache + base * kIndexerEntryBytes;
        const float k_scale = load_f32_unaligned(k_row + kIndexerHeadDim);
        for (int h = tid; h < IH; h += static_cast<int>(blockDim.x)) {
            float dot = 0.0f;
            for (int d = 0; d < ID; ++d) {
                dot += indexer_q_at(indexer_q, row, h, d, IH, ID) * fp8_e4m3_to_float(k_row[d], k_scale);
            }
            if (dot > 0.0f) {
                score_part += dot * csa_indexer_weights[row * IH + h];
            }
        }
    } else {
        for (int h = tid; h < IH; h += static_cast<int>(blockDim.x)) {
            float dot = 0.0f;
            for (int d = 0; d < ID; ++d) {
                dot += indexer_q_at(indexer_q, row, h, d, IH, ID)
                       * indexer_k_at(indexer_k,
                                      req,
                                      c,
                                      h,
                                      d,
                                      LI,
                                      IH,
                                      ID,
                                      symm_buffer_ptrs,
                                      use_symm_direct_indexer,
                                      symm_per_rank_buffer_bytes,
                                      symm_indexer_payload_offset,
                                      symm_indexer_l_local);
            }
            score_part += dot;
        }
    }
    return dsv4MegaBlockReduceSum(score_part, reduce_buf, result_buf);
}

template<typename scalar_t, bool CandidateTileAllValid>
__device__ __forceinline__ void dsv4MegaIndexerScoreCandidateTile(const scalar_t* __restrict__ indexer_q,
                                                                  const scalar_t* __restrict__ indexer_k,
                                                                  const uint8_t* __restrict__ csa_indexer_k_cache,
                                                                  const float* __restrict__ csa_indexer_weights,
                                                                  const int64_t* __restrict__ csa_indexer_cu_lens,
                                                                  const uint8_t* __restrict__ csa_indexer_k_pool,
                                                                  const int32_t* __restrict__ csa_indexer_block_table,
                                                                  int64_t row,
                                                                  int req,
                                                                  int candidate_base,
                                                                  const int* __restrict__ candidate_pos,
                                                                  const int* __restrict__ candidate_valid,
                                                                  int candidate_count,
                                                                  int csa_indexer_pool_block_size,
                                                                  int64_t csa_indexer_pool_block_stride,
                                                                  int64_t csa_indexer_block_table_stride,
                                                                  float* head_scores,
                                                                  uint8_t* metadata_storage,
                                                                  float* tile_scores) {
    const int tid = static_cast<int>(threadIdx.x);
    const int group = tid / (64 * kMegaIndexerCandidateHeadLanes);
    const int head_lane = tid - group * 64 * kMegaIndexerCandidateHeadLanes;
    const int head = head_lane / kMegaIndexerCandidateHeadLanes;
    const int lane_in_head = head_lane - head * kMegaIndexerCandidateHeadLanes;
    const bool use_fp8_indexer_pool = csa_indexer_k_pool != nullptr && csa_indexer_block_table != nullptr
                                      && csa_indexer_weights != nullptr;
    const bool use_fp8_indexer_cache = !use_fp8_indexer_pool && csa_indexer_k_cache != nullptr
                                       && csa_indexer_weights != nullptr && csa_indexer_cu_lens != nullptr;
    const uint8_t** candidate_data_rows = reinterpret_cast<const uint8_t**>(metadata_storage);
    float* candidate_scales = reinterpret_cast<float*>(candidate_data_rows + kMegaIndexerCandidateTile);

    if (tid < kMegaIndexerCandidateTile) {
        tile_scores[tid] = kInvalidCompressorScore;
    }
    if (tid < kMegaIndexerCandidateTile) {
        const int candidate_lane_init = tid;
        const bool candidate_in_range = CandidateTileAllValid || candidate_lane_init < candidate_count;
        int candidate_valid_init = 1;
        if constexpr (!CandidateTileAllValid) {
            candidate_valid_init = candidate_in_range ? candidate_valid[candidate_lane_init] : 0;
        }
        int c = 0;
        if constexpr (CandidateTileAllValid) {
            c = candidate_base + candidate_lane_init;
        } else {
            c = candidate_in_range ? candidate_pos[candidate_lane_init] : 0;
        }
        const uint8_t* data_ptr = nullptr;
        float scale = 1.0f;
        if (candidate_valid_init != 0 && use_fp8_indexer_pool) {
            const int64_t block_row = c / csa_indexer_pool_block_size;
            const int64_t in_block = c - block_row * csa_indexer_pool_block_size;
            const int32_t block_id = csa_indexer_block_table[static_cast<int64_t>(req) * csa_indexer_block_table_stride
                                                             + block_row];
            if (block_id >= 0) {
                const uint8_t* block = csa_indexer_k_pool + static_cast<int64_t>(block_id) * csa_indexer_pool_block_stride;
                data_ptr = block + in_block * kIndexerHeadDim;
                const uint8_t* scale_ptr = block + static_cast<int64_t>(csa_indexer_pool_block_size) * kIndexerHeadDim
                                           + in_block * 4;
                scale = load_f32_unaligned(scale_ptr);
            }
        } else if (candidate_valid_init != 0 && use_fp8_indexer_cache) {
            const int64_t base = csa_indexer_cu_lens[req] + c;
            data_ptr = csa_indexer_k_cache + base * kIndexerEntryBytes;
            scale = load_f32_unaligned(data_ptr + kIndexerHeadDim);
        }
        candidate_data_rows[candidate_lane_init] = data_ptr;
        candidate_scales[candidate_lane_init] = scale;
    }
    __syncthreads();

    float dot[kMegaIndexerCandidatesPerGroup];
#pragma unroll
    for (int j = 0; j < kMegaIndexerCandidatesPerGroup; ++j) {
        dot[j] = 0.0f;
    }
    const bool use_fp8_indexer = use_fp8_indexer_pool || use_fp8_indexer_cache;
    float head_weight = 0.0f;
    if (use_fp8_indexer && head < kMegaIndexerHeads) {
        head_weight = csa_indexer_weights[row * kMegaIndexerHeads + head];
    }
    if (group < kMegaIndexerCandidateGroups && use_fp8_indexer) {
        const uint8_t* candidate_rows[kMegaIndexerCandidatesPerGroup];
        float candidate_scale_values[kMegaIndexerCandidatesPerGroup];
        bool candidate_active_values[kMegaIndexerCandidatesPerGroup];
#pragma unroll
        for (int j = 0; j < kMegaIndexerCandidatesPerGroup; ++j) {
            const int candidate_lane = group * kMegaIndexerCandidatesPerGroup + j;
            bool candidate_active = candidate_data_rows[candidate_lane] != nullptr;
            if constexpr (!CandidateTileAllValid) {
                candidate_active = candidate_lane < candidate_count && candidate_valid[candidate_lane] != 0
                                   && candidate_data_rows[candidate_lane] != nullptr;
            }
            candidate_active_values[j] = candidate_active;
            candidate_rows[j] = candidate_active ? candidate_data_rows[candidate_lane] : nullptr;
            candidate_scale_values[j] = candidate_active ? candidate_scales[candidate_lane] : 1.0f;
        }
        for (int d = lane_in_head; d < kIndexerHeadDim; d += kMegaIndexerCandidateHeadLanes * 2) {
            const float q_value = indexer_q_at(indexer_q, row, head, d, kMegaIndexerHeads, kIndexerHeadDim);
#pragma unroll
            for (int j = 0; j < kMegaIndexerCandidatesPerGroup; ++j) {
                if (candidate_active_values[j]) {
                    dot[j] += q_value * fp8_e4m3_to_float(candidate_rows[j][d], candidate_scale_values[j]);
                }
            }
            const int d_next = d + kMegaIndexerCandidateHeadLanes;
            if (d_next < kIndexerHeadDim) {
                const float q_next =
                    indexer_q_at(indexer_q, row, head, d_next, kMegaIndexerHeads, kIndexerHeadDim);
#pragma unroll
                for (int j = 0; j < kMegaIndexerCandidatesPerGroup; ++j) {
                    if (candidate_active_values[j]) {
                        dot[j] += q_next * fp8_e4m3_to_float(candidate_rows[j][d_next], candidate_scale_values[j]);
                    }
                }
            }
        }
    }
#pragma unroll
    for (int j = 0; j < kMegaIndexerCandidatesPerGroup; ++j) {
        const int candidate_lane = group * kMegaIndexerCandidatesPerGroup + j;
        float reduced_dot = dsv4MegaHeadLaneReduceSum<kMegaIndexerCandidateHeadLanes>(dot[j]);
        bool candidate_active = true;
        if constexpr (!CandidateTileAllValid) {
            candidate_active = candidate_lane < candidate_count && candidate_valid[candidate_lane] != 0;
        }
        const float weighted_head_score =
            (candidate_active && use_fp8_indexer && lane_in_head == 0 && reduced_dot > 0.0f) ?
                reduced_dot * head_weight :
                0.0f;
        if (candidate_active && use_fp8_indexer && lane_in_head == 0) {
            head_scores[candidate_lane * kMegaIndexerHeads + head] = weighted_head_score;
        }
    }
    __syncthreads();

    float* candidate_warp_sums = head_scores + kMegaIndexerCandidateTile * kMegaIndexerHeads;
    for (int sum_idx = tid; sum_idx < kMegaIndexerCandidateTile * kMegaIndexerHeads;
         sum_idx += static_cast<int>(blockDim.x)) {
        const int sum_candidate = sum_idx / kMegaIndexerHeads;
        const int sum_head = sum_idx - sum_candidate * kMegaIndexerHeads;
        bool sum_active = use_fp8_indexer;
        if constexpr (!CandidateTileAllValid) {
            sum_active = sum_candidate < candidate_count && candidate_valid[sum_candidate] != 0
                         && use_fp8_indexer;
        }
        float head_score = sum_active ? head_scores[sum_candidate * kMegaIndexerHeads + sum_head] : 0.0f;
        head_score = dsv4MegaWarpReduceSum(head_score);
        if ((tid & (kMegaAttentionWarpSize - 1)) == 0 && sum_candidate < kMegaIndexerCandidateTile) {
            const int warp_in_candidate = sum_head / kMegaAttentionWarpSize;
            candidate_warp_sums[sum_candidate * kMegaIndexerWarpsPerCandidate + warp_in_candidate] = head_score;
        }
    }
    __syncthreads();
    if (tid < kMegaIndexerCandidateTile) {
        bool candidate_active = use_fp8_indexer;
        if constexpr (!CandidateTileAllValid) {
            candidate_active = tid < candidate_count && candidate_valid[tid] != 0 && use_fp8_indexer;
        }
        if (candidate_active) {
            float score = 0.0f;
#pragma unroll
            for (int warp_i = 0; warp_i < kMegaIndexerWarpsPerCandidate; ++warp_i) {
                score += candidate_warp_sums[tid * kMegaIndexerWarpsPerCandidate + warp_i];
            }
            tile_scores[tid] = score;
        }
    }
    __syncthreads();
}

template<typename scalar_t>
__device__ __forceinline__ int dsv4MegaBuildCompressedIndicesForRow(const scalar_t* __restrict__ indexer_q,
                                                                     const scalar_t* __restrict__ indexer_k,
                                                                     const int64_t* __restrict__ req_id_per_token,
                                                                     const int64_t* __restrict__ position_ids,
                                                                     const int64_t* __restrict__ prefix_lengths,
                                                                     const int64_t* __restrict__ input_lengths,
                                                                     const int64_t* __restrict__ local_rows,
                                                                     int local_row,
                                                                     int compress_ratio,
                                                                     int compressed_topk,
                                                                     int LI,
                                                                     int IH,
                                                                     int ID,
                                                                     const uint8_t* __restrict__ csa_indexer_k_cache,
                                                                     const float* __restrict__ csa_indexer_weights,
                                                                     const int64_t* __restrict__ csa_indexer_cu_lens,
                                                                     int csa_indexer_total_len,
                                                                     const uint8_t* __restrict__ csa_indexer_k_pool,
                                                                     const int32_t* __restrict__ csa_indexer_block_table,
                                                                     const int32_t* __restrict__ csa_indexer_seq_lens,
                                                                     int csa_indexer_pool_block_size,
                                                                     int64_t csa_indexer_pool_block_stride,
                                                                     int64_t csa_indexer_block_table_stride,
                                                                     const int32_t* __restrict__ attention_cmp_seq_lens,
                                                                     const uint8_t* const* __restrict__ symm_buffer_ptrs,
                                                                     int use_symm_direct_indexer,
                                                                     int64_t symm_per_rank_buffer_bytes,
                                                                     int64_t symm_indexer_payload_offset,
                                                                     int symm_indexer_l_local,
                                                                     int compressed_stride,
                                                                     unsigned int compressed_epoch,
                                                                     unsigned long long compressed_epoch_prefix,
                                                                     unsigned long long* __restrict__ compressed_meta,
                                                                     int* __restrict__ row_compressed_indices_base,
                                                                     int* __restrict__ scratch_indices,
                                                                     float* __restrict__ scratch_scores,
                                                                     int* __restrict__ reduce_indices,
                                                                     int* __restrict__ reduce_indices_second,
                                                                     float* __restrict__ reduce_buf,
                                                                     uint8_t* __restrict__ indexer_metadata_storage,
                                                                     float* __restrict__ reduce_result) {
    const int tid = static_cast<int>(threadIdx.x);
    int compressed_count = 0;
    const int64_t topk_row = local_rows[local_row];
    const int     topk_req = static_cast<int>(req_id_per_token[topk_row]);
    const int     topk_q_pos = static_cast<int>(position_ids[topk_row]);
    const int     topk_kv_len = static_cast<int>(prefix_lengths[topk_req] + input_lengths[topk_req]);

    if (compress_ratio == 0) {
        compressed_count = 0;
    } else if (compress_ratio == 128) {
        const int valid = min_int((topk_q_pos + 1) / compress_ratio, compressed_topk);
        int valid_bound = valid;
        if (attention_cmp_seq_lens != nullptr) {
            valid_bound = min_int(valid_bound, static_cast<int>(attention_cmp_seq_lens[topk_req]));
        }
        for (int c = tid; c < valid_bound && c < kMaxCompressedTopK; c += static_cast<int>(blockDim.x)) {
            const int boundary_pos = (c + 1) * compress_ratio - 1;
            if (boundary_pos >= 0 && boundary_pos < topk_kv_len) {
                scratch_indices[c] = c;
            }
        }
        compressed_count = min_int(min_int(valid_bound, kMaxCompressedTopK), max_int(0, topk_kv_len / compress_ratio));
    } else {
        const bool use_fp8_indexer_pool = csa_indexer_k_pool != nullptr && csa_indexer_block_table != nullptr
                                          && csa_indexer_weights != nullptr;
        const bool use_fp8_indexer = use_fp8_indexer_pool
                                     || (csa_indexer_k_cache != nullptr && csa_indexer_weights != nullptr
                                         && csa_indexer_cu_lens != nullptr);
        int req_k_len = LI;
        if (use_fp8_indexer_pool) {
            req_k_len = csa_indexer_seq_lens == nullptr ? LI : static_cast<int>(csa_indexer_seq_lens[topk_req]);
        } else if (use_fp8_indexer) {
            req_k_len = static_cast<int>(csa_indexer_cu_lens[topk_req + 1] - csa_indexer_cu_lens[topk_req]);
            req_k_len = min_int(req_k_len, csa_indexer_total_len);
        }
        const int valid = min_int((topk_q_pos + 1) / compress_ratio, req_k_len);
        const int k_eff = min_int(min_int(valid, compressed_topk), kMaxCompressedTopK);
        compressed_count = k_eff;
        if (k_eff <= 0) {
            compressed_count = 0;
        } else if (k_eff >= valid) {
            for (int c = tid; c < valid && c < kMaxCompressedTopK; c += static_cast<int>(blockDim.x)) {
                const int boundary_pos = (c + 1) * compress_ratio - 1;
                if (boundary_pos >= 0 && boundary_pos < topk_kv_len) {
                    scratch_indices[c] = c;
                }
            }
            compressed_count = min_int(valid, kMaxCompressedTopK);
        } else {
            for (int i = tid; i < k_eff; i += static_cast<int>(blockDim.x)) {
                scratch_indices[i] = -1;
                scratch_scores[i] = kInvalidCompressorScore;
            }
            __syncthreads();
            int cached_worst_slot = -1;
            int cached_second_worst_slot = -1;
            float cached_worst_score = 3.4028234663852886e38F;
            float cached_second_worst_score = 3.4028234663852886e38F;
            int cached_worst_idx = -2147483647;
            int cached_second_worst_idx = -2147483647;
            bool refresh_worst_slot = true;
            // Candidate tiling relies on the DSv4 CSA production shape: 64 indexer
            // heads, 2 lanes/head, and several candidates accumulated per head
            // group so each Q element is reused across candidate scores.
            const bool use_candidate_tile =
                use_fp8_indexer && IH == 64 && ID == kIndexerHeadDim
                && static_cast<int>(blockDim.x) == kMegaBlockThreads;
            for (int c = 0; c < valid;) {
                int candidate_count = 1;
                bool candidate_tile_all_valid = false;
                if (use_candidate_tile) {
                    candidate_count = min_int(kMegaIndexerCandidateTile, valid - c);
                    const int last_tile_boundary = (c + kMegaIndexerCandidateTile) * compress_ratio - 1;
                    candidate_tile_all_valid = candidate_count == kMegaIndexerCandidateTile && last_tile_boundary >= 0
                                               && last_tile_boundary < topk_kv_len;
                    if (candidate_tile_all_valid) {
                        dsv4MegaIndexerScoreCandidateTile<scalar_t, true>(indexer_q,
                                                                          indexer_k,
                                                                          csa_indexer_k_cache,
                                                                          csa_indexer_weights,
                                                                          csa_indexer_cu_lens,
                                                                          csa_indexer_k_pool,
                                                                          csa_indexer_block_table,
                                                                          topk_row,
                                                                          topk_req,
                                                                          c,
                                                                          nullptr,
                                                                          nullptr,
                                                                          candidate_count,
                                                                          csa_indexer_pool_block_size,
                                                                          csa_indexer_pool_block_stride,
                                                                          csa_indexer_block_table_stride,
                                                                          reduce_buf,
                                                                          indexer_metadata_storage,
                                                                          reduce_result);
                    } else {
                        int candidate_pos[kMegaIndexerCandidateTile];
                        int candidate_valid[kMegaIndexerCandidateTile];
#pragma unroll
                        for (int tile_i = 0; tile_i < kMegaIndexerCandidateTile; ++tile_i) {
                            const int candidate = c + tile_i;
                            candidate_pos[tile_i] = candidate;
                            const int boundary_pos = (candidate + 1) * compress_ratio - 1;
                            candidate_valid[tile_i] =
                                (tile_i < candidate_count && boundary_pos >= 0 && boundary_pos < topk_kv_len) ? 1 : 0;
                        }
                        dsv4MegaIndexerScoreCandidateTile<scalar_t, false>(indexer_q,
                                                                           indexer_k,
                                                                           csa_indexer_k_cache,
                                                                           csa_indexer_weights,
                                                                           csa_indexer_cu_lens,
                                                                           csa_indexer_k_pool,
                                                                           csa_indexer_block_table,
                                                                           topk_row,
                                                                           topk_req,
                                                                           c,
                                                                           candidate_pos,
                                                                           candidate_valid,
                                                                           candidate_count,
                                                                           csa_indexer_pool_block_size,
                                                                           csa_indexer_pool_block_stride,
                                                                           csa_indexer_block_table_stride,
                                                                           reduce_buf,
                                                                           indexer_metadata_storage,
                                                                           reduce_result);
                    }
                }
                const bool candidate_tile_ends_initial_fill = c < k_eff && c + candidate_count == k_eff;
                const bool candidate_tile_crosses_initial_fill = c < k_eff && c + candidate_count > k_eff;
                if (use_candidate_tile && c >= k_eff) {
                    if (refresh_worst_slot) {
                        dsv4MegaFindTwoWorstTopKSlots(scratch_indices,
                                                      scratch_scores,
                                                      k_eff,
                                                      reduce_indices,
                                                      reduce_indices_second,
                                                      reduce_buf,
                                                      cached_worst_slot,
                                                      cached_second_worst_slot);
                        cached_worst_score =
                            cached_worst_slot >= 0 ? scratch_scores[cached_worst_slot] : 3.4028234663852886e38F;
                        cached_worst_idx = cached_worst_slot >= 0 ? scratch_indices[cached_worst_slot] : -2147483647;
                        cached_second_worst_score =
                            cached_second_worst_slot >= 0 ? scratch_scores[cached_second_worst_slot] :
                                                            3.4028234663852886e38F;
                        cached_second_worst_idx =
                            cached_second_worst_slot >= 0 ? scratch_indices[cached_second_worst_slot] : -2147483647;
                        refresh_worst_slot = false;
                    }
                    int tile_valid_count = 0;
                    if (tid == 0) {
                        float tile_best_score = kInvalidCompressorScore;
                        int tile_best_idx = 2147483647;
                        if (candidate_tile_all_valid) {
                            tile_valid_count = kMegaIndexerCandidateTile;
#pragma unroll
                            for (int tile_i = 0; tile_i < kMegaIndexerCandidateTile; ++tile_i) {
                                reduce_indices_second[tile_i] = tile_i;
                                const int candidate = c + tile_i;
                                const float score = reduce_result[tile_i];
                                if (dsv4MegaTopKBetter(score, candidate, tile_best_score, tile_best_idx)) {
                                    tile_best_score = score;
                                    tile_best_idx = candidate;
                                }
                            }
                        } else {
#pragma unroll
                            for (int tile_i = 0; tile_i < kMegaIndexerCandidateTile; ++tile_i) {
                                if (tile_i < candidate_count) {
                                    const int candidate = c + tile_i;
                                    const int boundary_pos = (candidate + 1) * compress_ratio - 1;
                                    const bool valid_boundary = boundary_pos >= 0 && boundary_pos < topk_kv_len;
                                    if (valid_boundary) {
                                        reduce_indices_second[tile_valid_count] = tile_i;
                                        ++tile_valid_count;
                                        const float score = reduce_result[tile_i];
                                        if (dsv4MegaTopKBetter(score, candidate, tile_best_score, tile_best_idx)) {
                                            tile_best_score = score;
                                            tile_best_idx = candidate;
                                        }
                                    }
                                }
                            }
                        }
                        const bool tile_can_replace =
                            cached_worst_slot >= 0
                            && dsv4MegaTopKBetter(tile_best_score,
                                                  tile_best_idx,
                                                  cached_worst_score,
                                                  cached_worst_idx);
                        reduce_indices[0] = tile_can_replace ? tile_valid_count : 0;
                    }
                    __syncthreads();
                    tile_valid_count = reduce_indices[0];
                    if (tile_valid_count <= 0) {
                        c += candidate_count;
                        continue;
                    }
                    dsv4MegaFindWorstTopKSlotsTile(scratch_indices,
                                                   scratch_scores,
                                                   k_eff,
                                                   tile_valid_count,
                                                   reduce_indices_second + kMegaIndexerCandidateTile,
                                                   reduce_indices,
                                                   reduce_buf);
                    if (tid == 0) {
                        int sorted_tile_count = tile_valid_count;
                        if (sorted_tile_count == 1) {
                            const int worst_slot = reduce_indices[0];
                            const int tile_i = reduce_indices_second[0];
                            const int best_idx = c + tile_i;
                            const float best_score = reduce_result[tile_i];
                            if (worst_slot >= 0
                                && dsv4MegaTopKBetter(best_score,
                                                      best_idx,
                                                      scratch_scores[worst_slot],
                                                      scratch_indices[worst_slot])) {
                                scratch_indices[worst_slot] = best_idx;
                                scratch_scores[worst_slot] = best_score;
                            }
                        } else {
#pragma unroll
                            for (int sort_i = 1; sort_i < kMegaIndexerCandidateTile; ++sort_i) {
                                if (sort_i < sorted_tile_count) {
                                    const int tile_i = reduce_indices_second[sort_i];
                                    const int candidate = c + tile_i;
                                    const float score = reduce_result[tile_i];
                                    int insert = sort_i;
#pragma unroll
                                    for (int scan = sort_i - 1; scan >= 0; --scan) {
                                        if (scan < sorted_tile_count) {
                                            const int scan_tile_i = reduce_indices_second[scan];
                                            const int scan_candidate = c + scan_tile_i;
                                            const float scan_score = reduce_result[scan_tile_i];
                                            if (dsv4MegaTopKBetter(score, candidate, scan_score, scan_candidate)) {
                                                reduce_indices_second[scan + 1] = scan_tile_i;
                                                insert = scan;
                                            } else {
                                                break;
                                            }
                                        }
                                    }
                                    reduce_indices_second[insert] = tile_i;
                                }
                            }
#pragma unroll
                            for (int tile_rank = 0; tile_rank < kMegaIndexerCandidateTile; ++tile_rank) {
                                if (tile_rank >= sorted_tile_count) {
                                    break;
                                }
                                const int worst_slot = reduce_indices[tile_rank];
                                if (worst_slot < 0) {
                                    break;
                                }
                                const int tile_i = reduce_indices_second[tile_rank];
                                const int best_idx = c + tile_i;
                                const float best_score = reduce_result[tile_i];
                                if (!dsv4MegaTopKBetter(best_score,
                                                        best_idx,
                                                        scratch_scores[worst_slot],
                                                        scratch_indices[worst_slot])) {
                                    break;
                                }
                                scratch_indices[worst_slot] = best_idx;
                                scratch_scores[worst_slot] = best_score;
                            }
                        }
                        reduce_indices[0] = 1;
                    }
                    __syncthreads();
                    refresh_worst_slot = reduce_indices[0] != 0;
                    c += candidate_count;
                    continue;
                }
                for (int tile_i = 0; tile_i < candidate_count; ++tile_i) {
                    const int candidate = c + tile_i;
                    if (candidate == k_eff && candidate_tile_crosses_initial_fill) {
                        __syncthreads();
                    }
                    if (candidate >= k_eff && refresh_worst_slot) {
                        dsv4MegaFindTwoWorstTopKSlots(scratch_indices,
                                                      scratch_scores,
                                                      k_eff,
                                                      reduce_indices,
                                                      reduce_indices_second,
                                                      reduce_buf,
                                                      cached_worst_slot,
                                                      cached_second_worst_slot);
                        cached_worst_score =
                            cached_worst_slot >= 0 ? scratch_scores[cached_worst_slot] : 3.4028234663852886e38F;
                        cached_worst_idx = cached_worst_slot >= 0 ? scratch_indices[cached_worst_slot] : -2147483647;
                        cached_second_worst_score =
                            cached_second_worst_slot >= 0 ? scratch_scores[cached_second_worst_slot] :
                                                            3.4028234663852886e38F;
                        cached_second_worst_idx =
                            cached_second_worst_slot >= 0 ? scratch_indices[cached_second_worst_slot] : -2147483647;
                        refresh_worst_slot = false;
                    }
                    const int boundary_pos = (candidate + 1) * compress_ratio - 1;
                    const bool valid_boundary = boundary_pos >= 0 && boundary_pos < topk_kv_len;
                    const float score =
                        use_candidate_tile ?
                            reduce_result[tile_i] :
                            (valid_boundary ?
                                 dsv4MegaIndexerScoreParallel(indexer_q,
                                                              indexer_k,
                                                              csa_indexer_k_cache,
                                                              csa_indexer_weights,
                                                              csa_indexer_cu_lens,
                                                              csa_indexer_k_pool,
                                                              csa_indexer_block_table,
                                                              topk_row,
                                                              topk_req,
                                                              candidate,
                                                              IH,
                                                              ID,
                                                              LI,
                                                              csa_indexer_pool_block_size,
                                                              csa_indexer_pool_block_stride,
                                                              csa_indexer_block_table_stride,
                                                              symm_buffer_ptrs,
                                                              use_symm_direct_indexer,
                                                              symm_per_rank_buffer_bytes,
                                                              symm_indexer_payload_offset,
                                                              symm_indexer_l_local,
                                                              reduce_buf,
                                                              reduce_result) :
                                 kInvalidCompressorScore);
                    if (tid == 0 && valid_boundary) {
                        if (candidate < k_eff) {
                            scratch_indices[candidate] = candidate;
                            scratch_scores[candidate] = score;
                        } else {
                            const bool replace_worst =
                                score > scratch_scores[cached_worst_slot]
                                || (score == scratch_scores[cached_worst_slot]
                                    && candidate < scratch_indices[cached_worst_slot]);
                            if (replace_worst) {
                                scratch_indices[cached_worst_slot] = candidate;
                                scratch_scores[cached_worst_slot] = score;
                                if (cached_second_worst_slot < 0
                                    || dsv4MegaTopKWorse(score,
                                                         candidate,
                                                         cached_second_worst_score,
                                                         cached_second_worst_idx)) {
                                    cached_worst_score = score;
                                    cached_worst_idx = candidate;
                                } else {
                                    cached_worst_slot = cached_second_worst_slot;
                                    cached_worst_score = cached_second_worst_score;
                                    cached_worst_idx = cached_second_worst_idx;
                                    cached_second_worst_slot = -1;
                                    cached_second_worst_score = 3.4028234663852886e38F;
                                    cached_second_worst_idx = -2147483647;
                                    refresh_worst_slot = true;
                                }
                            }
                        }
                    }
                    const bool replacement_candidate = candidate >= k_eff;
                    if (tid == 0 && replacement_candidate) {
                        reduce_indices[0] = refresh_worst_slot ? 1 : 0;
                        reduce_indices[1] = cached_worst_slot;
                        reduce_indices[2] = cached_second_worst_slot;
                        reduce_buf[0] = cached_worst_score;
                        reduce_buf[1] = cached_second_worst_score;
                    }
                    if (replacement_candidate) {
                        __syncthreads();
                    }
                    if (replacement_candidate) {
                        refresh_worst_slot = reduce_indices[0] != 0;
                        cached_worst_slot = reduce_indices[1];
                        cached_second_worst_slot = reduce_indices[2];
                        cached_worst_score = reduce_buf[0];
                        cached_second_worst_score = reduce_buf[1];
                        cached_worst_idx =
                            cached_worst_slot >= 0 ? scratch_indices[cached_worst_slot] : -2147483647;
                        cached_second_worst_idx =
                            cached_second_worst_slot >= 0 ? scratch_indices[cached_second_worst_slot] : -2147483647;
                    }
                }
                if (candidate_tile_ends_initial_fill) {
                    __syncthreads();
                }
                c += candidate_count;
            }
        }
    }

    if (row_compressed_indices_base != nullptr) {
        int* dst = row_compressed_indices_base + static_cast<int64_t>(local_row) * compressed_stride;
        for (int i = tid; i < compressed_count; i += static_cast<int>(blockDim.x)) {
            dst[i] = scratch_indices[i];
        }
    }
    __threadfence();
    __syncthreads();
    if (tid == 0 && compressed_meta != nullptr) {
        cuda::atomic_ref<unsigned long long, cuda::thread_scope_device> meta_ref(compressed_meta[local_row]);
        meta_ref.store(compressed_epoch_prefix | kMegaCompressedDoneFlag
                           | (static_cast<unsigned int>(compressed_count) & kMegaCompressedCountMask),
                       cuda::std::memory_order_release);
    }
    __syncthreads();
    return compressed_count;
}

__global__ void writeProtocolRecordKernel(char* base, int64_t offset, int rank, int world_size) {
    if (threadIdx.x != 0) {
        return;
    }
    int64_t* record = reinterpret_cast<int64_t*>(base + offset);
    record[0]        = 0x445356344154544ELL;  // "DSV4ATTN" little-endian marker.
    record[1]        = static_cast<int64_t>(rank);
    record[2]        = static_cast<int64_t>(world_size);
    record[3]        = static_cast<int64_t>(blockIdx.x);
    record[4]        = 0;
    record[5]        = 0;
    record[6]        = 0;
    record[7]        = 0;
}

__global__ void verifyProtocolRecordsKernel(const char* __restrict__ base,
                                            const int64_t* __restrict__ offsets,
                                            int world_size,
                                            int* __restrict__ error) {
    const int rank = threadIdx.x;
    if (rank >= world_size) {
        return;
    }
    const int64_t* record = reinterpret_cast<const int64_t*>(base + offsets[rank]);
    if (record[0] != 0x445356344154544ELL || record[1] != rank || record[2] != world_size) {
        atomicCAS(error, 0, rank + 1);
    }
}

void runUserBufferProtocolExchange(int64_t                    cp_rank,
                                   int64_t                    cp_size,
                                   int64_t                    comm_ptr,
                                   int64_t                    buffer_handle,
                                   int64_t                    per_rank_buffer_bytes,
                                   const std::vector<int64_t>& rank_offsets,
                                   cudaStream_t                stream) {
    if (comm_ptr == 0) {
        return;
    }
    auto* comm = reinterpret_cast<rtp_llm::user_buffers::UbCommunicator*>(comm_ptr);
    TORCH_CHECK(comm != nullptr, "comm_ptr resolved to null communicator");
    TORCH_CHECK(comm->local_rank == static_cast<int32_t>(cp_rank),
                "communicator local_rank mismatch: comm=",
                comm->local_rank,
                " cp_rank=",
                cp_rank);
    TORCH_CHECK(comm->world_size == static_cast<int32_t>(cp_size),
                "communicator world_size mismatch: comm=",
                comm->world_size,
                " cp_size=",
                cp_size);
    TORCH_CHECK(buffer_handle >= 0 && buffer_handle < kMaxUserBufferRegions,
                "buffer_handle out of UserBuffers region range");
    TORCH_CHECK(per_rank_buffer_bytes >= kProtocolBytes,
                "per-rank symmetric buffer is too small for attention protocol");

    char* local_buffer = reinterpret_cast<char*>(comm->mem_ptr[buffer_handle]);
    TORCH_CHECK(local_buffer != nullptr, "registered attention buffer local pointer is null");
    const int64_t local_offset = rank_offsets[static_cast<size_t>(cp_rank)];
    writeProtocolRecordKernel<<<1, 1, 0, stream>>>(
        local_buffer, local_offset, static_cast<int>(cp_rank), static_cast<int>(cp_size));

    for (int64_t peer = 0; peer < cp_size; ++peer) {
        if (peer == cp_rank) {
            continue;
        }
        rtp_llm::user_buffers::userbuffers_send(static_cast<int>(buffer_handle),
                                                static_cast<size_t>(local_offset),
                                                static_cast<size_t>(local_offset),
                                                static_cast<size_t>(kProtocolBytes),
                                                comm,
                                                static_cast<int>(peer),
                                                stream);
    }
    for (int64_t peer = 0; peer < cp_size; ++peer) {
        if (peer == cp_rank) {
            continue;
        }
        rtp_llm::user_buffers::userbuffers_recv(
            static_cast<int>(buffer_handle), comm, static_cast<int>(peer), stream);
    }

    auto options = torch::TensorOptions().device(torch::kCUDA).dtype(torch::kInt32);
    auto error   = torch::empty({1}, options);
    AT_CUDA_CHECK(cudaMemsetAsync(error.data_ptr<int>(), 0, sizeof(int), stream));
    auto offset_options = torch::TensorOptions().device(torch::kCUDA).dtype(torch::kInt64);
    auto offsets        = torch::empty({cp_size}, offset_options);
    AT_CUDA_CHECK(cudaMemcpyAsync(offsets.data_ptr<int64_t>(),
                                  rank_offsets.data(),
                                  sizeof(int64_t) * static_cast<size_t>(cp_size),
                                  cudaMemcpyHostToDevice,
                                  stream));
    verifyProtocolRecordsKernel<<<1, static_cast<unsigned int>(cp_size), 0, stream>>>(
        local_buffer, offsets.data_ptr<int64_t>(), static_cast<int>(cp_size), error.data_ptr<int>());
    int host_error = 0;
    AT_CUDA_CHECK(cudaMemcpyAsync(&host_error, error.data_ptr<int>(), sizeof(int), cudaMemcpyDeviceToHost, stream));
    AT_CUDA_CHECK(cudaStreamSynchronize(stream));
    TORCH_CHECK(host_error == 0, "attention UserBuffers protocol verification failed at rank slot ", host_error - 1);
}

torch::Tensor allGather4DPayloadWithUserBuffers(const torch::Tensor&       tensor,
                                                int64_t                    cp_rank,
                                                int64_t                    cp_size,
                                                int64_t                    comm_ptr,
                                                int64_t                    buffer_handle,
                                                int64_t                    per_rank_buffer_bytes,
                                                const std::vector<int64_t>& rank_offsets,
                                                cudaStream_t                stream) {
    if (comm_ptr == 0) {
        return tensor;
    }
    auto* comm = reinterpret_cast<rtp_llm::user_buffers::UbCommunicator*>(comm_ptr);
    TORCH_CHECK(comm != nullptr, "comm_ptr resolved to null communicator");
    TORCH_CHECK(buffer_handle >= 0 && buffer_handle < kMaxUserBufferRegions,
                "buffer_handle out of UserBuffers region range");
    char* local_buffer = reinterpret_cast<char*>(comm->mem_ptr[buffer_handle]);
    TORCH_CHECK(local_buffer != nullptr, "registered attention buffer local pointer is null");

    const int64_t B       = tensor.size(0);
    const int64_t L_local = tensor.size(1);
    const int64_t H       = tensor.size(2);
    const int64_t D       = tensor.size(3);
    const int64_t elem    = tensor.element_size();
    const int64_t row_bytes = L_local * H * D * elem;
    const int64_t payload_bytes = B * row_bytes;
    TORCH_CHECK(kProtocolBytes + payload_bytes <= per_rank_buffer_bytes,
                "attention 4D payload exceeds per-rank symmetric buffer: need ",
                kProtocolBytes + payload_bytes,
                " bytes, have ",
                per_rank_buffer_bytes);

    const int64_t local_payload_offset = rank_offsets[static_cast<size_t>(cp_rank)] + kProtocolBytes;
    AT_CUDA_CHECK(cudaMemcpyAsync(local_buffer + local_payload_offset,
                                  tensor.data_ptr(),
                                  static_cast<size_t>(payload_bytes),
                                  cudaMemcpyDeviceToDevice,
                                  stream));
    for (int64_t peer = 0; peer < cp_size; ++peer) {
        if (peer == cp_rank) {
            continue;
        }
        rtp_llm::user_buffers::userbuffers_send(static_cast<int>(buffer_handle),
                                                static_cast<size_t>(local_payload_offset),
                                                static_cast<size_t>(local_payload_offset),
                                                static_cast<size_t>(payload_bytes),
                                                comm,
                                                static_cast<int>(peer),
                                                stream);
    }
    for (int64_t peer = 0; peer < cp_size; ++peer) {
        if (peer == cp_rank) {
            continue;
        }
        rtp_llm::user_buffers::userbuffers_recv(
            static_cast<int>(buffer_handle), comm, static_cast<int>(peer), stream);
    }

    auto gathered = torch::empty({B, L_local * cp_size, H, D}, tensor.options());
    char* dst_base = reinterpret_cast<char*>(gathered.data_ptr());
    for (int64_t peer = 0; peer < cp_size; ++peer) {
        const char* src_base = local_buffer + rank_offsets[static_cast<size_t>(peer)] + kProtocolBytes;
        for (int64_t b = 0; b < B; ++b) {
            char* dst = dst_base + ((b * L_local * cp_size + peer * L_local) * H * D * elem);
            const char* src = src_base + b * row_bytes;
            AT_CUDA_CHECK(cudaMemcpyAsync(dst, src, static_cast<size_t>(row_bytes), cudaMemcpyDeviceToDevice, stream));
        }
    }
    return gathered;
}

torch::Tensor allGather2DPayloadWithUserBuffers(const torch::Tensor&       tensor,
                                                int64_t                    cp_rank,
                                                int64_t                    cp_size,
                                                int64_t                    comm_ptr,
                                                int64_t                    buffer_handle,
                                                int64_t                    per_rank_buffer_bytes,
                                                const std::vector<int64_t>& rank_offsets,
                                                cudaStream_t                stream) {
    if (comm_ptr == 0) {
        return tensor;
    }
    auto* comm = reinterpret_cast<rtp_llm::user_buffers::UbCommunicator*>(comm_ptr);
    TORCH_CHECK(comm != nullptr, "comm_ptr resolved to null communicator");
    TORCH_CHECK(buffer_handle >= 0 && buffer_handle < kMaxUserBufferRegions,
                "buffer_handle out of UserBuffers region range");
    char* local_buffer = reinterpret_cast<char*>(comm->mem_ptr[buffer_handle]);
    TORCH_CHECK(local_buffer != nullptr, "registered attention buffer local pointer is null");

    const int64_t L_local = tensor.size(0);
    const int64_t D       = tensor.size(1);
    const int64_t elem    = tensor.element_size();
    const int64_t payload_bytes = L_local * D * elem;
    TORCH_CHECK(kProtocolBytes + payload_bytes <= per_rank_buffer_bytes,
                "attention 2D payload exceeds per-rank symmetric buffer: need ",
                kProtocolBytes + payload_bytes,
                " bytes, have ",
                per_rank_buffer_bytes);

    const int64_t local_payload_offset = rank_offsets[static_cast<size_t>(cp_rank)] + kProtocolBytes;
    AT_CUDA_CHECK(cudaMemcpyAsync(local_buffer + local_payload_offset,
                                  tensor.data_ptr(),
                                  static_cast<size_t>(payload_bytes),
                                  cudaMemcpyDeviceToDevice,
                                  stream));
    for (int64_t peer = 0; peer < cp_size; ++peer) {
        if (peer == cp_rank) {
            continue;
        }
        rtp_llm::user_buffers::userbuffers_send(static_cast<int>(buffer_handle),
                                                static_cast<size_t>(local_payload_offset),
                                                static_cast<size_t>(local_payload_offset),
                                                static_cast<size_t>(payload_bytes),
                                                comm,
                                                static_cast<int>(peer),
                                                stream);
    }
    for (int64_t peer = 0; peer < cp_size; ++peer) {
        if (peer == cp_rank) {
            continue;
        }
        rtp_llm::user_buffers::userbuffers_recv(
            static_cast<int>(buffer_handle), comm, static_cast<int>(peer), stream);
    }

    auto gathered = torch::empty({L_local * cp_size, D}, tensor.options());
    char* dst_base = reinterpret_cast<char*>(gathered.data_ptr());
    const int64_t row_bytes = L_local * D * elem;
    for (int64_t peer = 0; peer < cp_size; ++peer) {
        const char* src = local_buffer + rank_offsets[static_cast<size_t>(peer)] + kProtocolBytes;
        char*       dst = dst_base + peer * row_bytes;
        AT_CUDA_CHECK(cudaMemcpyAsync(dst, src, static_cast<size_t>(row_bytes), cudaMemcpyDeviceToDevice, stream));
    }
    return gathered;
}

bool hasSymmMemBackend(const std::optional<torch::Tensor>& symm_buffer_opt, int64_t symm_buffer_ptrs_dev) {
    return symm_buffer_opt.has_value() && symm_buffer_ptrs_dev != 0;
}

__global__ void symmMemDeviceBarrierKernel(uint8_t* const* __restrict__ buffer_ptrs,
                                           int cp_rank,
                                           int cp_size,
                                           int64_t per_rank_buffer_bytes,
                                           int barrier_channel) {
    if (threadIdx.x != 0 || blockIdx.x != 0) {
        return;
    }
    uint8_t* local_base = buffer_ptrs[cp_rank] + static_cast<int64_t>(cp_rank) * per_rank_buffer_bytes;
    auto* counter = reinterpret_cast<unsigned int*>(local_base);
    auto* signal0 = reinterpret_cast<int*>(local_base + 4);
    auto* signal1 = reinterpret_cast<int*>(local_base + 8);
    const unsigned int old = atomicAdd_system(counter, 1u);
    const int phase = static_cast<int>(old & 1u);
    const int sign = static_cast<int>((old >> 1) & 1u);
    volatile int* local_signal = phase == 0 ? signal0 : signal1;
    const int delta = sign == 0 ? 1 : -1;
    const int target = sign == 0 ? cp_size : 0;
    for (int peer = 0; peer < cp_size; ++peer) {
        uint8_t* peer_base = buffer_ptrs[peer] + static_cast<int64_t>(peer) * per_rank_buffer_bytes;
        auto* peer_signal = reinterpret_cast<int*>(peer_base + (phase == 0 ? 4 : 8));
        atomicAdd_system(peer_signal, delta);
    }
    __threadfence_system();
    const unsigned long long start = clock64();
    while (*local_signal != target) {
        if (clock64() - start > static_cast<unsigned long long>(kSymmMemBarrierTimeoutCycles)) {
            printf("DSV4 CP attention buffer barrier timeout: rank=%d channel=%d counter=%u signal=%d target=%d phase=%d sign=%d\n",
                   cp_rank,
                   barrier_channel,
                   *counter,
                   *local_signal,
                   target,
                   phase,
                   sign);
            asm("trap;");
        }
    }
    __threadfence_system();
}

template <cuda::std::memory_order Sem>
__device__ __forceinline__ uint32_t signal_pad_cas(uint32_t* addr, uint32_t compare, uint32_t val) {
    cuda::atomic_ref<uint32_t, cuda::thread_scope_system> ref(*addr);
    ref.compare_exchange_strong(compare, val, Sem, cuda::std::memory_order_relaxed);
    return compare;
}

template <cuda::std::memory_order Sem>
__device__ __forceinline__ bool signal_pad_try_put(uint32_t* addr, unsigned long long timeout_cycles) {
    const unsigned long long start = clock64();
    while (signal_pad_cas<Sem>(addr, 0u, 1u) != 0u) {
        if (timeout_cycles != 0 && clock64() - start > timeout_cycles) {
            return false;
        }
    }
    return true;
}

template <cuda::std::memory_order Sem>
__device__ __forceinline__ bool signal_pad_try_wait(uint32_t* addr, unsigned long long timeout_cycles) {
    const unsigned long long start = clock64();
    while (signal_pad_cas<Sem>(addr, 1u, 0u) != 1u) {
        if (timeout_cycles != 0 && clock64() - start > timeout_cycles) {
            return false;
        }
    }
    return true;
}

__global__ void symmMemSignalPadBarrierKernel(uint32_t* const* __restrict__ signal_pads,
                                              int cp_rank,
                                              int cp_size,
                                              int barrier_channel,
                                              unsigned long long timeout_cycles) {
    if (blockIdx.x != 0 || threadIdx.x >= static_cast<unsigned int>(cp_size)) {
        return;
    }
    const int peer = static_cast<int>(threadIdx.x);
    uint32_t* peer_signal = signal_pads[peer] + static_cast<int64_t>(barrier_channel) * cp_size + cp_rank;
    uint32_t* local_signal = signal_pads[cp_rank] + static_cast<int64_t>(barrier_channel) * cp_size + peer;
    if (!signal_pad_try_put<cuda::std::memory_order_release>(peer_signal, timeout_cycles)) {
        printf("DSV4 CP attention signal-pad put timeout: rank=%d peer=%d channel=%d\n",
               cp_rank,
               peer,
               barrier_channel);
        asm("trap;");
    }
    if (!signal_pad_try_wait<cuda::std::memory_order_acquire>(local_signal, timeout_cycles)) {
        printf("DSV4 CP attention signal-pad wait timeout: rank=%d peer=%d channel=%d value=%u\n",
               cp_rank,
               peer,
               barrier_channel,
               *local_signal);
        asm("trap;");
    }
}

void runSymmMemDeviceBarrier(int64_t symm_buffer_ptrs_dev,
                             int64_t symm_signal_pad_ptrs_dev,
                             int64_t cp_rank,
                             int64_t cp_size,
                             int64_t per_rank_buffer_bytes,
                             int64_t barrier_channel,
                             cudaStream_t stream) {
    TORCH_CHECK(symm_buffer_ptrs_dev != 0, "symm_buffer_ptrs_dev must be non-zero for device barrier");
    if (symm_signal_pad_ptrs_dev != 0) {
        const int64_t signal_pad_channel = static_cast<int64_t>(kAttentionSignalPadChannelBase) + barrier_channel;
        symmMemSignalPadBarrierKernel<<<1, 32, 0, stream>>>(
            reinterpret_cast<uint32_t* const*>(symm_signal_pad_ptrs_dev),
            static_cast<int>(cp_rank),
            static_cast<int>(cp_size),
            static_cast<int>(signal_pad_channel),
            static_cast<unsigned long long>(kSymmMemBarrierTimeoutCycles));
    } else {
        symmMemDeviceBarrierKernel<<<1, 32, 0, stream>>>(reinterpret_cast<uint8_t* const*>(symm_buffer_ptrs_dev),
                                                        static_cast<int>(cp_rank),
                                                        static_cast<int>(cp_size),
                                                        per_rank_buffer_bytes,
                                                        static_cast<int>(barrier_channel));
    }
    const cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "dsv4 symmetric barrier launch failed: ", cudaGetErrorString(err));
}

void validateSymmBuffer(const torch::Tensor& symm_buffer,
                        int64_t symm_buffer_ptrs_dev,
                        int64_t cp_size,
                        int64_t per_rank_buffer_bytes,
                        const std::vector<int64_t>& rank_offsets) {
    TORCH_CHECK(symm_buffer.is_cuda(), "symm_buffer must be a CUDA tensor");
    TORCH_CHECK(symm_buffer.scalar_type() == torch::kInt8, "symm_buffer must be int8");
    TORCH_CHECK(symm_buffer.is_contiguous(), "symm_buffer must be contiguous");
    TORCH_CHECK(symm_buffer_ptrs_dev != 0, "symm_buffer_ptrs_dev must be non-zero");
    TORCH_CHECK(cp_size == 8, "production symmetric-memory attention V0 requires cp_size=8");
    TORCH_CHECK(per_rank_buffer_bytes > 0, "per_rank_buffer_bytes must be positive when symm_buffer is set");
    TORCH_CHECK(symm_buffer.numel() >= per_rank_buffer_bytes * cp_size,
                "symm_buffer is smaller than cp_size * per_rank_buffer_bytes");
    TORCH_CHECK(static_cast<int64_t>(rank_offsets.size()) == cp_size,
                "rank_offsets length must equal cp_size when symm_buffer is set");
    for (int64_t rank = 0; rank < cp_size; ++rank) {
        TORCH_CHECK(rank_offsets[rank] == rank * per_rank_buffer_bytes,
                    "rank_offsets must be rank * per_rank_buffer_bytes for V0, got rank_offsets[",
                    rank,
                    "]=",
                    rank_offsets[rank]);
    }
}

__global__ void copySymmGather2DKernel(const uint8_t* const* __restrict__ buffer_ptrs,
                                       uint8_t* __restrict__ out,
                                       int64_t per_rank_buffer_bytes,
                                       int64_t row_bytes,
                                       int64_t total_bytes) {
    const int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= total_bytes) {
        return;
    }
    const int64_t peer = idx / row_bytes;
    const int64_t off  = idx - peer * row_bytes;
    const int peer_i = static_cast<int>(peer);
    const int64_t src_offset = peer * per_rank_buffer_bytes + static_cast<int64_t>(64) + off;
    const uint8_t* src = buffer_ptrs[peer_i];
    out[idx] = src[src_offset];
}

__global__ void copySymmGather4DKernel(const uint8_t* const* __restrict__ buffer_ptrs,
                                       uint8_t* __restrict__ out,
                                       int64_t per_rank_buffer_bytes,
                                       int64_t row_bytes,
                                       int64_t cp_size,
                                       int64_t total_bytes) {
    const int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= total_bytes) {
        return;
    }
    const int64_t bytes_per_batch = cp_size * row_bytes;
    const int64_t b               = idx / bytes_per_batch;
    const int64_t rem             = idx - b * bytes_per_batch;
    const int64_t peer            = rem / row_bytes;
    const int64_t off             = rem - peer * row_bytes;
    const int peer_i = static_cast<int>(peer);
    const int64_t src_offset = peer * per_rank_buffer_bytes + static_cast<int64_t>(64) + b * row_bytes + off;
    const uint8_t* src = buffer_ptrs[peer_i];
    out[idx] = src[src_offset];
}

torch::Tensor allGather2DPayloadWithSymmMem(const torch::Tensor& tensor,
                                            int64_t cp_rank,
                                            int64_t cp_size,
                                            const torch::Tensor& symm_buffer,
                                            int64_t symm_buffer_ptrs_dev,
                                            int64_t symm_signal_pad_ptrs_dev,
                                            int64_t per_rank_buffer_bytes,
                                            const std::vector<int64_t>& rank_offsets,
                                            const py::object& symm_handle,
                                            int64_t barrier_channel,
                                            cudaStream_t stream) {
    (void)symm_handle;
    (void)barrier_channel;
    validateSymmBuffer(symm_buffer, symm_buffer_ptrs_dev, cp_size, per_rank_buffer_bytes, rank_offsets);
    const int64_t L_local = tensor.size(0);
    const int64_t D       = tensor.size(1);
    const int64_t elem    = tensor.element_size();
    const int64_t payload_bytes = L_local * D * elem;
    TORCH_CHECK(kProtocolBytes + payload_bytes <= per_rank_buffer_bytes,
                "attention 2D payload exceeds per-rank symmetric buffer: need ",
                kProtocolBytes + payload_bytes,
                " bytes, have ",
                per_rank_buffer_bytes);

    const int64_t local_payload_offset = rank_offsets[static_cast<size_t>(cp_rank)] + kProtocolBytes;
    auto* local_buffer = reinterpret_cast<uint8_t*>(symm_buffer.data_ptr<int8_t>());
    AT_CUDA_CHECK(cudaMemcpyAsync(local_buffer + local_payload_offset,
                                  tensor.data_ptr(),
                                  static_cast<size_t>(payload_bytes),
                                  cudaMemcpyDeviceToDevice,
                                  stream));
    runSymmMemDeviceBarrier(
        symm_buffer_ptrs_dev, symm_signal_pad_ptrs_dev, cp_rank, cp_size, per_rank_buffer_bytes, barrier_channel * 2, stream);

    auto gathered = torch::empty({L_local * cp_size, D}, tensor.options());
    if (payload_bytes == 0) {
        runSymmMemDeviceBarrier(symm_buffer_ptrs_dev,
                                symm_signal_pad_ptrs_dev,
                                cp_rank,
                                cp_size,
                                per_rank_buffer_bytes,
                                barrier_channel * 2 + 1,
                                stream);
        return gathered;
    }
    const int64_t total_bytes = payload_bytes * cp_size;
    const int threads = 256;
    const int blocks = static_cast<int>((total_bytes + threads - 1) / threads);
    copySymmGather2DKernel<<<blocks, threads, 0, stream>>>(
        reinterpret_cast<const uint8_t* const*>(symm_buffer_ptrs_dev),
        reinterpret_cast<uint8_t*>(gathered.data_ptr()),
        per_rank_buffer_bytes,
        payload_bytes,
        total_bytes);
    runSymmMemDeviceBarrier(symm_buffer_ptrs_dev,
                            symm_signal_pad_ptrs_dev,
                            cp_rank,
                            cp_size,
                            per_rank_buffer_bytes,
                            barrier_channel * 2 + 1,
                            stream);
    return gathered;
}

torch::Tensor allGather4DPayloadWithSymmMem(const torch::Tensor& tensor,
                                            int64_t cp_rank,
                                            int64_t cp_size,
                                            const torch::Tensor& symm_buffer,
                                            int64_t symm_buffer_ptrs_dev,
                                            int64_t symm_signal_pad_ptrs_dev,
                                            int64_t per_rank_buffer_bytes,
                                            const std::vector<int64_t>& rank_offsets,
                                            const py::object& symm_handle,
                                            int64_t barrier_channel,
                                            cudaStream_t stream) {
    (void)symm_handle;
    (void)barrier_channel;
    validateSymmBuffer(symm_buffer, symm_buffer_ptrs_dev, cp_size, per_rank_buffer_bytes, rank_offsets);
    const int64_t B       = tensor.size(0);
    const int64_t L_local = tensor.size(1);
    const int64_t H       = tensor.size(2);
    const int64_t D       = tensor.size(3);
    const int64_t elem    = tensor.element_size();
    const int64_t row_bytes = L_local * H * D * elem;
    const int64_t payload_bytes = B * row_bytes;
    TORCH_CHECK(kProtocolBytes + payload_bytes <= per_rank_buffer_bytes,
                "attention 4D payload exceeds per-rank symmetric buffer: need ",
                kProtocolBytes + payload_bytes,
                " bytes, have ",
                per_rank_buffer_bytes);

    const int64_t local_payload_offset = rank_offsets[static_cast<size_t>(cp_rank)] + kProtocolBytes;
    auto* local_buffer = reinterpret_cast<uint8_t*>(symm_buffer.data_ptr<int8_t>());
    AT_CUDA_CHECK(cudaMemcpyAsync(local_buffer + local_payload_offset,
                                  tensor.data_ptr(),
                                  static_cast<size_t>(payload_bytes),
                                  cudaMemcpyDeviceToDevice,
                                  stream));
    runSymmMemDeviceBarrier(
        symm_buffer_ptrs_dev, symm_signal_pad_ptrs_dev, cp_rank, cp_size, per_rank_buffer_bytes, barrier_channel * 2, stream);

    auto gathered = torch::empty({B, L_local * cp_size, H, D}, tensor.options());
    if (payload_bytes == 0) {
        runSymmMemDeviceBarrier(symm_buffer_ptrs_dev,
                                symm_signal_pad_ptrs_dev,
                                cp_rank,
                                cp_size,
                                per_rank_buffer_bytes,
                                barrier_channel * 2 + 1,
                                stream);
        return gathered;
    }
    const int64_t total_bytes = payload_bytes * cp_size;
    const int threads = 256;
    const int blocks = static_cast<int>((total_bytes + threads - 1) / threads);
    copySymmGather4DKernel<<<blocks, threads, 0, stream>>>(
        reinterpret_cast<const uint8_t* const*>(symm_buffer_ptrs_dev),
        reinterpret_cast<uint8_t*>(gathered.data_ptr()),
        per_rank_buffer_bytes,
        row_bytes,
        cp_size,
        total_bytes);
    runSymmMemDeviceBarrier(symm_buffer_ptrs_dev,
                            symm_signal_pad_ptrs_dev,
                            cp_rank,
                            cp_size,
                            per_rank_buffer_bytes,
                            barrier_channel * 2 + 1,
                            stream);
    return gathered;
}

torch::Tensor allGather2DPayload(const torch::Tensor& tensor,
                                 int64_t cp_rank,
                                 int64_t cp_size,
                                 int64_t comm_ptr,
                                 int64_t buffer_handle,
                                 int64_t per_rank_buffer_bytes,
                                 const std::vector<int64_t>& rank_offsets,
                                 const std::optional<torch::Tensor>& symm_buffer,
                                 int64_t symm_buffer_ptrs_dev,
                                 int64_t symm_signal_pad_ptrs_dev,
                                 const py::object& symm_handle,
                                 int64_t barrier_channel,
                                 cudaStream_t stream) {
    if (hasSymmMemBackend(symm_buffer, symm_buffer_ptrs_dev)) {
        return allGather2DPayloadWithSymmMem(tensor,
                                            cp_rank,
                                            cp_size,
                                            *symm_buffer,
                                            symm_buffer_ptrs_dev,
                                            symm_signal_pad_ptrs_dev,
                                            per_rank_buffer_bytes,
                                            rank_offsets,
                                            symm_handle,
                                            barrier_channel,
                                            stream);
    }
    return allGather2DPayloadWithUserBuffers(
        tensor, cp_rank, cp_size, comm_ptr, buffer_handle, per_rank_buffer_bytes, rank_offsets, stream);
}

torch::Tensor allGather4DPayload(const torch::Tensor& tensor,
                                 int64_t cp_rank,
                                 int64_t cp_size,
                                 int64_t comm_ptr,
                                 int64_t buffer_handle,
                                 int64_t per_rank_buffer_bytes,
                                 const std::vector<int64_t>& rank_offsets,
                                 const std::optional<torch::Tensor>& symm_buffer,
                                 int64_t symm_buffer_ptrs_dev,
                                 int64_t symm_signal_pad_ptrs_dev,
                                 const py::object& symm_handle,
                                 int64_t barrier_channel,
                                 cudaStream_t stream) {
    if (hasSymmMemBackend(symm_buffer, symm_buffer_ptrs_dev)) {
        return allGather4DPayloadWithSymmMem(tensor,
                                            cp_rank,
                                            cp_size,
                                            *symm_buffer,
                                            symm_buffer_ptrs_dev,
                                            symm_signal_pad_ptrs_dev,
                                            per_rank_buffer_bytes,
                                            rank_offsets,
                                            symm_handle,
                                            barrier_channel,
                                            stream);
    }
    return allGather4DPayloadWithUserBuffers(
        tensor, cp_rank, cp_size, comm_ptr, buffer_handle, per_rank_buffer_bytes, rank_offsets, stream);
}

__global__ void dsv4SwaQuantizeAndInsertKernel(const c10::BFloat16* __restrict__ k,
                                               const int64_t* __restrict__ slot_mapping,
                                               uint8_t* __restrict__ k_cache,
                                               int num_tokens,
                                               int cache_block_size,
                                               int64_t block_stride,
                                               int num_cache_blocks) {
    const int token = blockIdx.x;
    const int tile  = blockIdx.y;
    const int lane  = threadIdx.x;
    if (token >= num_tokens || lane >= kSwaQuantBlock) {
        return;
    }

    const int64_t slot = slot_mapping[token];
    if (slot == -1) {
        return;
    }
    if (slot < 0) {
        return;
    }
    const int64_t block_idx = slot / cache_block_size;
    const int64_t pos_in_block = slot % cache_block_size;
    if (block_idx < 0 || block_idx >= num_cache_blocks) {
        return;
    }

    uint8_t* cache_block_ptr = k_cache + block_idx * block_stride;
    uint8_t* token_data_ptr  = cache_block_ptr + pos_in_block * kSwaTokenDataBytes;
    uint8_t* token_fp8_ptr   = token_data_ptr;
    uint8_t* token_bf16_ptr  = token_data_ptr + kSwaNopeDim;
    uint8_t* token_scale_ptr = cache_block_ptr + cache_block_size * kSwaTokenDataBytes + pos_in_block * kSwaScaleBytes;

    if (tile < kSwaNopeTiles) {
        const int offset = tile * kSwaQuantBlock + lane;
        const float x = static_cast<float>(k[static_cast<int64_t>(token) * kSwaHeadDim + offset]);

        __shared__ float abs_values[kSwaQuantBlock];
        __shared__ float tile_scale;
        __shared__ uint8_t encoded_scale;
        abs_values[lane] = fabsf(x);
        __syncthreads();
        for (int stride = kSwaQuantBlock / 2; stride > 0; stride >>= 1) {
            if (lane < stride) {
                abs_values[lane] = fmaxf(abs_values[lane], abs_values[lane + stride]);
            }
            __syncthreads();
        }
        if (lane == 0) {
            const float amax = fmaxf(abs_values[0], 1.0e-4f);
            float exponent = ceilf(log2f(amax / kSwaFp8Max));
            exponent = fminf(fmaxf(exponent, -127.0f), 128.0f);
            tile_scale = exp2f(exponent);
            int encoded = static_cast<int>(exponent + 127.0f);
            encoded     = encoded < 0 ? 0 : (encoded > 255 ? 255 : encoded);
            encoded_scale = static_cast<uint8_t>(encoded);
            token_scale_ptr[tile] = encoded_scale;
        }
        __syncthreads();

        float scaled = x / tile_scale;
        scaled = fminf(fmaxf(scaled, -kSwaFp8Max), kSwaFp8Max);
        __nv_fp8_e4m3 fp8_value = __nv_fp8_e4m3(scaled);
        token_fp8_ptr[offset] = fp8_value.__x;
        return;
    }

    if (tile == kSwaNopeTiles) {
        const uint16_t* k_u16 = reinterpret_cast<const uint16_t*>(k);
        store_u16_unaligned(token_bf16_ptr + lane * static_cast<int>(sizeof(uint16_t)),
                            k_u16[static_cast<int64_t>(token) * kSwaHeadDim + kSwaNopeDim + lane]);
        if (lane == 0) {
            token_scale_ptr[kSwaNopeTiles] = 0;
        }
    }
}

template<typename kv_t, typename score_t>
__global__ void dsv4CompressorSavePartialStatesKernel(const kv_t* __restrict__ kv,
                                                      const score_t* __restrict__ score,
                                                      const float* __restrict__ ape,
                                                      const int64_t* __restrict__ positions,
                                                      float* __restrict__ state_cache,
                                                      const int64_t* __restrict__ state_slots,
                                                      int num_tokens,
                                                      int head_size,
                                                      int state_width,
                                                      int cache_block_size,
                                                      int64_t block_stride,
                                                      int64_t row_stride,
                                                      int num_state_blocks,
                                                      int compress_ratio) {
    const int token = blockIdx.x;
    const int d = static_cast<int>(blockIdx.y) * blockDim.x + threadIdx.x;
    if (token >= num_tokens || d >= head_size) {
        return;
    }
    const int64_t slot = state_slots[token];
    if (slot < 0) {
        return;
    }
    const int64_t block_idx = slot / cache_block_size;
    const int64_t pos_in_block = slot % cache_block_size;
    if (block_idx < 0 || block_idx >= num_state_blocks) {
        return;
    }
    const int64_t position = positions[token];
    int ape_row = 0;
    if (compress_ratio > 0) {
        ape_row = static_cast<int>(position % compress_ratio);
        if (ape_row < 0) {
            ape_row += compress_ratio;
        }
    }

    float* row = state_cache + block_idx * block_stride + pos_in_block * row_stride;
    const int64_t src = static_cast<int64_t>(token) * head_size + d;
    row[d] = scalar_load_float(kv, src);
    row[state_width + d] = scalar_load_float(score, src) + ape[static_cast<int64_t>(ape_row) * head_size + d];
}

template<typename kv_t, typename score_t, typename norm_t>
__global__ void dsv4CompressorCompressNormRopeInsertKernel(const kv_t* __restrict__ kv_raw,
                                                           int64_t kv_raw_stride,
                                                           const score_t* __restrict__ score_raw,
                                                           int64_t score_raw_stride,
                                                           const float* __restrict__ ape,
                                                           int64_t ape_stride,
                                                           const int32_t* __restrict__ token_to_req,
                                                           const int64_t* __restrict__ positions,
                                                           const int64_t* __restrict__ state_slots,
                                                           const float* __restrict__ state_cache,
                                                           int64_t state_cache_stride0,
                                                           int64_t state_cache_stride1,
                                                           const int32_t* __restrict__ state_block_table,
                                                           int64_t state_block_table_stride,
                                                           const norm_t* __restrict__ norm_weight,
                                                           float rms_norm_eps,
                                                           const float* __restrict__ cos_sin_cache,
                                                           int64_t cos_sin_stride,
                                                           uint8_t* __restrict__ kv_cache,
                                                           const int64_t* __restrict__ kv_slots,
                                                           int num_tokens,
                                                           int head_dim,
                                                           int rope_head_dim,
                                                           int raw_width,
                                                           int state_width,
                                                           int compress_ratio,
                                                           int window_count,
                                                           int state_tokens_per_block,
                                                           int state_ring_entries,
                                                           int kv_cache_block_size,
                                                           int64_t kv_cache_block_stride,
                                                           int num_state_blocks,
                                                           int num_kv_blocks,
                                                           int64_t seq_start,
                                                           bool disable_raw_path,
                                                           bool batched_raw,
                                                           const int32_t* __restrict__ seq_start_per_req,
                                                           const int32_t* __restrict__ cu_seq_per_req,
                                                           const int64_t* __restrict__ raw_unpad_restore) {
    const int token = static_cast<int>(blockIdx.x);
    const int d     = static_cast<int>(threadIdx.x);

    __shared__ float compressed_shared[512];
    __shared__ float normed_raw_shared[512];
    __shared__ float quant_shared[512];
    __shared__ float reduce_shared[512];
    __shared__ float group_scale_shared[8];

    if (token >= num_tokens) {
        return;
    }

    const int64_t position = positions[token];
    if (compress_ratio <= 0 || ((position + 1) % compress_ratio) != 0) {
        return;
    }
    const int64_t kv_slot = kv_slots[token];
    if (kv_slot < 0) {
        return;
    }
    const int64_t kv_block_idx   = kv_slot / kv_cache_block_size;
    const int64_t kv_pos_in_blk  = kv_slot % kv_cache_block_size;
    if (kv_block_idx < 0 || kv_block_idx >= num_kv_blocks) {
        return;
    }

    const int req_idx = token_to_req == nullptr ? 0 : static_cast<int>(token_to_req[token]);
    const int start = static_cast<int>(position) - window_count + 1;
    float compressed = 0.0f;
    if (d < head_dim) {
        float max_score = kInvalidCompressorScore;
        int valid_count = 0;
        for (int t = 0; t < window_count; ++t) {
            const int pos = start + t;
            if (pos < 0) {
                continue;
            }
            const bool use_second_half = t >= compress_ratio;
            const int head_offset = use_second_half ? head_dim : 0;
            bool use_raw = false;
            int64_t flat_idx = 0;
            if (!disable_raw_path) {
                if (batched_raw) {
                    const int req_seq_start = seq_start_per_req[req_idx];
                    const int req_cu_lo     = cu_seq_per_req[req_idx];
                    const int req_cu_hi     = cu_seq_per_req[req_idx + 1];
                    const int flat_in_req   = pos - req_seq_start;
                    use_raw = flat_in_req >= 0 && flat_in_req < (req_cu_hi - req_cu_lo);
                    if (use_raw) {
                        const int64_t global_raw_idx = static_cast<int64_t>(req_cu_lo + flat_in_req);
                        flat_idx = raw_unpad_restore == nullptr ? global_raw_idx : raw_unpad_restore[global_raw_idx];
                        use_raw = flat_idx >= 0 && flat_idx < num_tokens;
                    }
                } else {
                    flat_idx = static_cast<int64_t>(pos) - seq_start;
                    use_raw = flat_idx >= 0 && flat_idx < num_tokens;
                }
            }

            float score_val = kInvalidCompressorScore;
            if (use_raw && head_offset + d < raw_width) {
                const int ape_row = ((pos % compress_ratio) + compress_ratio) % compress_ratio;
                const int64_t src = flat_idx * score_raw_stride + head_offset + d;
                score_val = scalar_load_float(score_raw, src)
                            + ape[static_cast<int64_t>(ape_row) * ape_stride + head_offset + d];
            } else {
                const bool use_cache = state_block_table != nullptr && pos >= 0 && state_tokens_per_block > 0;
                if (use_cache) {
                    const int64_t block_index = (pos / state_tokens_per_block) % state_block_table_stride;
                    const int block_number = state_block_table[static_cast<int64_t>(req_idx) * state_block_table_stride
                                                               + block_index];
                    const bool valid_block = block_number > 0 && block_number < num_state_blocks;
                    if (valid_block && head_offset + d < state_width) {
                        const int64_t row_offset = pos % state_ring_entries;
                        const float* row = state_cache + static_cast<int64_t>(block_number) * state_cache_stride0
                                           + row_offset * state_cache_stride1 + head_offset;
                        score_val = row[state_width + d];
                    }
                }
            }
            if (score_val != kInvalidCompressorScore) {
                max_score = fmaxf(max_score, score_val);
                ++valid_count;
            }
        }
        if (valid_count > 0) {
            float denom = 0.0f;
            float numer = 0.0f;
            for (int t = 0; t < window_count; ++t) {
                const int pos = start + t;
                if (pos < 0) {
                    continue;
                }
                const bool use_second_half = t >= compress_ratio;
                const int head_offset = use_second_half ? head_dim : 0;
                bool use_raw = false;
                int64_t flat_idx = 0;
                if (!disable_raw_path) {
                    if (batched_raw) {
                        const int req_seq_start = seq_start_per_req[req_idx];
                        const int req_cu_lo     = cu_seq_per_req[req_idx];
                        const int req_cu_hi     = cu_seq_per_req[req_idx + 1];
                        const int flat_in_req   = pos - req_seq_start;
                        use_raw = flat_in_req >= 0 && flat_in_req < (req_cu_hi - req_cu_lo);
                        if (use_raw) {
                            const int64_t global_raw_idx = static_cast<int64_t>(req_cu_lo + flat_in_req);
                            flat_idx = raw_unpad_restore == nullptr ? global_raw_idx : raw_unpad_restore[global_raw_idx];
                            use_raw = flat_idx >= 0 && flat_idx < num_tokens;
                        }
                    } else {
                        flat_idx = static_cast<int64_t>(pos) - seq_start;
                        use_raw = flat_idx >= 0 && flat_idx < num_tokens;
                    }
                }

                float kv_val = 0.0f;
                float score_val = kInvalidCompressorScore;
                if (use_raw && head_offset + d < raw_width) {
                    const int ape_row = ((pos % compress_ratio) + compress_ratio) % compress_ratio;
                    kv_val = scalar_load_float(kv_raw, flat_idx * kv_raw_stride + head_offset + d);
                    score_val = scalar_load_float(score_raw, flat_idx * score_raw_stride + head_offset + d)
                                + ape[static_cast<int64_t>(ape_row) * ape_stride + head_offset + d];
                } else {
                    const bool use_cache = state_block_table != nullptr && pos >= 0 && state_tokens_per_block > 0;
                    if (use_cache) {
                        const int64_t block_index = (pos / state_tokens_per_block) % state_block_table_stride;
                        const int block_number = state_block_table[static_cast<int64_t>(req_idx) * state_block_table_stride
                                                                   + block_index];
                        const bool valid_block = block_number > 0 && block_number < num_state_blocks;
                        if (valid_block && head_offset + d < state_width) {
                            const int64_t row_offset = pos % state_ring_entries;
                            const float* row = state_cache + static_cast<int64_t>(block_number) * state_cache_stride0
                                               + row_offset * state_cache_stride1 + head_offset;
                            kv_val = row[d];
                            score_val = row[state_width + d];
                        }
                    }
                }
                if (score_val != kInvalidCompressorScore) {
                    const float p = expf(score_val - max_score);
                    denom += p;
                    numer += kv_val * p;
                }
            }
            compressed = denom > 0.0f ? numer / denom : 0.0f;
        }
    }

    compressed_shared[d] = d < head_dim ? compressed : 0.0f;
    reduce_shared[d]    = d < head_dim ? compressed * compressed : 0.0f;
    __syncthreads();

    for (int stride = 256; stride > 0; stride >>= 1) {
        if (d < stride) {
            reduce_shared[d] += reduce_shared[d + stride];
        }
        __syncthreads();
    }
    const float rrms = rsqrtf(reduce_shared[0] / static_cast<float>(head_dim) + rms_norm_eps);
    if (d < head_dim) {
        normed_raw_shared[d] = compressed_shared[d] * rrms * scalar_load_float(norm_weight, d);
    }
    __syncthreads();

    const int nope_head_dim = head_dim - rope_head_dim;
    uint8_t* cache_block = kv_cache + kv_block_idx * kv_cache_block_stride;
    if (head_dim == 512) {
        constexpr int token_stride = kSwaTokenDataBytes;
        constexpr int scale_dim = kSwaScaleBytes;
        uint8_t* token_ptr = cache_block + kv_pos_in_blk * token_stride;
        uint8_t* scale_ptr = cache_block + kv_cache_block_size * token_stride + kv_pos_in_blk * scale_dim;

        if (d < head_dim) {
            quant_shared[d] = bf16_round_float(normed_raw_shared[d]);
        }
        __syncthreads();
        if (d < nope_head_dim / kSwaQuantBlock) {
            float amax = 0.0f;
            const int base = d * kSwaQuantBlock;
            for (int i = 0; i < kSwaQuantBlock; ++i) {
                amax = fmaxf(amax, fabsf(quant_shared[base + i]));
            }
            amax = fmaxf(amax, 1.0e-4f);
            float exponent = ceilf(log2f(amax / kSwaFp8Max));
            exponent = fminf(fmaxf(exponent, -127.0f), 128.0f);
            const float scale = exp2f(exponent);
            group_scale_shared[d] = scale;
            int encoded = static_cast<int>(exponent + 127.0f);
            encoded = encoded < 0 ? 0 : (encoded > 255 ? 255 : encoded);
            scale_ptr[d] = static_cast<uint8_t>(encoded);
        }
        if (d == 0) {
            scale_ptr[nope_head_dim / kSwaQuantBlock] = 0;
        }
        __syncthreads();
        if (d < nope_head_dim) {
            float scaled = quant_shared[d] / group_scale_shared[d / kSwaQuantBlock];
            scaled = fminf(fmaxf(scaled, -kSwaFp8Max), kSwaFp8Max);
            __nv_fp8_e4m3 fp8_value = __nv_fp8_e4m3(scaled);
            token_ptr[d] = fp8_value.__x;
        } else if (d < head_dim) {
            const int rope_local = d - nope_head_dim;
            const int pair_base = nope_head_dim + (rope_local / 2) * 2;
            const float even = normed_raw_shared[pair_base];
            const float odd  = normed_raw_shared[pair_base + 1];
            const int cs_idx = rope_local / 2;
            const int64_t compressed_pos = (position / compress_ratio) * compress_ratio;
            const float* cache_base = cos_sin_cache + compressed_pos * cos_sin_stride;
            const float cos_v = cache_base[cs_idx];
            const float sin_v = cache_base[rope_head_dim / 2 + cs_idx];
            const float result = (rope_local & 1) == 0 ? even * cos_v - odd * sin_v : odd * cos_v + even * sin_v;
            store_u16_unaligned(token_ptr + nope_head_dim + rope_local * static_cast<int>(sizeof(uint16_t)),
                                float_to_bf16_bits(result));
        }
    } else if (head_dim == 128) {
        constexpr int token_stride = 128;
        constexpr int scale_dim = 4;
        uint8_t* token_ptr = cache_block + kv_pos_in_blk * token_stride;
        uint8_t* scale_ptr = cache_block + kv_cache_block_size * token_stride + kv_pos_in_blk * scale_dim;
        float result = 0.0f;
        if (d < head_dim) {
            if (d < nope_head_dim) {
                result = normed_raw_shared[d];
            } else {
                const int rope_local = d - nope_head_dim;
                const int pair_base = nope_head_dim + (rope_local / 2) * 2;
                const float even = normed_raw_shared[pair_base];
                const float odd  = normed_raw_shared[pair_base + 1];
                const int cs_idx = rope_local / 2;
                const int64_t compressed_pos = (position / compress_ratio) * compress_ratio;
                const float* cache_base = cos_sin_cache + compressed_pos * cos_sin_stride;
                const float cos_v = cache_base[cs_idx];
                const float sin_v = cache_base[rope_head_dim / 2 + cs_idx];
                result = (rope_local & 1) == 0 ? even * cos_v - odd * sin_v : odd * cos_v + even * sin_v;
            }
            quant_shared[d] = bf16_round_float(result);
        }
        __syncthreads();
        if (d == 0) {
            float amax = 0.0f;
            for (int i = 0; i < 128; ++i) {
                amax = fmaxf(amax, fabsf(quant_shared[i]));
            }
            amax = fmaxf(amax, 1.0e-4f);
            const float exponent = ceilf(log2f((amax / kSwaFp8Max)));
            group_scale_shared[0] = exp2f(exponent);
            store_f32_unaligned(scale_ptr, group_scale_shared[0]);
        }
        __syncthreads();
        if (d < head_dim) {
            float scaled = quant_shared[d] / group_scale_shared[0];
            scaled = fminf(fmaxf(scaled, -kSwaFp8Max), kSwaFp8Max);
            __nv_fp8_e4m3 fp8_value = __nv_fp8_e4m3(scaled);
            token_ptr[d] = fp8_value.__x;
        }
    }
}

struct MegaSwaWriterArgs {
    const c10::BFloat16* k = nullptr;
    const int64_t*       slot_mapping = nullptr;
    uint8_t*             k_cache = nullptr;
    int                  enabled = 0;
    int                  local_rows = 0;
    int                  cache_block_size = 0;
    int64_t              block_stride = 0;
    int                  num_cache_blocks = 0;
};

struct MegaCompressorBf16Args {
    const c10::BFloat16* kv = nullptr;
    const c10::BFloat16* score = nullptr;
    const float*         ape = nullptr;
    const int64_t*       positions = nullptr;
    float*               state_cache = nullptr;
    const int64_t*       state_slots = nullptr;
    const int32_t*       token_to_req = nullptr;
    const int32_t*       state_block_table = nullptr;
    const c10::BFloat16* norm_weight = nullptr;
    const float*         cos_sin_cache = nullptr;
    uint8_t*             kv_cache = nullptr;
    const int64_t*       kv_slots = nullptr;
    const int32_t*       seq_start_per_req = nullptr;
    const int32_t*       cu_seq_per_req = nullptr;
    const int64_t*       raw_unpad_restore = nullptr;
    int                  enabled = 0;
    int                  state_enabled = 0;
    int                  kv_writer_enabled = 0;
    int                  local_rows = 0;
    int                  raw_width = 0;
    int                  state_width = 0;
    int                  state_cache_block_size = 0;
    int64_t              state_cache_stride0 = 0;
    int64_t              state_cache_stride1 = 0;
    int                  num_state_blocks = 0;
    int                  compressor_ratio = 0;
    int                  head_dim = 0;
    int                  rope_head_dim = 0;
    int                  window_count = 0;
    int                  state_tokens_per_block = 0;
    int                  state_ring_entries = 0;
    int64_t              state_block_table_stride = 0;
    int                  kv_cache_block_size = 0;
    int64_t              kv_cache_block_stride = 0;
    int                  num_kv_blocks = 0;
    int64_t              cos_sin_stride = 0;
    int64_t              seq_start = 0;
    int                  disable_raw_path = 0;
    int                  batched_raw = 0;
    float                rms_norm_eps = 0.0f;
    int64_t              kv_bytes = 0;
    int64_t              score_bytes = 0;
};

__device__ __forceinline__ void dsv4MegaInlineSignalPadBarrier(uint32_t* const* __restrict__ signal_pads,
                                                               int cp_rank,
                                                               int cp_size,
                                                               int barrier_channel) {
    if (cp_size <= 1) {
        __syncthreads();
        return;
    }
    if (signal_pads == nullptr) {
        if (threadIdx.x == 0) {
            printf("DSV4 mega attention requires signal pads for in-kernel CP barrier\n");
            asm("trap;");
        }
        __syncthreads();
        return;
    }
    if (threadIdx.x < static_cast<unsigned int>(cp_size)) {
        const int peer = static_cast<int>(threadIdx.x);
        const int channel = kAttentionSignalPadChannelBase + barrier_channel;
        uint32_t* peer_signal = signal_pads[peer] + static_cast<int64_t>(channel) * cp_size + cp_rank;
        uint32_t* local_signal = signal_pads[cp_rank] + static_cast<int64_t>(channel) * cp_size + peer;
        if (!signal_pad_try_put<cuda::std::memory_order_release>(
                peer_signal, static_cast<unsigned long long>(kSymmMemBarrierTimeoutCycles))) {
            printf("DSV4 mega inline signal put timeout: rank=%d peer=%d channel=%d\n", cp_rank, peer, channel);
            asm("trap;");
        }
        if (!signal_pad_try_wait<cuda::std::memory_order_acquire>(
                local_signal, static_cast<unsigned long long>(kSymmMemBarrierTimeoutCycles))) {
            printf("DSV4 mega inline signal wait timeout: rank=%d peer=%d channel=%d value=%u\n",
                   cp_rank,
                   peer,
                   channel,
                   *local_signal);
            asm("trap;");
        }
    }
    __threadfence_system();
    __syncthreads();
}

__device__ __forceinline__ void dsv4MegaStageBytes(const void* __restrict__ src,
                                                   int64_t bytes,
                                                   const uint8_t* const* __restrict__ symm_buffer_ptrs,
                                                   int64_t per_rank_buffer_bytes,
                                                   int64_t payload_offset,
                                                   int cp_rank,
                                                   int cp_size,
                                                   uint32_t* const* __restrict__ symm_signal_pads,
                                                   int barrier_channel) {
    if (cp_size <= 1) {
        __syncthreads();
        return;
    }
    uint8_t* dst = const_cast<uint8_t*>(symm_buffer_ptrs[cp_rank])
                   + static_cast<int64_t>(cp_rank) * per_rank_buffer_bytes + payload_offset;
    const uint8_t* src_bytes = reinterpret_cast<const uint8_t*>(src);
    if (bytes > 0) {
        for (int64_t off = static_cast<int64_t>(threadIdx.x); off < bytes; off += blockDim.x) {
            dst[off] = src_bytes[off];
        }
    }
    __threadfence_system();
    __syncthreads();
    dsv4MegaInlineSignalPadBarrier(symm_signal_pads, cp_rank, cp_size, barrier_channel);
}

__device__ __forceinline__ void dsv4MegaStageBytesParallel(const void* __restrict__ src,
                                                           int64_t bytes,
                                                           const uint8_t* const* __restrict__ symm_buffer_ptrs,
                                                           int64_t per_rank_buffer_bytes,
                                                           int64_t payload_offset,
                                                           int cp_rank) {
    if (bytes <= 0 || symm_buffer_ptrs == nullptr) {
        return;
    }
    uint8_t* dst = const_cast<uint8_t*>(symm_buffer_ptrs[cp_rank])
                   + static_cast<int64_t>(cp_rank) * per_rank_buffer_bytes + payload_offset;
    const uint8_t* src_bytes = reinterpret_cast<const uint8_t*>(src);
    const int64_t stride = static_cast<int64_t>(gridDim.x) * blockDim.x;
    for (int64_t off = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x; off < bytes; off += stride) {
        dst[off] = src_bytes[off];
    }
    __threadfence_system();
}

__device__ __forceinline__ void dsv4MegaResidentGridCpBarrier(uint8_t* grid_sync_base,
                                                              int grid_barrier_base,
                                                              uint32_t* const* __restrict__ signal_pads,
                                                              int cp_rank,
                                                              int cp_size,
                                                              int barrier_channel,
                                                              unsigned long long launch_epoch) {
    if (gridDim.x > 1) {
        dsv4MegaGridBarrier(grid_sync_base, grid_barrier_base, launch_epoch);
    } else {
        __syncthreads();
    }
    if (blockIdx.x == 0) {
        __threadfence_system();
        __syncthreads();
        dsv4MegaInlineSignalPadBarrier(signal_pads, cp_rank, cp_size, barrier_channel);
    }
    if (gridDim.x > 1) {
        dsv4MegaGridBarrier(grid_sync_base, grid_barrier_base + 1, launch_epoch);
    } else {
        __syncthreads();
    }
}

__device__ __forceinline__ unsigned long long*
dsv4MegaLocalPhasePtr(const uint8_t* const* __restrict__ symm_buffer_ptrs,
                      int64_t per_rank_buffer_bytes,
                      int cp_rank,
                      int64_t offset) {
    uint8_t* local_base = const_cast<uint8_t*>(symm_buffer_ptrs[cp_rank])
                          + static_cast<int64_t>(cp_rank) * per_rank_buffer_bytes;
    return reinterpret_cast<unsigned long long*>(local_base + offset);
}

__device__ __forceinline__ void dsv4MegaPublishLocalPhase(const uint8_t* const* __restrict__ symm_buffer_ptrs,
                                                          int64_t per_rank_buffer_bytes,
                                                          int cp_rank,
                                                          unsigned long long launch_epoch) {
    if (symm_buffer_ptrs == nullptr || per_rank_buffer_bytes < kProtocolBytes) {
        return;
    }
    if (threadIdx.x == 0) {
        unsigned long long* phase =
            dsv4MegaLocalPhasePtr(symm_buffer_ptrs, per_rank_buffer_bytes, cp_rank, kMegaLocalPhaseOffset);
        unsigned long long* phase_complement =
            dsv4MegaLocalPhasePtr(symm_buffer_ptrs, per_rank_buffer_bytes, cp_rank, kMegaLocalPhaseComplementOffset);
        cuda::atomic_ref<unsigned long long, cuda::thread_scope_system> complement_ref(*phase_complement);
        cuda::atomic_ref<unsigned long long, cuda::thread_scope_system> phase_ref(*phase);
        complement_ref.store(~launch_epoch, cuda::std::memory_order_release);
        phase_ref.store(launch_epoch, cuda::std::memory_order_release);
    }
    __threadfence_system();
    __syncthreads();
}

__device__ __forceinline__ void dsv4MegaWaitLocalPhase(const uint8_t* const* __restrict__ symm_buffer_ptrs,
                                                       int64_t per_rank_buffer_bytes,
                                                       int cp_rank,
                                                       unsigned long long launch_epoch) {
    if (gridDim.x <= 1 || blockIdx.x == 0 || symm_buffer_ptrs == nullptr || per_rank_buffer_bytes < kProtocolBytes) {
        return;
    }
    unsigned long long* phase =
        dsv4MegaLocalPhasePtr(symm_buffer_ptrs, per_rank_buffer_bytes, cp_rank, kMegaLocalPhaseOffset);
    unsigned long long* phase_complement =
        dsv4MegaLocalPhasePtr(symm_buffer_ptrs, per_rank_buffer_bytes, cp_rank, kMegaLocalPhaseComplementOffset);
    cuda::atomic_ref<unsigned long long, cuda::thread_scope_system> phase_ref(*phase);
    cuda::atomic_ref<unsigned long long, cuda::thread_scope_system> complement_ref(*phase_complement);
    const unsigned long long start = clock64();
    while (true) {
        const unsigned long long seen = phase_ref.load(cuda::std::memory_order_acquire);
        const unsigned long long seen_complement = complement_ref.load(cuda::std::memory_order_acquire);
        if (seen == launch_epoch && seen_complement == ~launch_epoch) {
            break;
        }
        if (clock64() - start > static_cast<unsigned long long>(kSymmMemBarrierTimeoutCycles)) {
            if (threadIdx.x == 0) {
                printf("DSV4 mega local phase wait timeout: block=%d rank=%d epoch=%llu seen=%llu complement=%llu\n",
                       static_cast<int>(blockIdx.x),
                       cp_rank,
                       launch_epoch,
                       seen,
                       seen_complement);
            }
            asm("trap;");
        }
    }
    __threadfence_system();
    __syncthreads();
}

template<typename T>
__device__ __forceinline__ const T* dsv4Mega2DPtr(const T* __restrict__ local,
                                                  const uint8_t* const* __restrict__ symm_buffer_ptrs,
                                                  int64_t per_rank_buffer_bytes,
                                                  int64_t payload_offset,
                                                  int local_rows,
                                                  int width,
                                                  int global_row,
                                                  int col,
                                                  int cp_size) {
    if (cp_size <= 1 || symm_buffer_ptrs == nullptr) {
        return local + static_cast<int64_t>(global_row) * width + col;
    }
    if (local_rows <= 0 || global_row < 0 || col < 0 || col >= width) {
        return nullptr;
    }
    const int owner = global_row / local_rows;
    const int local_row = global_row - owner * local_rows;
    if (owner < 0 || owner >= cp_size || local_row < 0 || local_row >= local_rows) {
        return nullptr;
    }
    const int64_t elem_offset = static_cast<int64_t>(local_row) * width + col;
    const int64_t byte_offset = static_cast<int64_t>(owner) * per_rank_buffer_bytes + payload_offset
                                + elem_offset * static_cast<int64_t>(sizeof(T));
    return reinterpret_cast<const T*>(symm_buffer_ptrs[owner] + byte_offset);
}

__device__ __forceinline__ float dsv4MegaBf16RawAt(const MegaCompressorBf16Args& args,
                                                  const uint8_t* const* __restrict__ symm_buffer_ptrs,
                                                  int64_t per_rank_buffer_bytes,
                                                  int64_t payload_offset,
                                                  int global_row,
                                                  int col,
                                                  int cp_size,
                                                  bool score) {
    const int64_t role_offset = score ? args.kv_bytes : 0;
    const c10::BFloat16* local = score ? args.score : args.kv;
    const c10::BFloat16* ptr = dsv4Mega2DPtr(local,
                                             symm_buffer_ptrs,
                                             per_rank_buffer_bytes,
                                             payload_offset + role_offset,
                                             args.local_rows,
                                             args.raw_width,
                                             global_row,
                                             col,
                                             cp_size);
    if (ptr == nullptr) {
        return 0.0f;
    }
    return to_float_device(*ptr);
}

__device__ __forceinline__ uint16_t dsv4MegaBf16Bits2D(const c10::BFloat16* __restrict__ local,
                                                       const uint8_t* const* __restrict__ symm_buffer_ptrs,
                                                       int64_t per_rank_buffer_bytes,
                                                       int64_t payload_offset,
                                                       int local_rows,
                                                       int width,
                                                       int global_row,
                                                       int col,
                                                       int cp_size) {
    const c10::BFloat16* ptr = dsv4Mega2DPtr(local,
                                             symm_buffer_ptrs,
                                             per_rank_buffer_bytes,
                                             payload_offset,
                                             local_rows,
                                             width,
                                             global_row,
                                             col,
                                             cp_size);
    if (ptr == nullptr) {
        return 0;
    }
    return *reinterpret_cast<const uint16_t*>(ptr);
}

__device__ __noinline__ void dsv4MegaWriteSwaCachePhase(const MegaSwaWriterArgs& args,
                                                        const uint8_t* const* __restrict__ symm_buffer_ptrs,
                                                        int64_t per_rank_buffer_bytes,
                                                        int64_t scratch_payload_offset,
                                                        int cp_rank,
                                                        int cp_size,
                                                        uint32_t* const* __restrict__ symm_signal_pads,
                                                        int barrier_channel,
                                                        int grid_parallel,
                                                        uint8_t* grid_sync_base,
                                                        unsigned long long launch_epoch,
                                                        int grid_barrier_base) {
    if (!args.enabled) {
        __syncthreads();
        return;
    }
    const int64_t payload_bytes = static_cast<int64_t>(args.local_rows) * kSwaHeadDim * sizeof(c10::BFloat16);
    if (grid_parallel && cp_size > 1) {
        dsv4MegaStageBytesParallel(args.k, payload_bytes, symm_buffer_ptrs, per_rank_buffer_bytes, scratch_payload_offset, cp_rank);
        dsv4MegaGridBarrier(grid_sync_base, grid_barrier_base, launch_epoch);
        if (blockIdx.x == 0) {
            __threadfence_system();
            __syncthreads();
            dsv4MegaInlineSignalPadBarrier(symm_signal_pads, cp_rank, cp_size, barrier_channel);
        }
        dsv4MegaGridBarrier(grid_sync_base, grid_barrier_base + 1, launch_epoch);
	    } else {
	        dsv4MegaStageBytes(args.k,
	                           payload_bytes,
                           symm_buffer_ptrs,
                           per_rank_buffer_bytes,
                           scratch_payload_offset,
                           cp_rank,
                           cp_size,
	                           symm_signal_pads,
	                           barrier_channel);
	    }
		    __shared__ float   abs_values[kSwaQuantBlock];
    __shared__ float   tile_scale;
    __shared__ uint8_t encoded_scale;

    const int tid = static_cast<int>(threadIdx.x);
    const int total_tokens = args.local_rows * cp_size;
    const int tiles = kSwaNopeTiles + 1;
    const int task_stride = grid_parallel ? static_cast<int>(gridDim.x) : 1;
    const int task_start = grid_parallel ? static_cast<int>(blockIdx.x) : 0;
    for (int task = task_start; task < total_tokens * tiles; task += task_stride) {
        const int token = task / tiles;
        const int tile = task - token * tiles;
        const bool lane_active = tid < kSwaQuantBlock;
        int64_t slot = -1;
        bool valid = false;
        int64_t block_idx = 0;
        int64_t pos_in_block = 0;
        uint8_t* token_data_ptr = nullptr;
        uint8_t* token_scale_ptr = nullptr;
        if (lane_active) {
            slot = args.slot_mapping[token];
            valid = slot >= 0;
            block_idx = valid ? slot / args.cache_block_size : 0;
            pos_in_block = valid ? slot % args.cache_block_size : 0;
            valid = valid && block_idx >= 0 && block_idx < args.num_cache_blocks;
            uint8_t* cache_block_ptr = args.k_cache + block_idx * args.block_stride;
            token_data_ptr = cache_block_ptr + pos_in_block * kSwaTokenDataBytes;
            token_scale_ptr = cache_block_ptr + static_cast<int64_t>(args.cache_block_size) * kSwaTokenDataBytes
                              + pos_in_block * kSwaScaleBytes;
        }

        if (tile < kSwaNopeTiles) {
            const int offset = tile * kSwaQuantBlock + tid;
            float x = 0.0f;
            if (lane_active && valid) {
                const c10::BFloat16* ptr = dsv4Mega2DPtr(args.k,
                                                         symm_buffer_ptrs,
                                                         per_rank_buffer_bytes,
                                                         scratch_payload_offset,
                                                         args.local_rows,
                                                         kSwaHeadDim,
                                                         token,
                                                         offset,
                                                         cp_size);
                x = to_float_device(*ptr);
            }
            if (lane_active) {
                abs_values[tid] = valid ? fabsf(x) : 0.0f;
            }
            const float tile_amax = dsv4MegaBlockReduceMax(lane_active ? abs_values[tid] : 0.0f, abs_values, &tile_scale);
            if (tid == 0) {
                const float amax = fmaxf(tile_amax, 1.0e-4f);
                float exponent = ceilf(log2f(amax / kSwaFp8Max));
                exponent = fminf(fmaxf(exponent, -127.0f), 128.0f);
                tile_scale = exp2f(exponent);
                int encoded = static_cast<int>(exponent + 127.0f);
                encoded = encoded < 0 ? 0 : (encoded > 255 ? 255 : encoded);
                encoded_scale = static_cast<uint8_t>(encoded);
                if (valid) {
                    token_scale_ptr[tile] = encoded_scale;
                }
            }
            __syncthreads();
            if (lane_active && valid) {
                float scaled = x / tile_scale;
                scaled = fminf(fmaxf(scaled, -kSwaFp8Max), kSwaFp8Max);
                __nv_fp8_e4m3 fp8_value = __nv_fp8_e4m3(scaled);
                token_data_ptr[offset] = fp8_value.__x;
            }
            __syncthreads();
        } else {
            if (lane_active && valid) {
                store_u16_unaligned(token_data_ptr + kSwaNopeDim + tid * static_cast<int>(sizeof(uint16_t)),
                                    dsv4MegaBf16Bits2D(args.k,
                                                       symm_buffer_ptrs,
                                                       per_rank_buffer_bytes,
                                                       scratch_payload_offset,
                                                       args.local_rows,
                                                       kSwaHeadDim,
                                                       token,
                                                       kSwaNopeDim + tid,
                                                       cp_size));
                if (tid == 0) {
                    token_scale_ptr[kSwaNopeTiles] = 0;
                }
            }
            __syncthreads();
        }
    }
}

__device__ __noinline__ void dsv4MegaRunCompressorBf16Phase(const MegaCompressorBf16Args& args,
                                                            const uint8_t* const* __restrict__ symm_buffer_ptrs,
                                                            int64_t per_rank_buffer_bytes,
                                                            int64_t scratch_payload_offset,
                                                            int cp_rank,
                                                            int cp_size,
                                                            uint32_t* const* __restrict__ symm_signal_pads,
                                                            int barrier_channel,
                                                            int grid_parallel,
                                                            uint8_t* grid_sync_base,
                                                            unsigned long long launch_epoch,
                                                            int grid_barrier_base) {
    if (!args.enabled) {
        __syncthreads();
        return;
    }
    if (cp_size > 1) {
        if (grid_parallel) {
            dsv4MegaStageBytesParallel(
                args.kv, args.kv_bytes, symm_buffer_ptrs, per_rank_buffer_bytes, scratch_payload_offset, cp_rank);
            if (args.score_bytes > 0) {
                dsv4MegaStageBytesParallel(args.score,
                                           args.score_bytes,
                                           symm_buffer_ptrs,
                                           per_rank_buffer_bytes,
                                           scratch_payload_offset + args.kv_bytes,
                                           cp_rank);
            }
            dsv4MegaGridBarrier(grid_sync_base, grid_barrier_base, launch_epoch);
            if (blockIdx.x == 0) {
                __threadfence_system();
                __syncthreads();
                dsv4MegaInlineSignalPadBarrier(symm_signal_pads, cp_rank, cp_size, barrier_channel);
            }
            dsv4MegaGridBarrier(grid_sync_base, grid_barrier_base + 1, launch_epoch);
        } else {
            uint8_t* dst = const_cast<uint8_t*>(symm_buffer_ptrs[cp_rank])
                           + static_cast<int64_t>(cp_rank) * per_rank_buffer_bytes + scratch_payload_offset;
            const uint8_t* kv_src = reinterpret_cast<const uint8_t*>(args.kv);
            for (int64_t off = static_cast<int64_t>(threadIdx.x); off < args.kv_bytes; off += blockDim.x) {
                dst[off] = kv_src[off];
            }
            if (args.score_bytes > 0) {
                const uint8_t* score_src = reinterpret_cast<const uint8_t*>(args.score);
                uint8_t* score_dst = dst + args.kv_bytes;
                for (int64_t off = static_cast<int64_t>(threadIdx.x); off < args.score_bytes; off += blockDim.x) {
                    score_dst[off] = score_src[off];
                }
            }
            __threadfence_system();
            __syncthreads();
            dsv4MegaInlineSignalPadBarrier(symm_signal_pads, cp_rank, cp_size, barrier_channel);
        }
    } else {
        __syncthreads();
    }

    const int tid = static_cast<int>(threadIdx.x);
    const int total_tokens = args.local_rows * cp_size;
    if (args.state_enabled) {
        const int token_stride = grid_parallel ? static_cast<int>(gridDim.x) : 1;
        const int token_start = grid_parallel ? static_cast<int>(blockIdx.x) : 0;
        for (int token = token_start; token < total_tokens; token += token_stride) {
            const int64_t slot = args.state_slots[token];
            if (slot < 0) {
                continue;
            }
            const int64_t block_idx = slot / args.state_cache_block_size;
            const int64_t pos_in_block = slot % args.state_cache_block_size;
            if (block_idx < 0 || block_idx >= args.num_state_blocks) {
                continue;
            }
            const int64_t position = args.positions[token];
            int ape_row = 0;
            if (args.compressor_ratio > 0) {
                ape_row = static_cast<int>(position % args.compressor_ratio);
                if (ape_row < 0) {
                    ape_row += args.compressor_ratio;
                }
            }
            float* row = args.state_cache + block_idx * args.state_cache_stride0 + pos_in_block * args.state_cache_stride1;
            for (int d = tid; d < args.raw_width; d += blockDim.x) {
                row[d] = dsv4MegaBf16RawAt(
                    args, symm_buffer_ptrs, per_rank_buffer_bytes, scratch_payload_offset, token, d, cp_size, false);
                row[args.state_width + d] =
                    dsv4MegaBf16RawAt(
                        args, symm_buffer_ptrs, per_rank_buffer_bytes, scratch_payload_offset, token, d, cp_size, true)
                    + args.ape[static_cast<int64_t>(ape_row) * args.raw_width + d];
            }
        }
    }
    __syncthreads();
    if (grid_parallel) {
        dsv4MegaGridBarrier(grid_sync_base, grid_barrier_base + 2, launch_epoch);
    }

    if (!args.kv_writer_enabled) {
        __syncthreads();
        return;
    }

    __shared__ float compressed_shared[512];
    __shared__ float normed_raw_shared[512];
    __shared__ float quant_shared[512];
    __shared__ float reduce_shared[512];
    __shared__ float group_scale_shared[8];

    const int token_stride = grid_parallel ? static_cast<int>(gridDim.x) : 1;
    const int token_start = grid_parallel ? static_cast<int>(blockIdx.x) : 0;
    for (int token = token_start; token < total_tokens; token += token_stride) {
        const int64_t position = args.positions[token];
        const int64_t kv_slot = args.kv_slots[token];
        const bool token_active = args.compressor_ratio > 0 && ((position + 1) % args.compressor_ratio) == 0
                                  && kv_slot >= 0;
        const int64_t kv_block_idx = token_active ? kv_slot / args.kv_cache_block_size : 0;
        const int64_t kv_pos_in_blk = token_active ? kv_slot % args.kv_cache_block_size : 0;
        const bool valid_kv_slot = token_active && kv_block_idx >= 0 && kv_block_idx < args.num_kv_blocks;
        const int req_idx = args.token_to_req == nullptr ? 0 : static_cast<int>(args.token_to_req[token]);
        const int start = static_cast<int>(position) - args.window_count + 1;
        if (!valid_kv_slot) {
            continue;
        }

        for (int d = tid; d < 512; d += blockDim.x) {
            compressed_shared[d] = 0.0f;
            normed_raw_shared[d] = 0.0f;
            quant_shared[d] = 0.0f;
            reduce_shared[d] = 0.0f;
        }
        __syncthreads();

        for (int d = tid; d < args.head_dim; d += blockDim.x) {
            float compressed = 0.0f;
            float max_score = kInvalidCompressorScore;
            int valid_count = 0;
            for (int t = 0; t < args.window_count; ++t) {
                    const int pos = start + t;
                    if (pos < 0) {
                        continue;
                    }
                    const bool use_second_half = t >= args.compressor_ratio;
                    const int head_offset = use_second_half ? args.head_dim : 0;
                    bool use_raw = false;
                    int64_t flat_idx = 0;
                    if (!args.disable_raw_path) {
                        if (args.batched_raw) {
                            const int req_seq_start = args.seq_start_per_req[req_idx];
                            const int req_cu_lo = args.cu_seq_per_req[req_idx];
                            const int req_cu_hi = args.cu_seq_per_req[req_idx + 1];
                            const int flat_in_req = pos - req_seq_start;
                            use_raw = flat_in_req >= 0 && flat_in_req < (req_cu_hi - req_cu_lo);
                            if (use_raw) {
                                const int64_t global_raw_idx = static_cast<int64_t>(req_cu_lo + flat_in_req);
                                flat_idx = args.raw_unpad_restore == nullptr ? global_raw_idx : args.raw_unpad_restore[global_raw_idx];
                                use_raw = flat_idx >= 0 && flat_idx < total_tokens;
                            }
                        } else {
                            flat_idx = static_cast<int64_t>(pos) - args.seq_start;
                            use_raw = flat_idx >= 0 && flat_idx < total_tokens;
                        }
                    }
                    float score_val = kInvalidCompressorScore;
                    if (use_raw && head_offset + d < args.raw_width) {
                        const int ape_row = ((pos % args.compressor_ratio) + args.compressor_ratio) % args.compressor_ratio;
                        score_val = dsv4MegaBf16RawAt(args,
                                                      symm_buffer_ptrs,
                                                      per_rank_buffer_bytes,
                                                      scratch_payload_offset,
                                                      static_cast<int>(flat_idx),
                                                      head_offset + d,
                                                      cp_size,
                                                      true)
                                    + args.ape[static_cast<int64_t>(ape_row) * args.raw_width + head_offset + d];
                    } else {
                        const bool use_cache = args.state_block_table != nullptr && pos >= 0 && args.state_tokens_per_block > 0;
                        if (use_cache) {
                            const int64_t block_index = (pos / args.state_tokens_per_block) % args.state_block_table_stride;
                            const int block_number =
                                args.state_block_table[static_cast<int64_t>(req_idx) * args.state_block_table_stride
                                                       + block_index];
                            const bool valid_block = block_number > 0 && block_number < args.num_state_blocks;
                            if (valid_block && head_offset + d < args.state_width) {
                                const int64_t row_offset = pos % args.state_ring_entries;
                                const float* row = args.state_cache
                                                   + static_cast<int64_t>(block_number) * args.state_cache_stride0
                                                   + row_offset * args.state_cache_stride1 + head_offset;
                                score_val = row[args.state_width + d];
                            }
                        }
                    }
                    if (score_val != kInvalidCompressorScore) {
                        max_score = fmaxf(max_score, score_val);
                        ++valid_count;
                    }
            }
            if (valid_count > 0) {
                float denom = 0.0f;
                float numer = 0.0f;
                for (int t = 0; t < args.window_count; ++t) {
                        const int pos = start + t;
                        if (pos < 0) {
                            continue;
                        }
                        const bool use_second_half = t >= args.compressor_ratio;
                        const int head_offset = use_second_half ? args.head_dim : 0;
                        bool use_raw = false;
                        int64_t flat_idx = 0;
                        if (!args.disable_raw_path) {
                            if (args.batched_raw) {
                                const int req_seq_start = args.seq_start_per_req[req_idx];
                                const int req_cu_lo = args.cu_seq_per_req[req_idx];
                                const int req_cu_hi = args.cu_seq_per_req[req_idx + 1];
                                const int flat_in_req = pos - req_seq_start;
                                use_raw = flat_in_req >= 0 && flat_in_req < (req_cu_hi - req_cu_lo);
                                if (use_raw) {
                                    const int64_t global_raw_idx = static_cast<int64_t>(req_cu_lo + flat_in_req);
                                    flat_idx = args.raw_unpad_restore == nullptr ? global_raw_idx :
                                                                                args.raw_unpad_restore[global_raw_idx];
                                    use_raw = flat_idx >= 0 && flat_idx < total_tokens;
                                }
                            } else {
                                flat_idx = static_cast<int64_t>(pos) - args.seq_start;
                                use_raw = flat_idx >= 0 && flat_idx < total_tokens;
                            }
                        }
                        float kv_val = 0.0f;
                        float score_val = kInvalidCompressorScore;
                        if (use_raw && head_offset + d < args.raw_width) {
                            const int ape_row =
                                ((pos % args.compressor_ratio) + args.compressor_ratio) % args.compressor_ratio;
                            kv_val = dsv4MegaBf16RawAt(args,
                                                       symm_buffer_ptrs,
                                                       per_rank_buffer_bytes,
                                                       scratch_payload_offset,
                                                       static_cast<int>(flat_idx),
                                                       head_offset + d,
                                                       cp_size,
                                                       false);
                            score_val = dsv4MegaBf16RawAt(args,
                                                          symm_buffer_ptrs,
                                                          per_rank_buffer_bytes,
                                                          scratch_payload_offset,
                                                          static_cast<int>(flat_idx),
                                                          head_offset + d,
                                                          cp_size,
                                                          true)
                                        + args.ape[static_cast<int64_t>(ape_row) * args.raw_width + head_offset + d];
                        } else {
                            const bool use_cache =
                                args.state_block_table != nullptr && pos >= 0 && args.state_tokens_per_block > 0;
                            if (use_cache) {
                                const int64_t block_index =
                                    (pos / args.state_tokens_per_block) % args.state_block_table_stride;
                                const int block_number =
                                    args.state_block_table[static_cast<int64_t>(req_idx)
                                                               * args.state_block_table_stride
                                                           + block_index];
                                const bool valid_block = block_number > 0 && block_number < args.num_state_blocks;
                                if (valid_block && head_offset + d < args.state_width) {
                                    const int64_t row_offset = pos % args.state_ring_entries;
                                    const float* row =
                                        args.state_cache + static_cast<int64_t>(block_number) * args.state_cache_stride0
                                        + row_offset * args.state_cache_stride1 + head_offset;
                                    kv_val = row[d];
                                    score_val = row[args.state_width + d];
                                }
                            }
                        }
                        if (score_val != kInvalidCompressorScore) {
                            const float p = expf(score_val - max_score);
                            denom += p;
                            numer += kv_val * p;
                        }
                }
                compressed = denom > 0.0f ? numer / denom : 0.0f;
            }
            compressed_shared[d] = compressed;
            reduce_shared[d] = compressed * compressed;
        }
        for (int stride = 256; stride > 0; stride >>= 1) {
            if (tid < stride) {
                reduce_shared[tid] += reduce_shared[tid + stride];
            }
            __syncthreads();
        }
        const float sumsq = reduce_shared[0];
        const float rrms = rsqrtf(sumsq / static_cast<float>(args.head_dim) + args.rms_norm_eps);
        for (int d = tid; d < args.head_dim; d += blockDim.x) {
            const float norm = scalar_load_float(args.norm_weight, d);
            normed_raw_shared[d] = compressed_shared[d] * rrms * norm;
        }
        __syncthreads();
        const int nope_head_dim = args.head_dim - args.rope_head_dim;
        uint8_t* cache_block = args.kv_cache + kv_block_idx * args.kv_cache_block_stride;
        if (args.head_dim == 512) {
            uint8_t* token_ptr = cache_block + kv_pos_in_blk * kSwaTokenDataBytes;
            uint8_t* scale_ptr = cache_block + static_cast<int64_t>(args.kv_cache_block_size) * kSwaTokenDataBytes
                                 + kv_pos_in_blk * kSwaScaleBytes;
            for (int d = tid; d < args.head_dim; d += blockDim.x) {
                quant_shared[d] = bf16_round_float(normed_raw_shared[d]);
            }
            __syncthreads();
            if (tid < nope_head_dim / kSwaQuantBlock) {
                float amax = 0.0f;
                const int base = tid * kSwaQuantBlock;
                for (int i = 0; i < kSwaQuantBlock; ++i) {
                    amax = fmaxf(amax, fabsf(quant_shared[base + i]));
                }
                amax = fmaxf(amax, 1.0e-4f);
                float exponent = ceilf(log2f(amax / kSwaFp8Max));
                exponent = fminf(fmaxf(exponent, -127.0f), 128.0f);
                const float scale = exp2f(exponent);
                group_scale_shared[tid] = scale;
                int encoded = static_cast<int>(exponent + 127.0f);
                encoded = encoded < 0 ? 0 : (encoded > 255 ? 255 : encoded);
                scale_ptr[tid] = static_cast<uint8_t>(encoded);
            }
            if (tid == 0) {
                scale_ptr[nope_head_dim / kSwaQuantBlock] = 0;
            }
            __syncthreads();
            for (int d = tid; d < args.head_dim; d += blockDim.x) {
            if (d < nope_head_dim) {
                float scaled = quant_shared[d] / group_scale_shared[d / kSwaQuantBlock];
                scaled = fminf(fmaxf(scaled, -kSwaFp8Max), kSwaFp8Max);
                __nv_fp8_e4m3 fp8_value = __nv_fp8_e4m3(scaled);
                token_ptr[d] = fp8_value.__x;
            } else {
                const int rope_local = d - nope_head_dim;
                const int pair_base = nope_head_dim + (rope_local / 2) * 2;
                const float even = normed_raw_shared[pair_base];
                const float odd = normed_raw_shared[pair_base + 1];
                const int cs_idx = rope_local / 2;
                const int64_t compressed_pos = (position / args.compressor_ratio) * args.compressor_ratio;
                const float* cache_base = args.cos_sin_cache + compressed_pos * args.cos_sin_stride;
                const float cos_v = cache_base[cs_idx];
                const float sin_v = cache_base[args.rope_head_dim / 2 + cs_idx];
                const float result =
                    (rope_local & 1) == 0 ? even * cos_v - odd * sin_v : odd * cos_v + even * sin_v;
                store_u16_unaligned(token_ptr + nope_head_dim + rope_local * static_cast<int>(sizeof(uint16_t)),
                                    float_to_bf16_bits(result));
            }
            }
        } else if (args.head_dim == 128) {
            uint8_t* token_ptr = cache_block + kv_pos_in_blk * 128;
            uint8_t* scale_ptr = cache_block + static_cast<int64_t>(args.kv_cache_block_size) * 128 + kv_pos_in_blk * 4;
            for (int d = tid; d < args.head_dim; d += blockDim.x) {
                float result = 0.0f;
                if (d < nope_head_dim) {
                    result = normed_raw_shared[d];
                } else {
                    const int rope_local = d - nope_head_dim;
                    const int pair_base = nope_head_dim + (rope_local / 2) * 2;
                    const float even = normed_raw_shared[pair_base];
                    const float odd = normed_raw_shared[pair_base + 1];
                    const int cs_idx = rope_local / 2;
                    const int64_t compressed_pos = (position / args.compressor_ratio) * args.compressor_ratio;
                    const float* cache_base = args.cos_sin_cache + compressed_pos * args.cos_sin_stride;
                    const float cos_v = cache_base[cs_idx];
                    const float sin_v = cache_base[args.rope_head_dim / 2 + cs_idx];
                    result = (rope_local & 1) == 0 ? even * cos_v - odd * sin_v : odd * cos_v + even * sin_v;
                }
                quant_shared[d] = bf16_round_float(result);
            }
            __syncthreads();
            if (tid == 0) {
                float amax = 0.0f;
                for (int i = 0; i < 128; ++i) {
                    amax = fmaxf(amax, fabsf(quant_shared[i]));
                }
                amax = fmaxf(amax, 1.0e-4f);
                const float exponent = ceilf(log2f(amax / kSwaFp8Max));
                group_scale_shared[0] = exp2f(exponent);
                store_f32_unaligned(scale_ptr, group_scale_shared[0]);
            }
            __syncthreads();
            for (int d = tid; d < args.head_dim; d += blockDim.x) {
                float scaled = quant_shared[d] / group_scale_shared[0];
                scaled = fminf(fmaxf(scaled, -kSwaFp8Max), kSwaFp8Max);
                __nv_fp8_e4m3 fp8_value = __nv_fp8_e4m3(scaled);
                token_ptr[d] = fp8_value.__x;
            }
        }
        __syncthreads();
    }
}

void writeSwaCacheIfRequested(const std::optional<torch::Tensor>& swa_k_opt,
                              const std::optional<torch::Tensor>& swa_k_cache_opt,
                              const std::optional<torch::Tensor>& swa_slot_mapping_opt,
                              int64_t cp_rank,
                              int64_t cp_size,
                              int64_t comm_ptr,
                              int64_t buffer_handle,
                              int64_t per_rank_buffer_bytes,
                              const std::vector<int64_t>& rank_offsets,
                              const std::optional<torch::Tensor>& symm_buffer,
                              int64_t symm_buffer_ptrs_dev,
                              int64_t symm_signal_pad_ptrs_dev,
                              const py::object& symm_handle,
                              cudaStream_t stream) {
    const bool has_swa = swa_k_opt.has_value() || swa_k_cache_opt.has_value() || swa_slot_mapping_opt.has_value();
    if (!has_swa) {
        return;
    }
    TORCH_CHECK(swa_k_opt.has_value() && swa_k_cache_opt.has_value() && swa_slot_mapping_opt.has_value(),
                "swa_k, swa_k_cache, and swa_slot_mapping must be provided together");

    const torch::Tensor& local_swa_k = *swa_k_opt;
    const torch::Tensor& swa_k_cache = *swa_k_cache_opt;
    const torch::Tensor& swa_slot_mapping = *swa_slot_mapping_opt;
    validateTensor(local_swa_k, "swa_k", torch::kBFloat16, 2);
    validateTensor(swa_slot_mapping, "swa_slot_mapping", torch::kInt64, 1);
    TORCH_CHECK(swa_k_cache.is_cuda(), "swa_k_cache must be a CUDA tensor");
    TORCH_CHECK(swa_k_cache.scalar_type() == torch::kUInt8, "swa_k_cache must be uint8");
    TORCH_CHECK(swa_k_cache.dim() == 3, "swa_k_cache must have rank 3");
    TORCH_CHECK(swa_k_cache.size(2) == kSwaEntryBytes, "swa_k_cache last dim must be 584 bytes");
    TORCH_CHECK(swa_k_cache.stride(2) == 1 && swa_k_cache.stride(1) == kSwaEntryBytes,
                "swa_k_cache must expose packed token rows with stride[1]=584 and stride[2]=1");
    TORCH_CHECK(local_swa_k.size(1) == kSwaHeadDim, "swa_k must be [num_tokens, 512] bf16");

    torch::Tensor gathered_swa_k = allGather2DPayload(local_swa_k,
                                                      cp_rank,
                                                      cp_size,
                                                      comm_ptr,
                                                      buffer_handle,
                                                      per_rank_buffer_bytes,
                                                      rank_offsets,
                                                      symm_buffer,
                                                      symm_buffer_ptrs_dev,
                                                      symm_signal_pad_ptrs_dev,
                                                      symm_handle,
                                                      2,
                                                      stream);
    TORCH_CHECK(swa_slot_mapping.size(0) == gathered_swa_k.size(0),
                "swa_slot_mapping length must match gathered swa_k rows");
    if (gathered_swa_k.size(0) == 0) {
        return;
    }

    const dim3 grid(static_cast<unsigned int>(gathered_swa_k.size(0)), kSwaNopeTiles + 1);
    const dim3 block(kSwaQuantBlock);
    dsv4SwaQuantizeAndInsertKernel<<<grid, block, 0, stream>>>(
        gathered_swa_k.data_ptr<c10::BFloat16>(),
        swa_slot_mapping.data_ptr<int64_t>(),
        swa_k_cache.data_ptr<uint8_t>(),
        static_cast<int>(gathered_swa_k.size(0)),
        static_cast<int>(swa_k_cache.size(1)),
        static_cast<int64_t>(swa_k_cache.stride(0)),
        static_cast<int>(swa_k_cache.size(0)));
    const cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "dsv4 SWA cache write launch failed: ", cudaGetErrorString(err));
}

void writeCompressorKvCacheIfRequested(const torch::Tensor& gathered_kv,
                                       const torch::Tensor& gathered_score,
                                       const torch::Tensor& ape,
                                       const torch::Tensor& positions,
                                       const torch::Tensor& state_cache,
                                       const std::optional<torch::Tensor>& token_to_req_opt,
                                       const std::optional<torch::Tensor>& state_block_table_opt,
                                       const std::optional<torch::Tensor>& norm_weight_opt,
                                       const std::optional<torch::Tensor>& cos_sin_cache_opt,
                                       const std::optional<torch::Tensor>& kv_cache_opt,
                                       const std::optional<torch::Tensor>& kv_slots_opt,
                                       int64_t compressor_ratio,
                                       int64_t seq_start,
                                       bool disable_raw_path,
                                       double rms_norm_eps,
                                       int64_t head_dim,
                                       int64_t rope_head_dim,
                                       bool overlap,
                                       int64_t state_tokens_per_block,
                                       const std::optional<torch::Tensor>& seq_start_per_req_opt,
                                       const std::optional<torch::Tensor>& cu_seq_per_req_opt,
                                       const std::optional<torch::Tensor>& unpad_restore_opt,
                                       cudaStream_t stream) {
    const bool has_writer = token_to_req_opt.has_value() || state_block_table_opt.has_value()
                            || norm_weight_opt.has_value() || cos_sin_cache_opt.has_value()
                            || kv_cache_opt.has_value() || kv_slots_opt.has_value()
                            || seq_start_per_req_opt.has_value() || cu_seq_per_req_opt.has_value()
                            || unpad_restore_opt.has_value()
                            || head_dim != 0 || rope_head_dim != 0 || state_tokens_per_block != 0;
    if (!has_writer) {
        return;
    }
    TORCH_CHECK(token_to_req_opt.has_value() && state_block_table_opt.has_value() && norm_weight_opt.has_value()
                    && cos_sin_cache_opt.has_value() && kv_cache_opt.has_value() && kv_slots_opt.has_value(),
                "compressor_token_to_req, compressor_state_block_table, compressor_norm_weight, "
                "compressor_cos_sin_cache, compressor_kv_cache, and compressor_kv_slots must be provided together");
    TORCH_CHECK((seq_start_per_req_opt.has_value() && cu_seq_per_req_opt.has_value())
                    || (!seq_start_per_req_opt.has_value() && !cu_seq_per_req_opt.has_value()),
                "compressor_seq_start_per_req and compressor_cu_seq_per_req must be provided together");
    const torch::Tensor& token_to_req = *token_to_req_opt;
    const torch::Tensor& state_block_table = *state_block_table_opt;
    const torch::Tensor& norm_weight = *norm_weight_opt;
    const torch::Tensor& cos_sin_cache = *cos_sin_cache_opt;
    const torch::Tensor& kv_cache = *kv_cache_opt;
    const torch::Tensor& kv_slots = *kv_slots_opt;
    validateTensor(token_to_req, "compressor_token_to_req", torch::kInt32, 1);
    validateTensor(state_block_table, "compressor_state_block_table", torch::kInt32, 2);
    validateTensor(cos_sin_cache, "compressor_cos_sin_cache", torch::kFloat32, 2);
    validateTensor(kv_slots, "compressor_kv_slots", torch::kInt64, 1);
    validateFloatPayloadTensor(norm_weight, "compressor_norm_weight", 1);
    TORCH_CHECK(kv_cache.is_cuda(), "compressor_kv_cache must be a CUDA tensor");
    TORCH_CHECK(kv_cache.scalar_type() == torch::kUInt8, "compressor_kv_cache must be uint8");
    TORCH_CHECK(kv_cache.dim() == 3, "compressor_kv_cache must be [num_blocks, block_size, entry_bytes]");
    TORCH_CHECK(kv_cache.stride(2) == 1, "compressor_kv_cache last dimension must be contiguous");
    TORCH_CHECK(token_to_req.size(0) == gathered_kv.size(0), "compressor_token_to_req length mismatch");
    TORCH_CHECK(kv_slots.size(0) == gathered_kv.size(0), "compressor_kv_slots length mismatch");
    TORCH_CHECK(head_dim == 512 || head_dim == 128, "compressor_head_dim must be 512 or 128");
    TORCH_CHECK(rope_head_dim >= 0 && rope_head_dim <= head_dim && (rope_head_dim % 2) == 0,
                "compressor_rope_head_dim must be an even value in [0, head_dim]");
    const int64_t expected_raw_width = head_dim * (overlap ? 2 : 1);
    TORCH_CHECK(gathered_kv.size(1) >= expected_raw_width,
                "compressor_kv width must cover head_dim * (1 + overlap)");
    TORCH_CHECK(gathered_score.size(1) >= expected_raw_width,
                "compressor_score width must cover head_dim * (1 + overlap)");
    TORCH_CHECK(ape.size(1) >= expected_raw_width, "compressor_ape width must cover head_dim * (1 + overlap)");
    TORCH_CHECK(norm_weight.size(0) == head_dim, "compressor_norm_weight length must match compressor_head_dim");
    TORCH_CHECK(cos_sin_cache.size(1) >= rope_head_dim, "compressor_cos_sin_cache width must cover rope_head_dim");
    TORCH_CHECK(state_cache.size(2) >= expected_raw_width * 2,
                "compressor_state_cache width must cover 2 * head_dim * (1 + overlap)");
    TORCH_CHECK(state_tokens_per_block > 0, "compressor_state_tokens_per_block must be positive");
    if (head_dim == 512) {
        TORCH_CHECK(kv_cache.size(2) == kSwaEntryBytes, "512-dim compressor_kv_cache entry must be 584 bytes");
    } else {
        TORCH_CHECK(kv_cache.size(2) == 132, "128-dim compressor_kv_cache entry must be 132 bytes");
    }
    const bool batched_raw = seq_start_per_req_opt.has_value();
    const int32_t* seq_start_per_req = nullptr;
    const int32_t* cu_seq_per_req = nullptr;
    const int64_t* unpad_restore = nullptr;
    if (batched_raw) {
        validateTensor(*seq_start_per_req_opt, "compressor_seq_start_per_req", torch::kInt32, 1);
        validateTensor(*cu_seq_per_req_opt, "compressor_cu_seq_per_req", torch::kInt32, 1);
        seq_start_per_req = seq_start_per_req_opt->data_ptr<int32_t>();
        cu_seq_per_req = cu_seq_per_req_opt->data_ptr<int32_t>();
        if (unpad_restore_opt.has_value()) {
            validateTensor(*unpad_restore_opt, "compressor_unpad_restore", torch::kInt64, 1);
            unpad_restore = unpad_restore_opt->data_ptr<int64_t>();
        }
    } else {
        TORCH_CHECK(!unpad_restore_opt.has_value(),
                    "compressor_unpad_restore requires compressor_seq_start_per_req and compressor_cu_seq_per_req");
    }
    if (gathered_kv.size(0) == 0) {
        return;
    }

    const dim3 grid(static_cast<unsigned int>(gathered_kv.size(0)));
    const dim3 block(512);
#define LAUNCH_COMPRESSOR_KV_WRITER(KV_T, SCORE_T, NORM_T)                                                              \
    dsv4CompressorCompressNormRopeInsertKernel<KV_T, SCORE_T, NORM_T><<<grid, block, 0, stream>>>(                     \
        reinterpret_cast<const KV_T*>(gathered_kv.data_ptr()),                                                          \
        static_cast<int64_t>(gathered_kv.stride(0)),                                                                    \
        reinterpret_cast<const SCORE_T*>(gathered_score.data_ptr()),                                                    \
        static_cast<int64_t>(gathered_score.stride(0)),                                                                 \
        ape.data_ptr<float>(),                                                                                          \
        static_cast<int64_t>(ape.stride(0)),                                                                            \
        token_to_req.data_ptr<int32_t>(),                                                                               \
        positions.data_ptr<int64_t>(),                                                                                  \
        nullptr,                                                                                                        \
        state_cache.data_ptr<float>(),                                                                                  \
        static_cast<int64_t>(state_cache.stride(0)),                                                                    \
        static_cast<int64_t>(state_cache.stride(1)),                                                                    \
        state_block_table.data_ptr<int32_t>(),                                                                          \
        static_cast<int64_t>(state_block_table.stride(0)),                                                              \
        reinterpret_cast<const NORM_T*>(norm_weight.data_ptr()),                                                        \
        static_cast<float>(rms_norm_eps),                                                                               \
        cos_sin_cache.data_ptr<float>(),                                                                                \
        static_cast<int64_t>(cos_sin_cache.stride(0)),                                                                  \
        kv_cache.data_ptr<uint8_t>(),                                                                                   \
        kv_slots.data_ptr<int64_t>(),                                                                                   \
        static_cast<int>(gathered_kv.size(0)),                                                                          \
        static_cast<int>(head_dim),                                                                                     \
        static_cast<int>(rope_head_dim),                                                                                \
        static_cast<int>(gathered_kv.size(1)),                                                                          \
        static_cast<int>(state_cache.size(2) / 2),                                                                      \
        static_cast<int>(compressor_ratio),                                                                             \
        static_cast<int>((overlap ? 2 : 1) * compressor_ratio),                                                         \
        static_cast<int>(state_tokens_per_block),                                                                       \
        static_cast<int>(state_cache.size(1)),                                                                          \
        static_cast<int>(kv_cache.size(1)),                                                                             \
        static_cast<int64_t>(kv_cache.stride(0)),                                                                       \
        static_cast<int>(state_cache.size(0)),                                                                          \
        static_cast<int>(kv_cache.size(0)),                                                                             \
        static_cast<int64_t>(seq_start),                                                                                \
        disable_raw_path,                                                                                               \
        batched_raw,                                                                                                    \
        seq_start_per_req,                                                                                              \
        cu_seq_per_req,                                                                                                 \
        unpad_restore)
    if (gathered_kv.scalar_type() == torch::kBFloat16 && gathered_score.scalar_type() == torch::kBFloat16
        && norm_weight.scalar_type() == torch::kBFloat16) {
        LAUNCH_COMPRESSOR_KV_WRITER(c10::BFloat16, c10::BFloat16, c10::BFloat16);
    } else if (gathered_kv.scalar_type() == torch::kFloat32 && gathered_score.scalar_type() == torch::kFloat32
               && norm_weight.scalar_type() == torch::kFloat32) {
        LAUNCH_COMPRESSOR_KV_WRITER(float, float, float);
    } else if (gathered_kv.scalar_type() == torch::kBFloat16 && gathered_score.scalar_type() == torch::kBFloat16
               && norm_weight.scalar_type() == torch::kFloat32) {
        LAUNCH_COMPRESSOR_KV_WRITER(c10::BFloat16, c10::BFloat16, float);
    } else if (gathered_kv.scalar_type() == torch::kBFloat16 && gathered_score.scalar_type() == torch::kFloat32
               && norm_weight.scalar_type() == torch::kBFloat16) {
        LAUNCH_COMPRESSOR_KV_WRITER(c10::BFloat16, float, c10::BFloat16);
    } else {
        TORCH_CHECK(false, "unsupported compressor KV writer dtype combination");
    }
#undef LAUNCH_COMPRESSOR_KV_WRITER
    const cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "dsv4 compressor KV cache write launch failed: ", cudaGetErrorString(err));
}

void writeCompressorStateIfRequested(const std::optional<torch::Tensor>& compressor_kv_opt,
                                     const std::optional<torch::Tensor>& compressor_score_opt,
                                     const std::optional<torch::Tensor>& compressor_ape_opt,
                                     const std::optional<torch::Tensor>& compressor_positions_opt,
                                     const std::optional<torch::Tensor>& compressor_state_cache_opt,
                                     const std::optional<torch::Tensor>& compressor_state_slots_opt,
                                     const std::optional<torch::Tensor>& compressor_token_to_req_opt,
                                     const std::optional<torch::Tensor>& compressor_state_block_table_opt,
                                     const std::optional<torch::Tensor>& compressor_norm_weight_opt,
                                     const std::optional<torch::Tensor>& compressor_cos_sin_cache_opt,
                                     const std::optional<torch::Tensor>& compressor_kv_cache_opt,
                                     const std::optional<torch::Tensor>& compressor_kv_slots_opt,
                                     int64_t cp_rank,
                                     int64_t cp_size,
                                     int64_t comm_ptr,
                                     int64_t buffer_handle,
                                     int64_t per_rank_buffer_bytes,
                                     const std::vector<int64_t>& rank_offsets,
                                     const std::optional<torch::Tensor>& symm_buffer,
                                     int64_t symm_buffer_ptrs_dev,
                                     int64_t symm_signal_pad_ptrs_dev,
                                     const py::object& symm_handle,
                                     int64_t compressor_ratio,
                                     int64_t compressor_seq_start,
                                     bool compressor_disable_raw_path,
                                     double compressor_rms_norm_eps,
                                     int64_t compressor_head_dim,
                                     int64_t compressor_rope_head_dim,
                                     bool compressor_overlap,
                                         int64_t compressor_state_tokens_per_block,
                                         const std::optional<torch::Tensor>& compressor_seq_start_per_req_opt,
                                         const std::optional<torch::Tensor>& compressor_cu_seq_per_req_opt,
                                         const std::optional<torch::Tensor>& compressor_unpad_restore_opt,
                                         int64_t barrier_channel_base,
                                         cudaStream_t stream) {
    const bool has_compressor = compressor_kv_opt.has_value() || compressor_score_opt.has_value()
                                || compressor_ape_opt.has_value() || compressor_positions_opt.has_value()
                                || compressor_state_cache_opt.has_value() || compressor_state_slots_opt.has_value();
    if (!has_compressor) {
        return;
    }
    TORCH_CHECK(compressor_kv_opt.has_value() && compressor_score_opt.has_value() && compressor_ape_opt.has_value()
                    && compressor_positions_opt.has_value() && compressor_state_cache_opt.has_value()
                    && compressor_state_slots_opt.has_value(),
                "compressor_kv, compressor_score, compressor_ape, compressor_positions, "
                "compressor_state_cache, and compressor_state_slots must be provided together");

    const torch::Tensor& local_kv = *compressor_kv_opt;
    const torch::Tensor& local_score = *compressor_score_opt;
    const torch::Tensor& ape = *compressor_ape_opt;
    const torch::Tensor& positions = *compressor_positions_opt;
    const torch::Tensor& state_cache = *compressor_state_cache_opt;
    const torch::Tensor& state_slots = *compressor_state_slots_opt;
    validateFloatPayloadTensor(local_kv, "compressor_kv", 2);
    validateFloatPayloadTensor(local_score, "compressor_score", 2);
    validateTensor(ape, "compressor_ape", torch::kFloat32, 2);
    validateTensor(positions, "compressor_positions", torch::kInt64, 1);
    validateTensor(state_slots, "compressor_state_slots", torch::kInt64, 1);
    TORCH_CHECK(state_cache.is_cuda(), "compressor_state_cache must be a CUDA tensor");
    TORCH_CHECK(state_cache.scalar_type() == torch::kFloat32, "compressor_state_cache must be float32");
    TORCH_CHECK(state_cache.dim() == 3, "compressor_state_cache must be [num_blocks, entries_per_block, width]");
    TORCH_CHECK(state_cache.stride(2) == 1, "compressor_state_cache last dimension must be contiguous");
    TORCH_CHECK(local_score.scalar_type() == local_kv.scalar_type() || local_score.scalar_type() == torch::kFloat32
                    || local_score.scalar_type() == torch::kBFloat16,
                "compressor_score must be float32 or bfloat16");
    TORCH_CHECK(local_score.size(0) == local_kv.size(0) && local_score.size(1) == local_kv.size(1),
                "compressor_score shape must match compressor_kv shape");
    TORCH_CHECK(ape.size(0) >= compressor_ratio && compressor_ratio > 0,
                "compressor_ape rows must cover compressor_ratio");
    TORCH_CHECK(ape.size(1) == local_kv.size(1), "compressor_ape width must match compressor_kv");
    TORCH_CHECK(state_cache.size(2) >= local_kv.size(1) * 2,
                "compressor_state_cache width must be at least 2 * compressor_kv width");

    torch::Tensor gathered_kv = allGather2DPayload(local_kv,
                                                   cp_rank,
                                                   cp_size,
                                                   comm_ptr,
                                                   buffer_handle,
                                                   per_rank_buffer_bytes,
                                                   rank_offsets,
                                                       symm_buffer,
                                                       symm_buffer_ptrs_dev,
                                                       symm_signal_pad_ptrs_dev,
                                                       symm_handle,
                                                       barrier_channel_base,
                                                       stream);
    torch::Tensor gathered_score = allGather2DPayload(local_score,
                                                      cp_rank,
                                                      cp_size,
                                                      comm_ptr,
                                                      buffer_handle,
                                                      per_rank_buffer_bytes,
                                                      rank_offsets,
                                                          symm_buffer,
                                                          symm_buffer_ptrs_dev,
                                                          symm_signal_pad_ptrs_dev,
                                                          symm_handle,
                                                          barrier_channel_base + 1,
                                                          stream);
    TORCH_CHECK(positions.size(0) == gathered_kv.size(0),
                "compressor_positions length must match gathered compressor rows");
    TORCH_CHECK(state_slots.size(0) == gathered_kv.size(0),
                "compressor_state_slots length must match gathered compressor rows");
    if (gathered_kv.size(0) == 0) {
        return;
    }

    const int head_size = static_cast<int>(gathered_kv.size(1));
    const dim3 grid(static_cast<unsigned int>(gathered_kv.size(0)),
                    static_cast<unsigned int>((head_size + 255) / 256));
    const dim3 block(256);
    if (gathered_kv.scalar_type() == torch::kFloat32 && gathered_score.scalar_type() == torch::kFloat32) {
        dsv4CompressorSavePartialStatesKernel<float, float><<<grid, block, 0, stream>>>(
            gathered_kv.data_ptr<float>(),
            gathered_score.data_ptr<float>(),
            ape.data_ptr<float>(),
            positions.data_ptr<int64_t>(),
            state_cache.data_ptr<float>(),
            state_slots.data_ptr<int64_t>(),
            static_cast<int>(gathered_kv.size(0)),
            head_size,
            static_cast<int>(state_cache.size(2) / 2),
            static_cast<int>(state_cache.size(1)),
            static_cast<int64_t>(state_cache.stride(0)),
            static_cast<int64_t>(state_cache.stride(1)),
            static_cast<int>(state_cache.size(0)),
            static_cast<int>(compressor_ratio));
    } else if (gathered_kv.scalar_type() == torch::kBFloat16 && gathered_score.scalar_type() == torch::kBFloat16) {
        dsv4CompressorSavePartialStatesKernel<c10::BFloat16, c10::BFloat16><<<grid, block, 0, stream>>>(
            gathered_kv.data_ptr<c10::BFloat16>(),
            gathered_score.data_ptr<c10::BFloat16>(),
            ape.data_ptr<float>(),
            positions.data_ptr<int64_t>(),
            state_cache.data_ptr<float>(),
            state_slots.data_ptr<int64_t>(),
            static_cast<int>(gathered_kv.size(0)),
            head_size,
            static_cast<int>(state_cache.size(2) / 2),
            static_cast<int>(state_cache.size(1)),
            static_cast<int64_t>(state_cache.stride(0)),
            static_cast<int64_t>(state_cache.stride(1)),
            static_cast<int>(state_cache.size(0)),
            static_cast<int>(compressor_ratio));
    } else if (gathered_kv.scalar_type() == torch::kBFloat16 && gathered_score.scalar_type() == torch::kFloat32) {
        dsv4CompressorSavePartialStatesKernel<c10::BFloat16, float><<<grid, block, 0, stream>>>(
            gathered_kv.data_ptr<c10::BFloat16>(),
            gathered_score.data_ptr<float>(),
            ape.data_ptr<float>(),
            positions.data_ptr<int64_t>(),
            state_cache.data_ptr<float>(),
            state_slots.data_ptr<int64_t>(),
            static_cast<int>(gathered_kv.size(0)),
            head_size,
            static_cast<int>(state_cache.size(2) / 2),
            static_cast<int>(state_cache.size(1)),
            static_cast<int64_t>(state_cache.stride(0)),
            static_cast<int64_t>(state_cache.stride(1)),
            static_cast<int>(state_cache.size(0)),
            static_cast<int>(compressor_ratio));
    } else if (gathered_kv.scalar_type() == torch::kFloat32 && gathered_score.scalar_type() == torch::kBFloat16) {
        dsv4CompressorSavePartialStatesKernel<float, c10::BFloat16><<<grid, block, 0, stream>>>(
            gathered_kv.data_ptr<float>(),
            gathered_score.data_ptr<c10::BFloat16>(),
            ape.data_ptr<float>(),
            positions.data_ptr<int64_t>(),
            state_cache.data_ptr<float>(),
            state_slots.data_ptr<int64_t>(),
            static_cast<int>(gathered_kv.size(0)),
            head_size,
            static_cast<int>(state_cache.size(2) / 2),
            static_cast<int>(state_cache.size(1)),
            static_cast<int64_t>(state_cache.stride(0)),
            static_cast<int64_t>(state_cache.stride(1)),
            static_cast<int>(state_cache.size(0)),
            static_cast<int>(compressor_ratio));
    } else {
        TORCH_CHECK(false, "unsupported compressor kv/score dtype combination");
    }
    const cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "dsv4 compressor state write launch failed: ", cudaGetErrorString(err));

    writeCompressorKvCacheIfRequested(gathered_kv,
                                      gathered_score,
                                      ape,
                                      positions,
                                      state_cache,
                                      compressor_token_to_req_opt,
                                      compressor_state_block_table_opt,
                                      compressor_norm_weight_opt,
                                      compressor_cos_sin_cache_opt,
                                      compressor_kv_cache_opt,
                                      compressor_kv_slots_opt,
                                      compressor_ratio,
                                      compressor_seq_start,
                                      compressor_disable_raw_path,
                                      compressor_rms_norm_eps,
                                      compressor_head_dim,
                                      compressor_rope_head_dim,
                                      compressor_overlap,
                                      compressor_state_tokens_per_block,
                                      compressor_seq_start_per_req_opt,
                                      compressor_cu_seq_per_req_opt,
                                      compressor_unpad_restore_opt,
                                      stream);
}

bool hasSwaWriterArgs(const std::optional<torch::Tensor>& swa_k_opt,
                      const std::optional<torch::Tensor>& swa_k_cache_opt,
                      const std::optional<torch::Tensor>& swa_slot_mapping_opt) {
    return swa_k_opt.has_value() || swa_k_cache_opt.has_value() || swa_slot_mapping_opt.has_value();
}

bool hasCompressorArgs(const std::optional<torch::Tensor>& compressor_kv_opt,
                       const std::optional<torch::Tensor>& compressor_score_opt,
                       const std::optional<torch::Tensor>& compressor_ape_opt,
                       const std::optional<torch::Tensor>& compressor_positions_opt,
                       const std::optional<torch::Tensor>& compressor_state_cache_opt,
                       const std::optional<torch::Tensor>& compressor_state_slots_opt) {
    return compressor_kv_opt.has_value() || compressor_score_opt.has_value() || compressor_ape_opt.has_value()
           || compressor_positions_opt.has_value() || compressor_state_cache_opt.has_value()
           || compressor_state_slots_opt.has_value();
}

MegaSwaWriterArgs makeMegaSwaWriterArgs(const std::optional<torch::Tensor>& swa_k_opt,
                                        const std::optional<torch::Tensor>& swa_k_cache_opt,
                                        const std::optional<torch::Tensor>& swa_slot_mapping_opt,
                                        int64_t cp_size) {
    MegaSwaWriterArgs args{};
    if (!hasSwaWriterArgs(swa_k_opt, swa_k_cache_opt, swa_slot_mapping_opt)) {
        return args;
    }
    TORCH_CHECK(swa_k_opt.has_value() && swa_k_cache_opt.has_value() && swa_slot_mapping_opt.has_value(),
                "mega SWA writer requires swa_k, swa_k_cache, and swa_slot_mapping together");
    const torch::Tensor& local_swa_k = *swa_k_opt;
    const torch::Tensor& swa_k_cache = *swa_k_cache_opt;
    const torch::Tensor& swa_slot_mapping = *swa_slot_mapping_opt;
    validateTensor(local_swa_k, "swa_k", torch::kBFloat16, 2);
    validateTensor(swa_slot_mapping, "swa_slot_mapping", torch::kInt64, 1);
    TORCH_CHECK(swa_k_cache.is_cuda() && swa_k_cache.scalar_type() == torch::kUInt8 && swa_k_cache.dim() == 3,
                "mega swa_k_cache must be a rank-3 uint8 CUDA tensor");
    TORCH_CHECK(swa_k_cache.size(2) == kSwaEntryBytes && swa_k_cache.stride(2) == 1
                    && swa_k_cache.stride(1) == kSwaEntryBytes,
                "mega swa_k_cache must expose packed 584B rows");
    TORCH_CHECK(local_swa_k.size(1) == kSwaHeadDim, "mega swa_k must be [num_tokens, 512] bf16");
    TORCH_CHECK(swa_slot_mapping.size(0) == local_swa_k.size(0) * cp_size,
                "mega swa_slot_mapping length must match gathered SWA rows");
    args.enabled = 1;
    args.k = local_swa_k.data_ptr<c10::BFloat16>();
    args.slot_mapping = swa_slot_mapping.data_ptr<int64_t>();
    args.k_cache = swa_k_cache.data_ptr<uint8_t>();
    args.local_rows = static_cast<int>(local_swa_k.size(0));
    args.cache_block_size = static_cast<int>(swa_k_cache.size(1));
    args.block_stride = static_cast<int64_t>(swa_k_cache.stride(0));
    args.num_cache_blocks = static_cast<int>(swa_k_cache.size(0));
    return args;
}

MegaCompressorBf16Args makeMegaCompressorBf16Args(
    const std::optional<torch::Tensor>& compressor_kv_opt,
    const std::optional<torch::Tensor>& compressor_score_opt,
    const std::optional<torch::Tensor>& compressor_ape_opt,
    const std::optional<torch::Tensor>& compressor_positions_opt,
    const std::optional<torch::Tensor>& compressor_state_cache_opt,
    const std::optional<torch::Tensor>& compressor_state_slots_opt,
    int64_t compressor_ratio,
    const std::optional<torch::Tensor>& compressor_token_to_req_opt,
    const std::optional<torch::Tensor>& compressor_state_block_table_opt,
    const std::optional<torch::Tensor>& compressor_norm_weight_opt,
    const std::optional<torch::Tensor>& compressor_cos_sin_cache_opt,
    const std::optional<torch::Tensor>& compressor_kv_cache_opt,
    const std::optional<torch::Tensor>& compressor_kv_slots_opt,
    int64_t compressor_seq_start,
    bool compressor_disable_raw_path,
    double compressor_rms_norm_eps,
    int64_t compressor_head_dim,
    int64_t compressor_rope_head_dim,
    bool compressor_overlap,
    int64_t compressor_state_tokens_per_block,
    const std::optional<torch::Tensor>& compressor_seq_start_per_req_opt,
    const std::optional<torch::Tensor>& compressor_cu_seq_per_req_opt,
    const std::optional<torch::Tensor>& compressor_unpad_restore_opt,
    int64_t cp_size) {
    MegaCompressorBf16Args args{};
    if (!hasCompressorArgs(compressor_kv_opt,
                           compressor_score_opt,
                           compressor_ape_opt,
                           compressor_positions_opt,
                           compressor_state_cache_opt,
                           compressor_state_slots_opt)) {
        return args;
    }
    TORCH_CHECK(compressor_kv_opt.has_value() && compressor_score_opt.has_value() && compressor_ape_opt.has_value()
                    && compressor_positions_opt.has_value() && compressor_state_cache_opt.has_value()
                    && compressor_state_slots_opt.has_value(),
                "mega compressor requires kv, score, ape, positions, state_cache, and state_slots together");
    const torch::Tensor& local_kv = *compressor_kv_opt;
    const torch::Tensor& local_score = *compressor_score_opt;
    const torch::Tensor& ape = *compressor_ape_opt;
    const torch::Tensor& positions = *compressor_positions_opt;
    const torch::Tensor& state_cache = *compressor_state_cache_opt;
    const torch::Tensor& state_slots = *compressor_state_slots_opt;
    validateTensor(local_kv, "compressor_kv", torch::kBFloat16, 2);
    validateTensor(local_score, "compressor_score", torch::kBFloat16, 2);
    validateTensor(ape, "compressor_ape", torch::kFloat32, 2);
    validateTensor(positions, "compressor_positions", torch::kInt64, 1);
    validateTensor(state_slots, "compressor_state_slots", torch::kInt64, 1);
    TORCH_CHECK(state_cache.is_cuda() && state_cache.scalar_type() == torch::kFloat32 && state_cache.dim() == 3
                    && state_cache.stride(2) == 1,
                "mega compressor_state_cache must be [num_blocks, entries, width] float32 with contiguous width");
    TORCH_CHECK(local_score.sizes() == local_kv.sizes(), "mega compressor_score shape must match compressor_kv");
    TORCH_CHECK(ape.size(0) >= compressor_ratio && compressor_ratio > 0,
                "mega compressor_ape rows must cover compressor_ratio");
    TORCH_CHECK(ape.size(1) == local_kv.size(1), "mega compressor_ape width must match compressor_kv");
    TORCH_CHECK(state_cache.size(2) >= local_kv.size(1) * 2,
                "mega compressor_state_cache width must cover 2 * raw width");
    const int64_t total_rows = local_kv.size(0) * cp_size;
    TORCH_CHECK(positions.size(0) == total_rows, "mega compressor_positions length must match gathered rows");
    TORCH_CHECK(state_slots.size(0) == total_rows, "mega compressor_state_slots length must match gathered rows");

    args.enabled = 1;
    args.state_enabled = 1;
    args.kv = local_kv.data_ptr<c10::BFloat16>();
    args.score = local_score.data_ptr<c10::BFloat16>();
    args.ape = ape.data_ptr<float>();
    args.positions = positions.data_ptr<int64_t>();
    args.state_cache = state_cache.data_ptr<float>();
    args.state_slots = state_slots.data_ptr<int64_t>();
    args.local_rows = static_cast<int>(local_kv.size(0));
    args.raw_width = static_cast<int>(local_kv.size(1));
    args.state_width = static_cast<int>(state_cache.size(2) / 2);
    args.state_cache_block_size = static_cast<int>(state_cache.size(1));
    args.state_cache_stride0 = static_cast<int64_t>(state_cache.stride(0));
    args.state_cache_stride1 = static_cast<int64_t>(state_cache.stride(1));
    args.num_state_blocks = static_cast<int>(state_cache.size(0));
    args.compressor_ratio = static_cast<int>(compressor_ratio);
    args.kv_bytes = local_kv.numel() * local_kv.element_size();
    args.score_bytes = local_score.numel() * local_score.element_size();

    const bool has_kv_writer = compressor_token_to_req_opt.has_value() || compressor_state_block_table_opt.has_value()
                               || compressor_norm_weight_opt.has_value() || compressor_cos_sin_cache_opt.has_value()
                               || compressor_kv_cache_opt.has_value() || compressor_kv_slots_opt.has_value()
                               || compressor_seq_start_per_req_opt.has_value()
                               || compressor_cu_seq_per_req_opt.has_value() || compressor_unpad_restore_opt.has_value()
                               || compressor_head_dim != 0 || compressor_rope_head_dim != 0
                               || compressor_state_tokens_per_block != 0;
    if (!has_kv_writer) {
        return args;
    }
    TORCH_CHECK(compressor_token_to_req_opt.has_value() && compressor_state_block_table_opt.has_value()
                    && compressor_norm_weight_opt.has_value() && compressor_cos_sin_cache_opt.has_value()
                    && compressor_kv_cache_opt.has_value() && compressor_kv_slots_opt.has_value(),
                "mega compressor KV writer requires token_to_req, state_block_table, norm_weight, cos_sin_cache, "
                "kv_cache, and kv_slots together");
    const torch::Tensor& token_to_req = *compressor_token_to_req_opt;
    const torch::Tensor& state_block_table = *compressor_state_block_table_opt;
    const torch::Tensor& norm_weight = *compressor_norm_weight_opt;
    const torch::Tensor& cos_sin_cache = *compressor_cos_sin_cache_opt;
    const torch::Tensor& kv_cache = *compressor_kv_cache_opt;
    const torch::Tensor& kv_slots = *compressor_kv_slots_opt;
    validateTensor(token_to_req, "compressor_token_to_req", torch::kInt32, 1);
    validateTensor(state_block_table, "compressor_state_block_table", torch::kInt32, 2);
    validateTensor(norm_weight, "compressor_norm_weight", torch::kBFloat16, 1);
    validateTensor(cos_sin_cache, "compressor_cos_sin_cache", torch::kFloat32, 2);
    validateTensor(kv_slots, "compressor_kv_slots", torch::kInt64, 1);
    TORCH_CHECK(token_to_req.size(0) == total_rows, "mega compressor_token_to_req length must match gathered rows");
    TORCH_CHECK(kv_slots.size(0) == total_rows, "mega compressor_kv_slots length must match gathered rows");
    TORCH_CHECK(compressor_head_dim == 512 || compressor_head_dim == 128,
                "mega compressor_head_dim must be 512 or 128");
    TORCH_CHECK(compressor_rope_head_dim >= 0 && compressor_rope_head_dim <= compressor_head_dim
                    && (compressor_rope_head_dim % 2) == 0,
                "mega compressor_rope_head_dim must be an even value in [0, head_dim]");
    const int64_t expected_raw_width = compressor_head_dim * (compressor_overlap ? 2 : 1);
    TORCH_CHECK(local_kv.size(1) >= expected_raw_width,
                "mega compressor raw width must cover head_dim * (1 + overlap)");
    TORCH_CHECK(norm_weight.size(0) == compressor_head_dim,
                "mega compressor_norm_weight length must match head_dim");
    TORCH_CHECK(cos_sin_cache.size(1) >= compressor_rope_head_dim,
                "mega compressor_cos_sin_cache width must cover rope_head_dim");
    TORCH_CHECK(kv_cache.is_cuda() && kv_cache.scalar_type() == torch::kUInt8 && kv_cache.dim() == 3
                    && kv_cache.stride(2) == 1,
                "mega compressor_kv_cache must be [num_blocks, block_size, entry_bytes] uint8");
    if (compressor_head_dim == 512) {
        TORCH_CHECK(kv_cache.size(2) == kSwaEntryBytes, "mega 512-dim compressor cache entry must be 584 bytes");
    } else {
        TORCH_CHECK(kv_cache.size(2) == kIndexerEntryBytes, "mega 128-dim compressor cache entry must be 132 bytes");
    }
    TORCH_CHECK((compressor_seq_start_per_req_opt.has_value() && compressor_cu_seq_per_req_opt.has_value())
                    || (!compressor_seq_start_per_req_opt.has_value() && !compressor_cu_seq_per_req_opt.has_value()),
                "mega compressor_seq_start_per_req and compressor_cu_seq_per_req must be provided together");
    args.kv_writer_enabled = 1;
    args.token_to_req = token_to_req.data_ptr<int32_t>();
    args.state_block_table = state_block_table.data_ptr<int32_t>();
    args.norm_weight = norm_weight.data_ptr<c10::BFloat16>();
    args.cos_sin_cache = cos_sin_cache.data_ptr<float>();
    args.kv_cache = kv_cache.data_ptr<uint8_t>();
    args.kv_slots = kv_slots.data_ptr<int64_t>();
    args.head_dim = static_cast<int>(compressor_head_dim);
    args.rope_head_dim = static_cast<int>(compressor_rope_head_dim);
    args.window_count = static_cast<int>((compressor_overlap ? 2 : 1) * compressor_ratio);
    args.state_tokens_per_block = static_cast<int>(compressor_state_tokens_per_block);
    args.state_ring_entries = static_cast<int>(state_cache.size(1));
    args.state_block_table_stride = static_cast<int64_t>(state_block_table.stride(0));
    args.kv_cache_block_size = static_cast<int>(kv_cache.size(1));
    args.kv_cache_block_stride = static_cast<int64_t>(kv_cache.stride(0));
    args.num_kv_blocks = static_cast<int>(kv_cache.size(0));
    args.cos_sin_stride = static_cast<int64_t>(cos_sin_cache.stride(0));
    args.seq_start = compressor_seq_start;
    args.disable_raw_path = compressor_disable_raw_path ? 1 : 0;
    args.rms_norm_eps = static_cast<float>(compressor_rms_norm_eps);
    args.batched_raw = compressor_seq_start_per_req_opt.has_value() ? 1 : 0;
    if (args.batched_raw) {
        validateTensor(*compressor_seq_start_per_req_opt, "compressor_seq_start_per_req", torch::kInt32, 1);
        validateTensor(*compressor_cu_seq_per_req_opt, "compressor_cu_seq_per_req", torch::kInt32, 1);
        args.seq_start_per_req = compressor_seq_start_per_req_opt->data_ptr<int32_t>();
        args.cu_seq_per_req = compressor_cu_seq_per_req_opt->data_ptr<int32_t>();
        if (compressor_unpad_restore_opt.has_value()) {
            validateTensor(*compressor_unpad_restore_opt, "compressor_unpad_restore", torch::kInt64, 1);
            args.raw_unpad_restore = compressor_unpad_restore_opt->data_ptr<int64_t>();
        }
    } else {
        TORCH_CHECK(!compressor_unpad_restore_opt.has_value(),
                    "mega compressor_unpad_restore requires seq_start_per_req/cu_seq_per_req");
    }
    return args;
}

template<typename scalar_t>
__global__ void dsv4CpDistributedPrefillAttentionKernel(const scalar_t* __restrict__ q,
                                                        const scalar_t* __restrict__ kv,
                                                        const scalar_t* __restrict__ indexer_q,
                                                        const scalar_t* __restrict__ indexer_k,
                                                        const float* __restrict__ attn_sink,
                                                        const int64_t* __restrict__ req_id_per_token,
                                                        const int64_t* __restrict__ position_ids,
                                                        const int64_t* __restrict__ prefix_lengths,
                                                        const int64_t* __restrict__ input_lengths,
                                                        const int64_t* __restrict__ local_rows,
                                                        scalar_t* __restrict__ output,
                                                        int R,
                                                        int H,
                                                        int D,
                                                        int L,
                                                        int KH,
                                                        int IH,
                                                        int ID,
                                                        int LI,
                                                        int compress_ratio,
                                                        int window_size,
                                                        int compressed_topk,
                                                        const uint8_t* __restrict__ csa_indexer_k_cache,
                                                        const float* __restrict__ csa_indexer_weights,
                                                        const int64_t* __restrict__ csa_indexer_cu_lens,
                                                        int csa_indexer_total_len,
                                                        const uint8_t* __restrict__ csa_indexer_k_pool,
                                                        const int32_t* __restrict__ csa_indexer_block_table,
                                                        const int32_t* __restrict__ csa_indexer_seq_lens,
                                                        int csa_indexer_pool_block_size,
                                                        int64_t csa_indexer_pool_block_stride,
                                                        int64_t csa_indexer_block_table_stride,
                                                        const uint8_t* __restrict__ attention_cmp_k_cache,
                                                        const int64_t* __restrict__ attention_cmp_cu_lens,
                                                        const uint8_t* __restrict__ attention_cmp_k_pool,
                                                        const int32_t* __restrict__ attention_cmp_block_table,
                                                        const int32_t* __restrict__ attention_cmp_seq_lens,
                                                        int attention_cmp_pool_block_size,
                                                        int64_t attention_cmp_pool_block_stride,
                                                        int64_t attention_cmp_block_table_stride,
                                                        const uint8_t* __restrict__ attention_swa_k_cache,
                                                        const int64_t* __restrict__ attention_swa_cu_lens,
                                                        const uint8_t* __restrict__ attention_swa_k_pool,
                                                        const int64_t* __restrict__ attention_swa_slot_mapping,
                                                        const int32_t* __restrict__ attention_swa_gather_lens,
                                                        int attention_swa_pool_block_size,
                                                        int64_t attention_swa_pool_block_stride,
                                                        int64_t attention_swa_slot_mapping_stride,
                                                        const int64_t* __restrict__ kv_unpad_restore,
                                                        const int64_t* __restrict__ kv_cu_lens) {
    const int local_row = blockIdx.x;
    const int h         = blockIdx.y;
    const int tid       = threadIdx.x;
    if (local_row >= R || h >= H) {
        return;
    }

    __shared__ int   key_pos[kMaxAttentionKeys];
    __shared__ int   key_is_compressed[kMaxAttentionKeys];
    __shared__ int   key_cmp_idx[kMaxAttentionKeys];
    __shared__ float logits[kMaxAttentionKeys];
    __shared__ float reduce_buf[256];
    __shared__ int   key_count_s;
    __shared__ float max_logit_s;
    __shared__ float denom_s;

    const int64_t row   = local_rows[local_row];
    const int     req   = static_cast<int>(req_id_per_token[row]);
    const int     q_pos = static_cast<int>(position_ids[row]);
    const int     prefix_len = static_cast<int>(prefix_lengths[req]);
    const int     kv_len = static_cast<int>(prefix_lengths[req] + input_lengths[req]);

    if (tid == 0) {
        int key_count = 0;
        if (compress_ratio == 128) {
            const int valid = min_int((q_pos + 1) / compress_ratio, compressed_topk);
            int valid_bound = valid;
            if (attention_cmp_seq_lens != nullptr) {
                valid_bound = min_int(valid_bound, static_cast<int>(attention_cmp_seq_lens[req]));
            }
            for (int c = 0; c < valid_bound && key_count < kMaxAttentionKeys; ++c) {
                const int boundary_pos = (c + 1) * compress_ratio - 1;
                if (boundary_pos >= 0 && boundary_pos < kv_len) {
                    const bool use_compressed_cache = attention_cmp_k_pool != nullptr || attention_cmp_k_cache != nullptr;
                    key_pos[key_count] = use_compressed_cache ? c : boundary_pos;
                    key_is_compressed[key_count] = use_compressed_cache ? 1 : 0;
                    key_cmp_idx[key_count] = c;
                    ++key_count;
                }
            }
        } else {
            const bool use_fp8_indexer_pool = csa_indexer_k_pool != nullptr && csa_indexer_block_table != nullptr
                                              && csa_indexer_weights != nullptr;
            const bool use_fp8_indexer = use_fp8_indexer_pool
                                         || (csa_indexer_k_cache != nullptr && csa_indexer_weights != nullptr
                                             && csa_indexer_cu_lens != nullptr);
            int req_k_len = LI;
            if (use_fp8_indexer_pool) {
                req_k_len = csa_indexer_seq_lens == nullptr ? LI : static_cast<int>(csa_indexer_seq_lens[req]);
            } else if (use_fp8_indexer) {
                req_k_len = static_cast<int>(csa_indexer_cu_lens[req + 1] - csa_indexer_cu_lens[req]);
                req_k_len = min_int(req_k_len, csa_indexer_total_len);
            }
            const int valid = min_int((q_pos + 1) / compress_ratio, req_k_len);
            const int k_eff = min_int(valid, compressed_topk);
            for (int kth = 0; kth < k_eff && key_count < kMaxAttentionKeys; ++kth) {
                float best_score = -3.402823466e38F;
                int   best_idx   = -1;
                for (int c = 0; c < valid; ++c) {
                    bool used = false;
                    for (int prev = 0; prev < kth; ++prev) {
                        if (key_cmp_idx[prev] == c) {
                            used = true;
                        }
                    }
                    if (used) {
                        continue;
                    }
                    float score = 0.0f;
                    if (use_fp8_indexer_pool) {
                        score = indexer_score_fp8_pool(indexer_q,
                                                       csa_indexer_k_pool,
                                                       csa_indexer_block_table,
                                                       csa_indexer_weights,
                                                       row,
                                                       req,
                                                       c,
                                                       IH,
                                                       ID,
                                                       csa_indexer_pool_block_size,
                                                       csa_indexer_pool_block_stride,
                                                       csa_indexer_block_table_stride);
                    } else if (use_fp8_indexer) {
                        score = indexer_score_fp8_cache(indexer_q,
                                                        csa_indexer_k_cache,
                                                        csa_indexer_weights,
                                                        csa_indexer_cu_lens,
                                                        row,
                                                        req,
                                                        c,
                                                        IH,
                                                        ID);
                    } else {
                        score = indexer_score(indexer_q,
                                              indexer_k,
                                              row,
                                              req,
                                              c,
                                              IH,
                                              ID,
                                              LI,
                                              nullptr,
                                              0,
                                              0,
                                              0,
                                              0);
                    }
                    if (score > best_score || (score == best_score && (best_idx < 0 || c < best_idx))) {
                        best_score = score;
                        best_idx   = c;
                    }
                }
                if (best_idx >= 0) {
                    const int boundary_pos = (best_idx + 1) * compress_ratio - 1;
                    if (boundary_pos >= 0 && boundary_pos < kv_len) {
                        const bool use_compressed_cache = attention_cmp_k_pool != nullptr || attention_cmp_k_cache != nullptr;
                        key_pos[key_count] = use_compressed_cache ? best_idx : boundary_pos;
                        key_is_compressed[key_count] = use_compressed_cache ? 1 : 0;
                        key_cmp_idx[key_count] = best_idx;
                        ++key_count;
                    }
                }
            }
        }
        const int swa_start = max_int(0, q_pos - window_size + 1);
        const int swa_end   = min_int(q_pos + 1, kv_len);
        for (int pos = swa_start; pos < swa_end && key_count < kMaxAttentionKeys; ++pos) {
            key_pos[key_count] = pos;
            key_is_compressed[key_count] = 0;
            key_cmp_idx[key_count] = -1;
            ++key_count;
        }
        key_count_s = key_count;
        max_logit_s = attn_sink[h];
    }
    __syncthreads();

    const int   key_count = key_count_s;
    const float scale     = rsqrtf(static_cast<float>(D));
    if (key_count <= kWarpDotAttentionKeyThreshold) {
        for (int i = 0; i < key_count; ++i) {
            float dot_part = 0.0f;
            for (int d = tid; d < D; d += blockDim.x) {
                dot_part += q_at(q, row, h, d, H, D)
                            * attention_key_at(kv,
                                               attention_cmp_k_cache,
                                               attention_cmp_cu_lens,
                                               attention_cmp_k_pool,
                                               attention_cmp_block_table,
                                               attention_cmp_seq_lens,
                                               attention_cmp_pool_block_size,
                                               attention_cmp_pool_block_stride,
                                               attention_cmp_block_table_stride,
                                               attention_swa_k_cache,
                                               attention_swa_cu_lens,
                                               attention_swa_k_pool,
                                               attention_swa_slot_mapping,
                                               attention_swa_gather_lens,
                                               attention_swa_pool_block_size,
                                               attention_swa_pool_block_stride,
                                               attention_swa_slot_mapping_stride,
                                               key_is_compressed[i],
                                               req,
                                               key_pos[i],
                                               prefix_len,
                                               h,
                                               d,
                                               L,
                                               KH,
                                               D,
                                               kv_unpad_restore,
                                               kv_cu_lens,
                                               nullptr,
                                               0,
                                               0,
                                               0,
                                               0);
            }
            reduce_buf[tid] = dot_part;
            __syncthreads();
            for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
                if (tid < stride) {
                    reduce_buf[tid] += reduce_buf[tid + stride];
                }
                __syncthreads();
            }
            if (tid == 0) {
                logits[i] = reduce_buf[0] * scale;
                max_logit_s = fmaxf(max_logit_s, logits[i]);
            }
            __syncthreads();
        }
    } else {
        const int lane      = tid & 31;
        const int warp_id   = tid >> 5;
        const int num_warps = blockDim.x >> 5;
        for (int i = warp_id; i < key_count; i += num_warps) {
            float dot_part = 0.0f;
            for (int d = lane; d < D; d += 32) {
                dot_part += q_at(q, row, h, d, H, D)
                            * attention_key_at(kv,
                                               attention_cmp_k_cache,
                                               attention_cmp_cu_lens,
                                               attention_cmp_k_pool,
                                               attention_cmp_block_table,
                                               attention_cmp_seq_lens,
                                               attention_cmp_pool_block_size,
                                               attention_cmp_pool_block_stride,
                                               attention_cmp_block_table_stride,
                                               attention_swa_k_cache,
                                               attention_swa_cu_lens,
                                               attention_swa_k_pool,
                                               attention_swa_slot_mapping,
                                               attention_swa_gather_lens,
                                               attention_swa_pool_block_size,
                                               attention_swa_pool_block_stride,
                                               attention_swa_slot_mapping_stride,
                                               key_is_compressed[i],
                                               req,
                                               key_pos[i],
                                               prefix_len,
                                               h,
                                               d,
                                               L,
                                               KH,
                                               D,
                                               kv_unpad_restore,
                                               kv_cu_lens,
                                               nullptr,
                                               0,
                                               0,
                                               0,
                                               0);
            }
            dot_part = warp_sum(dot_part);
            if (lane == 0) {
                logits[i] = dot_part * scale;
            }
        }
        __syncthreads();

        float max_part = tid == 0 ? attn_sink[h] : -3.402823466e38F;
        for (int i = tid; i < key_count; i += blockDim.x) {
            max_part = fmaxf(max_part, logits[i]);
        }
        reduce_buf[tid] = max_part;
        __syncthreads();
        for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
            if (tid < stride) {
                reduce_buf[tid] = fmaxf(reduce_buf[tid], reduce_buf[tid + stride]);
            }
            __syncthreads();
        }
        if (tid == 0) {
            max_logit_s = reduce_buf[0];
        }
    }
    __syncthreads();

    float denom_part = 0.0f;
    const float max_logit = max_logit_s;
    for (int i = tid; i < key_count; i += blockDim.x) {
        denom_part += expf(logits[i] - max_logit);
    }
    reduce_buf[tid] = denom_part;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            reduce_buf[tid] += reduce_buf[tid + stride];
        }
        __syncthreads();
    }
    if (tid == 0) {
        denom_s = expf(attn_sink[h] - max_logit) + reduce_buf[0];
    }
    __syncthreads();

    const float denom = denom_s;
    for (int d = tid; d < D; d += blockDim.x) {
        float acc = 0.0f;
        for (int i = 0; i < key_count; ++i) {
            const float prob = expf(logits[i] - max_logit) / denom;
            acc += prob
                   * attention_key_at(kv,
                                      attention_cmp_k_cache,
                                      attention_cmp_cu_lens,
                                      attention_cmp_k_pool,
                                      attention_cmp_block_table,
                                      attention_cmp_seq_lens,
                                      attention_cmp_pool_block_size,
                                      attention_cmp_pool_block_stride,
                                      attention_cmp_block_table_stride,
                                      attention_swa_k_cache,
                                      attention_swa_cu_lens,
                                      attention_swa_k_pool,
                                      attention_swa_slot_mapping,
                                      attention_swa_gather_lens,
                                      attention_swa_pool_block_size,
                                      attention_swa_pool_block_stride,
                                      attention_swa_slot_mapping_stride,
                                      key_is_compressed[i],
                                      req,
                                      key_pos[i],
                                      prefix_len,
                                      h,
                                      d,
                                      L,
                                      KH,
                                      D,
                                      kv_unpad_restore,
                                      kv_cu_lens,
                                      nullptr,
                                      0,
                                      0,
                                      0,
                                      0);
        }
        output[(static_cast<int64_t>(local_row) * H + h) * D + d] = from_float_device<scalar_t>(acc);
    }
}


template<typename scalar_t>
__global__ __launch_bounds__(kMegaBlockThreads, 1) void dsv4CpDistributedPrefillMegaAttentionKernel(const scalar_t* __restrict__ q,
                                                        const scalar_t* __restrict__ kv,
                                                        const scalar_t* __restrict__ indexer_q,
                                                        const scalar_t* __restrict__ indexer_k,
                                                        const float* __restrict__ attn_sink,
                                                        const int64_t* __restrict__ req_id_per_token,
                                                        const int64_t* __restrict__ position_ids,
                                                        const int64_t* __restrict__ prefix_lengths,
                                                        const int64_t* __restrict__ input_lengths,
                                                        const int64_t* __restrict__ local_rows,
                                                        scalar_t* __restrict__ output,
                                                        int R,
                                                        int H,
                                                        int D,
                                                        int L,
                                                        int KH,
                                                        int IH,
                                                        int ID,
                                                        int LI,
                                                        int indexer_batch,
                                                        int compress_ratio,
                                                        int window_size,
                                                        int compressed_topk,
                                                        const uint8_t* __restrict__ csa_indexer_k_cache,
                                                        const float* __restrict__ csa_indexer_weights,
                                                        const int64_t* __restrict__ csa_indexer_cu_lens,
                                                        int csa_indexer_total_len,
                                                        const uint8_t* __restrict__ csa_indexer_k_pool,
                                                        const int32_t* __restrict__ csa_indexer_block_table,
                                                        const int32_t* __restrict__ csa_indexer_seq_lens,
                                                        int csa_indexer_pool_block_size,
                                                        int64_t csa_indexer_pool_block_stride,
                                                        int64_t csa_indexer_block_table_stride,
                                                        const uint8_t* __restrict__ attention_cmp_k_cache,
                                                        const int64_t* __restrict__ attention_cmp_cu_lens,
                                                        const uint8_t* __restrict__ attention_cmp_k_pool,
                                                        const int32_t* __restrict__ attention_cmp_block_table,
                                                        const int32_t* __restrict__ attention_cmp_seq_lens,
                                                        int attention_cmp_pool_block_size,
                                                        int64_t attention_cmp_pool_block_stride,
                                                        int64_t attention_cmp_block_table_stride,
                                                        const uint8_t* __restrict__ attention_swa_k_cache,
                                                        const int64_t* __restrict__ attention_swa_cu_lens,
                                                        const uint8_t* __restrict__ attention_swa_k_pool,
                                                        const int64_t* __restrict__ attention_swa_slot_mapping,
                                                        const int32_t* __restrict__ attention_swa_gather_lens,
                                                        int attention_swa_pool_block_size,
                                                        int64_t attention_swa_pool_block_stride,
                                                        int64_t attention_swa_slot_mapping_stride,
                                                        const int64_t* __restrict__ kv_unpad_restore,
                                                        const int64_t* __restrict__ kv_cu_lens,
                                                        const uint8_t* const* __restrict__ symm_buffer_ptrs,
                                                        int use_symm_direct_fresh,
                                                        int64_t symm_per_rank_buffer_bytes,
                                                        int64_t symm_fresh_payload_offset,
                                                        int symm_l_local,
                                                        int use_symm_direct_indexer,
                                                        int64_t symm_indexer_payload_offset,
                                                        int symm_indexer_l_local,
                                                        int64_t symm_scratch_payload_offset,
                                                        int64_t symm_splitk_payload_offset,
                                                        MegaSwaWriterArgs swa_writer,
                                                        MegaCompressorBf16Args csa_indexer_compressor_writer,
                                                        MegaCompressorBf16Args main_compressor_writer,
                                                        unsigned long long launch_epoch,
                                                        int cp_rank,
                                                        int cp_size,
                                                        uint32_t* const* __restrict__ symm_signal_pads,
                                                        int enable_grid_side_effects,
                                                        int enable_split_k_attention,
                                                        int splitk_keys_per_block) {
		    const int tid = threadIdx.x;
            if (static_cast<int>(blockDim.x) != kMegaBlockThreads) {
                if (tid == 0) {
                    printf("DSV4 mega attention launched with unsupported blockDim.x=%d expected=%d\n",
                           static_cast<int>(blockDim.x),
                           kMegaBlockThreads);
                }
                asm("trap;");
            }
    extern __shared__ unsigned char mega_dynamic_smem[];
    uint8_t* local_scratch_base = nullptr;
    if (symm_buffer_ptrs != nullptr && symm_per_rank_buffer_bytes > symm_scratch_payload_offset) {
        local_scratch_base = const_cast<uint8_t*>(symm_buffer_ptrs[cp_rank])
                             + static_cast<int64_t>(cp_rank) * symm_per_rank_buffer_bytes
                             + symm_scratch_payload_offset;
    }
    const int grid_parallel_side_effects =
        enable_grid_side_effects && local_scratch_base != nullptr && gridDim.x > 1;
    const int64_t grid_sync_bytes = grid_parallel_side_effects ? kMegaGridSyncBytes : 0;
    const int64_t swa_scratch_payload_offset = symm_scratch_payload_offset + grid_sync_bytes;
    const int64_t csa_compressor_scratch_payload_offset =
        alignUpInt64(swa_scratch_payload_offset
                         + static_cast<int64_t>(swa_writer.local_rows) * kSwaHeadDim
                               * static_cast<int64_t>(sizeof(c10::BFloat16)),
                     kMegaPayloadAlignBytes);
    const int64_t main_compressor_scratch_payload_offset =
        alignUpInt64(csa_compressor_scratch_payload_offset + csa_indexer_compressor_writer.kv_bytes
                         + csa_indexer_compressor_writer.score_bytes,
                     kMegaPayloadAlignBytes);
    const int64_t compressed_index_scratch_payload_offset =
        alignUpInt64(main_compressor_scratch_payload_offset + main_compressor_writer.kv_bytes
                         + main_compressor_writer.score_bytes,
                     kMegaPayloadAlignBytes);

    if (grid_parallel_side_effects || blockIdx.x == 0) {
        dsv4MegaWriteSwaCachePhase(swa_writer,
                                   symm_buffer_ptrs,
                                   symm_per_rank_buffer_bytes,
                                   swa_scratch_payload_offset,
                                   cp_rank,
                                   cp_size,
                                   symm_signal_pads,
                                   2,
                                   grid_parallel_side_effects,
                                   local_scratch_base,
                                   launch_epoch,
                                   0);
        dsv4MegaRunCompressorBf16Phase(csa_indexer_compressor_writer,
                                       symm_buffer_ptrs,
                                       symm_per_rank_buffer_bytes,
                                       csa_compressor_scratch_payload_offset,
                                       cp_rank,
                                       cp_size,
                                       symm_signal_pads,
                                       3,
                                       grid_parallel_side_effects,
                                       local_scratch_base,
                                       launch_epoch,
                                       2);
        dsv4MegaRunCompressorBf16Phase(main_compressor_writer,
                                       symm_buffer_ptrs,
                                       symm_per_rank_buffer_bytes,
                                       main_compressor_scratch_payload_offset,
                                       cp_rank,
                                       cp_size,
                                       symm_signal_pads,
                                       5,
                                       grid_parallel_side_effects,
                                       local_scratch_base,
                                       launch_epoch,
                                       7);
        if (use_symm_direct_fresh) {
            const int64_t payload_bytes = static_cast<int64_t>(symm_l_local) * KH * D * sizeof(scalar_t);
            if (grid_parallel_side_effects) {
                dsv4MegaStageBytesParallel(kv,
                                           payload_bytes,
                                           symm_buffer_ptrs,
                                           symm_per_rank_buffer_bytes,
                                           symm_fresh_payload_offset,
                                           cp_rank);
                dsv4MegaGridBarrier(local_scratch_base, 12, launch_epoch);
                if (blockIdx.x == 0) {
                    __threadfence_system();
                    __syncthreads();
                    dsv4MegaInlineSignalPadBarrier(symm_signal_pads, cp_rank, cp_size, 0);
                }
                dsv4MegaGridBarrier(local_scratch_base, 13, launch_epoch);
            } else {
                dsv4MegaStageBytes(kv,
                                   payload_bytes,
                                   symm_buffer_ptrs,
                                   symm_per_rank_buffer_bytes,
                                   symm_fresh_payload_offset,
                                   cp_rank,
                                   cp_size,
                                   symm_signal_pads,
                                   0);
            }
        }
        if (use_symm_direct_indexer) {
            const int64_t indexer_payload_bytes =
                static_cast<int64_t>(indexer_batch) * symm_indexer_l_local * IH * ID * sizeof(scalar_t);
            if (grid_parallel_side_effects) {
                dsv4MegaStageBytesParallel(indexer_k,
                                           indexer_payload_bytes,
                                           symm_buffer_ptrs,
                                           symm_per_rank_buffer_bytes,
                                           symm_indexer_payload_offset,
                                           cp_rank);
                dsv4MegaGridBarrier(local_scratch_base, 14, launch_epoch);
                if (blockIdx.x == 0) {
                    __threadfence_system();
                    __syncthreads();
                    dsv4MegaInlineSignalPadBarrier(symm_signal_pads, cp_rank, cp_size, 1);
                }
                dsv4MegaGridBarrier(local_scratch_base, 15, launch_epoch);
            } else {
                dsv4MegaStageBytes(indexer_k,
                                   indexer_payload_bytes,
                                   symm_buffer_ptrs,
                                   symm_per_rank_buffer_bytes,
                                   symm_indexer_payload_offset,
                                   cp_rank,
                                   cp_size,
                                   symm_signal_pads,
                                   1);
            }
        }
        if (grid_parallel_side_effects) {
            dsv4MegaGridBarrier(local_scratch_base, 16, launch_epoch);
        }
        if (blockIdx.x == 0) {
            dsv4MegaPublishLocalPhase(symm_buffer_ptrs, symm_per_rank_buffer_bytes, cp_rank, launch_epoch);
        }
    }
    if (grid_parallel_side_effects) {
        dsv4MegaResidentGridCpBarrier(local_scratch_base,
                                      21,
                                      symm_signal_pads,
                                      cp_rank,
                                      cp_size,
                                      kMegaSideEffectReleaseChannel,
                                      launch_epoch);
    }
    dsv4MegaWaitLocalPhase(symm_buffer_ptrs, symm_per_rank_buffer_bytes, cp_rank, launch_epoch);

    __shared__ int   compressed_indices[kMaxCompressedTopK];
    __shared__ int   reduce_indices[kMegaBlockThreads];
    __shared__ int   reduce_indices_second[kMegaBlockThreads];
    __shared__ float compressed_scores[kMaxCompressedTopK];
    __shared__ float reduce_buf[kMegaReduceBufSlots];
    __shared__ float reduce_result[kMegaReduceResultSlots];
    __shared__ float tile_logits[kMegaAttentionKeyTile];
    __shared__ float q_shared[kSwaHeadDim];
    __shared__ __align__(8) uint8_t indexer_metadata_storage[kMegaIndexerMetadataBytes];
    float* grouped_key_tile = reinterpret_cast<float*>(mega_dynamic_smem);

    const int compressed_stride = max_int(compressed_topk, 1);
	    uint8_t* scratch_work_base =
	        local_scratch_base == nullptr ?
	            nullptr :
	            local_scratch_base + (compressed_index_scratch_payload_offset - symm_scratch_payload_offset);
    unsigned long long* compressed_meta = reinterpret_cast<unsigned long long*>(scratch_work_base);
    const int64_t compressed_indices_offset =
        alignUpInt64(static_cast<int64_t>(R) * static_cast<int64_t>(sizeof(unsigned long long)),
                     kMegaPayloadAlignBytes);
    int* row_compressed_indices_base =
        scratch_work_base == nullptr ? nullptr : reinterpret_cast<int*>(scratch_work_base + compressed_indices_offset);
    const unsigned int compressed_epoch = static_cast<unsigned int>(launch_epoch & 0xffffffffull);
    const unsigned long long compressed_epoch_prefix = static_cast<unsigned long long>(compressed_epoch) << 32;

    const bool use_grouped_attention =
        D == kSwaHeadDim && KH == 1 && H >= kMegaAttentionHeadsPerCta
        && (H % kMegaAttentionHeadsPerCta) == 0 && static_cast<int>(blockDim.x) == kMegaBlockThreads;
    const int prebuild_compressed_indices =
        compress_ratio == 4 && compressed_topk > 0 && grid_parallel_side_effects && compressed_meta != nullptr
        && row_compressed_indices_base != nullptr;
    if (prebuild_compressed_indices) {
        for (int prebuild_row = static_cast<int>(blockIdx.x); prebuild_row < R; prebuild_row += static_cast<int>(gridDim.x)) {
            const int64_t prebuild_global_row = local_rows[prebuild_row];
            const int prebuild_req = static_cast<int>(req_id_per_token[prebuild_global_row]);
            const int prebuild_q_pos = static_cast<int>(position_ids[prebuild_global_row]);
            const int prebuild_kv_len =
                static_cast<int>(prefix_lengths[prebuild_req] + input_lengths[prebuild_req]);
            const int prebuild_sequential_count = dsv4MegaSequentialCompressedCount(compress_ratio,
                                                                                    compressed_topk,
                                                                                    prebuild_q_pos,
                                                                                    prebuild_kv_len,
                                                                                    prebuild_req,
                                                                                    LI,
                                                                                    csa_indexer_k_cache,
                                                                                    csa_indexer_weights,
                                                                                    csa_indexer_cu_lens,
                                                                                    csa_indexer_total_len,
                                                                                    csa_indexer_k_pool,
                                                                                    csa_indexer_block_table,
                                                                                    csa_indexer_seq_lens,
                                                                                    attention_cmp_seq_lens);
            if (prebuild_sequential_count < 0) {
                (void)dsv4MegaBuildCompressedIndicesForRow<scalar_t>(indexer_q,
                                                                     indexer_k,
                                                                     req_id_per_token,
                                                                     position_ids,
                                                                     prefix_lengths,
                                                                     input_lengths,
                                                                     local_rows,
                                                                     prebuild_row,
                                                                     compress_ratio,
                                                                     compressed_topk,
                                                                     LI,
                                                                     IH,
                                                                     ID,
                                                                     csa_indexer_k_cache,
                                                                     csa_indexer_weights,
                                                                     csa_indexer_cu_lens,
                                                                     csa_indexer_total_len,
                                                                     csa_indexer_k_pool,
                                                                     csa_indexer_block_table,
                                                                     csa_indexer_seq_lens,
                                                                     csa_indexer_pool_block_size,
                                                                     csa_indexer_pool_block_stride,
                                                                     csa_indexer_block_table_stride,
                                                                     attention_cmp_seq_lens,
                                                                     symm_buffer_ptrs,
                                                                     use_symm_direct_indexer,
                                                                     symm_per_rank_buffer_bytes,
                                                                     symm_indexer_payload_offset,
                                                                     symm_indexer_l_local,
                                                                     compressed_stride,
                                                                     compressed_epoch,
                                                                     compressed_epoch_prefix,
                                                                     compressed_meta,
                                                                     row_compressed_indices_base,
                                                                     compressed_indices,
                                                                     compressed_scores,
                                                                     reduce_indices,
                                                                     reduce_indices_second,
                                                                     reduce_buf,
                                                                     indexer_metadata_storage,
                                                                     reduce_result);
            } else {
                __syncthreads();
            }
        }
        dsv4MegaGridBarrier(local_scratch_base, 18, launch_epoch);
    }
    const int heads_per_task = use_grouped_attention ? kMegaAttentionHeadsPerCta : 1;
    const int head_task_count = use_grouped_attention ? H / kMegaAttentionHeadsPerCta : H;
    const int task_count = R * head_task_count;
    uint8_t* splitk_base =
        enable_split_k_attention && local_scratch_base != nullptr ?
            local_scratch_base + (symm_splitk_payload_offset - symm_scratch_payload_offset) :
            nullptr;
    if (enable_split_k_attention && use_grouped_attention && splitk_base != nullptr) {
        const int physical_group = static_cast<int>(blockIdx.x) / kMegaSplitKGroupSize;
        const int role = static_cast<int>(blockIdx.x) - physical_group * kMegaSplitKGroupSize;
        const int physical_group_count = max_int(1, static_cast<int>(gridDim.x) / kMegaSplitKGroupSize);
        uint8_t* splitk_group_sync_base =
            splitk_base + static_cast<int64_t>(gridDim.x) * kMegaSplitKRecordBytes;
        const int task_slots = max_int(1, static_cast<int>(gridDim.x) / kMegaSplitKGroupSize);
        const int max_split_keys = max_int(1, compressed_topk + window_size);
        const int max_split_key_blocks =
            max_int(1, (max_split_keys + splitk_keys_per_block - 1) / splitk_keys_per_block);
        for (int task_base = 0; task_base < task_count; task_base += task_slots) {
            const int split_task = task_base + physical_group;
            int split_local_row = 0;
            int split_head_task = 0;
            int split_h = 0;
            int64_t split_row = 0;
            int split_req = 0;
            int split_q_pos = 0;
            int split_prefix_len = 0;
            int split_kv_len = 0;
            int split_sequential_compressed_count = 0;
            int split_compressed_count = 0;
            const int* split_row_compressed_indices = nullptr;
            int split_swa_start = 0;
            int split_swa_end = 0;
            int split_total_keys = 0;
            int split_task_keys_per_block = splitk_keys_per_block;
            int split_key_blocks = max_split_key_blocks;
            const bool split_task_valid = split_task < task_count;
            if (split_task_valid) {
                split_local_row = split_task / head_task_count;
                split_head_task = split_task - split_local_row * head_task_count;
                split_h = split_head_task * heads_per_task;
                split_row = local_rows[split_local_row];
                split_req = static_cast<int>(req_id_per_token[split_row]);
                split_q_pos = static_cast<int>(position_ids[split_row]);
                split_prefix_len = static_cast<int>(prefix_lengths[split_req]);
                split_kv_len = static_cast<int>(prefix_lengths[split_req] + input_lengths[split_req]);
                split_sequential_compressed_count =
                    dsv4MegaSequentialCompressedCount(compress_ratio,
                                                      compressed_topk,
                                                      split_q_pos,
                                                      split_kv_len,
                                                      split_req,
                                                      LI,
                                                      csa_indexer_k_cache,
                                                      csa_indexer_weights,
                                                      csa_indexer_cu_lens,
                                                      csa_indexer_total_len,
                                                      csa_indexer_k_pool,
                                                      csa_indexer_block_table,
                                                      csa_indexer_seq_lens,
                                                      attention_cmp_seq_lens);
                if (split_sequential_compressed_count >= 0) {
                    split_compressed_count = split_sequential_compressed_count;
                } else {
                    cuda::atomic_ref<unsigned long long, cuda::thread_scope_device> meta_ref(
                        compressed_meta[split_local_row]);
                    unsigned long long meta_value = meta_ref.load(cuda::std::memory_order_acquire);
                    if ((meta_value & 0xffffffff00000000ull) != compressed_epoch_prefix
                        || (meta_value & kMegaCompressedDoneFlag) == 0) {
                        if (threadIdx.x == 0) {
                            printf("DSV4 split-K compressed-index meta not ready: block=%d row=%d epoch=%u meta=%llu\n",
                                   static_cast<int>(blockIdx.x),
                                   split_local_row,
                                   compressed_epoch,
                                   meta_value);
                        }
                        asm("trap;");
                    }
                    split_compressed_count = static_cast<int>(meta_value & kMegaCompressedCountMask);
                    split_row_compressed_indices =
                        row_compressed_indices_base + static_cast<int64_t>(split_local_row) * compressed_stride;
                }
                split_swa_start = max_int(0, split_q_pos - window_size + 1);
                split_swa_end = min_int(split_q_pos + 1, split_kv_len);
                split_total_keys = split_compressed_count + max_int(0, split_swa_end - split_swa_start);
                const MegaSplitKTiling split_tiling =
                    dsv4MegaSplitKActualTiling(split_total_keys, splitk_keys_per_block);
                split_task_keys_per_block = split_tiling.keys_per_block;
                split_key_blocks = split_tiling.key_blocks;
            }
            float* merge_record = dsv4MegaSplitKRecord(splitk_base, physical_group * kMegaSplitKGroupSize);
            if (role == 0) {
                if (split_task_valid) {
                    dsv4MegaInitializeSplitKMergeRecord(merge_record, attn_sink, split_h);
                } else {
                    dsv4MegaInitializeSplitKRecord(merge_record, 0);
                }
            }
            dsv4MegaSplitKGroupBarrier(splitk_group_sync_base, physical_group, physical_group_count, 0, launch_epoch);
            const int split_wave_limit = split_task_valid ? split_key_blocks : 0;
            for (int key_wave = 0; key_wave < split_wave_limit; key_wave += kMegaSplitKBlocksPerWave) {
                if (role > 0) {
                    const int partial_block = key_wave + role - 1;
                    float* partial_record =
                        dsv4MegaSplitKRecord(splitk_base, physical_group * kMegaSplitKGroupSize + role);
                    if (split_task_valid && partial_block < split_key_blocks) {
                        dsv4MegaComputeSplitKPartial(q,
                                                     kv,
                                                     attention_cmp_k_cache,
                                                     attention_cmp_cu_lens,
                                                     attention_cmp_k_pool,
                                                     attention_cmp_block_table,
                                                     attention_cmp_seq_lens,
                                                     attention_cmp_pool_block_size,
                                                     attention_cmp_pool_block_stride,
                                                     attention_cmp_block_table_stride,
                                                     attention_swa_k_cache,
                                                     attention_swa_cu_lens,
                                                     attention_swa_k_pool,
                                                     attention_swa_slot_mapping,
                                                     attention_swa_gather_lens,
                                                     attention_swa_pool_block_size,
                                                     attention_swa_pool_block_stride,
                                                     attention_swa_slot_mapping_stride,
                                                     split_row_compressed_indices,
                                                     split_sequential_compressed_count,
                                                     kv_unpad_restore,
                                                     kv_cu_lens,
                                                     symm_buffer_ptrs,
                                                     use_symm_direct_fresh,
                                                     symm_per_rank_buffer_bytes,
                                                     symm_fresh_payload_offset,
                                                     symm_l_local,
                                                     split_row,
                                                     split_req,
                                                     split_q_pos,
                                                     split_prefix_len,
                                                     split_h,
                                                     H,
                                                     D,
                                                     L,
                                                     KH,
                                                     compress_ratio,
                                                     split_compressed_count,
                                                     split_total_keys,
                                                     partial_block * split_task_keys_per_block,
                                                     (partial_block + 1) * split_task_keys_per_block,
                                                     attention_cmp_k_pool != nullptr || attention_cmp_k_cache != nullptr,
                                                     split_swa_start,
                                                     rsqrtf(static_cast<float>(D)),
                                                     grouped_key_tile,
                                                     partial_record);
                    }
                }
                dsv4MegaSplitKGroupBarrier(splitk_group_sync_base, physical_group, physical_group_count, 1, launch_epoch);
                if (role == 0 && split_task_valid) {
                    const int wave_blocks = min_int(kMegaSplitKBlocksPerWave, split_key_blocks - key_wave);
                    dsv4MegaMergeSplitKWave(merge_record, splitk_base, physical_group, wave_blocks);
                }
                dsv4MegaSplitKGroupBarrier(splitk_group_sync_base, physical_group, physical_group_count, 2, launch_epoch);
            }
            if (role == 0 && split_task_valid) {
                const int tid_local = static_cast<int>(threadIdx.x);
                for (int out_idx = tid_local; out_idx < kMegaAttentionHeadsPerCta * kSwaHeadDim;
                     out_idx += static_cast<int>(blockDim.x)) {
                    const int out_head_offset = out_idx / kSwaHeadDim;
                    const int out_d = out_idx - out_head_offset * kSwaHeadDim;
                    const float denom = dsv4MegaSplitKRecordDenom(merge_record)[out_head_offset];
                    const float value =
                        denom > 0.0f ?
                            dsv4MegaSplitKRecordAcc(merge_record)[out_head_offset * kSwaHeadDim + out_d] / denom :
                            0.0f;
                    output[(static_cast<int64_t>(split_local_row) * H + split_h + out_head_offset) * D + out_d] =
                        from_float_device<scalar_t>(value);
                }
            }
            dsv4MegaSplitKGroupBarrier(splitk_group_sync_base, physical_group, physical_group_count, 3, launch_epoch);
        }
        if (grid_parallel_side_effects) {
            dsv4MegaResidentGridCpBarrier(local_scratch_base,
                                          19,
                                          symm_signal_pads,
                                          cp_rank,
                                          cp_size,
                                          kMegaKernelDoneChannel,
                                          launch_epoch);
        }
        return;
    }
    for (int task = blockIdx.x; task < task_count; task += gridDim.x) {
    const int local_row = task / head_task_count;
    const int head_task = task - local_row * head_task_count;
    const int h = use_grouped_attention ? head_task * heads_per_task : head_task;

    const int64_t row   = local_rows[local_row];
    const int     req   = static_cast<int>(req_id_per_token[row]);
	    const int     q_pos = static_cast<int>(position_ids[row]);
	    const int     prefix_len = static_cast<int>(prefix_lengths[req]);
	    const int     kv_len = static_cast<int>(prefix_lengths[req] + input_lengths[req]);

			    unsigned long long compressed_meta_value = 0;
			    int compressed_count = 0;
			    const int sequential_compressed_count = dsv4MegaSequentialCompressedCount(compress_ratio,
			                                                                              compressed_topk,
			                                                                              q_pos,
			                                                                              kv_len,
		                                                                              req,
		                                                                              LI,
		                                                                              csa_indexer_k_cache,
		                                                                              csa_indexer_weights,
		                                                                              csa_indexer_cu_lens,
		                                                                              csa_indexer_total_len,
		                                                                              csa_indexer_k_pool,
		                                                                              csa_indexer_block_table,
		                                                                              csa_indexer_seq_lens,
		                                                                              attention_cmp_seq_lens);
			    if (sequential_compressed_count >= 0) {
			        compressed_count = sequential_compressed_count;
			    } else if (compressed_meta != nullptr && row_compressed_indices_base != nullptr) {
			        cuda::atomic_ref<unsigned long long, cuda::thread_scope_device> meta_ref(compressed_meta[local_row]);
			        const unsigned long long claimed_value = compressed_epoch_prefix;
		        bool build_this_row = false;
		        if (!prebuild_compressed_indices) {
		        if (tid == 0) {
		            while (true) {
		                unsigned long long expected = meta_ref.load(cuda::std::memory_order_acquire);
		                if ((expected & 0xffffffff00000000ull) == compressed_epoch_prefix
		                    && (expected & kMegaCompressedDoneFlag) != 0) {
		                    compressed_meta_value = expected;
		                    build_this_row = false;
		                    break;
		                }
		                if ((expected & 0xffffffff00000000ull) == compressed_epoch_prefix) {
		                    compressed_meta_value = expected;
		                    build_this_row = false;
		                    break;
		                }
		                unsigned long long desired = claimed_value;
		                if (meta_ref.compare_exchange_strong(expected,
		                                                     desired,
		                                                     cuda::std::memory_order_acq_rel,
		                                                     cuda::std::memory_order_acquire)) {
		                    compressed_meta_value = desired;
		                    build_this_row = true;
		                    break;
		                }
		            }
		        }
		        build_this_row = __syncthreads_or(build_this_row);
		        } else {
		            __syncthreads();
		        }
			        if (build_this_row) {
			            (void)dsv4MegaBuildCompressedIndicesForRow<scalar_t>(indexer_q,
			                                                                 indexer_k,
			                                                                 req_id_per_token,
			                                                                 position_ids,
			                                                                 prefix_lengths,
			                                                                 input_lengths,
			                                                                 local_rows,
			                                                                 local_row,
			                                                                 compress_ratio,
			                                                                 compressed_topk,
			                                                                 LI,
			                                                                 IH,
			                                                                 ID,
			                                                                 csa_indexer_k_cache,
			                                                                 csa_indexer_weights,
			                                                                 csa_indexer_cu_lens,
			                                                                 csa_indexer_total_len,
			                                                                 csa_indexer_k_pool,
			                                                                 csa_indexer_block_table,
			                                                                 csa_indexer_seq_lens,
			                                                                 csa_indexer_pool_block_size,
			                                                                 csa_indexer_pool_block_stride,
			                                                                 csa_indexer_block_table_stride,
			                                                                 attention_cmp_seq_lens,
			                                                                 symm_buffer_ptrs,
			                                                                 use_symm_direct_indexer,
			                                                                 symm_per_rank_buffer_bytes,
			                                                                 symm_indexer_payload_offset,
			                                                                 symm_indexer_l_local,
			                                                                 compressed_stride,
			                                                                 compressed_epoch,
			                                                                 compressed_epoch_prefix,
			                                                                 compressed_meta,
	                                                                 row_compressed_indices_base,
		                                                                 compressed_indices,
	                                                                 compressed_scores,
	                                                                 reduce_indices,
	                                                                 reduce_indices_second,
	                                                                 reduce_buf,
	                                                                 indexer_metadata_storage,
	                                                                 reduce_result);
			        } else {
			            __syncthreads();
			        }
		        const unsigned long long start = clock64();
		        while (true) {
		            compressed_meta_value = meta_ref.load(cuda::std::memory_order_acquire);
		            if ((compressed_meta_value & 0xffffffff00000000ull) == compressed_epoch_prefix
		                && (compressed_meta_value & kMegaCompressedDoneFlag) != 0) {
		                break;
		            }
		            if (clock64() - start > static_cast<unsigned long long>(kSymmMemBarrierTimeoutCycles)) {
		                if (tid == 0) {
		                    printf("DSV4 mega compressed-index wait timeout: block=%d row=%d epoch=%u meta=%llu\n",
		                           static_cast<int>(blockIdx.x),
		                           local_row,
		                           compressed_epoch,
		                           compressed_meta_value);
		                }
		                asm("trap;");
			            }
			        }
			        compressed_count = static_cast<int>(compressed_meta_value & kMegaCompressedCountMask);
			    } else {
			        compressed_count = dsv4MegaBuildCompressedIndicesForRow<scalar_t>(indexer_q,
			                                                                          indexer_k,
			                                                                          req_id_per_token,
			                                                                          position_ids,
			                                                                          prefix_lengths,
			                                                                          input_lengths,
			                                                                          local_rows,
			                                                                          local_row,
			                                                                          compress_ratio,
			                                                                          compressed_topk,
			                                                                          LI,
			                                                                          IH,
			                                                                          ID,
			                                                                          csa_indexer_k_cache,
			                                                                          csa_indexer_weights,
			                                                                          csa_indexer_cu_lens,
			                                                                          csa_indexer_total_len,
			                                                                          csa_indexer_k_pool,
			                                                                          csa_indexer_block_table,
			                                                                          csa_indexer_seq_lens,
			                                                                          csa_indexer_pool_block_size,
			                                                                          csa_indexer_pool_block_stride,
			                                                                          csa_indexer_block_table_stride,
			                                                                          attention_cmp_seq_lens,
			                                                                          symm_buffer_ptrs,
			                                                                          use_symm_direct_indexer,
			                                                                          symm_per_rank_buffer_bytes,
			                                                                          symm_indexer_payload_offset,
			                                                                          symm_indexer_l_local,
			                                                                          compressed_stride,
			                                                                          compressed_epoch,
			                                                                          compressed_epoch_prefix,
			                                                                          nullptr,
			                                                                          nullptr,
			                                                                          compressed_indices,
			                                                                          compressed_scores,
			                                                                          reduce_indices,
			                                                                          reduce_indices_second,
			                                                                          reduce_buf,
			                                                                          indexer_metadata_storage,
			                                                                          reduce_result);
			    }

			    const int*  row_compressed_indices =
			        sequential_compressed_count >= 0 ?
			            nullptr :
			            (row_compressed_indices_base != nullptr ?
			                 row_compressed_indices_base + static_cast<int64_t>(local_row) * compressed_stride :
			                 compressed_indices);
	    const float scale            = rsqrtf(static_cast<float>(D));
	    const bool  use_compressed_cache = attention_cmp_k_pool != nullptr || attention_cmp_k_cache != nullptr;
	    const int   swa_start = max_int(0, q_pos - window_size + 1);
	    const int   swa_end   = min_int(q_pos + 1, kv_len);

        if (use_grouped_attention) {
            const int warp_id = tid / kMegaAttentionWarpSize;
            const int lane_id = tid & (kMegaAttentionWarpSize - 1);
            const int head = h + warp_id;
            float q_values[kMegaAttentionDimsPerWarp];
            float acc[kMegaAttentionDimsPerWarp];
#pragma unroll
            for (int chunk = 0; chunk < kMegaAttentionDimsPerWarp; ++chunk) {
                const int d = lane_id + chunk * kMegaAttentionWarpSize;
                q_values[chunk] = q_at(q, row, head, d, H, D);
                acc[chunk] = 0.0f;
            }
            float online_max = attn_sink[head];
            float online_denom = 1.0f;

            int tile_key_pos[kMegaAttentionGroupedKeyTile];
            int tile_key_is_compressed[kMegaAttentionGroupedKeyTile];
#define DSV4_MEGA_CONSUME_GROUPED_TILE(KEY_COUNT)                                                                        \
    do {                                                                                                                 \
        dsv4MegaLoadAttentionKeyTile<scalar_t>(kv,                                                                      \
                                               attention_cmp_k_cache,                                                    \
                                               attention_cmp_cu_lens,                                                    \
                                               attention_cmp_k_pool,                                                     \
                                               attention_cmp_block_table,                                                \
                                               attention_cmp_seq_lens,                                                   \
                                               attention_cmp_pool_block_size,                                            \
                                               attention_cmp_pool_block_stride,                                          \
                                               attention_cmp_block_table_stride,                                         \
                                               attention_swa_k_cache,                                                    \
                                               attention_swa_cu_lens,                                                    \
                                               attention_swa_k_pool,                                                     \
                                               attention_swa_slot_mapping,                                               \
                                               attention_swa_gather_lens,                                                \
                                               attention_swa_pool_block_size,                                            \
                                               attention_swa_pool_block_stride,                                          \
                                               attention_swa_slot_mapping_stride,                                        \
                                               tile_key_is_compressed,                                                   \
                                               tile_key_pos,                                                             \
                                               KEY_COUNT,                                                                \
                                               req,                                                                      \
                                               prefix_len,                                                               \
                                               D,                                                                        \
                                               L,                                                                        \
                                               KH,                                                                       \
                                               kv_unpad_restore,                                                         \
                                               kv_cu_lens,                                                               \
                                               symm_buffer_ptrs,                                                         \
                                               use_symm_direct_fresh,                                                    \
                                               symm_per_rank_buffer_bytes,                                               \
                                               symm_fresh_payload_offset,                                                \
                                               symm_l_local,                                                             \
                                               grouped_key_tile);                                                        \
        dsv4MegaConsumeLoadedKeyTileGrouped(grouped_key_tile,                                                            \
                                            q_values,                                                                    \
                                            KEY_COUNT,                                                                   \
                                            scale,                                                                       \
                                            online_max,                                                                  \
                                            online_denom,                                                                \
                                            acc);                                                                        \
        __syncthreads();                                                                                                 \
    } while (0)
            for (int i = 0; i < compressed_count;) {
                int tile_count = 0;
#pragma unroll
                for (int key_i = 0; key_i < kMegaAttentionGroupedKeyTile; ++key_i) {
                    if (i + key_i < compressed_count) {
                        const int cmp_idx =
                            sequential_compressed_count >= 0 ? i + key_i : row_compressed_indices[i + key_i];
                        tile_key_pos[key_i] = use_compressed_cache ? cmp_idx : (cmp_idx + 1) * compress_ratio - 1;
                        tile_key_is_compressed[key_i] = use_compressed_cache ? 1 : 0;
                        ++tile_count;
                    }
                }
                DSV4_MEGA_CONSUME_GROUPED_TILE(tile_count);
                i += tile_count;
            }
            for (int pos = swa_start; pos < swa_end;) {
                int tile_count = 0;
#pragma unroll
                for (int key_i = 0; key_i < kMegaAttentionGroupedKeyTile; ++key_i) {
                    const int key_pos = pos + key_i;
                    if (key_pos < swa_end) {
                        tile_key_pos[key_i] = key_pos;
                        tile_key_is_compressed[key_i] = 0;
                        ++tile_count;
                    }
                }
                DSV4_MEGA_CONSUME_GROUPED_TILE(tile_count);
                pos += tile_count;
            }
#pragma unroll
            for (int chunk = 0; chunk < kMegaAttentionDimsPerWarp; ++chunk) {
                const int d = lane_id + chunk * kMegaAttentionWarpSize;
                output[(static_cast<int64_t>(local_row) * H + head) * D + d] =
                    from_float_device<scalar_t>(acc[chunk] / online_denom);
            }
#undef DSV4_MEGA_CONSUME_GROUPED_TILE
            continue;
        }

			    const int d0 = tid;
		    const int d1 = tid + blockDim.x;
		    const int lane_id = tid & 31;
		    float online_max = attn_sink[h];
		    float online_denom = 1.0f;
			    float acc0 = 0.0f;
			    float acc1 = 0.0f;

        for (int d = tid; d < D; d += static_cast<int>(blockDim.x)) {
            q_shared[d] = q_at(q, row, h, d, H, D);
        }
        __syncthreads();

#define DSV4_MEGA_CONSUME_TILE(KEY_COUNT)                                                                            \
    do {                                                                                                             \
        dsv4MegaAttentionLogitsForKeyTile<scalar_t>(q_shared,                                                       \
                                                    kv,                                                             \
                                                    attention_cmp_k_cache,                                          \
                                                    attention_cmp_cu_lens,                                          \
                                                    attention_cmp_k_pool,                                           \
                                                    attention_cmp_block_table,                                      \
                                                    attention_cmp_seq_lens,                                         \
                                                    attention_cmp_pool_block_size,                                  \
                                                    attention_cmp_pool_block_stride,                                \
                                                    attention_cmp_block_table_stride,                               \
                                                    attention_swa_k_cache,                                          \
                                                    attention_swa_cu_lens,                                          \
                                                    attention_swa_k_pool,                                           \
                                                    attention_swa_slot_mapping,                                     \
                                                    attention_swa_gather_lens,                                      \
                                                    attention_swa_pool_block_size,                                  \
                                                    attention_swa_pool_block_stride,                                \
                                                    attention_swa_slot_mapping_stride,                              \
                                                    tile_key_is_compressed,                                         \
                                                    tile_key_pos,                                                   \
                                                    KEY_COUNT,                                                      \
                                                    req,                                                            \
                                                    prefix_len,                                                     \
                                                    h,                                                              \
                                                    D,                                                              \
                                                    L,                                                              \
                                                    KH,                                                             \
                                                    kv_unpad_restore,                                               \
                                                    kv_cu_lens,                                                     \
                                                    symm_buffer_ptrs,                                               \
                                                    use_symm_direct_fresh,                                          \
                                                    symm_per_rank_buffer_bytes,                                     \
                                                    symm_fresh_payload_offset,                                      \
                                                    symm_l_local,                                                   \
                                                    scale,                                                          \
                                                    reduce_buf,                                                     \
                                                    reduce_result,                                                  \
                                                    tile_logits,                                                    \
                                                    tile_value0,                                                    \
                                                    tile_value1);                                                   \
        for (int key_i = 0; key_i < KEY_COUNT; ++key_i) {                                                           \
            const float online_logit = tile_logits[key_i];                                                          \
            float new_max = 0.0f;                                                                                    \
            float old_scale = 0.0f;                                                                                  \
            float key_scale = 0.0f;                                                                                  \
            float new_denom = 0.0f;                                                                                  \
            if (lane_id == 0) {                                                                                      \
                new_max = fmaxf(online_max, online_logit);                                                           \
                old_scale = expf(online_max - new_max);                                                              \
                key_scale = expf(online_logit - new_max);                                                            \
                new_denom = online_denom * old_scale + key_scale;                                                    \
            }                                                                                                        \
            new_max = __shfl_sync(0xffffffffu, new_max, 0);                                                         \
            old_scale = __shfl_sync(0xffffffffu, old_scale, 0);                                                     \
            key_scale = __shfl_sync(0xffffffffu, key_scale, 0);                                                     \
            new_denom = __shfl_sync(0xffffffffu, new_denom, 0);                                                     \
            online_denom = new_denom;                                                                                \
            acc0 *= old_scale;                                                                                       \
            acc1 *= old_scale;                                                                                       \
            if (d0 < D) {                                                                                            \
                acc0 += key_scale * tile_value0[key_i];                                                              \
            }                                                                                                        \
            if (d1 < D) {                                                                                            \
                acc1 += key_scale * tile_value1[key_i];                                                              \
            }                                                                                                        \
            online_max = new_max;                                                                                    \
        }                                                                                                            \
        __syncthreads();                                                                                             \
    } while (0)

        int tile_key_pos[kMegaAttentionKeyTile];
        int tile_key_is_compressed[kMegaAttentionKeyTile];
        float tile_value0[kMegaAttentionKeyTile];
        float tile_value1[kMegaAttentionKeyTile];

        for (int i = 0; i < compressed_count;) {
            int tile_count = 0;
#pragma unroll
            for (int key_i = 0; key_i < kMegaAttentionKeyTile; ++key_i) {
                if (i + key_i < compressed_count) {
                    const int cmp_idx = sequential_compressed_count >= 0 ? i + key_i : row_compressed_indices[i + key_i];
                    tile_key_pos[key_i] = use_compressed_cache ? cmp_idx : (cmp_idx + 1) * compress_ratio - 1;
                    tile_key_is_compressed[key_i] = use_compressed_cache ? 1 : 0;
                    ++tile_count;
                }
            }
            DSV4_MEGA_CONSUME_TILE(tile_count);
            i += tile_count;
        }
        for (int pos = swa_start; pos < swa_end;) {
            int tile_count = 0;
#pragma unroll
            for (int key_i = 0; key_i < kMegaAttentionKeyTile; ++key_i) {
                const int key_pos = pos + key_i;
                if (key_pos < swa_end) {
                    tile_key_pos[key_i] = key_pos;
                    tile_key_is_compressed[key_i] = 0;
                    ++tile_count;
                }
            }
            DSV4_MEGA_CONSUME_TILE(tile_count);
            pos += tile_count;
        }
	    if (d0 < D) {
	        output[(static_cast<int64_t>(local_row) * H + h) * D + d0] =
	            from_float_device<scalar_t>(acc0 / online_denom);
	    }
	    if (d1 < D) {
	        output[(static_cast<int64_t>(local_row) * H + h) * D + d1] =
	            from_float_device<scalar_t>(acc1 / online_denom);
	    }
#undef DSV4_MEGA_CONSUME_TILE
	    }
        if (grid_parallel_side_effects) {
            dsv4MegaResidentGridCpBarrier(local_scratch_base,
                                          19,
                                          symm_signal_pads,
                                          cp_rank,
                                          cp_size,
                                          kMegaKernelDoneChannel,
                                          launch_epoch);
        }
	}

template<typename scalar_t>
__global__ void dsv4BuildFlashMlaWorkspaceKernel(const scalar_t* __restrict__ kv,
                                                 scalar_t* __restrict__ workspace,
                                                 const int64_t* __restrict__ prefix_lengths,
                                                 const int64_t* __restrict__ input_lengths,
                                                 int B,
                                                 int M,
                                                 int D,
                                                 int L,
                                                 int KH,
                                                 int compressed_topk,
                                                 int window_size,
                                                 const uint8_t* __restrict__ attention_cmp_k_cache,
                                                 const int64_t* __restrict__ attention_cmp_cu_lens,
                                                 const uint8_t* __restrict__ attention_cmp_k_pool,
                                                 const int32_t* __restrict__ attention_cmp_block_table,
                                                 const int32_t* __restrict__ attention_cmp_seq_lens,
                                                 int attention_cmp_pool_block_size,
                                                 int64_t attention_cmp_pool_block_stride,
                                                 int64_t attention_cmp_block_table_stride,
                                                 const uint8_t* __restrict__ attention_swa_k_cache,
                                                 const int64_t* __restrict__ attention_swa_cu_lens,
                                                 const uint8_t* __restrict__ attention_swa_k_pool,
                                                 const int64_t* __restrict__ attention_swa_slot_mapping,
                                                 const int32_t* __restrict__ attention_swa_gather_lens,
                                                 int attention_swa_pool_block_size,
                                                 int64_t attention_swa_pool_block_stride,
                                                 int64_t attention_swa_slot_mapping_stride,
                                                 const int64_t* __restrict__ kv_unpad_restore,
                                                 const int64_t* __restrict__ kv_cu_lens) {
    const int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const int64_t total = static_cast<int64_t>(B) * M * D;
    if (idx >= total) {
        return;
    }
    const int d = static_cast<int>(idx % D);
    const int slot = static_cast<int>((idx / D) % M);
    const int req = static_cast<int>(idx / (static_cast<int64_t>(M) * D));
    const int prefix_len = static_cast<int>(prefix_lengths[req]);
    const int input_len = static_cast<int>(input_lengths[req]);
    float value = 0.0f;
    if (slot < compressed_topk) {
        value = attention_key_at(kv,
                                 attention_cmp_k_cache,
                                 attention_cmp_cu_lens,
                                 attention_cmp_k_pool,
                                 attention_cmp_block_table,
                                 attention_cmp_seq_lens,
                                 attention_cmp_pool_block_size,
                                 attention_cmp_pool_block_stride,
                                 attention_cmp_block_table_stride,
                                 attention_swa_k_cache,
                                 attention_swa_cu_lens,
                                 attention_swa_k_pool,
                                 attention_swa_slot_mapping,
                                 attention_swa_gather_lens,
                                 attention_swa_pool_block_size,
                                 attention_swa_pool_block_stride,
                                 attention_swa_slot_mapping_stride,
                                 1,
                                 req,
                                 slot,
                                 prefix_len,
                                 0,
                                 d,
                                 L,
                                 KH,
                                 D,
                                 kv_unpad_restore,
                                 kv_cu_lens,
                                 nullptr,
                                 0,
                                 0,
                                 0,
                                 0);
    } else {
        const int rel = slot - compressed_topk;
        const int gather_len = attention_swa_gather_lens == nullptr ? min_int(prefix_len, window_size) :
                                                                      static_cast<int>(attention_swa_gather_lens[req]);
        const int start_pos = prefix_len - gather_len;
        const int key_pos = start_pos + rel;
        if (key_pos >= 0 && key_pos < prefix_len + input_len) {
            value = attention_key_at(kv,
                                     attention_cmp_k_cache,
                                     attention_cmp_cu_lens,
                                     attention_cmp_k_pool,
                                     attention_cmp_block_table,
                                     attention_cmp_seq_lens,
                                     attention_cmp_pool_block_size,
                                     attention_cmp_pool_block_stride,
                                     attention_cmp_block_table_stride,
                                     attention_swa_k_cache,
                                     attention_swa_cu_lens,
                                     attention_swa_k_pool,
                                     attention_swa_slot_mapping,
                                     attention_swa_gather_lens,
                                     attention_swa_pool_block_size,
                                     attention_swa_pool_block_stride,
                                     attention_swa_slot_mapping_stride,
                                     0,
                                     req,
                                     key_pos,
                                     prefix_len,
                                     0,
                                     d,
                                     L,
                                     KH,
                                     D,
                                     kv_unpad_restore,
                                     kv_cu_lens,
                                     nullptr,
                                     0,
                                     0,
                                     0,
                                     0);
        }
    }
    workspace[idx] = from_float_device<scalar_t>(value);
}

template<typename scalar_t>
__global__ void dsv4BuildFlashMlaIndicesKernel(const scalar_t* __restrict__ indexer_q,
                                               const scalar_t* __restrict__ indexer_k,
                                               const int64_t* __restrict__ req_id_per_token,
                                               const int64_t* __restrict__ position_ids,
                                               const int64_t* __restrict__ prefix_lengths,
                                               const int64_t* __restrict__ input_lengths,
                                               const int64_t* __restrict__ local_rows,
                                               int32_t* __restrict__ indices,
                                               int32_t* __restrict__ topk_lens,
                                               int R,
                                               int M,
                                               int L,
                                               int IH,
                                               int ID,
                                               int LI,
                                               int compress_ratio,
                                               int window_size,
                                               int compressed_topk,
                                               int compressed_region_width,
                                               int key_width,
                                               const uint8_t* __restrict__ csa_indexer_k_cache,
                                               const float* __restrict__ csa_indexer_weights,
                                               const int64_t* __restrict__ csa_indexer_cu_lens,
                                               int csa_indexer_total_len,
                                               const uint8_t* __restrict__ csa_indexer_k_pool,
                                               const int32_t* __restrict__ csa_indexer_block_table,
                                               const int32_t* __restrict__ csa_indexer_seq_lens,
                                               int csa_indexer_pool_block_size,
                                               int64_t csa_indexer_pool_block_stride,
                                               int64_t csa_indexer_block_table_stride,
                                               const int32_t* __restrict__ attention_cmp_seq_lens,
                                               const int32_t* __restrict__ attention_swa_gather_lens) {
    const int local_row = blockIdx.x;
    const int tid = threadIdx.x;
    if (local_row >= R) {
        return;
    }
    for (int i = tid; i < key_width; i += blockDim.x) {
        indices[static_cast<int64_t>(local_row) * key_width + i] = -1;
    }
    __syncthreads();
    if (tid != 0) {
        return;
    }

    const int64_t row = local_rows[local_row];
    const int req = static_cast<int>(req_id_per_token[row]);
    const int q_pos = static_cast<int>(position_ids[row]);
    const int prefix_len = static_cast<int>(prefix_lengths[req]);
    const int kv_len = static_cast<int>(prefix_lengths[req] + input_lengths[req]);
    int key_count = 0;

    if (compress_ratio == 128) {
        int valid_bound = min_int((q_pos + 1) / compress_ratio, compressed_region_width);
        if (attention_cmp_seq_lens != nullptr) {
            valid_bound = min_int(valid_bound, static_cast<int>(attention_cmp_seq_lens[req]));
        }
        for (int c = 0; c < valid_bound && key_count < key_width; ++c) {
            const int boundary_pos = (c + 1) * compress_ratio - 1;
            if (boundary_pos >= 0 && boundary_pos < kv_len) {
                indices[static_cast<int64_t>(local_row) * key_width + key_count] =
                    static_cast<int32_t>(req * M + c);
                ++key_count;
            }
        }
    } else {
        const bool use_fp8_indexer_pool = csa_indexer_k_pool != nullptr && csa_indexer_block_table != nullptr
                                          && csa_indexer_weights != nullptr;
        const bool use_fp8_indexer = use_fp8_indexer_pool
                                     || (csa_indexer_k_cache != nullptr && csa_indexer_weights != nullptr
                                         && csa_indexer_cu_lens != nullptr);
        int req_k_len = LI;
        if (use_fp8_indexer_pool) {
            req_k_len = csa_indexer_seq_lens == nullptr ? LI : static_cast<int>(csa_indexer_seq_lens[req]);
        } else if (use_fp8_indexer) {
            req_k_len = static_cast<int>(csa_indexer_cu_lens[req + 1] - csa_indexer_cu_lens[req]);
            req_k_len = min_int(req_k_len, csa_indexer_total_len);
        }
        const int valid = min_int((q_pos + 1) / compress_ratio, req_k_len);
        const int k_eff = min_int(valid, compressed_topk);
        for (int kth = 0; kth < k_eff && key_count < key_width; ++kth) {
            float best_score = -3.402823466e38F;
            int best_idx = -1;
            for (int c = 0; c < valid; ++c) {
                bool used = false;
                for (int prev = 0; prev < kth; ++prev) {
                    const int32_t prev_global = indices[static_cast<int64_t>(local_row) * key_width + prev];
                    if (prev_global == static_cast<int32_t>(req * M + c)) {
                        used = true;
                    }
                }
                if (used) {
                    continue;
                }
                float score = 0.0f;
                if (use_fp8_indexer_pool) {
                    score = indexer_score_fp8_pool(indexer_q,
                                                   csa_indexer_k_pool,
                                                   csa_indexer_block_table,
                                                   csa_indexer_weights,
                                                   row,
                                                   req,
                                                   c,
                                                   IH,
                                                   ID,
                                                   csa_indexer_pool_block_size,
                                                   csa_indexer_pool_block_stride,
                                                   csa_indexer_block_table_stride);
                } else if (use_fp8_indexer) {
                    score = indexer_score_fp8_cache(
                        indexer_q, csa_indexer_k_cache, csa_indexer_weights, csa_indexer_cu_lens, row, req, c, IH, ID);
                } else {
                    score = indexer_score(
                        indexer_q, indexer_k, row, req, c, IH, ID, LI, nullptr, 0, 0, 0, 0);
                }
                if (score > best_score || (score == best_score && (best_idx < 0 || c < best_idx))) {
                    best_score = score;
                    best_idx = c;
                }
            }
            if (best_idx >= 0) {
                const int boundary_pos = (best_idx + 1) * compress_ratio - 1;
                if (boundary_pos >= 0 && boundary_pos < kv_len) {
                    indices[static_cast<int64_t>(local_row) * key_width + key_count] =
                        static_cast<int32_t>(req * M + best_idx);
                    ++key_count;
                }
            }
        }
    }

    const int gather_len = attention_swa_gather_lens == nullptr ? min_int(prefix_len, window_size) :
                                                                  static_cast<int>(attention_swa_gather_lens[req]);
    const int swa_base = compressed_region_width;
    const int swa_start = max_int(0, q_pos - window_size + 1);
    const int swa_end = min_int(q_pos + 1, kv_len);
    const int workspace_swa_start = prefix_len - gather_len;
    for (int pos = swa_start; pos < swa_end && key_count < key_width; ++pos) {
        const int rel = pos - workspace_swa_start;
        if (rel >= 0 && swa_base + rel < M) {
            indices[static_cast<int64_t>(local_row) * key_width + key_count] =
                static_cast<int32_t>(req * M + swa_base + rel);
            ++key_count;
        }
    }
    topk_lens[local_row] = static_cast<int32_t>(key_count);
}

__global__ void dsv4GatherIndexerKvForDeepGemmKernel(const uint8_t* __restrict__ pool,
                                                     const int32_t* __restrict__ block_table,
                                                     const int32_t* __restrict__ seq_lens,
                                                     uint8_t* __restrict__ k_quant,
                                                     float* __restrict__ k_scale,
                                                     int B,
                                                     int rows_per_req,
                                                     int block_size,
                                                     int64_t block_stride,
                                                     int64_t block_table_stride) {
    const int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const int64_t total = static_cast<int64_t>(B) * rows_per_req * kIndexerHeadDim;
    if (idx >= total) {
        return;
    }
    const int d = static_cast<int>(idx % kIndexerHeadDim);
    const int64_t row = idx / kIndexerHeadDim;
    const int req = static_cast<int>(row / rows_per_req);
    const int c = static_cast<int>(row - static_cast<int64_t>(req) * rows_per_req);
    const int64_t out_row = static_cast<int64_t>(req) * rows_per_req + c;
    if (c >= seq_lens[req]) {
        k_quant[out_row * kIndexerHeadDim + d] = 0;
        if (d == 0) {
            k_scale[out_row] = 1.0f;
        }
        return;
    }
    const int64_t block_row = c / block_size;
    const int64_t in_block = c - block_row * block_size;
    const int32_t block_id = block_table[static_cast<int64_t>(req) * block_table_stride + block_row];
    if (block_id < 0) {
        k_quant[out_row * kIndexerHeadDim + d] = 0;
        if (d == 0) {
            k_scale[out_row] = 1.0f;
        }
        return;
    }
    const uint8_t* block = pool + static_cast<int64_t>(block_id) * block_stride;
    const uint8_t* token_ptr = block + in_block * kIndexerHeadDim;
    const uint8_t* scale_ptr = block + static_cast<int64_t>(block_size) * kIndexerHeadDim + in_block * 4;
    k_quant[out_row * kIndexerHeadDim + d] = token_ptr[d];
    if (d == 0) {
        k_scale[out_row] = load_f32_unaligned(scale_ptr);
    }
}

__global__ void dsv4BuildIndexerDeepGemmRangesKernel(const int64_t* __restrict__ req_id_per_token,
                                                     const int64_t* __restrict__ position_ids,
                                                     const int64_t* __restrict__ local_rows,
                                                     const int32_t* __restrict__ seq_lens,
                                                     int32_t* __restrict__ row_starts,
                                                     int32_t* __restrict__ row_ends,
                                                     int R,
                                                     int rows_per_req,
                                                     int compress_ratio) {
    const int r = static_cast<int>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (r >= R) {
        return;
    }
    const int64_t row = local_rows[r];
    const int req = static_cast<int>(req_id_per_token[row]);
    const int q_pos = static_cast<int>(position_ids[row]);
    const int base = req * rows_per_req;
    const int visible = min_int((q_pos + 1) / compress_ratio, static_cast<int>(seq_lens[req]));
    row_starts[r] = base;
    row_ends[r] = base + visible;
}

__global__ void dsv4BuildFlashMlaIndicesFromTopkKernel(const int32_t* __restrict__ topk_local,
                                                       const int32_t* __restrict__ row_starts,
                                                       const int32_t* __restrict__ row_ends,
                                                       const int64_t* __restrict__ req_id_per_token,
                                                       const int64_t* __restrict__ position_ids,
                                                       const int64_t* __restrict__ prefix_lengths,
                                                       const int64_t* __restrict__ input_lengths,
                                                       const int64_t* __restrict__ local_rows,
                                                       const int32_t* __restrict__ attention_swa_gather_lens,
                                                       int32_t* __restrict__ indices,
                                                       int32_t* __restrict__ topk_lens,
                                                       int R,
                                                       int M,
                                                       int key_width,
                                                       int compressed_topk,
                                                       int compressed_region_width,
                                                       int window_size) {
    const int r = static_cast<int>(blockIdx.x);
    const int tid = threadIdx.x;
    if (r >= R) {
        return;
    }
    for (int i = tid; i < key_width; i += blockDim.x) {
        indices[static_cast<int64_t>(r) * key_width + i] = -1;
    }
    __syncthreads();
    const int64_t row = local_rows[r];
    const int req = static_cast<int>(req_id_per_token[row]);
    const int q_pos = static_cast<int>(position_ids[row]);
    const int prefix_len = static_cast<int>(prefix_lengths[req]);
    const int kv_len = static_cast<int>(prefix_lengths[req] + input_lengths[req]);
    const int visible_compressed = max_int(0, row_ends[r] - row_starts[r]);
    const int cmp_count = min_int(min_int(visible_compressed, compressed_topk), compressed_region_width);
    for (int kth = tid; kth < cmp_count; kth += blockDim.x) {
        const int local_idx = topk_local[static_cast<int64_t>(r) * compressed_topk + kth];
        indices[static_cast<int64_t>(r) * key_width + kth] =
            (local_idx >= 0 && local_idx < compressed_region_width) ? static_cast<int32_t>(req * M + local_idx) :
                                                                      -1;
    }

    const int gather_len = attention_swa_gather_lens == nullptr ? min_int(prefix_len, window_size) :
                                                                  static_cast<int>(attention_swa_gather_lens[req]);
    const int swa_base = compressed_region_width;
    const int swa_start = max_int(0, q_pos - window_size + 1);
    const int swa_end = min_int(q_pos + 1, kv_len);
    const int workspace_swa_start = prefix_len - gather_len;
    const int first_swa_pos = max_int(swa_start, workspace_swa_start);
    const int max_workspace_swa_end = workspace_swa_start + max_int(0, M - swa_base);
    const int last_swa_pos_exclusive = min_int(swa_end, max_workspace_swa_end);
    const int swa_count = max_int(0, last_swa_pos_exclusive - first_swa_pos);
    for (int off = tid; off < swa_count; off += blockDim.x) {
        const int pos = first_swa_pos + off;
        const int rel = pos - workspace_swa_start;
        indices[static_cast<int64_t>(r) * key_width + cmp_count + off] =
            static_cast<int32_t>(req * M + swa_base + rel);
    }
    if (tid == 0) {
        topk_lens[r] = static_cast<int32_t>(cmp_count + swa_count);
    }
}

void validateTensor(const torch::Tensor& t, const char* name, c10::ScalarType dtype, int64_t dim) {
    TORCH_CHECK(t.is_cuda(), name, " must be a CUDA tensor");
    TORCH_CHECK(t.scalar_type() == dtype, name, " has wrong dtype");
    TORCH_CHECK(t.dim() == dim, name, " has wrong rank");
    TORCH_CHECK(t.is_contiguous(), name, " must be contiguous");
}

void validateFloatPayloadTensor(const torch::Tensor& t, const char* name, int64_t dim) {
    TORCH_CHECK(t.is_cuda(), name, " must be a CUDA tensor");
    TORCH_CHECK(t.scalar_type() == torch::kFloat32 || t.scalar_type() == torch::kBFloat16,
                name,
                " must be float32 or bfloat16");
    TORCH_CHECK(t.dim() == dim, name, " has wrong rank");
    TORCH_CHECK(t.is_contiguous(), name, " must be contiguous");
}

void validateProductionAbi(int64_t                    cp_rank,
                           int64_t                    cp_size,
                           int64_t                    comm_ptr,
                           int64_t                    buffer_handle,
                           int64_t                    signal_handle,
                           int64_t                    per_rank_buffer_bytes,
                           const std::vector<int64_t>& rank_offsets,
                           bool                        has_symm_backend) {
    TORCH_CHECK(cp_size >= 1, "cp_size must be positive");
    TORCH_CHECK(cp_rank >= 0 && cp_rank < cp_size, "cp_rank must be in [0, cp_size)");
    if (has_symm_backend) {
        TORCH_CHECK(cp_size == 8, "production distributed attention V0 requires cp_size=8");
        TORCH_CHECK(comm_ptr == 0, "symm-memory attention path must not also pass UserBuffers comm_ptr");
        TORCH_CHECK(buffer_handle < 0, "buffer_handle must be omitted for symm-memory attention path");
        TORCH_CHECK(signal_handle < 0, "signal_handle must be omitted for symm-memory attention path");
        TORCH_CHECK(per_rank_buffer_bytes > 0, "per_rank_buffer_bytes must be positive for symm-memory path");
        TORCH_CHECK(static_cast<int64_t>(rank_offsets.size()) == cp_size,
                    "rank_offsets length must equal cp_size for symm-memory path");
        return;
    }
    if (comm_ptr == 0) {
        TORCH_CHECK(cp_size == 1, "semantic no-communication mode requires cp_size=1");
        TORCH_CHECK(buffer_handle < 0,
                    "buffer_handle must be omitted when comm_ptr is 0; "
                    "semantic no-communication mode does not own a buffer");
        TORCH_CHECK(signal_handle < 0,
                    "signal_handle must be omitted when comm_ptr is 0; "
                    "semantic no-communication mode does not own a signal buffer");
        TORCH_CHECK(per_rank_buffer_bytes == 0,
                    "per_rank_buffer_bytes must be 0 in semantic no-communication mode");
        TORCH_CHECK(rank_offsets.empty(), "rank_offsets must be empty in semantic no-communication mode");
        return;
    }

    TORCH_CHECK(cp_size == 8, "production distributed attention V0 requires cp_size=8");
    TORCH_CHECK(buffer_handle >= 0, "buffer_handle must be non-negative when comm_ptr is set");
    TORCH_CHECK(signal_handle >= 0, "signal_handle must be non-negative when comm_ptr is set");
    TORCH_CHECK(per_rank_buffer_bytes > 0, "per_rank_buffer_bytes must be positive when comm_ptr is set");
    TORCH_CHECK(static_cast<int64_t>(rank_offsets.size()) == cp_size,
                "rank_offsets length must equal cp_size when comm_ptr is set");
    for (int64_t rank = 0; rank < cp_size; ++rank) {
        TORCH_CHECK(rank_offsets[rank] == rank * per_rank_buffer_bytes,
                    "rank_offsets must be rank * per_rank_buffer_bytes for V0, got rank_offsets[",
                    rank,
                    "]=",
                    rank_offsets[rank]);
        TORCH_CHECK(rank_offsets[rank] + per_rank_buffer_bytes <= per_rank_buffer_bytes * cp_size,
                    "rank_offsets[",
                    rank,
                    "] exceeds registered symmetric buffer size");
    }
}

torch::Tensor launchDsv4CpDistributedPrefillAttention(const torch::Tensor& q,
                                                      const torch::Tensor& kv,
                                                      const torch::Tensor& indexer_q,
                                                      const torch::Tensor& indexer_k,
                                                      const torch::Tensor& attn_sink,
                                                      const torch::Tensor& req_id_per_token,
                                                      const torch::Tensor& position_ids,
                                                      const torch::Tensor& prefix_lengths,
                                                      const torch::Tensor& input_lengths,
                                                      const torch::Tensor& local_rows,
                                                      int64_t              compress_ratio,
                                                      int64_t              window_size,
                                                      int64_t              compressed_topk,
                                                      int64_t              compressed_region_width,
                                                      int64_t              cp_rank,
                                                      int64_t              cp_size,
                                                      int64_t              comm_ptr,
                                                      int64_t              buffer_handle,
                                                      int64_t              signal_handle,
                                                      int64_t              per_rank_buffer_bytes,
                                                      const std::vector<int64_t>& rank_offsets,
                                                      const std::optional<torch::Tensor>& swa_k,
                                                      const std::optional<torch::Tensor>& swa_k_cache,
                                                      const std::optional<torch::Tensor>& swa_slot_mapping,
                                                      const std::optional<torch::Tensor>& symm_buffer,
                                                      int64_t symm_buffer_ptrs_dev,
                                                      int64_t symm_signal_pad_ptrs_dev,
                                                      const py::object& symm_handle,
                                                      const std::optional<torch::Tensor>& compressor_kv,
                                                      const std::optional<torch::Tensor>& compressor_score,
                                                      const std::optional<torch::Tensor>& compressor_ape,
                                                      const std::optional<torch::Tensor>& compressor_positions,
                                                      const std::optional<torch::Tensor>& compressor_state_cache,
                                                      const std::optional<torch::Tensor>& compressor_state_slots,
                                                      int64_t compressor_ratio,
                                                      const std::optional<torch::Tensor>& compressor_token_to_req,
                                                      const std::optional<torch::Tensor>& compressor_state_block_table,
                                                      const std::optional<torch::Tensor>& compressor_norm_weight,
                                                      const std::optional<torch::Tensor>& compressor_cos_sin_cache,
                                                      const std::optional<torch::Tensor>& compressor_kv_cache,
                                                      const std::optional<torch::Tensor>& compressor_kv_slots,
                                                      int64_t compressor_seq_start,
                                                      bool compressor_disable_raw_path,
                                                      double compressor_rms_norm_eps,
                                                      int64_t compressor_head_dim,
                                                      int64_t compressor_rope_head_dim,
                                                      bool compressor_overlap,
                                                      int64_t compressor_state_tokens_per_block,
                                                          const std::optional<torch::Tensor>& compressor_seq_start_per_req,
                                                          const std::optional<torch::Tensor>& compressor_cu_seq_per_req,
                                                          const std::optional<torch::Tensor>& compressor_unpad_restore,
                                                          const std::optional<torch::Tensor>& csa_indexer_compressor_kv,
                                                          const std::optional<torch::Tensor>& csa_indexer_compressor_score,
                                                          const std::optional<torch::Tensor>& csa_indexer_compressor_ape,
                                                          const std::optional<torch::Tensor>& csa_indexer_compressor_positions,
                                                          const std::optional<torch::Tensor>& csa_indexer_compressor_state_cache,
                                                          const std::optional<torch::Tensor>& csa_indexer_compressor_state_slots,
                                                          int64_t csa_indexer_compressor_ratio,
                                                          const std::optional<torch::Tensor>& csa_indexer_compressor_token_to_req,
                                                          const std::optional<torch::Tensor>& csa_indexer_compressor_state_block_table,
                                                          const std::optional<torch::Tensor>& csa_indexer_compressor_norm_weight,
                                                          const std::optional<torch::Tensor>& csa_indexer_compressor_cos_sin_cache,
                                                          const std::optional<torch::Tensor>& csa_indexer_compressor_kv_cache,
                                                          const std::optional<torch::Tensor>& csa_indexer_compressor_kv_slots,
                                                          int64_t csa_indexer_compressor_seq_start,
                                                          bool csa_indexer_compressor_disable_raw_path,
                                                          double csa_indexer_compressor_rms_norm_eps,
                                                          int64_t csa_indexer_compressor_head_dim,
                                                          int64_t csa_indexer_compressor_rope_head_dim,
                                                          bool csa_indexer_compressor_overlap,
                                                          int64_t csa_indexer_compressor_state_tokens_per_block,
                                                          const std::optional<torch::Tensor>& csa_indexer_compressor_seq_start_per_req,
                                                          const std::optional<torch::Tensor>& csa_indexer_compressor_cu_seq_per_req,
                                                          const std::optional<torch::Tensor>& csa_indexer_compressor_unpad_restore,
                                                          const std::optional<torch::Tensor>& csa_indexer_k_cache,
                                                      const std::optional<torch::Tensor>& csa_indexer_weights,
                                                      const std::optional<torch::Tensor>& csa_indexer_cu_lens,
                                                      const std::optional<torch::Tensor>& csa_indexer_k_pool,
                                                      const std::optional<torch::Tensor>& csa_indexer_block_table,
                                                      const std::optional<torch::Tensor>& csa_indexer_seq_lens,
                                                      const std::optional<torch::Tensor>& attention_cmp_k_cache,
                                                      const std::optional<torch::Tensor>& attention_cmp_cu_lens,
                                                      const std::optional<torch::Tensor>& attention_cmp_k_pool,
                                                      const std::optional<torch::Tensor>& attention_cmp_block_table,
                                                      const std::optional<torch::Tensor>& attention_cmp_seq_lens,
                                                      const std::optional<torch::Tensor>& attention_swa_k_cache,
                                                      const std::optional<torch::Tensor>& attention_swa_cu_lens,
                                                      const std::optional<torch::Tensor>& attention_swa_k_pool,
                                                      const std::optional<torch::Tensor>& attention_swa_slot_mapping,
                                                      const std::optional<torch::Tensor>& attention_swa_gather_lens,
                                                      const std::optional<torch::Tensor>& kv_unpad_restore,
                                                      const std::optional<torch::Tensor>& kv_cu_lens) {
    validateFloatPayloadTensor(q, "q", 3);
    validateFloatPayloadTensor(kv, "kv", 4);
    validateFloatPayloadTensor(indexer_q, "indexer_q", 3);
    validateFloatPayloadTensor(indexer_k, "indexer_k", 4);
    validateTensor(attn_sink, "attn_sink", torch::kFloat32, 1);
    validateTensor(req_id_per_token, "req_id_per_token", torch::kInt64, 1);
    validateTensor(position_ids, "position_ids", torch::kInt64, 1);
    validateTensor(prefix_lengths, "prefix_lengths", torch::kInt64, 1);
    validateTensor(input_lengths, "input_lengths", torch::kInt64, 1);
    validateTensor(local_rows, "local_rows", torch::kInt64, 1);

    TORCH_CHECK(compress_ratio == 0 || compress_ratio == 4 || compress_ratio == 128,
                "compress_ratio must be 0, 4, or 128");
    TORCH_CHECK(window_size >= 0, "window_size must be non-negative");
    TORCH_CHECK(compressed_topk >= 0, "compressed_topk must be non-negative");
    if (compressed_region_width < 0) {
        compressed_region_width = compressed_topk;
    }
    TORCH_CHECK(compressed_region_width >= 0, "compressed_region_width must be non-negative");
    validateProductionAbi(cp_rank,
                          cp_size,
                          comm_ptr,
                          buffer_handle,
                          signal_handle,
                          per_rank_buffer_bytes,
                          rank_offsets,
                          hasSymmMemBackend(symm_buffer, symm_buffer_ptrs_dev));
    TORCH_CHECK(compressed_topk <= kMaxCompressedTopK, "compressed_topk exceeds semantic mega kernel capacity");

    const int64_t T = q.size(0);
    const int64_t H = q.size(1);
    const int64_t D = q.size(2);
    const bool has_kv_restore = kv_unpad_restore.has_value() || kv_cu_lens.has_value();
    const int64_t logical_batch = has_kv_restore ? input_lengths.size(0) : kv.size(0);
    TORCH_CHECK(kv.scalar_type() == q.scalar_type(), "kv dtype must match q dtype");
    TORCH_CHECK(indexer_q.scalar_type() == q.scalar_type(), "indexer_q dtype must match q dtype");
    TORCH_CHECK(indexer_k.scalar_type() == q.scalar_type(), "indexer_k dtype must match q dtype");
    TORCH_CHECK(D <= kSwaHeadDim, "DSV4 mega attention supports head_dim <= 512");
    TORCH_CHECK((kv.size(2) == H || kv.size(2) == 1) && kv.size(3) == D,
                "kv shape must match q head_dim and have either q heads or one MQA head");
    TORCH_CHECK(attn_sink.size(0) == H, "attn_sink size must match q heads");
    TORCH_CHECK(req_id_per_token.size(0) == T, "req_id_per_token size must match q rows");
    TORCH_CHECK(position_ids.size(0) == T, "position_ids size must match q rows");
    if (!has_kv_restore) {
        TORCH_CHECK(prefix_lengths.size(0) == kv.size(0), "prefix_lengths size must match kv batch");
        TORCH_CHECK(input_lengths.size(0) == kv.size(0), "input_lengths size must match kv batch");
    } else {
        TORCH_CHECK(prefix_lengths.size(0) == input_lengths.size(0),
                    "prefix_lengths size must match input_lengths in fresh-K restore mode");
    }
    TORCH_CHECK(indexer_q.size(0) == T, "indexer_q rows must match q rows");
    TORCH_CHECK(indexer_k.size(0) == logical_batch, "indexer_k batch must match logical batch");
    TORCH_CHECK(indexer_q.size(1) == indexer_k.size(2), "indexer head count mismatch");
    TORCH_CHECK(indexer_q.size(2) == indexer_k.size(3), "indexer head dim mismatch");
    const bool has_csa_fp8_indexer = csa_indexer_k_cache.has_value() || csa_indexer_cu_lens.has_value();
    if (has_csa_fp8_indexer) {
        TORCH_CHECK(compress_ratio == 4, "csa_indexer_* tensors are only valid for CSA compress_ratio=4");
        TORCH_CHECK(csa_indexer_k_cache.has_value() && csa_indexer_weights.has_value()
                        && csa_indexer_cu_lens.has_value(),
                    "csa_indexer_k_cache, csa_indexer_weights, and csa_indexer_cu_lens must be provided together");
        validateTensor(*csa_indexer_k_cache, "csa_indexer_k_cache", torch::kUInt8, 2);
        validateTensor(*csa_indexer_weights, "csa_indexer_weights", torch::kFloat32, 2);
        validateTensor(*csa_indexer_cu_lens, "csa_indexer_cu_lens", torch::kInt64, 1);
        TORCH_CHECK(csa_indexer_k_cache->size(1) == kIndexerEntryBytes,
                    "csa_indexer_k_cache must expose flat 132B INDEXER_KV rows");
        TORCH_CHECK(csa_indexer_weights->size(0) == T, "csa_indexer_weights rows must match q rows");
        TORCH_CHECK(csa_indexer_weights->size(1) == indexer_q.size(1),
                    "csa_indexer_weights columns must match indexer_q heads");
        TORCH_CHECK(csa_indexer_cu_lens->size(0) == logical_batch + 1,
                    "csa_indexer_cu_lens must be [batch + 1]");
    }
    const bool has_csa_fp8_indexer_pool = csa_indexer_k_pool.has_value() || csa_indexer_block_table.has_value()
                                          || csa_indexer_seq_lens.has_value();
    if (has_csa_fp8_indexer_pool) {
        TORCH_CHECK(compress_ratio == 4, "csa_indexer_pool tensors are only valid for CSA compress_ratio=4");
        TORCH_CHECK(csa_indexer_k_pool.has_value() && csa_indexer_weights.has_value()
                        && csa_indexer_block_table.has_value() && csa_indexer_seq_lens.has_value(),
                    "csa_indexer_k_pool, csa_indexer_weights, csa_indexer_block_table, and "
                    "csa_indexer_seq_lens must be provided together");
        const torch::Tensor& pool = *csa_indexer_k_pool;
        TORCH_CHECK(pool.is_cuda(), "csa_indexer_k_pool must be a CUDA tensor");
        TORCH_CHECK(pool.scalar_type() == torch::kUInt8, "csa_indexer_k_pool must be uint8");
        TORCH_CHECK(pool.dim() == 3 && pool.size(2) == kIndexerEntryBytes,
                    "csa_indexer_k_pool must be [num_blocks, block_size, 132]");
        TORCH_CHECK(pool.stride(2) == 1 && pool.stride(1) == kIndexerEntryBytes,
                    "csa_indexer_k_pool must have stride[1]=132 and stride[2]=1");
        validateTensor(*csa_indexer_weights, "csa_indexer_weights", torch::kFloat32, 2);
        validateTensor(*csa_indexer_block_table, "csa_indexer_block_table", torch::kInt32, 2);
        validateTensor(*csa_indexer_seq_lens, "csa_indexer_seq_lens", torch::kInt32, 1);
        TORCH_CHECK(csa_indexer_weights->size(0) == T, "csa_indexer_weights rows must match q rows");
        TORCH_CHECK(csa_indexer_weights->size(1) == indexer_q.size(1),
                    "csa_indexer_weights columns must match indexer_q heads");
        TORCH_CHECK(csa_indexer_block_table->size(0) == logical_batch,
                    "csa_indexer_block_table batch must match logical batch");
        TORCH_CHECK(csa_indexer_seq_lens->size(0) == logical_batch,
                    "csa_indexer_seq_lens length must match logical batch");
    }
    const bool has_attention_cmp_cache = attention_cmp_k_cache.has_value() || attention_cmp_cu_lens.has_value();
    if (has_attention_cmp_cache) {
        TORCH_CHECK(attention_cmp_k_cache.has_value() && attention_cmp_cu_lens.has_value(),
                    "attention_cmp_k_cache and attention_cmp_cu_lens must be provided together");
        validateTensor(*attention_cmp_k_cache, "attention_cmp_k_cache", torch::kUInt8, 2);
        validateTensor(*attention_cmp_cu_lens, "attention_cmp_cu_lens", torch::kInt64, 1);
        TORCH_CHECK(attention_cmp_k_cache->size(1) == kSwaEntryBytes,
                    "attention_cmp_k_cache must expose flat 584B compressed-K rows");
        TORCH_CHECK(attention_cmp_cu_lens->size(0) == logical_batch + 1,
                    "attention_cmp_cu_lens must be [batch + 1]");
        TORCH_CHECK(D <= kSwaHeadDim, "attention_cmp_k_cache only supports head_dim <= 512");
    }
    const bool has_attention_cmp_pool = attention_cmp_k_pool.has_value() || attention_cmp_block_table.has_value()
                                        || attention_cmp_seq_lens.has_value();
    if (has_attention_cmp_pool) {
        TORCH_CHECK(attention_cmp_k_pool.has_value() && attention_cmp_block_table.has_value()
                        && attention_cmp_seq_lens.has_value(),
                    "attention_cmp_k_pool, attention_cmp_block_table, and attention_cmp_seq_lens must be provided together");
        const torch::Tensor& pool = *attention_cmp_k_pool;
        TORCH_CHECK(pool.is_cuda(), "attention_cmp_k_pool must be a CUDA tensor");
        TORCH_CHECK(pool.scalar_type() == torch::kUInt8, "attention_cmp_k_pool must be uint8");
        TORCH_CHECK(pool.dim() == 3 && pool.size(2) == kSwaEntryBytes,
                    "attention_cmp_k_pool must be [num_blocks, block_size, 584]");
        TORCH_CHECK(pool.stride(2) == 1 && pool.stride(1) == kSwaEntryBytes,
                    "attention_cmp_k_pool must have stride[1]=584 and stride[2]=1");
        validateTensor(*attention_cmp_block_table, "attention_cmp_block_table", torch::kInt32, 2);
        validateTensor(*attention_cmp_seq_lens, "attention_cmp_seq_lens", torch::kInt32, 1);
        TORCH_CHECK(attention_cmp_block_table->size(0) == logical_batch,
                    "attention_cmp_block_table batch must match logical batch");
        TORCH_CHECK(attention_cmp_seq_lens->size(0) == logical_batch,
                    "attention_cmp_seq_lens length must match logical batch");
        TORCH_CHECK(D <= kSwaHeadDim, "attention_cmp_k_pool only supports head_dim <= 512");
    }
    const bool has_attention_swa_cache = attention_swa_k_cache.has_value() || attention_swa_cu_lens.has_value();
    if (has_attention_swa_cache) {
        TORCH_CHECK(attention_swa_k_cache.has_value() && attention_swa_cu_lens.has_value(),
                    "attention_swa_k_cache and attention_swa_cu_lens must be provided together");
        validateTensor(*attention_swa_k_cache, "attention_swa_k_cache", torch::kUInt8, 2);
        validateTensor(*attention_swa_cu_lens, "attention_swa_cu_lens", torch::kInt64, 1);
        TORCH_CHECK(attention_swa_k_cache->size(1) == kSwaEntryBytes,
                    "attention_swa_k_cache must expose flat 584B SWA_KV rows");
        TORCH_CHECK(attention_swa_cu_lens->size(0) == logical_batch + 1,
                    "attention_swa_cu_lens must be [batch + 1]");
        TORCH_CHECK(D <= kSwaHeadDim, "attention_swa_k_cache only supports head_dim <= 512");
    }
    const bool has_attention_swa_pool = attention_swa_k_pool.has_value() || attention_swa_slot_mapping.has_value()
                                        || attention_swa_gather_lens.has_value();
    if (has_attention_swa_pool) {
        TORCH_CHECK(attention_swa_k_pool.has_value() && attention_swa_slot_mapping.has_value()
                        && attention_swa_gather_lens.has_value(),
                    "attention_swa_k_pool, attention_swa_slot_mapping, and attention_swa_gather_lens must be provided together");
        const torch::Tensor& pool = *attention_swa_k_pool;
        TORCH_CHECK(pool.is_cuda(), "attention_swa_k_pool must be a CUDA tensor");
        TORCH_CHECK(pool.scalar_type() == torch::kUInt8, "attention_swa_k_pool must be uint8");
        TORCH_CHECK(pool.dim() == 3 && pool.size(2) == kSwaEntryBytes,
                    "attention_swa_k_pool must be [num_blocks, block_size, 584]");
        TORCH_CHECK(pool.stride(2) == 1 && pool.stride(1) == kSwaEntryBytes,
                    "attention_swa_k_pool must have stride[1]=584 and stride[2]=1");
        validateTensor(*attention_swa_slot_mapping, "attention_swa_slot_mapping", torch::kInt64, 2);
        validateTensor(*attention_swa_gather_lens, "attention_swa_gather_lens", torch::kInt32, 1);
        TORCH_CHECK(attention_swa_slot_mapping->size(0) == logical_batch,
                    "attention_swa_slot_mapping batch must match logical batch");
        TORCH_CHECK(attention_swa_gather_lens->size(0) == logical_batch,
                    "attention_swa_gather_lens length must match logical batch");
        TORCH_CHECK(D <= kSwaHeadDim, "attention_swa_k_pool only supports head_dim <= 512");
    }
    if (has_kv_restore) {
        TORCH_CHECK(kv_unpad_restore.has_value() && kv_cu_lens.has_value(),
                    "kv_unpad_restore and kv_cu_lens must be provided together");
        validateTensor(*kv_unpad_restore, "kv_unpad_restore", torch::kInt64, 1);
        validateTensor(*kv_cu_lens, "kv_cu_lens", torch::kInt64, 1);
        TORCH_CHECK(kv.size(0) == 1,
                    "kv_unpad_restore fresh-K mode expects gathered KV in a single flat batch dimension");
        TORCH_CHECK(kv_cu_lens->size(0) == input_lengths.size(0) + 1,
                    "kv_cu_lens must be [batch + 1]");
    }

    const c10::cuda::CUDAGuard device_guard(q.device());
    const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    const bool debug_sync = dsv4CpAttentionDebugSyncEnabled();
    if (!hasSymmMemBackend(symm_buffer, symm_buffer_ptrs_dev)) {
        debugStage(debug_sync, cp_rank, "user_buffer_protocol", "begin", stream, false);
        runUserBufferProtocolExchange(cp_rank,
                                      cp_size,
                                      comm_ptr,
                                      buffer_handle,
                                      per_rank_buffer_bytes,
                                      rank_offsets,
                                      stream);
        debugStage(debug_sync, cp_rank, "user_buffer_protocol", "end", stream, true);
    }
    const bool use_mega_kernel = dsv4CpAttentionMegaKernelEnabled();
    const bool use_symm_backend = hasSymmMemBackend(symm_buffer, symm_buffer_ptrs_dev);
    const bool needs_semantic_indexer_k = compress_ratio == 4 && !has_csa_fp8_indexer && !has_csa_fp8_indexer_pool;
    if (use_mega_kernel && cp_size > 1) {
        TORCH_CHECK(use_symm_backend && symm_signal_pad_ptrs_dev != 0,
                    "DSV4_CP_ATTENTION_MEGA_KERNEL requires symmetric memory and signal pads for CP>1");
    }
    const bool use_symm_direct_fresh_kv = use_mega_kernel && use_symm_backend && has_kv_restore && cp_size > 1;
    const bool use_symm_direct_indexer_k =
        use_mega_kernel && use_symm_backend && needs_semantic_indexer_k && cp_size > 1;
    debugStage(debug_sync, cp_rank, "main_kv_gather", "begin", stream, false);
    torch::Tensor gathered_kv = kv;
    if (use_symm_direct_fresh_kv) {
        const torch::Tensor& symm = *symm_buffer;
        validateSymmBuffer(symm, symm_buffer_ptrs_dev, cp_size, per_rank_buffer_bytes, rank_offsets);
        const int64_t payload_bytes = kv.numel() * kv.element_size();
        TORCH_CHECK(kProtocolBytes + payload_bytes <= per_rank_buffer_bytes,
                    "attention direct mega KV payload exceeds per-rank symmetric buffer: need ",
                    kProtocolBytes + payload_bytes,
                    " bytes, have ",
                    per_rank_buffer_bytes);
    } else {
        gathered_kv = allGather4DPayload(kv,
                                         cp_rank,
                                         cp_size,
                                         comm_ptr,
                                         buffer_handle,
                                         per_rank_buffer_bytes,
                                         rank_offsets,
                                         symm_buffer,
                                         symm_buffer_ptrs_dev,
                                         symm_signal_pad_ptrs_dev,
                                         symm_handle,
                                         0,
                                         stream);
    }
    debugStage(debug_sync, cp_rank, "main_kv_gather", "end", stream, true);
    if (dsv4CpAttentionReturnAfter("main_kv_gather")) {
        return makeZeroAttentionOutput(q, local_rows.size(0), H, D);
    }
    torch::Tensor gathered_indexer_k = indexer_k;
    if (needs_semantic_indexer_k) {
        debugStage(debug_sync, cp_rank, "csa_indexer_k_gather", "begin", stream, false);
        if (use_symm_direct_indexer_k) {
            const torch::Tensor& symm = *symm_buffer;
            validateSymmBuffer(symm, symm_buffer_ptrs_dev, cp_size, per_rank_buffer_bytes, rank_offsets);
            const int64_t payload_bytes = indexer_k.numel() * indexer_k.element_size();
            TORCH_CHECK(kProtocolBytes + payload_bytes <= per_rank_buffer_bytes,
                        "attention direct mega INDEXER_K payload exceeds per-rank symmetric buffer: need ",
                        kProtocolBytes + payload_bytes,
                        " bytes, have ",
                        per_rank_buffer_bytes);
        } else {
            gathered_indexer_k = allGather4DPayload(indexer_k,
                                                    cp_rank,
                                                    cp_size,
                                                    comm_ptr,
                                                    buffer_handle,
                                                    per_rank_buffer_bytes,
                                                    rank_offsets,
                                                    symm_buffer,
                                                    symm_buffer_ptrs_dev,
                                                    symm_signal_pad_ptrs_dev,
                                                    symm_handle,
                                                    1,
                                                    stream);
        }
        debugStage(debug_sync, cp_rank, "csa_indexer_k_gather", "end", stream, true);
        if (dsv4CpAttentionReturnAfter("csa_indexer_k_gather")) {
            return makeZeroAttentionOutput(q, local_rows.size(0), H, D);
        }
    }
    debugStage(debug_sync, cp_rank, "swa_writer", "begin", stream, false);
    if (!use_mega_kernel) {
        writeSwaCacheIfRequested(swa_k,
                                 swa_k_cache,
                                 swa_slot_mapping,
                             cp_rank,
                             cp_size,
                             comm_ptr,
                             buffer_handle,
                             per_rank_buffer_bytes,
                             rank_offsets,
                             symm_buffer,
                             symm_buffer_ptrs_dev,
                                 symm_signal_pad_ptrs_dev,
                                 symm_handle,
                                 stream);
    }
    debugStage(debug_sync, cp_rank, "swa_writer", "end", stream, true);
    if (dsv4CpAttentionReturnAfter("swa_writer")) {
        return makeZeroAttentionOutput(q, local_rows.size(0), H, D);
    }
    debugStage(debug_sync, cp_rank, "csa_indexer_compressor_writer", "begin", stream, false);
    if (!use_mega_kernel) {
        writeCompressorStateIfRequested(csa_indexer_compressor_kv,
                                        csa_indexer_compressor_score,
                                        csa_indexer_compressor_ape,
                                        csa_indexer_compressor_positions,
                                        csa_indexer_compressor_state_cache,
                                        csa_indexer_compressor_state_slots,
                                        csa_indexer_compressor_token_to_req,
                                        csa_indexer_compressor_state_block_table,
                                        csa_indexer_compressor_norm_weight,
                                        csa_indexer_compressor_cos_sin_cache,
                                        csa_indexer_compressor_kv_cache,
                                        csa_indexer_compressor_kv_slots,
                                        cp_rank,
                                        cp_size,
                                        comm_ptr,
                                        buffer_handle,
                                        per_rank_buffer_bytes,
                                        rank_offsets,
                                        symm_buffer,
                                        symm_buffer_ptrs_dev,
                                        symm_signal_pad_ptrs_dev,
                                        symm_handle,
                                        csa_indexer_compressor_ratio,
                                        csa_indexer_compressor_seq_start,
                                        csa_indexer_compressor_disable_raw_path,
                                        csa_indexer_compressor_rms_norm_eps,
                                        csa_indexer_compressor_head_dim,
                                        csa_indexer_compressor_rope_head_dim,
                                        csa_indexer_compressor_overlap,
                                        csa_indexer_compressor_state_tokens_per_block,
                                        csa_indexer_compressor_seq_start_per_req,
                                        csa_indexer_compressor_cu_seq_per_req,
                                        csa_indexer_compressor_unpad_restore,
                                        3,
                                        stream);
    }
    debugStage(debug_sync, cp_rank, "csa_indexer_compressor_writer", "end", stream, true);
    if (dsv4CpAttentionReturnAfter("csa_indexer_compressor_writer")) {
        return makeZeroAttentionOutput(q, local_rows.size(0), H, D);
    }
    debugStage(debug_sync, cp_rank, "main_compressor_writer", "begin", stream, false);
    if (!use_mega_kernel) {
        writeCompressorStateIfRequested(compressor_kv,
                                        compressor_score,
                                        compressor_ape,
                                    compressor_positions,
                                    compressor_state_cache,
                                    compressor_state_slots,
                                    compressor_token_to_req,
                                    compressor_state_block_table,
                                    compressor_norm_weight,
                                    compressor_cos_sin_cache,
                                    compressor_kv_cache,
                                    compressor_kv_slots,
                                    cp_rank,
                                    cp_size,
                                    comm_ptr,
                                    buffer_handle,
                                    per_rank_buffer_bytes,
                                    rank_offsets,
                                    symm_buffer,
                                    symm_buffer_ptrs_dev,
                                    symm_signal_pad_ptrs_dev,
                                    symm_handle,
                                    compressor_ratio,
                                    compressor_seq_start,
                                    compressor_disable_raw_path,
                                    compressor_rms_norm_eps,
                                    compressor_head_dim,
                                    compressor_rope_head_dim,
                                    compressor_overlap,
                                        compressor_state_tokens_per_block,
                                        compressor_seq_start_per_req,
                                        compressor_cu_seq_per_req,
                                        compressor_unpad_restore,
                                        5,
                                        stream);
    }
    debugStage(debug_sync, cp_rank, "main_compressor_writer", "end", stream, true);
    if (dsv4CpAttentionReturnAfter("main_compressor_writer")) {
        return makeZeroAttentionOutput(q, local_rows.size(0), H, D);
    }
    MegaSwaWriterArgs mega_swa_writer{};
    MegaCompressorBf16Args mega_csa_indexer_compressor_writer{};
    MegaCompressorBf16Args mega_main_compressor_writer{};
    if (use_mega_kernel) {
        mega_swa_writer = makeMegaSwaWriterArgs(swa_k, swa_k_cache, swa_slot_mapping, cp_size);
        mega_csa_indexer_compressor_writer =
            makeMegaCompressorBf16Args(csa_indexer_compressor_kv,
                                       csa_indexer_compressor_score,
                                       csa_indexer_compressor_ape,
                                       csa_indexer_compressor_positions,
                                       csa_indexer_compressor_state_cache,
                                       csa_indexer_compressor_state_slots,
                                       csa_indexer_compressor_ratio,
                                       csa_indexer_compressor_token_to_req,
                                       csa_indexer_compressor_state_block_table,
                                       csa_indexer_compressor_norm_weight,
                                       csa_indexer_compressor_cos_sin_cache,
                                       csa_indexer_compressor_kv_cache,
                                       csa_indexer_compressor_kv_slots,
                                       csa_indexer_compressor_seq_start,
                                       csa_indexer_compressor_disable_raw_path,
                                       csa_indexer_compressor_rms_norm_eps,
                                       csa_indexer_compressor_head_dim,
                                       csa_indexer_compressor_rope_head_dim,
                                       csa_indexer_compressor_overlap,
                                       csa_indexer_compressor_state_tokens_per_block,
                                       csa_indexer_compressor_seq_start_per_req,
                                       csa_indexer_compressor_cu_seq_per_req,
                                       csa_indexer_compressor_unpad_restore,
                                       cp_size);
        mega_main_compressor_writer =
            makeMegaCompressorBf16Args(compressor_kv,
                                       compressor_score,
                                       compressor_ape,
                                       compressor_positions,
                                       compressor_state_cache,
                                       compressor_state_slots,
                                       compressor_ratio,
                                       compressor_token_to_req,
                                       compressor_state_block_table,
                                       compressor_norm_weight,
                                       compressor_cos_sin_cache,
                                       compressor_kv_cache,
                                       compressor_kv_slots,
                                       compressor_seq_start,
                                       compressor_disable_raw_path,
                                       compressor_rms_norm_eps,
                                       compressor_head_dim,
                                       compressor_rope_head_dim,
                                       compressor_overlap,
                                       compressor_state_tokens_per_block,
                                       compressor_seq_start_per_req,
                                       compressor_cu_seq_per_req,
                                       compressor_unpad_restore,
                                       cp_size);
    }
    const bool needs_mega_side_effects = use_mega_kernel
                                         && (use_symm_direct_fresh_kv || use_symm_direct_indexer_k
                                             || mega_swa_writer.enabled
                                             || mega_csa_indexer_compressor_writer.enabled
                                             || mega_main_compressor_writer.enabled);
    auto output = torch::empty({local_rows.size(0), H, D}, q.options());
    if (local_rows.size(0) == 0 && !needs_mega_side_effects) {
        return output;
    }
        const uint8_t* csa_indexer_k_cache_ptr =
            has_csa_fp8_indexer ? csa_indexer_k_cache->data_ptr<uint8_t>() : nullptr;
        const float* csa_indexer_weights_ptr =
            (has_csa_fp8_indexer || has_csa_fp8_indexer_pool) ? csa_indexer_weights->data_ptr<float>() : nullptr;
    const int64_t* csa_indexer_cu_lens_ptr =
        has_csa_fp8_indexer ? csa_indexer_cu_lens->data_ptr<int64_t>() : nullptr;
    const int csa_indexer_total_len =
        has_csa_fp8_indexer ? static_cast<int>(csa_indexer_k_cache->size(0)) : 0;
    const uint8_t* csa_indexer_k_pool_ptr =
        has_csa_fp8_indexer_pool ? csa_indexer_k_pool->data_ptr<uint8_t>() : nullptr;
    const int32_t* csa_indexer_block_table_ptr =
        has_csa_fp8_indexer_pool ? csa_indexer_block_table->data_ptr<int32_t>() : nullptr;
    const int32_t* csa_indexer_seq_lens_ptr =
        has_csa_fp8_indexer_pool ? csa_indexer_seq_lens->data_ptr<int32_t>() : nullptr;
    const int csa_indexer_pool_block_size =
        has_csa_fp8_indexer_pool ? static_cast<int>(csa_indexer_k_pool->size(1)) : 0;
    const int64_t csa_indexer_pool_block_stride =
        has_csa_fp8_indexer_pool ? static_cast<int64_t>(csa_indexer_k_pool->stride(0)) : 0;
    const int64_t csa_indexer_block_table_stride =
        has_csa_fp8_indexer_pool ? static_cast<int64_t>(csa_indexer_block_table->stride(0)) : 0;
    const uint8_t* attention_cmp_k_cache_ptr =
        has_attention_cmp_cache ? attention_cmp_k_cache->data_ptr<uint8_t>() : nullptr;
    const int64_t* attention_cmp_cu_lens_ptr =
        has_attention_cmp_cache ? attention_cmp_cu_lens->data_ptr<int64_t>() : nullptr;
    const uint8_t* attention_cmp_k_pool_ptr =
        has_attention_cmp_pool ? attention_cmp_k_pool->data_ptr<uint8_t>() : nullptr;
    const int32_t* attention_cmp_block_table_ptr =
        has_attention_cmp_pool ? attention_cmp_block_table->data_ptr<int32_t>() : nullptr;
    const int32_t* attention_cmp_seq_lens_ptr =
        has_attention_cmp_pool ? attention_cmp_seq_lens->data_ptr<int32_t>() : nullptr;
    const int attention_cmp_pool_block_size =
        has_attention_cmp_pool ? static_cast<int>(attention_cmp_k_pool->size(1)) : 0;
    const int64_t attention_cmp_pool_block_stride =
        has_attention_cmp_pool ? static_cast<int64_t>(attention_cmp_k_pool->stride(0)) : 0;
    const int64_t attention_cmp_block_table_stride =
        has_attention_cmp_pool ? static_cast<int64_t>(attention_cmp_block_table->stride(0)) : 0;
    const uint8_t* attention_swa_k_cache_ptr =
        has_attention_swa_cache ? attention_swa_k_cache->data_ptr<uint8_t>() : nullptr;
    const int64_t* attention_swa_cu_lens_ptr =
        has_attention_swa_cache ? attention_swa_cu_lens->data_ptr<int64_t>() : nullptr;
    const uint8_t* attention_swa_k_pool_ptr =
        has_attention_swa_pool ? attention_swa_k_pool->data_ptr<uint8_t>() : nullptr;
    const int64_t* attention_swa_slot_mapping_ptr =
        has_attention_swa_pool ? attention_swa_slot_mapping->data_ptr<int64_t>() : nullptr;
    const int32_t* attention_swa_gather_lens_ptr =
        has_attention_swa_pool ? attention_swa_gather_lens->data_ptr<int32_t>() : nullptr;
    const int attention_swa_pool_block_size =
        has_attention_swa_pool ? static_cast<int>(attention_swa_k_pool->size(1)) : 0;
    const int64_t attention_swa_pool_block_stride =
        has_attention_swa_pool ? static_cast<int64_t>(attention_swa_k_pool->stride(0)) : 0;
    const int64_t attention_swa_slot_mapping_stride =
        has_attention_swa_pool ? static_cast<int64_t>(attention_swa_slot_mapping->stride(0)) : 0;
    const int64_t* kv_unpad_restore_ptr = has_kv_restore ? kv_unpad_restore->data_ptr<int64_t>() : nullptr;
    const int64_t* kv_cu_lens_ptr = has_kv_restore ? kv_cu_lens->data_ptr<int64_t>() : nullptr;
    const int64_t symm_fresh_payload_bytes = use_symm_direct_fresh_kv ? kv.numel() * kv.element_size() : 0;
    const int64_t symm_indexer_payload_bytes =
        use_symm_direct_indexer_k ? indexer_k.numel() * indexer_k.element_size() : 0;
    const int64_t symm_fresh_payload_offset = kProtocolBytes;
    const int64_t symm_indexer_payload_offset =
        alignUpInt64(symm_fresh_payload_offset + symm_fresh_payload_bytes, kMegaPayloadAlignBytes);
    const int64_t symm_scratch_payload_offset =
        alignUpInt64(symm_indexer_payload_offset + symm_indexer_payload_bytes, kMegaPayloadAlignBytes);
    const int enable_grid_side_effects =
        use_mega_kernel && dsv4CpAttentionMegaGridSideEffectsEnabled() ? 1 : 0;
    const bool split_k_explicitly_enabled = dsv4CpAttentionMegaSplitKEnabled();
    const bool split_k_explicitly_disabled = dsv4CpAttentionMegaSplitKDisabled();
    const int64_t mega_swa_bytes =
        static_cast<int64_t>(mega_swa_writer.local_rows) * kSwaHeadDim * static_cast<int64_t>(sizeof(c10::BFloat16));
    const int64_t mega_csa_compressor_bytes =
        mega_csa_indexer_compressor_writer.kv_bytes + mega_csa_indexer_compressor_writer.score_bytes;
    const int64_t mega_main_compressor_bytes =
        mega_main_compressor_writer.kv_bytes + mega_main_compressor_writer.score_bytes;
    const int64_t mega_compressed_index_bytes =
        alignUpInt64(static_cast<int64_t>(local_rows.size(0)) * static_cast<int64_t>(sizeof(unsigned long long)),
                     kMegaPayloadAlignBytes)
        + static_cast<int64_t>(local_rows.size(0))
              * std::max<int64_t>(static_cast<int64_t>(compressed_topk), 1)
              * static_cast<int64_t>(sizeof(int));
    const int64_t mega_grid_sync_bytes = enable_grid_side_effects ? kMegaGridSyncBytes : 0;
    const int64_t mega_scratch_bytes =
        alignUpInt64(mega_grid_sync_bytes + mega_swa_bytes, kMegaPayloadAlignBytes)
        + alignUpInt64(mega_csa_compressor_bytes, kMegaPayloadAlignBytes)
        + alignUpInt64(mega_main_compressor_bytes, kMegaPayloadAlignBytes)
        + mega_compressed_index_bytes;
    const int64_t mega_splitk_scratch_offset =
        alignUpInt64(symm_scratch_payload_offset + mega_scratch_bytes, kMegaPayloadAlignBytes);
    if (use_mega_kernel && cp_size > 1) {
        TORCH_CHECK(symm_scratch_payload_offset + mega_scratch_bytes <= per_rank_buffer_bytes,
                    "DSV4 mega attention symmetric buffer too small: need ",
                    symm_scratch_payload_offset + mega_scratch_bytes,
                    " bytes per rank, have ",
                    per_rank_buffer_bytes);
    }

    const bool use_flash_mla_attention = !dsv4CpAttentionMegaKernelEnabled() && dsv4CpAttentionFlashMlaEnabled()
                                         && q.scalar_type() == torch::kBFloat16
                                         && H == 128 && D == kSwaHeadDim && gathered_kv.size(1) >= 512
                                         && local_rows.size(0) > 0;
    if (use_flash_mla_attention) {
        const int key_width_raw = static_cast<int>(window_size + compressed_topk);
        const int key_width = ((key_width_raw + 63) / 64) * 64;
        const int workspace_m = static_cast<int>(compressed_region_width + window_size + gathered_kv.size(1));
        TORCH_CHECK(key_width > 0, "flash_mla attention path requires a positive key width");
        TORCH_CHECK(workspace_m > 0, "flash_mla attention path requires a positive workspace width");
        auto workspace = torch::empty({logical_batch * workspace_m, 1, D}, q.options());
        const int threads = 256;
        const int64_t workspace_elems = workspace.numel();
        const int workspace_blocks = static_cast<int>((workspace_elems + threads - 1) / threads);
        dsv4BuildFlashMlaWorkspaceKernel<c10::BFloat16><<<workspace_blocks, threads, 0, stream>>>(
            gathered_kv.data_ptr<c10::BFloat16>(),
            workspace.data_ptr<c10::BFloat16>(),
            prefix_lengths.data_ptr<int64_t>(),
            input_lengths.data_ptr<int64_t>(),
            static_cast<int>(logical_batch),
            workspace_m,
            static_cast<int>(D),
            static_cast<int>(gathered_kv.size(1)),
            static_cast<int>(gathered_kv.size(2)),
            static_cast<int>(compressed_region_width),
            static_cast<int>(window_size),
            attention_cmp_k_cache_ptr,
            attention_cmp_cu_lens_ptr,
            attention_cmp_k_pool_ptr,
            attention_cmp_block_table_ptr,
            attention_cmp_seq_lens_ptr,
            attention_cmp_pool_block_size,
            attention_cmp_pool_block_stride,
            attention_cmp_block_table_stride,
            attention_swa_k_cache_ptr,
            attention_swa_cu_lens_ptr,
            attention_swa_k_pool_ptr,
            attention_swa_slot_mapping_ptr,
            attention_swa_gather_lens_ptr,
            attention_swa_pool_block_size,
            attention_swa_pool_block_stride,
            attention_swa_slot_mapping_stride,
            kv_unpad_restore_ptr,
            kv_cu_lens_ptr);
        auto indices = torch::empty({local_rows.size(0), key_width},
                                    torch::TensorOptions().device(q.device()).dtype(torch::kInt32));
        auto topk_lens = torch::empty({local_rows.size(0)},
                                      torch::TensorOptions().device(q.device()).dtype(torch::kInt32));
        const bool use_deepgemm_indexer =
            compress_ratio == 4 && compressed_topk > 0 && has_csa_fp8_indexer_pool
            && indexer_q.size(1) == 64 && indexer_q.size(2) == kIndexerHeadDim
            && local_rows.size(0) == q.size(0);
        if (use_deepgemm_indexer) {
            const int rows_per_req = static_cast<int>(csa_indexer_block_table->size(1)) * csa_indexer_pool_block_size;
            auto k_quant = torch::empty({logical_batch * rows_per_req, kIndexerHeadDim},
                                        torch::TensorOptions().device(q.device()).dtype(torch::kFloat8_e4m3fn));
            auto k_scale = torch::empty({logical_batch * rows_per_req},
                                        torch::TensorOptions().device(q.device()).dtype(torch::kFloat32));
            const int64_t gather_elems = static_cast<int64_t>(logical_batch) * rows_per_req * kIndexerHeadDim;
            dsv4GatherIndexerKvForDeepGemmKernel<<<static_cast<int>((gather_elems + 255) / 256), 256, 0, stream>>>(
                csa_indexer_k_pool_ptr,
                csa_indexer_block_table_ptr,
                csa_indexer_seq_lens_ptr,
                reinterpret_cast<uint8_t*>(k_quant.data_ptr()),
                k_scale.data_ptr<float>(),
                static_cast<int>(logical_batch),
                rows_per_req,
                csa_indexer_pool_block_size,
                csa_indexer_pool_block_stride,
                csa_indexer_block_table_stride);
            auto row_starts = torch::empty({local_rows.size(0)},
                                           torch::TensorOptions().device(q.device()).dtype(torch::kInt32));
            auto row_ends = torch::empty({local_rows.size(0)},
                                         torch::TensorOptions().device(q.device()).dtype(torch::kInt32));
            dsv4BuildIndexerDeepGemmRangesKernel<<<static_cast<int>((local_rows.size(0) + 255) / 256), 256, 0, stream>>>(
                req_id_per_token.data_ptr<int64_t>(),
                position_ids.data_ptr<int64_t>(),
                local_rows.data_ptr<int64_t>(),
                csa_indexer_seq_lens_ptr,
                row_starts.data_ptr<int32_t>(),
                row_ends.data_ptr<int32_t>(),
                static_cast<int>(local_rows.size(0)),
                rows_per_req,
                static_cast<int>(compress_ratio));
            const cudaError_t indexer_prep_err = cudaGetLastError();
            TORCH_CHECK(indexer_prep_err == cudaSuccess,
                        "dsv4 DeepGEMM indexer prep launch failed: ",
                        cudaGetErrorString(indexer_prep_err));
            torch::Tensor q_indexer_local = indexer_q.index_select(0, local_rows).to(torch::kFloat8_e4m3fn);
            torch::Tensor w_fold_local = csa_indexer_weights->index_select(0, local_rows).contiguous();
            py::object deep_gemm = py::module_::import("deep_gemm");
            py::object logits_obj = deep_gemm.attr("fp8_mqa_logits")(tensorToPyObject(q_indexer_local),
                                                                     py::make_tuple(tensorToPyObject(k_quant),
                                                                                    tensorToPyObject(k_scale)),
                                                                     tensorToPyObject(w_fold_local),
                                                                     tensorToPyObject(row_starts),
                                                                     tensorToPyObject(row_ends),
                                                                     false,
                                                                     0);
            torch::Tensor logits = logits_obj.cast<torch::Tensor>();
            auto topk_local = torch::empty({local_rows.size(0), compressed_topk},
                                           torch::TensorOptions().device(q.device()).dtype(torch::kInt32));
            dsv4_top_k_per_row_prefill(logits,
                                       row_starts,
                                       row_ends,
                                       topk_local,
                                       local_rows.size(0),
                                       logits.stride(0),
                                       logits.stride(1),
                                       compressed_topk,
                                       false);
            dsv4BuildFlashMlaIndicesFromTopkKernel<<<static_cast<unsigned int>(local_rows.size(0)), 256, 0, stream>>>(
                topk_local.data_ptr<int32_t>(),
                row_starts.data_ptr<int32_t>(),
                row_ends.data_ptr<int32_t>(),
                req_id_per_token.data_ptr<int64_t>(),
                position_ids.data_ptr<int64_t>(),
                prefix_lengths.data_ptr<int64_t>(),
                input_lengths.data_ptr<int64_t>(),
                local_rows.data_ptr<int64_t>(),
                attention_swa_gather_lens_ptr,
                indices.data_ptr<int32_t>(),
                topk_lens.data_ptr<int32_t>(),
                static_cast<int>(local_rows.size(0)),
                workspace_m,
                key_width,
                static_cast<int>(compressed_topk),
                static_cast<int>(compressed_region_width),
                static_cast<int>(window_size));
        } else {
            dsv4BuildFlashMlaIndicesKernel<c10::BFloat16><<<static_cast<unsigned int>(local_rows.size(0)), 256, 0, stream>>>(
                indexer_q.data_ptr<c10::BFloat16>(),
                gathered_indexer_k.data_ptr<c10::BFloat16>(),
                req_id_per_token.data_ptr<int64_t>(),
                position_ids.data_ptr<int64_t>(),
                prefix_lengths.data_ptr<int64_t>(),
                input_lengths.data_ptr<int64_t>(),
                local_rows.data_ptr<int64_t>(),
                indices.data_ptr<int32_t>(),
                topk_lens.data_ptr<int32_t>(),
                static_cast<int>(local_rows.size(0)),
                workspace_m,
                static_cast<int>(gathered_kv.size(1)),
                static_cast<int>(indexer_q.size(1)),
                static_cast<int>(indexer_q.size(2)),
                static_cast<int>(gathered_indexer_k.size(1)),
                static_cast<int>(compress_ratio),
                static_cast<int>(window_size),
                static_cast<int>(compressed_topk),
                static_cast<int>(compressed_region_width),
                key_width,
                csa_indexer_k_cache_ptr,
                csa_indexer_weights_ptr,
                csa_indexer_cu_lens_ptr,
                csa_indexer_total_len,
                csa_indexer_k_pool_ptr,
                csa_indexer_block_table_ptr,
                csa_indexer_seq_lens_ptr,
                csa_indexer_pool_block_size,
                csa_indexer_pool_block_stride,
                csa_indexer_block_table_stride,
                attention_cmp_seq_lens_ptr,
                attention_swa_gather_lens_ptr);
        }
        const cudaError_t flash_prep_err = cudaGetLastError();
        TORCH_CHECK(flash_prep_err == cudaSuccess,
                    "dsv4 flash_mla attention prep launch failed: ",
                    cudaGetErrorString(flash_prep_err));
        debugStage(debug_sync, cp_rank, "flash_mla_sparse_fwd", "begin", stream, false);
        try {
            py::object flash_mla_sparse_fwd = py::module_::import("flash_mla").attr("flash_mla_sparse_fwd");
            torch::Tensor indices_3d = indices.view({local_rows.size(0), 1, key_width});
            py::object result = flash_mla_sparse_fwd(tensorToPyObject(q),
                                                     tensorToPyObject(workspace),
                                                     tensorToPyObject(indices_3d),
                                                     1.0 / std::sqrt(static_cast<double>(D)),
                                                     static_cast<int>(D),
                                                     tensorToPyObject(attn_sink),
                                                     tensorToPyObject(topk_lens));
            py::tuple result_tuple = result.cast<py::tuple>();
            torch::Tensor output = result_tuple[0].cast<torch::Tensor>();
            debugStage(debug_sync, cp_rank, "flash_mla_sparse_fwd", "end", stream, true);
            return output;
        } catch (const py::error_already_set& e) {
            if (std::getenv("DSV4_CP_ATTENTION_FLASHMLA_STRICT") != nullptr) {
                throw;
            }
            PyErr_Clear();
            std::fprintf(stderr,
                         "[DSV4 CP Attention Cuda] rank=%ld flash_mla path unavailable, falling back: %s\n",
                         static_cast<long>(cp_rank),
                         e.what());
            std::fflush(stderr);
        }
    }

    const dim3 block(kMegaBlockThreads);
    const bool host_use_grouped_attention =
        D == kSwaHeadDim && gathered_kv.size(2) == 1 && H >= kMegaAttentionHeadsPerCta
        && (H % kMegaAttentionHeadsPerCta) == 0 && static_cast<int>(block.x) == kMegaBlockThreads;
    const int splitk_max_keys = static_cast<int>(std::max<int64_t>(
        1, static_cast<int64_t>(compressed_topk) + static_cast<int64_t>(window_size)));
    const bool splitk_supported_ratio =
        compress_ratio == 0 || compress_ratio == 4 || compress_ratio == 128;
    if (split_k_explicitly_enabled && !split_k_explicitly_disabled) {
        TORCH_CHECK(splitk_supported_ratio,
                    "DSV4_CP_ATTENTION_MEGA_SPLITK supports only compress_ratio 0, 4, or 128");
        TORCH_CHECK(use_symm_backend && enable_grid_side_effects,
                    "DSV4_CP_ATTENTION_MEGA_SPLITK requires symmetric backend and grid side effects");
        TORCH_CHECK(host_use_grouped_attention,
                    "DSV4_CP_ATTENTION_MEGA_SPLITK currently supports only production grouped MQA shape");
    }
    const int allow_split_k_attention =
        use_mega_kernel && !split_k_explicitly_disabled
        && splitk_supported_ratio
        && (split_k_explicitly_enabled
            || (use_symm_backend && enable_grid_side_effects && host_use_grouped_attention
                && splitk_max_keys >= kMegaSplitKMinKeys)) ?
            1 :
            0;
    const int enable_split_k_attention =
        allow_split_k_attention && splitk_max_keys >= kMegaSplitKMinKeys ? 1 : 0;
    int splitk_keys_per_block = kMegaSplitKKeysPerBlock;
    if (splitk_max_keys >= 2048) {
        splitk_keys_per_block = 256;
    } else if (splitk_max_keys >= 1024) {
        splitk_keys_per_block = 128;
    }
    if (allow_split_k_attention) {
        TORCH_CHECK(use_symm_backend && enable_grid_side_effects,
                    "DSV4_CP_ATTENTION_MEGA_SPLITK requires symmetric backend and grid side effects");
        TORCH_CHECK(host_use_grouped_attention,
                    "DSV4_CP_ATTENTION_MEGA_SPLITK currently supports only production grouped MQA shape");
        TORCH_CHECK(splitk_keys_per_block > 0 && kMegaSplitKBlocksPerWave > 0,
                    "DSV4_CP_ATTENTION_MEGA_SPLITK requires positive split-K tiling constants");
    }
    const int64_t host_head_task_count =
        host_use_grouped_attention ? H / kMegaAttentionHeadsPerCta : H;
    const int64_t semantic_task_count = local_rows.size(0) * host_head_task_count;
    const size_t mega_dynamic_smem_bytes =
        host_use_grouped_attention ?
            static_cast<size_t>(kMegaAttentionGroupedKeyTile) * static_cast<size_t>(kSwaHeadDim) * sizeof(float) :
            0;
    int device_id = -1;
    AT_CUDA_CHECK(cudaGetDevice(&device_id));
    cudaDeviceProp device_prop{};
    AT_CUDA_CHECK(cudaGetDeviceProperties(&device_prop, device_id));
    int mega_active_blocks_per_sm = 1;
    if (q.scalar_type() == torch::kFloat32) {
        if (use_mega_kernel && mega_dynamic_smem_bytes > 0) {
            AT_CUDA_CHECK(cudaFuncSetAttribute(dsv4CpDistributedPrefillMegaAttentionKernel<float>,
                                               cudaFuncAttributeMaxDynamicSharedMemorySize,
                                               static_cast<int>(mega_dynamic_smem_bytes)));
        }
        cudaError_t occ_err = cudaOccupancyMaxActiveBlocksPerMultiprocessor(
            &mega_active_blocks_per_sm,
            dsv4CpDistributedPrefillMegaAttentionKernel<float>,
            static_cast<int>(block.x),
            mega_dynamic_smem_bytes);
        if (occ_err != cudaSuccess || mega_active_blocks_per_sm <= 0) {
            mega_active_blocks_per_sm = 1;
        }
    } else if (q.scalar_type() == torch::kBFloat16) {
        if (use_mega_kernel && mega_dynamic_smem_bytes > 0) {
            AT_CUDA_CHECK(cudaFuncSetAttribute(dsv4CpDistributedPrefillMegaAttentionKernel<c10::BFloat16>,
                                               cudaFuncAttributeMaxDynamicSharedMemorySize,
                                               static_cast<int>(mega_dynamic_smem_bytes)));
        }
        cudaError_t occ_err = cudaOccupancyMaxActiveBlocksPerMultiprocessor(
            &mega_active_blocks_per_sm,
            dsv4CpDistributedPrefillMegaAttentionKernel<c10::BFloat16>,
            static_cast<int>(block.x),
            mega_dynamic_smem_bytes);
        if (occ_err != cudaSuccess || mega_active_blocks_per_sm <= 0) {
            mega_active_blocks_per_sm = 1;
        }
    }
    const int64_t max_resident_mega_blocks =
        std::max<int64_t>(1, static_cast<int64_t>(mega_active_blocks_per_sm) * device_prop.multiProcessorCount);
    const int64_t splitk_grid_count =
        enable_split_k_attention ?
            std::max<int64_t>(
                kMegaSplitKGroupSize,
                (max_resident_mega_blocks / kMegaSplitKGroupSize) * kMegaSplitKGroupSize) :
            0;
    const int64_t splitk_physical_group_count =
        enable_split_k_attention ? std::max<int64_t>(1, splitk_grid_count / kMegaSplitKGroupSize) : 0;
    const int64_t splitk_group_barrier_slots =
        splitk_physical_group_count * static_cast<int64_t>(kMegaSplitKGroupBarrierSlots);
    const int64_t splitk_group_counter_bytes =
        ((splitk_group_barrier_slots * static_cast<int64_t>(sizeof(unsigned int)) + 63) / 64) * 64;
    const int64_t splitk_group_epoch_bytes =
        ((splitk_group_barrier_slots * static_cast<int64_t>(sizeof(unsigned long long)) + 63) / 64) * 64;
    if (enable_split_k_attention) {
        const int64_t mega_splitk_scratch_bytes = max_resident_mega_blocks * kMegaSplitKRecordBytes
                                                  + splitk_group_counter_bytes + splitk_group_epoch_bytes;
        TORCH_CHECK(mega_splitk_scratch_offset + mega_splitk_scratch_bytes <= per_rank_buffer_bytes,
                    "DSV4 split-K mega attention symmetric buffer too small: need ",
                    mega_splitk_scratch_offset + mega_splitk_scratch_bytes,
                    " bytes per rank, have ",
                    per_rank_buffer_bytes);
        TORCH_CHECK(max_resident_mega_blocks >= kMegaSplitKGroupSize,
                    "DSV4 split-K mega attention needs at least ",
                    kMegaSplitKGroupSize,
                    " resident CTAs, got ",
                    max_resident_mega_blocks);
    }
    const bool serialize_mega_side_effects_without_grid_barrier =
        needs_mega_side_effects && !use_symm_backend;
    const int64_t mega_grid_limit =
        enable_grid_side_effects ? max_resident_mega_blocks :
                                   max_resident_mega_blocks * kMegaNonCoopGridWaveMultiplier;
    const int64_t mega_grid_count =
        use_mega_kernel ?
            (enable_split_k_attention ?
                 splitk_grid_count :
             (serialize_mega_side_effects_without_grid_barrier ?
                 1 :
                 std::max<int64_t>(1, std::min<int64_t>(semantic_task_count, mega_grid_limit)))) :
            1;
    const dim3 grid(use_mega_kernel ? static_cast<unsigned int>(mega_grid_count) :
                                      static_cast<unsigned int>(semantic_task_count));
    const dim3 legacy_grid(static_cast<unsigned int>(local_rows.size(0)), static_cast<unsigned int>(H));
    static std::atomic<unsigned long long> mega_launch_epoch_counter{0xD504C00000000000ull};
    const unsigned long long mega_launch_epoch =
        use_mega_kernel ? mega_launch_epoch_counter.fetch_add(1, std::memory_order_relaxed) + 1ull : 0ull;
    const int attention_kv_len = use_symm_direct_fresh_kv ? static_cast<int>(kv.size(1) * cp_size) :
                                                            static_cast<int>(gathered_kv.size(1));
    const bool needs_symm_for_mega =
        use_symm_direct_fresh_kv || use_symm_direct_indexer_k || needs_mega_side_effects || enable_split_k_attention;
    const uint8_t* const* symm_buffer_ptrs_for_mega =
        needs_symm_for_mega ?
            reinterpret_cast<const uint8_t* const*>(symm_buffer_ptrs_dev) :
            nullptr;
    const int symm_l_local = use_symm_direct_fresh_kv ? static_cast<int>(kv.size(1)) : 0;
    const int attention_indexer_len = use_symm_direct_indexer_k ? static_cast<int>(indexer_k.size(1) * cp_size) :
                                                                  static_cast<int>(gathered_indexer_k.size(1));
    const int symm_indexer_l_local = use_symm_direct_indexer_k ? static_cast<int>(indexer_k.size(1)) : 0;
    uint32_t* const* symm_signal_pads_for_mega =
        needs_symm_for_mega ?
            reinterpret_cast<uint32_t* const*>(symm_signal_pad_ptrs_dev) :
            nullptr;
    (void)cudaGetLastError();
    debugStage(debug_sync, cp_rank, "attention_body", "begin", stream, false);
    if (q.scalar_type() == torch::kFloat32) {
        if (dsv4CpAttentionDebugSyncEnabled()) {
            cudaFuncAttributes attr{};
            cudaError_t attr_err = cudaFuncGetAttributes(&attr, dsv4CpDistributedPrefillMegaAttentionKernel<float>);
            int active_blocks = -1;
            cudaError_t occ_err = cudaOccupancyMaxActiveBlocksPerMultiprocessor(
                &active_blocks, dsv4CpDistributedPrefillMegaAttentionKernel<float>, static_cast<int>(block.x), 0);
            std::fprintf(stderr,
                         "[DSV4 CP Attention Cuda] mega kernel<float> attr_err=%s occ_err=%s max_threads=%d "
                         "shared=%zu regs=%d const=%zu local=%zu active_blocks=%d block=%u grid=%u\n",
                         cudaGetErrorString(attr_err),
                         cudaGetErrorString(occ_err),
                         attr.maxThreadsPerBlock,
                         attr.sharedSizeBytes,
                         attr.numRegs,
                         attr.constSizeBytes,
                         attr.localSizeBytes,
                         active_blocks,
                         block.x,
                         grid.x);
            std::fflush(stderr);
        }
        if (use_mega_kernel) {
            dsv4CpDistributedPrefillMegaAttentionKernel<float><<<grid, block, mega_dynamic_smem_bytes, stream>>>(
                q.data_ptr<float>(),
                gathered_kv.data_ptr<float>(),
                indexer_q.data_ptr<float>(),
                gathered_indexer_k.data_ptr<float>(),
                attn_sink.data_ptr<float>(),
                req_id_per_token.data_ptr<int64_t>(),
                position_ids.data_ptr<int64_t>(),
                prefix_lengths.data_ptr<int64_t>(),
                input_lengths.data_ptr<int64_t>(),
                local_rows.data_ptr<int64_t>(),
                output.data_ptr<float>(),
                static_cast<int>(local_rows.size(0)),
                static_cast<int>(H),
                static_cast<int>(D),
                attention_kv_len,
                static_cast<int>(gathered_kv.size(2)),
                static_cast<int>(indexer_q.size(1)),
                static_cast<int>(indexer_q.size(2)),
                attention_indexer_len,
                static_cast<int>(logical_batch),
                static_cast<int>(compress_ratio),
                static_cast<int>(window_size),
                static_cast<int>(compressed_topk),
                csa_indexer_k_cache_ptr,
                csa_indexer_weights_ptr,
                csa_indexer_cu_lens_ptr,
                csa_indexer_total_len,
                csa_indexer_k_pool_ptr,
                csa_indexer_block_table_ptr,
                csa_indexer_seq_lens_ptr,
                csa_indexer_pool_block_size,
                csa_indexer_pool_block_stride,
                csa_indexer_block_table_stride,
                attention_cmp_k_cache_ptr,
                attention_cmp_cu_lens_ptr,
                attention_cmp_k_pool_ptr,
                attention_cmp_block_table_ptr,
                attention_cmp_seq_lens_ptr,
                attention_cmp_pool_block_size,
                attention_cmp_pool_block_stride,
                attention_cmp_block_table_stride,
                attention_swa_k_cache_ptr,
                attention_swa_cu_lens_ptr,
                attention_swa_k_pool_ptr,
                attention_swa_slot_mapping_ptr,
                attention_swa_gather_lens_ptr,
                attention_swa_pool_block_size,
                attention_swa_pool_block_stride,
                attention_swa_slot_mapping_stride,
                kv_unpad_restore_ptr,
                kv_cu_lens_ptr,
                symm_buffer_ptrs_for_mega,
                use_symm_direct_fresh_kv ? 1 : 0,
                per_rank_buffer_bytes,
                symm_fresh_payload_offset,
                symm_l_local,
                use_symm_direct_indexer_k ? 1 : 0,
                symm_indexer_payload_offset,
                symm_indexer_l_local,
                symm_scratch_payload_offset,
                mega_splitk_scratch_offset,
                mega_swa_writer,
                mega_csa_indexer_compressor_writer,
                mega_main_compressor_writer,
                mega_launch_epoch,
                static_cast<int>(cp_rank),
                static_cast<int>(cp_size),
                symm_signal_pads_for_mega,
                enable_grid_side_effects,
                enable_split_k_attention,
                splitk_keys_per_block);
        } else {
            dsv4CpDistributedPrefillAttentionKernel<float><<<legacy_grid, block, 0, stream>>>(
                q.data_ptr<float>(),
                gathered_kv.data_ptr<float>(),
                indexer_q.data_ptr<float>(),
                gathered_indexer_k.data_ptr<float>(),
                attn_sink.data_ptr<float>(),
                req_id_per_token.data_ptr<int64_t>(),
                position_ids.data_ptr<int64_t>(),
                prefix_lengths.data_ptr<int64_t>(),
                input_lengths.data_ptr<int64_t>(),
                local_rows.data_ptr<int64_t>(),
                output.data_ptr<float>(),
                static_cast<int>(local_rows.size(0)),
                static_cast<int>(H),
                static_cast<int>(D),
                static_cast<int>(gathered_kv.size(1)),
                static_cast<int>(gathered_kv.size(2)),
                static_cast<int>(indexer_q.size(1)),
                static_cast<int>(indexer_q.size(2)),
                static_cast<int>(gathered_indexer_k.size(1)),
                static_cast<int>(compress_ratio),
                static_cast<int>(window_size),
                static_cast<int>(compressed_topk),
                csa_indexer_k_cache_ptr,
                csa_indexer_weights_ptr,
                csa_indexer_cu_lens_ptr,
                csa_indexer_total_len,
                csa_indexer_k_pool_ptr,
                csa_indexer_block_table_ptr,
                csa_indexer_seq_lens_ptr,
                csa_indexer_pool_block_size,
                csa_indexer_pool_block_stride,
                csa_indexer_block_table_stride,
                attention_cmp_k_cache_ptr,
                attention_cmp_cu_lens_ptr,
                attention_cmp_k_pool_ptr,
                attention_cmp_block_table_ptr,
                attention_cmp_seq_lens_ptr,
                attention_cmp_pool_block_size,
                attention_cmp_pool_block_stride,
                attention_cmp_block_table_stride,
                attention_swa_k_cache_ptr,
                attention_swa_cu_lens_ptr,
                attention_swa_k_pool_ptr,
                attention_swa_slot_mapping_ptr,
                attention_swa_gather_lens_ptr,
                attention_swa_pool_block_size,
                attention_swa_pool_block_stride,
                attention_swa_slot_mapping_stride,
                kv_unpad_restore_ptr,
                kv_cu_lens_ptr);
        }
    } else if (q.scalar_type() == torch::kBFloat16) {
        if (dsv4CpAttentionDebugSyncEnabled()) {
            cudaFuncAttributes attr{};
            cudaError_t attr_err =
                cudaFuncGetAttributes(&attr, dsv4CpDistributedPrefillMegaAttentionKernel<c10::BFloat16>);
            int active_blocks = -1;
            cudaError_t occ_err = cudaOccupancyMaxActiveBlocksPerMultiprocessor(
                &active_blocks, dsv4CpDistributedPrefillMegaAttentionKernel<c10::BFloat16>, static_cast<int>(block.x), 0);
            std::fprintf(stderr,
                         "[DSV4 CP Attention Cuda] mega kernel<bf16> attr_err=%s occ_err=%s max_threads=%d "
                         "shared=%zu regs=%d const=%zu local=%zu active_blocks=%d block=%u grid=%u\n",
                         cudaGetErrorString(attr_err),
                         cudaGetErrorString(occ_err),
                         attr.maxThreadsPerBlock,
                         attr.sharedSizeBytes,
                         attr.numRegs,
                         attr.constSizeBytes,
                         attr.localSizeBytes,
                         active_blocks,
                         block.x,
                         grid.x);
            std::fflush(stderr);
        }
        if (use_mega_kernel) {
            dsv4CpDistributedPrefillMegaAttentionKernel<c10::BFloat16><<<grid, block, mega_dynamic_smem_bytes, stream>>>(
                q.data_ptr<c10::BFloat16>(),
                gathered_kv.data_ptr<c10::BFloat16>(),
                indexer_q.data_ptr<c10::BFloat16>(),
                gathered_indexer_k.data_ptr<c10::BFloat16>(),
                attn_sink.data_ptr<float>(),
                req_id_per_token.data_ptr<int64_t>(),
                position_ids.data_ptr<int64_t>(),
                prefix_lengths.data_ptr<int64_t>(),
                input_lengths.data_ptr<int64_t>(),
                local_rows.data_ptr<int64_t>(),
                output.data_ptr<c10::BFloat16>(),
                static_cast<int>(local_rows.size(0)),
                static_cast<int>(H),
                static_cast<int>(D),
                attention_kv_len,
                static_cast<int>(gathered_kv.size(2)),
                static_cast<int>(indexer_q.size(1)),
                static_cast<int>(indexer_q.size(2)),
                attention_indexer_len,
                static_cast<int>(logical_batch),
                static_cast<int>(compress_ratio),
                static_cast<int>(window_size),
                static_cast<int>(compressed_topk),
                csa_indexer_k_cache_ptr,
                csa_indexer_weights_ptr,
                csa_indexer_cu_lens_ptr,
                csa_indexer_total_len,
                csa_indexer_k_pool_ptr,
                csa_indexer_block_table_ptr,
                csa_indexer_seq_lens_ptr,
                csa_indexer_pool_block_size,
                csa_indexer_pool_block_stride,
                csa_indexer_block_table_stride,
                attention_cmp_k_cache_ptr,
                attention_cmp_cu_lens_ptr,
                attention_cmp_k_pool_ptr,
                attention_cmp_block_table_ptr,
                attention_cmp_seq_lens_ptr,
                attention_cmp_pool_block_size,
                attention_cmp_pool_block_stride,
                attention_cmp_block_table_stride,
                attention_swa_k_cache_ptr,
                attention_swa_cu_lens_ptr,
                attention_swa_k_pool_ptr,
                attention_swa_slot_mapping_ptr,
                attention_swa_gather_lens_ptr,
                attention_swa_pool_block_size,
                attention_swa_pool_block_stride,
                attention_swa_slot_mapping_stride,
                kv_unpad_restore_ptr,
                kv_cu_lens_ptr,
                symm_buffer_ptrs_for_mega,
                use_symm_direct_fresh_kv ? 1 : 0,
                per_rank_buffer_bytes,
                symm_fresh_payload_offset,
                symm_l_local,
                use_symm_direct_indexer_k ? 1 : 0,
                symm_indexer_payload_offset,
                symm_indexer_l_local,
                symm_scratch_payload_offset,
                mega_splitk_scratch_offset,
                mega_swa_writer,
                mega_csa_indexer_compressor_writer,
                mega_main_compressor_writer,
                mega_launch_epoch,
                static_cast<int>(cp_rank),
                static_cast<int>(cp_size),
                symm_signal_pads_for_mega,
                enable_grid_side_effects,
                enable_split_k_attention,
                splitk_keys_per_block);
        } else {
            dsv4CpDistributedPrefillAttentionKernel<c10::BFloat16><<<legacy_grid, block, 0, stream>>>(
                q.data_ptr<c10::BFloat16>(),
                gathered_kv.data_ptr<c10::BFloat16>(),
                indexer_q.data_ptr<c10::BFloat16>(),
                gathered_indexer_k.data_ptr<c10::BFloat16>(),
                attn_sink.data_ptr<float>(),
                req_id_per_token.data_ptr<int64_t>(),
                position_ids.data_ptr<int64_t>(),
                prefix_lengths.data_ptr<int64_t>(),
                input_lengths.data_ptr<int64_t>(),
                local_rows.data_ptr<int64_t>(),
                output.data_ptr<c10::BFloat16>(),
                static_cast<int>(local_rows.size(0)),
                static_cast<int>(H),
                static_cast<int>(D),
                static_cast<int>(gathered_kv.size(1)),
                static_cast<int>(gathered_kv.size(2)),
                static_cast<int>(indexer_q.size(1)),
                static_cast<int>(indexer_q.size(2)),
                static_cast<int>(gathered_indexer_k.size(1)),
                static_cast<int>(compress_ratio),
                static_cast<int>(window_size),
                static_cast<int>(compressed_topk),
                csa_indexer_k_cache_ptr,
                csa_indexer_weights_ptr,
                csa_indexer_cu_lens_ptr,
                csa_indexer_total_len,
                csa_indexer_k_pool_ptr,
                csa_indexer_block_table_ptr,
                csa_indexer_seq_lens_ptr,
                csa_indexer_pool_block_size,
                csa_indexer_pool_block_stride,
                csa_indexer_block_table_stride,
                attention_cmp_k_cache_ptr,
                attention_cmp_cu_lens_ptr,
                attention_cmp_k_pool_ptr,
                attention_cmp_block_table_ptr,
                attention_cmp_seq_lens_ptr,
                attention_cmp_pool_block_size,
                attention_cmp_pool_block_stride,
                attention_cmp_block_table_stride,
                attention_swa_k_cache_ptr,
                attention_swa_cu_lens_ptr,
                attention_swa_k_pool_ptr,
                attention_swa_slot_mapping_ptr,
                attention_swa_gather_lens_ptr,
                attention_swa_pool_block_size,
                attention_swa_pool_block_stride,
                attention_swa_slot_mapping_stride,
                kv_unpad_restore_ptr,
                kv_cu_lens_ptr);
        }
    }
    const cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess,
                "dsv4_cp_distributed_prefill_attention launch failed: ",
                cudaGetErrorString(err));
    debugStage(debug_sync, cp_rank, "attention_body", "end", stream, true);
    return output;
}

}  // namespace
#endif

torch::Tensor dsv4_cp_distributed_prefill_attention(const torch::Tensor& q,
                                                    const torch::Tensor& kv,
                                                    const torch::Tensor& indexer_q,
                                                    const torch::Tensor& indexer_k,
                                                    const torch::Tensor& attn_sink,
                                                    const torch::Tensor& req_id_per_token,
                                                    const torch::Tensor& position_ids,
                                                    const torch::Tensor& prefix_lengths,
                                                    const torch::Tensor& input_lengths,
                                                    const torch::Tensor& local_rows,
                                                    int64_t              compress_ratio,
                                                    int64_t              window_size,
                                                    int64_t              compressed_topk,
                                                    int64_t              compressed_region_width,
                                                    int64_t              cp_rank,
                                                    int64_t              cp_size,
                                                    int64_t              comm_ptr,
                                                    int64_t              buffer_handle,
                                                    int64_t              signal_handle,
                                                    int64_t              per_rank_buffer_bytes,
                                                    std::vector<int64_t> rank_offsets,
                                                    const std::optional<torch::Tensor>& swa_k,
                                                    const std::optional<torch::Tensor>& swa_k_cache,
                                                    const std::optional<torch::Tensor>& swa_slot_mapping,
                                                    const std::optional<torch::Tensor>& symm_buffer,
                                                    int64_t symm_buffer_ptrs_dev,
                                                    int64_t symm_signal_pad_ptrs_dev,
                                                    py::object symm_handle,
                                                    const std::optional<torch::Tensor>& compressor_kv,
                                                    const std::optional<torch::Tensor>& compressor_score,
                                                    const std::optional<torch::Tensor>& compressor_ape,
                                                    const std::optional<torch::Tensor>& compressor_positions,
                                                    const std::optional<torch::Tensor>& compressor_state_cache,
                                                    const std::optional<torch::Tensor>& compressor_state_slots,
                                                    int64_t compressor_ratio,
                                                    const std::optional<torch::Tensor>& compressor_token_to_req,
                                                    const std::optional<torch::Tensor>& compressor_state_block_table,
                                                    const std::optional<torch::Tensor>& compressor_norm_weight,
                                                    const std::optional<torch::Tensor>& compressor_cos_sin_cache,
                                                    const std::optional<torch::Tensor>& compressor_kv_cache,
                                                    const std::optional<torch::Tensor>& compressor_kv_slots,
                                                    int64_t compressor_seq_start,
                                                    bool compressor_disable_raw_path,
                                                    double compressor_rms_norm_eps,
                                                    int64_t compressor_head_dim,
                                                    int64_t compressor_rope_head_dim,
                                                    bool compressor_overlap,
                                                    int64_t compressor_state_tokens_per_block,
                                                        const std::optional<torch::Tensor>& compressor_seq_start_per_req,
                                                        const std::optional<torch::Tensor>& compressor_cu_seq_per_req,
                                                        const std::optional<torch::Tensor>& compressor_unpad_restore,
                                                        const std::optional<torch::Tensor>& csa_indexer_compressor_kv,
                                                        const std::optional<torch::Tensor>& csa_indexer_compressor_score,
                                                        const std::optional<torch::Tensor>& csa_indexer_compressor_ape,
                                                        const std::optional<torch::Tensor>& csa_indexer_compressor_positions,
                                                        const std::optional<torch::Tensor>& csa_indexer_compressor_state_cache,
                                                        const std::optional<torch::Tensor>& csa_indexer_compressor_state_slots,
                                                        int64_t csa_indexer_compressor_ratio,
                                                        const std::optional<torch::Tensor>& csa_indexer_compressor_token_to_req,
                                                        const std::optional<torch::Tensor>& csa_indexer_compressor_state_block_table,
                                                        const std::optional<torch::Tensor>& csa_indexer_compressor_norm_weight,
                                                        const std::optional<torch::Tensor>& csa_indexer_compressor_cos_sin_cache,
                                                        const std::optional<torch::Tensor>& csa_indexer_compressor_kv_cache,
                                                        const std::optional<torch::Tensor>& csa_indexer_compressor_kv_slots,
                                                        int64_t csa_indexer_compressor_seq_start,
                                                        bool csa_indexer_compressor_disable_raw_path,
                                                        double csa_indexer_compressor_rms_norm_eps,
                                                        int64_t csa_indexer_compressor_head_dim,
                                                        int64_t csa_indexer_compressor_rope_head_dim,
                                                        bool csa_indexer_compressor_overlap,
                                                        int64_t csa_indexer_compressor_state_tokens_per_block,
                                                        const std::optional<torch::Tensor>& csa_indexer_compressor_seq_start_per_req,
                                                        const std::optional<torch::Tensor>& csa_indexer_compressor_cu_seq_per_req,
                                                        const std::optional<torch::Tensor>& csa_indexer_compressor_unpad_restore,
                                                        const std::optional<torch::Tensor>& csa_indexer_k_cache,
                                                    const std::optional<torch::Tensor>& csa_indexer_weights,
                                                    const std::optional<torch::Tensor>& csa_indexer_cu_lens,
                                                    const std::optional<torch::Tensor>& csa_indexer_k_pool,
                                                    const std::optional<torch::Tensor>& csa_indexer_block_table,
                                                    const std::optional<torch::Tensor>& csa_indexer_seq_lens,
                                                    const std::optional<torch::Tensor>& attention_cmp_k_cache,
                                                      const std::optional<torch::Tensor>& attention_cmp_cu_lens,
                                                      const std::optional<torch::Tensor>& attention_cmp_k_pool,
                                                      const std::optional<torch::Tensor>& attention_cmp_block_table,
                                                      const std::optional<torch::Tensor>& attention_cmp_seq_lens,
                                                      const std::optional<torch::Tensor>& attention_swa_k_cache,
                                                      const std::optional<torch::Tensor>& attention_swa_cu_lens,
                                                      const std::optional<torch::Tensor>& attention_swa_k_pool,
                                                      const std::optional<torch::Tensor>& attention_swa_slot_mapping,
                                                      const std::optional<torch::Tensor>& attention_swa_gather_lens,
                                                      const std::optional<torch::Tensor>& kv_unpad_restore,
                                                      const std::optional<torch::Tensor>& kv_cu_lens) {
#ifndef USE_ROCM
    if (dsv4CpAttentionDebugSyncEnabled()) {
        std::fprintf(stderr,
                     "[DSV4 CP Attention Cuda] rank=%ld entry wrapper q=(%ld,%ld,%ld) kv=(%ld,%ld,%ld,%ld) "
                     "ratio=%ld cp_size=%ld symm_ptrs=%ld signal_ptrs=%ld\n",
                     static_cast<long>(cp_rank),
                     static_cast<long>(q.dim() > 0 ? q.size(0) : -1),
                     static_cast<long>(q.dim() > 1 ? q.size(1) : -1),
                     static_cast<long>(q.dim() > 2 ? q.size(2) : -1),
                     static_cast<long>(kv.dim() > 0 ? kv.size(0) : -1),
                     static_cast<long>(kv.dim() > 1 ? kv.size(1) : -1),
                     static_cast<long>(kv.dim() > 2 ? kv.size(2) : -1),
                     static_cast<long>(kv.dim() > 3 ? kv.size(3) : -1),
                     static_cast<long>(compress_ratio),
                     static_cast<long>(cp_size),
                     static_cast<long>(symm_buffer_ptrs_dev),
                     static_cast<long>(symm_signal_pad_ptrs_dev));
        std::fflush(stderr);
    }
    return launchDsv4CpDistributedPrefillAttention(q,
                                                   kv,
                                                   indexer_q,
                                                   indexer_k,
                                                   attn_sink,
                                                   req_id_per_token,
                                                   position_ids,
                                                   prefix_lengths,
                                                   input_lengths,
                                                   local_rows,
                                                   compress_ratio,
                                                   window_size,
                                                   compressed_topk,
                                                   compressed_region_width,
                                                   cp_rank,
                                                   cp_size,
                                                   comm_ptr,
                                                   buffer_handle,
                                                   signal_handle,
                                                   per_rank_buffer_bytes,
                                                   rank_offsets,
                                                   swa_k,
                                                   swa_k_cache,
                                                   swa_slot_mapping,
                                                   symm_buffer,
                                                   symm_buffer_ptrs_dev,
                                                   symm_signal_pad_ptrs_dev,
                                                   symm_handle,
                                                   compressor_kv,
                                                   compressor_score,
                                                   compressor_ape,
                                                   compressor_positions,
                                                   compressor_state_cache,
                                                   compressor_state_slots,
                                                   compressor_ratio,
                                                   compressor_token_to_req,
                                                   compressor_state_block_table,
                                                   compressor_norm_weight,
                                                   compressor_cos_sin_cache,
                                                   compressor_kv_cache,
                                                   compressor_kv_slots,
                                                   compressor_seq_start,
                                                   compressor_disable_raw_path,
                                                   compressor_rms_norm_eps,
                                                   compressor_head_dim,
                                                   compressor_rope_head_dim,
                                                   compressor_overlap,
                                                   compressor_state_tokens_per_block,
                                                       compressor_seq_start_per_req,
                                                       compressor_cu_seq_per_req,
                                                       compressor_unpad_restore,
                                                       csa_indexer_compressor_kv,
                                                       csa_indexer_compressor_score,
                                                       csa_indexer_compressor_ape,
                                                       csa_indexer_compressor_positions,
                                                       csa_indexer_compressor_state_cache,
                                                       csa_indexer_compressor_state_slots,
                                                       csa_indexer_compressor_ratio,
                                                       csa_indexer_compressor_token_to_req,
                                                       csa_indexer_compressor_state_block_table,
                                                       csa_indexer_compressor_norm_weight,
                                                       csa_indexer_compressor_cos_sin_cache,
                                                       csa_indexer_compressor_kv_cache,
                                                       csa_indexer_compressor_kv_slots,
                                                       csa_indexer_compressor_seq_start,
                                                       csa_indexer_compressor_disable_raw_path,
                                                       csa_indexer_compressor_rms_norm_eps,
                                                       csa_indexer_compressor_head_dim,
                                                       csa_indexer_compressor_rope_head_dim,
                                                       csa_indexer_compressor_overlap,
                                                       csa_indexer_compressor_state_tokens_per_block,
                                                       csa_indexer_compressor_seq_start_per_req,
                                                       csa_indexer_compressor_cu_seq_per_req,
                                                       csa_indexer_compressor_unpad_restore,
                                                       csa_indexer_k_cache,
                                                   csa_indexer_weights,
                                                   csa_indexer_cu_lens,
                                                   csa_indexer_k_pool,
                                                   csa_indexer_block_table,
                                                   csa_indexer_seq_lens,
                                                   attention_cmp_k_cache,
                                                   attention_cmp_cu_lens,
                                                   attention_cmp_k_pool,
                                                   attention_cmp_block_table,
                                                   attention_cmp_seq_lens,
                                                   attention_swa_k_cache,
                                                   attention_swa_cu_lens,
                                                   attention_swa_k_pool,
                                                   attention_swa_slot_mapping,
                                                   attention_swa_gather_lens,
                                                   kv_unpad_restore,
                                                   kv_cu_lens);
#else
    TORCH_CHECK(false, "dsv4_cp_distributed_prefill_attention is not supported on ROCm");
#endif
}

}  // namespace torch_ext
