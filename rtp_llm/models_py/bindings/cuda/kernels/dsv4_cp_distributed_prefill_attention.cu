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
constexpr int64_t kSymmMemBarrierTimeoutCycles = 60000000000ll;
constexpr float kInvalidCompressorScore = -3.4028234663852886e38F;
constexpr int64_t kMegaPayloadAlignBytes = 256;
constexpr int64_t kMegaLocalPhaseOffset = 16;
constexpr int64_t kMegaLocalPhaseComplementOffset = 24;

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

__device__ __forceinline__ float fp8_e4m3_to_float(uint8_t x, float scale) {
    __half_raw raw = __nv_cvt_fp8_to_halfraw(x, __NV_E4M3);
    __half h = __ushort_as_half(raw.x);
    return __half2float(h) * scale;
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
    if (owner < 0 || local_row < 0) {
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
    if (owner < 0 || local_pos < 0) {
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
        const uint16_t bits = reinterpret_cast<const uint16_t*>(row + kSwaNopeDim)[rope_d];
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
        const uint16_t bits = reinterpret_cast<const uint16_t*>(token_data + kSwaNopeDim)[rope_d];
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
__device__ __forceinline__ float dsv4MegaAttentionLogitForKey(const scalar_t* q,
                                                              const scalar_t* kv,
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
                                                              int64_t row,
                                                              int req,
                                                              int key_idx,
                                                              int prefix_len,
                                                              int h,
                                                              int H,
                                                              int D,
                                                              int L,
                                                              int KH,
                                                              const int64_t* kv_unpad_restore,
                                                              const int64_t* kv_cu_lens,
                                                              const uint8_t* const* __restrict__ symm_buffer_ptrs,
                                                              int use_symm_direct_fresh,
                                                              int64_t per_rank_buffer_bytes,
                                                              int64_t fresh_payload_offset,
                                                              int symm_l_local,
                                                              float scale,
                                                              float* reduce_buf,
                                                              float* logit_s) {
    const int tid = threadIdx.x;
    float dot_part = 0.0f;
    for (int d = tid; d < D; d += blockDim.x) {
        dot_part += q_at(q, row, h, d, H, D)
                    * attention_key_at(kv,
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
                                       key_is_compressed,
                                       req,
                                       key_idx,
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
        *logit_s = reduce_buf[0] * scale;
    }
    __syncthreads();
    return *logit_s;
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
    const float scale = *reinterpret_cast<const float*>(scale_ptr);
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
    const float k_scale = *reinterpret_cast<const float*>(k_row + kIndexerHeadDim);
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
        uint16_t* rope_u16 = reinterpret_cast<uint16_t*>(token_bf16_ptr);
        rope_u16[lane] = k_u16[static_cast<int64_t>(token) * kSwaHeadDim + kSwaNopeDim + lane];
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
                    const int64_t global_raw_idx = static_cast<int64_t>(req_cu_lo + flat_in_req);
                    flat_idx = raw_unpad_restore == nullptr ? global_raw_idx : raw_unpad_restore[global_raw_idx];
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
                        const int64_t global_raw_idx = static_cast<int64_t>(req_cu_lo + flat_in_req);
                        flat_idx = raw_unpad_restore == nullptr ? global_raw_idx : raw_unpad_restore[global_raw_idx];
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
            uint16_t* rope_ptr = reinterpret_cast<uint16_t*>(token_ptr + nope_head_dim);
            rope_ptr[rope_local] = float_to_bf16_bits(result);
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
            *reinterpret_cast<float*>(scale_ptr) = group_scale_shared[0];
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
    if (local_rows <= 0) {
        return local;
    }
    const int owner = global_row / local_rows;
    const int local_row = global_row - owner * local_rows;
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
    return *reinterpret_cast<const uint16_t*>(ptr);
}

__device__ __noinline__ void dsv4MegaWriteSwaCachePhase(const MegaSwaWriterArgs& args,
                                                        const uint8_t* const* __restrict__ symm_buffer_ptrs,
                                                        int64_t per_rank_buffer_bytes,
                                                        int64_t scratch_payload_offset,
                                                        int cp_rank,
                                                        int cp_size,
                                                        uint32_t* const* __restrict__ symm_signal_pads,
                                                        int barrier_channel) {
    if (!args.enabled) {
        __syncthreads();
        return;
    }
    const int64_t payload_bytes = static_cast<int64_t>(args.local_rows) * kSwaHeadDim * sizeof(c10::BFloat16);
    dsv4MegaStageBytes(args.k,
                       payload_bytes,
                       symm_buffer_ptrs,
                       per_rank_buffer_bytes,
                       scratch_payload_offset,
                       cp_rank,
                       cp_size,
                       symm_signal_pads,
                       barrier_channel);

    __shared__ float   abs_values[kSwaQuantBlock];
    __shared__ float   tile_scale;
    __shared__ uint8_t encoded_scale;

    const int tid = static_cast<int>(threadIdx.x);
    const int total_tokens = args.local_rows * cp_size;
    const int tiles = kSwaNopeTiles + 1;
    for (int task = 0; task < total_tokens * tiles; ++task) {
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
            __syncthreads();
            for (int stride = kSwaQuantBlock / 2; stride > 0; stride >>= 1) {
                if (tid < stride) {
                    abs_values[tid] = fmaxf(abs_values[tid], abs_values[tid + stride]);
                }
                __syncthreads();
            }
            if (tid == 0) {
                const float amax = fmaxf(abs_values[0], 1.0e-4f);
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
                uint16_t* rope_u16 = reinterpret_cast<uint16_t*>(token_data_ptr + kSwaNopeDim);
                rope_u16[tid] = dsv4MegaBf16Bits2D(args.k,
                                                   symm_buffer_ptrs,
                                                   per_rank_buffer_bytes,
                                                   scratch_payload_offset,
                                                   args.local_rows,
                                                   kSwaHeadDim,
                                                   token,
                                                   kSwaNopeDim + tid,
                                                   cp_size);
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
                                                            int barrier_channel) {
    if (!args.enabled) {
        __syncthreads();
        return;
    }
    dsv4MegaStageBytes(args.kv,
                       args.kv_bytes,
                       symm_buffer_ptrs,
                       per_rank_buffer_bytes,
                       scratch_payload_offset,
                       cp_rank,
                       cp_size,
                       symm_signal_pads,
                       barrier_channel);
    if (cp_size > 1 && args.score_bytes > 0) {
        uint8_t* dst = const_cast<uint8_t*>(symm_buffer_ptrs[cp_rank])
                       + static_cast<int64_t>(cp_rank) * per_rank_buffer_bytes + scratch_payload_offset + args.kv_bytes;
        const uint8_t* src = reinterpret_cast<const uint8_t*>(args.score);
        for (int64_t off = static_cast<int64_t>(threadIdx.x); off < args.score_bytes; off += blockDim.x) {
            dst[off] = src[off];
        }
        __threadfence_system();
        __syncthreads();
        dsv4MegaInlineSignalPadBarrier(symm_signal_pads, cp_rank, cp_size, barrier_channel + 1);
    } else {
        __syncthreads();
    }

    const int tid = static_cast<int>(threadIdx.x);
    const int total_tokens = args.local_rows * cp_size;
    if (args.state_enabled) {
        for (int token = 0; token < total_tokens; ++token) {
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

    if (!args.kv_writer_enabled) {
        __syncthreads();
        return;
    }

    __shared__ float compressed_shared[512];
    __shared__ float normed_raw_shared[512];
    __shared__ float quant_shared[512];
    __shared__ float reduce_shared[512];
    __shared__ float group_scale_shared[8];

    for (int token = 0; token < total_tokens; ++token) {
        const int64_t position = args.positions[token];
        const int64_t kv_slot = args.kv_slots[token];
        const bool token_active = args.compressor_ratio > 0 && ((position + 1) % args.compressor_ratio) == 0
                                  && kv_slot >= 0;
        const int64_t kv_block_idx = token_active ? kv_slot / args.kv_cache_block_size : 0;
        const int64_t kv_pos_in_blk = token_active ? kv_slot % args.kv_cache_block_size : 0;
        const bool valid_kv_slot = token_active && kv_block_idx >= 0 && kv_block_idx < args.num_kv_blocks;
        const int req_idx = args.token_to_req == nullptr ? 0 : static_cast<int>(args.token_to_req[token]);
        const int start = static_cast<int>(position) - args.window_count + 1;

        for (int d = tid; d < 512; d += blockDim.x) {
            compressed_shared[d] = 0.0f;
            normed_raw_shared[d] = 0.0f;
            quant_shared[d] = 0.0f;
            reduce_shared[d] = 0.0f;
        }
        __syncthreads();

        for (int d = tid; d < args.head_dim; d += blockDim.x) {
            float compressed = 0.0f;
            if (valid_kv_slot) {
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
                            const int64_t global_raw_idx = static_cast<int64_t>(req_cu_lo + flat_in_req);
                            flat_idx = args.raw_unpad_restore == nullptr ? global_raw_idx : args.raw_unpad_restore[global_raw_idx];
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
                                const int64_t global_raw_idx = static_cast<int64_t>(req_cu_lo + flat_in_req);
                                flat_idx = args.raw_unpad_restore == nullptr ? global_raw_idx :
                                                                            args.raw_unpad_restore[global_raw_idx];
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
            }
            compressed_shared[d] = compressed;
            reduce_shared[d] = compressed * compressed;
        }
        __syncthreads();
        for (int stride = 256; stride > 0; stride >>= 1) {
            if (tid < stride) {
                reduce_shared[tid] += reduce_shared[tid + stride];
            }
            __syncthreads();
        }
        const float rrms = rsqrtf(reduce_shared[0] / static_cast<float>(args.head_dim) + args.rms_norm_eps);
        if (valid_kv_slot) {
            for (int d = tid; d < args.head_dim; d += blockDim.x) {
                const float norm = scalar_load_float(args.norm_weight, d);
                normed_raw_shared[d] = compressed_shared[d] * rrms * norm;
            }
        }
        __syncthreads();

        if (!valid_kv_slot) {
            __syncthreads();
            continue;
        }
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
                uint16_t* rope_ptr = reinterpret_cast<uint16_t*>(token_ptr + nope_head_dim);
                rope_ptr[rope_local] = float_to_bf16_bits(result);
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
                *reinterpret_cast<float*>(scale_ptr) = group_scale_shared[0];
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
__global__ __launch_bounds__(256, 1) void dsv4CpDistributedPrefillMegaAttentionKernel(const scalar_t* __restrict__ q,
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
                                                        MegaSwaWriterArgs swa_writer,
                                                        MegaCompressorBf16Args csa_indexer_compressor_writer,
                                                        MegaCompressorBf16Args main_compressor_writer,
                                                        unsigned long long launch_epoch,
                                                        int cp_rank,
                                                        int cp_size,
                                                        uint32_t* const* __restrict__ symm_signal_pads) {
	    const int tid = threadIdx.x;
	    if (blockIdx.x == 0) {
	        dsv4MegaWriteSwaCachePhase(swa_writer,
	                                   symm_buffer_ptrs,
	                                   symm_per_rank_buffer_bytes,
	                                   symm_scratch_payload_offset,
	                                   cp_rank,
	                                   cp_size,
	                                   symm_signal_pads,
	                                   2);
	        dsv4MegaRunCompressorBf16Phase(csa_indexer_compressor_writer,
	                                       symm_buffer_ptrs,
	                                       symm_per_rank_buffer_bytes,
	                                       symm_scratch_payload_offset,
	                                       cp_rank,
	                                       cp_size,
	                                       symm_signal_pads,
	                                       3);
	        dsv4MegaRunCompressorBf16Phase(main_compressor_writer,
	                                       symm_buffer_ptrs,
	                                       symm_per_rank_buffer_bytes,
	                                       symm_scratch_payload_offset,
	                                       cp_rank,
	                                       cp_size,
	                                       symm_signal_pads,
	                                       5);
	        if (use_symm_direct_fresh) {
	            const int64_t payload_bytes = static_cast<int64_t>(symm_l_local) * KH * D * sizeof(scalar_t);
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
	        if (use_symm_direct_indexer) {
	            const int64_t indexer_payload_bytes =
	                static_cast<int64_t>(indexer_batch) * symm_indexer_l_local * IH * ID * sizeof(scalar_t);
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
	        dsv4MegaPublishLocalPhase(symm_buffer_ptrs, symm_per_rank_buffer_bytes, cp_rank, launch_epoch);
	    }
	    dsv4MegaWaitLocalPhase(symm_buffer_ptrs, symm_per_rank_buffer_bytes, cp_rank, launch_epoch);

    __shared__ int   compressed_indices[kMaxCompressedTopK];
    __shared__ float compressed_scores[kMaxCompressedTopK];
    __shared__ float reduce_buf[512];
    __shared__ float logit_s;

    const int compressed_stride = max_int(compressed_topk, 1);
    uint8_t* local_scratch_base = nullptr;
    if (symm_buffer_ptrs != nullptr && symm_per_rank_buffer_bytes > symm_scratch_payload_offset) {
        local_scratch_base = const_cast<uint8_t*>(symm_buffer_ptrs[cp_rank])
                             + static_cast<int64_t>(cp_rank) * symm_per_rank_buffer_bytes
                             + symm_scratch_payload_offset;
    }
    unsigned long long* compressed_meta = reinterpret_cast<unsigned long long*>(local_scratch_base);
    const int64_t compressed_indices_offset =
        alignUpInt64(static_cast<int64_t>(R) * static_cast<int64_t>(sizeof(unsigned long long)),
                     kMegaPayloadAlignBytes);
    int* row_compressed_indices_base =
        local_scratch_base == nullptr ? nullptr : reinterpret_cast<int*>(local_scratch_base + compressed_indices_offset);
    const unsigned int compressed_epoch = static_cast<unsigned int>(launch_epoch & 0xffffffffull);
    const unsigned long long compressed_epoch_prefix = static_cast<unsigned long long>(compressed_epoch) << 32;

    for (int topk_local_row = static_cast<int>(blockIdx.x); topk_local_row < R; topk_local_row += gridDim.x) {
        const int64_t topk_row = local_rows[topk_local_row];
        const int     topk_req = static_cast<int>(req_id_per_token[topk_row]);
        const int     topk_q_pos = static_cast<int>(position_ids[topk_row]);
        const int     topk_kv_len = static_cast<int>(prefix_lengths[topk_req] + input_lengths[topk_req]);
        if (tid == 0) {
            int compressed_count = 0;
            if (compress_ratio == 0) {
                compressed_count = 0;
            } else if (compress_ratio == 128) {
                const int valid = min_int((topk_q_pos + 1) / compress_ratio, compressed_topk);
                int valid_bound = valid;
                if (attention_cmp_seq_lens != nullptr) {
                    valid_bound = min_int(valid_bound, static_cast<int>(attention_cmp_seq_lens[topk_req]));
                }
                for (int c = 0; c < valid_bound && compressed_count < kMaxCompressedTopK; ++c) {
                    const int boundary_pos = (c + 1) * compress_ratio - 1;
                    if (boundary_pos >= 0 && boundary_pos < topk_kv_len) {
                        compressed_indices[compressed_count++] = c;
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
                    req_k_len =
                        csa_indexer_seq_lens == nullptr ? LI : static_cast<int>(csa_indexer_seq_lens[topk_req]);
                } else if (use_fp8_indexer) {
                    req_k_len = static_cast<int>(csa_indexer_cu_lens[topk_req + 1] - csa_indexer_cu_lens[topk_req]);
                    req_k_len = min_int(req_k_len, csa_indexer_total_len);
                }
                const int valid = min_int((topk_q_pos + 1) / compress_ratio, req_k_len);
                const int k_eff = min_int(min_int(valid, compressed_topk), kMaxCompressedTopK);
                for (int c = 0; c < valid; ++c) {
                    const int boundary_pos = (c + 1) * compress_ratio - 1;
                    if (boundary_pos < 0 || boundary_pos >= topk_kv_len) {
                        continue;
                    }
                    float score = 0.0f;
                    if (use_fp8_indexer_pool) {
                        score = indexer_score_fp8_pool(indexer_q,
                                                       csa_indexer_k_pool,
                                                       csa_indexer_block_table,
                                                       csa_indexer_weights,
                                                       topk_row,
                                                       topk_req,
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
                                                        topk_row,
                                                        topk_req,
                                                        c,
                                                        IH,
                                                        ID);
                    } else {
                        score = indexer_score(indexer_q,
                                              indexer_k,
                                              topk_row,
                                              topk_req,
                                              c,
                                              IH,
                                              ID,
                                              LI,
                                              symm_buffer_ptrs,
                                              use_symm_direct_indexer,
                                              symm_per_rank_buffer_bytes,
                                              symm_indexer_payload_offset,
                                              symm_indexer_l_local);
                    }
                    if (compressed_count < k_eff) {
                        compressed_indices[compressed_count] = c;
                        compressed_scores[compressed_count] = score;
                        ++compressed_count;
                        continue;
                    }
                    if (k_eff <= 0) {
                        continue;
                    }
                    int worst_slot = 0;
                    for (int i = 1; i < compressed_count; ++i) {
                        const bool current_is_worse =
                            compressed_scores[i] < compressed_scores[worst_slot]
                            || (compressed_scores[i] == compressed_scores[worst_slot]
                                && compressed_indices[i] > compressed_indices[worst_slot]);
                        if (current_is_worse) {
                            worst_slot = i;
                        }
                    }
                    const bool replace_worst =
                        score > compressed_scores[worst_slot]
                        || (score == compressed_scores[worst_slot] && c < compressed_indices[worst_slot]);
                    if (replace_worst) {
                        compressed_indices[worst_slot] = c;
                        compressed_scores[worst_slot] = score;
                    }
                }
            }
            if (row_compressed_indices_base != nullptr) {
                int* dst = row_compressed_indices_base + static_cast<int64_t>(topk_local_row) * compressed_stride;
                for (int i = 0; i < compressed_count; ++i) {
                    dst[i] = compressed_indices[i];
                }
                cuda::atomic_ref<unsigned long long, cuda::thread_scope_system> meta_ref(
                    compressed_meta[topk_local_row]);
                meta_ref.store(compressed_epoch_prefix | static_cast<unsigned int>(compressed_count),
                               cuda::std::memory_order_release);
            }
        }
        __syncthreads();
    }

    const int task_count = R * H;
    for (int task = blockIdx.x; task < task_count; task += gridDim.x) {
    const int local_row = task / H;
    const int h = task - local_row * H;

    const int64_t row   = local_rows[local_row];
    const int     req   = static_cast<int>(req_id_per_token[row]);
    const int     q_pos = static_cast<int>(position_ids[row]);
    const int     prefix_len = static_cast<int>(prefix_lengths[req]);
    const int     kv_len = static_cast<int>(prefix_lengths[req] + input_lengths[req]);

	    unsigned long long compressed_meta_value = 0;
	    if (compressed_meta != nullptr) {
	        cuda::atomic_ref<unsigned long long, cuda::thread_scope_system> meta_ref(compressed_meta[local_row]);
	        const unsigned long long start = clock64();
	        while (true) {
	            compressed_meta_value = meta_ref.load(cuda::std::memory_order_acquire);
	            if ((compressed_meta_value & 0xffffffff00000000ull) == compressed_epoch_prefix) {
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
	    }

	    const int   compressed_count = static_cast<int>(compressed_meta_value & 0xffffffffu);
	    const int*  row_compressed_indices =
	        row_compressed_indices_base + static_cast<int64_t>(local_row) * compressed_stride;
	    const float scale            = rsqrtf(static_cast<float>(D));
	    const bool  use_compressed_cache = attention_cmp_k_pool != nullptr || attention_cmp_k_cache != nullptr;
	    const int   swa_start = max_int(0, q_pos - window_size + 1);
	    const int   swa_end   = min_int(q_pos + 1, kv_len);

#define DSV4_MEGA_ATTENTION_LOGIT(KEY_IS_COMPRESSED, KEY_POS)                                                        \
    dsv4MegaAttentionLogitForKey<scalar_t>(q,                                                                       \
                                           kv,                                                                      \
                                           attention_cmp_k_cache,                                                   \
                                           attention_cmp_cu_lens,                                                   \
                                           attention_cmp_k_pool,                                                    \
                                           attention_cmp_block_table,                                               \
                                           attention_cmp_seq_lens,                                                  \
                                           attention_cmp_pool_block_size,                                           \
                                           attention_cmp_pool_block_stride,                                         \
                                           attention_cmp_block_table_stride,                                        \
                                           attention_swa_k_cache,                                                   \
                                           attention_swa_cu_lens,                                                   \
                                           attention_swa_k_pool,                                                    \
                                           attention_swa_slot_mapping,                                              \
                                           attention_swa_gather_lens,                                               \
                                           attention_swa_pool_block_size,                                           \
                                           attention_swa_pool_block_stride,                                         \
                                           attention_swa_slot_mapping_stride,                                       \
                                           KEY_IS_COMPRESSED,                                                       \
                                           row,                                                                     \
                                           req,                                                                     \
                                           KEY_POS,                                                                 \
                                           prefix_len,                                                              \
                                           h,                                                                       \
                                           H,                                                                       \
                                           D,                                                                       \
                                           L,                                                                       \
                                           KH,                                                                      \
                                           kv_unpad_restore,                                                        \
                                           kv_cu_lens,                                                              \
                                           symm_buffer_ptrs,                                                        \
                                           use_symm_direct_fresh,                                                   \
                                           symm_per_rank_buffer_bytes,                                              \
                                           symm_fresh_payload_offset,                                               \
                                           symm_l_local,                                                            \
                                           scale,                                                                   \
                                           reduce_buf,                                                              \
                                           &logit_s)

#define DSV4_MEGA_ATTENTION_VALUE(KEY_IS_COMPRESSED, KEY_POS, DIM)                                                   \
    attention_key_at(kv,                                                                                             \
                     attention_cmp_k_cache,                                                                          \
                     attention_cmp_cu_lens,                                                                          \
                     attention_cmp_k_pool,                                                                           \
                     attention_cmp_block_table,                                                                      \
                     attention_cmp_seq_lens,                                                                         \
                     attention_cmp_pool_block_size,                                                                  \
                     attention_cmp_pool_block_stride,                                                                \
                     attention_cmp_block_table_stride,                                                               \
                     attention_swa_k_cache,                                                                          \
                     attention_swa_cu_lens,                                                                          \
                     attention_swa_k_pool,                                                                           \
                     attention_swa_slot_mapping,                                                                     \
                     attention_swa_gather_lens,                                                                      \
                     attention_swa_pool_block_size,                                                                  \
                     attention_swa_pool_block_stride,                                                                \
                     attention_swa_slot_mapping_stride,                                                              \
                     KEY_IS_COMPRESSED,                                                                              \
                     req,                                                                                            \
                     KEY_POS,                                                                                        \
                     prefix_len,                                                                                     \
                     h,                                                                                              \
                     DIM,                                                                                            \
                     L,                                                                                              \
                     KH,                                                                                             \
                     D,                                                                                              \
                     kv_unpad_restore,                                                                               \
                     kv_cu_lens,                                                                                     \
                     symm_buffer_ptrs,                                                                               \
                     use_symm_direct_fresh,                                                                          \
                     symm_per_rank_buffer_bytes,                                                                     \
                     symm_fresh_payload_offset,                                                                      \
                     symm_l_local)

#define DSV4_MEGA_ONLINE_CONSUME(KEY_IS_COMPRESSED, KEY_POS)                                                        \
    do {                                                                                                             \
        const float online_logit = DSV4_MEGA_ATTENTION_LOGIT(KEY_IS_COMPRESSED, KEY_POS);                           \
        const float new_max = fmaxf(online_max, online_logit);                                                       \
        const float old_scale = expf(online_max - new_max);                                                         \
        const float key_scale = expf(online_logit - new_max);                                                        \
        online_denom = online_denom * old_scale + key_scale;                                                        \
        acc0 *= old_scale;                                                                                           \
        acc1 *= old_scale;                                                                                           \
        if (d0 < D) {                                                                                                \
            acc0 += key_scale * DSV4_MEGA_ATTENTION_VALUE(KEY_IS_COMPRESSED, KEY_POS, d0);                           \
        }                                                                                                            \
        if (d1 < D) {                                                                                                \
            acc1 += key_scale * DSV4_MEGA_ATTENTION_VALUE(KEY_IS_COMPRESSED, KEY_POS, d1);                           \
        }                                                                                                            \
        online_max = new_max;                                                                                        \
    } while (0)

	    const int d0 = tid;
	    const int d1 = tid + blockDim.x;
	    float online_max = attn_sink[h];
	    float online_denom = 1.0f;
	    float acc0 = 0.0f;
	    float acc1 = 0.0f;
	    for (int i = 0; i < compressed_count; ++i) {
	        const int cmp_idx = row_compressed_indices[i];
	        const int key_pos = use_compressed_cache ? cmp_idx : (cmp_idx + 1) * compress_ratio - 1;
	        const int key_is_compressed = use_compressed_cache ? 1 : 0;
	        DSV4_MEGA_ONLINE_CONSUME(key_is_compressed, key_pos);
	    }
	    for (int pos = swa_start; pos < swa_end; ++pos) {
	        DSV4_MEGA_ONLINE_CONSUME(0, pos);
	    }
	    if (d0 < D) {
	        output[(static_cast<int64_t>(local_row) * H + h) * D + d0] =
	            from_float_device<scalar_t>(acc0 / online_denom);
	    }
	    if (d1 < D) {
	        output[(static_cast<int64_t>(local_row) * H + h) * D + d1] =
	            from_float_device<scalar_t>(acc1 / online_denom);
	    }
#undef DSV4_MEGA_ONLINE_CONSUME
#undef DSV4_MEGA_ATTENTION_VALUE
#undef DSV4_MEGA_ATTENTION_LOGIT
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
        k_scale[out_row] = *reinterpret_cast<const float*>(scale_ptr);
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
    int64_t mega_scratch_bytes = 0;
    const int64_t mega_swa_bytes =
        static_cast<int64_t>(mega_swa_writer.local_rows) * kSwaHeadDim * static_cast<int64_t>(sizeof(c10::BFloat16));
    mega_scratch_bytes = mega_scratch_bytes > mega_swa_bytes ? mega_scratch_bytes : mega_swa_bytes;
    const int64_t mega_csa_compressor_bytes =
        mega_csa_indexer_compressor_writer.kv_bytes + mega_csa_indexer_compressor_writer.score_bytes;
    mega_scratch_bytes =
        mega_scratch_bytes > mega_csa_compressor_bytes ? mega_scratch_bytes : mega_csa_compressor_bytes;
    const int64_t mega_main_compressor_bytes =
        mega_main_compressor_writer.kv_bytes + mega_main_compressor_writer.score_bytes;
    mega_scratch_bytes =
        mega_scratch_bytes > mega_main_compressor_bytes ? mega_scratch_bytes : mega_main_compressor_bytes;
    const int64_t mega_compressed_index_bytes =
        alignUpInt64(static_cast<int64_t>(local_rows.size(0)) * static_cast<int64_t>(sizeof(unsigned long long)),
                     kMegaPayloadAlignBytes)
        + static_cast<int64_t>(local_rows.size(0))
              * std::max<int64_t>(static_cast<int64_t>(compressed_topk), 1)
              * static_cast<int64_t>(sizeof(int));
    mega_scratch_bytes =
        mega_scratch_bytes > mega_compressed_index_bytes ? mega_scratch_bytes : mega_compressed_index_bytes;
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

    const dim3 block(256);
    const int64_t semantic_task_count = local_rows.size(0) * H;
    int device_id = -1;
    AT_CUDA_CHECK(cudaGetDevice(&device_id));
    cudaDeviceProp device_prop{};
    AT_CUDA_CHECK(cudaGetDeviceProperties(&device_prop, device_id));
    int mega_active_blocks_per_sm = 1;
    if (q.scalar_type() == torch::kFloat32) {
        cudaError_t occ_err = cudaOccupancyMaxActiveBlocksPerMultiprocessor(
            &mega_active_blocks_per_sm,
            dsv4CpDistributedPrefillMegaAttentionKernel<float>,
            static_cast<int>(block.x),
            0);
        if (occ_err != cudaSuccess || mega_active_blocks_per_sm <= 0) {
            mega_active_blocks_per_sm = 1;
        }
    } else if (q.scalar_type() == torch::kBFloat16) {
        cudaError_t occ_err = cudaOccupancyMaxActiveBlocksPerMultiprocessor(
            &mega_active_blocks_per_sm,
            dsv4CpDistributedPrefillMegaAttentionKernel<c10::BFloat16>,
            static_cast<int>(block.x),
            0);
        if (occ_err != cudaSuccess || mega_active_blocks_per_sm <= 0) {
            mega_active_blocks_per_sm = 1;
        }
    }
    const int64_t max_resident_mega_blocks =
        std::max<int64_t>(1, static_cast<int64_t>(mega_active_blocks_per_sm) * device_prop.multiProcessorCount);
    const int64_t mega_grid_count =
        use_mega_kernel ? std::max<int64_t>(1, std::min<int64_t>(semantic_task_count, max_resident_mega_blocks)) : 1;
    const dim3 grid(use_mega_kernel ? static_cast<unsigned int>(mega_grid_count) :
                                      static_cast<unsigned int>(semantic_task_count));
    static std::atomic<unsigned long long> mega_launch_epoch_counter{0xD504C00000000000ull};
    const unsigned long long mega_launch_epoch =
        use_mega_kernel ? mega_launch_epoch_counter.fetch_add(1, std::memory_order_relaxed) + 1ull : 0ull;
    const int attention_kv_len = use_symm_direct_fresh_kv ? static_cast<int>(kv.size(1) * cp_size) :
                                                            static_cast<int>(gathered_kv.size(1));
    const uint8_t* const* symm_buffer_ptrs_for_mega =
        (use_symm_direct_fresh_kv || use_symm_direct_indexer_k || needs_mega_side_effects) ?
            reinterpret_cast<const uint8_t* const*>(symm_buffer_ptrs_dev) :
            nullptr;
    const int symm_l_local = use_symm_direct_fresh_kv ? static_cast<int>(kv.size(1)) : 0;
    const int attention_indexer_len = use_symm_direct_indexer_k ? static_cast<int>(indexer_k.size(1) * cp_size) :
                                                                  static_cast<int>(gathered_indexer_k.size(1));
    const int symm_indexer_l_local = use_symm_direct_indexer_k ? static_cast<int>(indexer_k.size(1)) : 0;
    uint32_t* const* symm_signal_pads_for_mega =
        (use_symm_direct_fresh_kv || use_symm_direct_indexer_k || needs_mega_side_effects) ?
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
        dsv4CpDistributedPrefillMegaAttentionKernel<float><<<grid, block, 0, stream>>>(
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
	            mega_swa_writer,
	            mega_csa_indexer_compressor_writer,
	            mega_main_compressor_writer,
	            mega_launch_epoch,
	            static_cast<int>(cp_rank),
	            static_cast<int>(cp_size),
	            symm_signal_pads_for_mega);
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
        dsv4CpDistributedPrefillMegaAttentionKernel<c10::BFloat16><<<grid, block, 0, stream>>>(
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
	                mega_swa_writer,
	                mega_csa_indexer_compressor_writer,
	                mega_main_compressor_writer,
	                mega_launch_epoch,
	                static_cast<int>(cp_rank),
	                static_cast<int>(cp_size),
	                symm_signal_pads_for_mega);
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
