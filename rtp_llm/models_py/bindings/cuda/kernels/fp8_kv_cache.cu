#include "rtp_llm/models_py/bindings/cuda/kernels/fp8_kv_cache.h"

#include "rtp_llm/models_py/bindings/common/kernels/rotary_position_embedding.h"
#include "rtp_llm/models_py/bindings/cuda/cuda_host_utils.h"
#include "rtp_llm/models_py/bindings/cuda/kernels/scaled_fp8_quant_utils.h"

#include <ATen/Functions.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/macros/Macros.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>
#include <tuple>

#ifdef ENABLE_FP8
#include <cuda_fp8.h>
#endif

namespace rtp_llm {
namespace {

constexpr int   kThreads = 256;
constexpr float kFp8Max  = 448.0f;

enum class CacheLayout : int {
    PhysicalPage = 0,
    KernelPage   = 1,
};

struct CacheGeometry {
    CacheLayout layout;
    int64_t     physical_pages;
    int64_t     storage_pages;
    int64_t     storage_page_size;
    int64_t     cache_page_stride;
    int64_t     scale_page_stride;
};

void check_cuda_contiguous(const at::Tensor& tensor, const char* name) {
    TORCH_CHECK(tensor.is_cuda(), name, " must be a CUDA tensor");
    TORCH_CHECK(tensor.is_contiguous(), name, " must be contiguous");
}

void check_same_device(const at::Tensor& reference, const at::Tensor& tensor, const char* name) {
    TORCH_CHECK(tensor.get_device() == reference.get_device(), name, " must be on the same CUDA device as kv_cache");
}

void check_page_mapping(int64_t physical_page_size, int64_t kernel_page_size, int64_t subdivision) {
    TORCH_CHECK(physical_page_size > 0, "physical_page_size must be positive");
    TORCH_CHECK(kernel_page_size > 0, "kernel_page_size must be positive");
    TORCH_CHECK(subdivision > 0, "subdivision must be positive");
    TORCH_CHECK(physical_page_size == kernel_page_size * subdivision,
                "physical_page_size must equal kernel_page_size * subdivision, got ",
                physical_page_size,
                ", ",
                kernel_page_size,
                ", and ",
                subdivision);
}

CacheGeometry validate_cache_geometry(const at::Tensor& kv_cache,
                                      const at::Tensor& kv_scales,
                                      int64_t           num_heads,
                                      int64_t           head_dim,
                                      int64_t           physical_page_size,
                                      int64_t           kernel_page_size,
                                      int64_t           subdivision) {
    check_cuda_contiguous(kv_cache, "kv_cache");
    check_cuda_contiguous(kv_scales, "kv_scales");
    TORCH_CHECK(kv_cache.scalar_type() == at::ScalarType::Float8_e4m3fn,
                "kv_cache must have dtype torch.float8_e4m3fn");
    TORCH_CHECK(kv_scales.scalar_type() == at::ScalarType::Float, "kv_scales must have dtype torch.float32");
    TORCH_CHECK(kv_scales.dim() == 2, "kv_scales must be a 2D tensor");
    TORCH_CHECK(kv_cache.dim() == 2 || kv_cache.dim() == 5, "kv_cache must be a packed 2D tensor or a 5D paged tensor");
    TORCH_CHECK(num_heads > 0 && head_dim > 0, "num_heads and head_dim must be positive");
    check_page_mapping(physical_page_size, kernel_page_size, subdivision);

    const int64_t physical_scale_stride = 2 * num_heads * physical_page_size;
    const int64_t kernel_scale_stride   = 2 * num_heads * kernel_page_size;

    CacheGeometry geometry{};
    if (kv_scales.size(1) == physical_scale_stride) {
        geometry.layout            = CacheLayout::PhysicalPage;
        geometry.physical_pages    = kv_scales.size(0);
        geometry.storage_pages     = geometry.physical_pages;
        geometry.storage_page_size = physical_page_size;
    } else {
        TORCH_CHECK(kv_scales.size(1) == kernel_scale_stride,
                    "kv_scales.size(1) must be 2 * H * physical_page_size (",
                    physical_scale_stride,
                    ") or 2 * H * kernel_page_size (",
                    kernel_scale_stride,
                    "), got ",
                    kv_scales.size(1));
        TORCH_CHECK(kv_scales.size(0) % subdivision == 0,
                    "kernel-page kv_scales.size(0) must be divisible by subdivision");
        geometry.layout            = CacheLayout::KernelPage;
        geometry.storage_pages     = kv_scales.size(0);
        geometry.physical_pages    = geometry.storage_pages / subdivision;
        geometry.storage_page_size = kernel_page_size;
    }
    TORCH_CHECK(geometry.physical_pages > 0, "kv_cache must contain at least one physical page");

    const int64_t payload_per_storage_page = 2 * num_heads * geometry.storage_page_size * head_dim;
    if (kv_cache.dim() == 5) {
        TORCH_CHECK(kv_cache.size(0) == geometry.storage_pages && kv_cache.size(1) == 2 && kv_cache.size(2) == num_heads
                        && kv_cache.size(3) == geometry.storage_page_size && kv_cache.size(4) == head_dim,
                    "5D kv_cache shape does not match the inferred cache layout");
        geometry.cache_page_stride = payload_per_storage_page;
    } else {
        TORCH_CHECK(kv_cache.size(0) == geometry.storage_pages, "packed kv_cache page count does not match kv_scales");
        TORCH_CHECK(kv_cache.size(1) >= payload_per_storage_page,
                    "packed kv_cache row is smaller than the required KV payload");
        geometry.cache_page_stride = kv_cache.size(1);
    }
    geometry.scale_page_stride = kv_scales.size(1);
    return geometry;
}

#ifndef NDEBUG
void validate_unique_destinations_async(const at::Tensor& physical_page_ids,
                                        const at::Tensor& token_offsets,
                                        int64_t           physical_page_size) {
    if (physical_page_ids.numel() <= 1) {
        return;
    }
    const auto destinations        = physical_page_ids.to(at::kLong) * physical_page_size + token_offsets.to(at::kLong);
    const auto sorted_destinations = std::get<0>(destinations.sort());
    const auto unique_destinations = sorted_destinations.slice(0, 1, sorted_destinations.numel())
                                         .ne(sorted_destinations.slice(0, 0, sorted_destinations.numel() - 1))
                                         .all();
    at::_assert_async(unique_destinations, "FP8 KV cache write destinations must be unique");
}
#endif

#ifdef ENABLE_FP8

template<typename T>
struct FusedVec;

template<>
struct FusedVec<half> {
    using Type                = uint32_t;
    static constexpr int size = 2;
};

#ifdef ENABLE_BF16
template<>
struct FusedVec<__nv_bfloat16> {
    using Type                = __nv_bfloat162;
    static constexpr int size = 2;
};
#endif

template<typename T, RopeStyle ROPE_STYLE>
__global__ void fused_rope_quantize_and_write_fp8_kv_cache_kernel(const T* __restrict__ qkv,
                                                                  T* __restrict__ q_output,
                                                                  __nv_fp8_e4m3* __restrict__ kv_cache,
                                                                  float* __restrict__ kv_scales,
                                                                  const int32_t* __restrict__ batch_indices,
                                                                  const int32_t* __restrict__ positions,
                                                                  const int32_t* __restrict__ page_indptr,
                                                                  const int32_t* __restrict__ page_indices,
                                                                  int64_t       num_tokens,
                                                                  int64_t       num_q_heads,
                                                                  int64_t       num_kv_heads,
                                                                  int64_t       head_dim,
                                                                  int64_t       storage_pages,
                                                                  int64_t       kernel_page_size,
                                                                  int64_t       cache_page_stride,
                                                                  int64_t       scale_page_stride,
                                                                  int64_t       page_indptr_size,
                                                                  int64_t       page_indices_size,
                                                                  int64_t       cos_sin_rows,
                                                                  RopeConfig    rope_config,
                                                                  const float2* cos_sin_cache) {
    extern __shared__ __align__(sizeof(float2)) char rope_smem[];
    __shared__ float                                 maxima[2][WARP_SIZE];

    using Vec              = typename FusedVec<T>::Type;
    constexpr int vec_size = FusedVec<T>::size;

    const int64_t token_idx    = blockIdx.x;
    const int64_t head_idx     = blockIdx.y;
    const int64_t vec_offset   = static_cast<int64_t>(threadIdx.x) * vec_size;
    const bool    in_head      = vec_offset < head_dim;
    const bool    owns_kv      = head_idx < num_kv_heads;
    const int64_t packed_width = (num_q_heads + 2 * num_kv_heads) * head_dim;

    Vec q{};
    Vec k{};
    Vec v{};
    if (in_head) {
        const int64_t q_offset = token_idx * packed_width + head_idx * head_dim + vec_offset;
        q                      = *reinterpret_cast<const Vec*>(qkv + q_offset);
        if (owns_kv) {
            const int64_t k_offset =
                token_idx * packed_width + num_q_heads * head_dim + head_idx * head_dim + vec_offset;
            const int64_t v_offset = k_offset + num_kv_heads * head_dim;
            k                      = *reinterpret_cast<const Vec*>(qkv + k_offset);
            v                      = *reinterpret_cast<const Vec*>(qkv + v_offset);
        }
    }

    const int32_t position       = positions[token_idx];
    const bool    valid_position = position >= 0 && (cos_sin_cache == nullptr || position < cos_sin_rows);
    CUDA_KERNEL_ASSERT_MSG(valid_position, "fused FP8 KV cache RoPE position is out of bounds");
    if (!valid_position) {
        return;
    }
    apply_rope<T, Vec, ROPE_STYLE>(
        rope_config, q, reinterpret_cast<T*>(rope_smem), threadIdx.x, position, position + 1, cos_sin_cache);
    if (owns_kv) {
        apply_rope<T, Vec, ROPE_STYLE>(
            rope_config, k, reinterpret_cast<T*>(rope_smem), threadIdx.x, position, position + 1, cos_sin_cache);
    }

    if (in_head) {
        const int64_t q_offset                       = (token_idx * num_q_heads + head_idx) * head_dim + vec_offset;
        *reinterpret_cast<Vec*>(q_output + q_offset) = q;
    }

    if (!owns_kv) {
        return;
    }

    float k_max = 0.0f;
    float v_max = 0.0f;
    if (in_head) {
#pragma unroll
        for (int i = 0; i < vec_size; ++i) {
            const float k_value = static_cast<float>(reinterpret_cast<T*>(&k)[i]);
            const float v_value = static_cast<float>(reinterpret_cast<T*>(&v)[i]);
            if (isfinite(k_value)) {
                k_max = fmaxf(k_max, fabsf(k_value));
            }
            if (isfinite(v_value)) {
                v_max = fmaxf(v_max, fabsf(v_value));
            }
        }
    }
    k_max                = warpReduceMax(k_max);
    v_max                = warpReduceMax(v_max);
    const int lane_id    = threadIdx.x % WARP_SIZE;
    const int warp_id    = threadIdx.x / WARP_SIZE;
    const int warp_count = blockDim.x / WARP_SIZE;
    if (lane_id == 0) {
        maxima[0][warp_id] = k_max;
        maxima[1][warp_id] = v_max;
    }
    __syncthreads();
    if (warp_id == 0) {
        k_max = lane_id < warp_count ? maxima[0][lane_id] : 0.0f;
        v_max = lane_id < warp_count ? maxima[1][lane_id] : 0.0f;
        k_max = warpReduceMax(k_max);
        v_max = warpReduceMax(v_max);
        if (lane_id == 0) {
            maxima[0][0] = k_max;
            maxima[1][0] = v_max;
        }
    }
    __syncthreads();

    const int64_t batch_idx   = static_cast<int64_t>(batch_indices[token_idx]);
    const bool    valid_batch = batch_idx >= 0 && batch_idx + 1 < page_indptr_size;
    CUDA_KERNEL_ASSERT_MSG(valid_batch, "fused FP8 KV cache batch index is out of bounds");
    if (!valid_batch || position < 0) {
        return;
    }
    const int64_t page_offset = position / kernel_page_size;
    const int64_t page_slot   = static_cast<int64_t>(page_indptr[batch_idx]) + page_offset;
    const bool    valid_slot =
        page_slot >= 0 && page_slot < page_indices_size && page_slot < static_cast<int64_t>(page_indptr[batch_idx + 1]);
    CUDA_KERNEL_ASSERT_MSG(valid_slot, "fused FP8 KV cache page slot is out of bounds");
    if (!valid_slot) {
        return;
    }
    const int64_t storage_page  = static_cast<int64_t>(page_indices[page_slot]);
    const int64_t storage_token = position % kernel_page_size;
    const bool    valid_page    = storage_page >= 0 && storage_page < storage_pages;
    CUDA_KERNEL_ASSERT_MSG(valid_page, "fused FP8 KV cache page id is out of bounds");
    if (!valid_page) {
        return;
    }

    const float k_scale     = maxima[0][0] == 0.0f ? 1.0f : maxima[0][0] / kFp8Max;
    const float v_scale     = maxima[1][0] == 0.0f ? 1.0f : maxima[1][0] / kFp8Max;
    const float k_inv_scale = 1.0f / k_scale;
    const float v_inv_scale = 1.0f / v_scale;
    if (threadIdx.x == 0) {
        kv_scales[storage_page * scale_page_stride + head_idx * kernel_page_size + storage_token] = k_scale;
        kv_scales[storage_page * scale_page_stride + (num_kv_heads + head_idx) * kernel_page_size + storage_token] =
            v_scale;
    }

    if (in_head) {
        const int64_t k_cache_row =
            storage_page * cache_page_stride + (head_idx * kernel_page_size + storage_token) * head_dim;
        const int64_t v_cache_row = storage_page * cache_page_stride
                                    + ((num_kv_heads + head_idx) * kernel_page_size + storage_token) * head_dim;
#pragma unroll
        for (int i = 0; i < vec_size; ++i) {
            const float k_value                    = static_cast<float>(reinterpret_cast<T*>(&k)[i]);
            const float v_value                    = static_cast<float>(reinterpret_cast<T*>(&v)[i]);
            const float quantized_k                = isnan(k_value) ? 0.0f :
                                                     isinf(k_value) ? copysignf(kFp8Max, k_value) :
                                                                      fmaxf(-kFp8Max, fminf(kFp8Max, k_value * k_inv_scale));
            const float quantized_v                = isnan(v_value) ? 0.0f :
                                                     isinf(v_value) ? copysignf(kFp8Max, v_value) :
                                                                      fmaxf(-kFp8Max, fminf(kFp8Max, v_value * v_inv_scale));
            kv_cache[k_cache_row + vec_offset + i] = static_cast<__nv_fp8_e4m3>(quantized_k);
            kv_cache[v_cache_row + vec_offset + i] = static_cast<__nv_fp8_e4m3>(quantized_v);
        }
    }
}

template<typename T, RopeStyle ROPE_STYLE>
void launch_fused_rope_quantize_and_write(const at::Tensor&    qkv,
                                          at::Tensor&          q_output,
                                          at::Tensor&          kv_cache,
                                          at::Tensor&          kv_scales,
                                          const at::Tensor&    batch_indices,
                                          const at::Tensor&    positions,
                                          const at::Tensor&    page_indptr,
                                          const at::Tensor&    page_indices,
                                          int64_t              num_q_heads,
                                          int64_t              num_kv_heads,
                                          int64_t              head_dim,
                                          int64_t              kernel_page_size,
                                          const CacheGeometry& geometry,
                                          const RopeConfig&    rope_config,
                                          const float2*        cos_sin_cache,
                                          int64_t              cos_sin_rows,
                                          cudaStream_t         stream) {
    if (qkv.size(0) == 0) {
        return;
    }
    int threads = 32;
    while (threads < (head_dim + FusedVec<T>::size - 1) / FusedVec<T>::size) {
        threads *= 2;
    }
    const dim3   grid(qkv.size(0), num_q_heads);
    const size_t smem_size = ROPE_STYLE == RopeStyle::No ? 0 : 2 * rope_config.dim * sizeof(T);
    fused_rope_quantize_and_write_fp8_kv_cache_kernel<T, ROPE_STYLE>
        <<<grid, threads, smem_size, stream>>>(reinterpret_cast<const T*>(qkv.data_ptr()),
                                               reinterpret_cast<T*>(q_output.data_ptr()),
                                               reinterpret_cast<__nv_fp8_e4m3*>(kv_cache.data_ptr()),
                                               kv_scales.data_ptr<float>(),
                                               batch_indices.data_ptr<int32_t>(),
                                               positions.data_ptr<int32_t>(),
                                               page_indptr.data_ptr<int32_t>(),
                                               page_indices.data_ptr<int32_t>(),
                                               qkv.size(0),
                                               num_q_heads,
                                               num_kv_heads,
                                               head_dim,
                                               geometry.storage_pages,
                                               kernel_page_size,
                                               geometry.cache_page_stride,
                                               geometry.scale_page_stride,
                                               page_indptr.numel(),
                                               page_indices.numel(),
                                               cos_sin_rows,
                                               rope_config,
                                               cos_sin_cache);
}

template<typename T, typename IndexT>
__global__ void quantize_and_write_fp8_kv_cache_kernel(const T* __restrict__ k,
                                                       const T* __restrict__ v,
                                                       __nv_fp8_e4m3* __restrict__ kv_cache,
                                                       float* __restrict__ kv_scales,
                                                       const IndexT* __restrict__ physical_page_ids,
                                                       const IndexT* __restrict__ token_offsets,
                                                       int64_t     num_tokens,
                                                       int64_t     num_heads,
                                                       int64_t     head_dim,
                                                       int64_t     physical_pages,
                                                       int64_t     physical_page_size,
                                                       int64_t     kernel_page_size,
                                                       int64_t     subdivision,
                                                       int64_t     cache_page_stride,
                                                       int64_t     scale_page_stride,
                                                       CacheLayout layout) {
    const int64_t row = blockIdx.x;
    const int64_t h   = row % num_heads;
    const int64_t kv  = (row / num_heads) % 2;
    const int64_t n   = row / (2 * num_heads);
    if (n >= num_tokens) {
        return;
    }

    const int64_t physical_page = static_cast<int64_t>(physical_page_ids[n]);
    const int64_t token_offset  = static_cast<int64_t>(token_offsets[n]);
    const bool    valid_mapping =
        physical_page >= 0 && physical_page < physical_pages && token_offset >= 0 && token_offset < physical_page_size;
    CUDA_KERNEL_ASSERT_MSG(valid_mapping, "FP8 KV cache write mapping is out of bounds");
    if (!valid_mapping) {
        return;
    }

    int64_t storage_page  = physical_page;
    int64_t storage_token = token_offset;
    if (layout == CacheLayout::KernelPage) {
        storage_page  = physical_page * subdivision + token_offset / kernel_page_size;
        storage_token = token_offset % kernel_page_size;
    }

    const T* input_row = (kv == 0 ? k : v) + (n * num_heads + h) * head_dim;

    __shared__ float maxima[kThreads];
    float            local_max = 0.0f;
    for (int64_t d = threadIdx.x; d < head_dim; d += blockDim.x) {
        const float value = static_cast<float>(input_row[d]);
        if (isfinite(value)) {
            local_max = fmaxf(local_max, fabsf(value));
        }
    }
    maxima[threadIdx.x] = local_max;
    __syncthreads();
    for (int offset = blockDim.x / 2; offset > 0; offset /= 2) {
        if (threadIdx.x < offset) {
            maxima[threadIdx.x] = fmaxf(maxima[threadIdx.x], maxima[threadIdx.x + offset]);
        }
        __syncthreads();
    }

    const float scale = maxima[0] == 0.0f ? 1.0f : maxima[0] / kFp8Max;
    if (threadIdx.x == 0) {
        const int64_t scale_index =
            storage_page * scale_page_stride
            + (kv * num_heads + h) * (layout == CacheLayout::PhysicalPage ? physical_page_size : kernel_page_size)
            + storage_token;
        kv_scales[scale_index] = scale;
    }

    const float   inv_scale = 1.0f / scale;
    const int64_t cache_row =
        storage_page * cache_page_stride
        + ((kv * num_heads + h) * (layout == CacheLayout::PhysicalPage ? physical_page_size : kernel_page_size)
           + storage_token)
              * head_dim;
    for (int64_t d = threadIdx.x; d < head_dim; d += blockDim.x) {
        const float input_value = static_cast<float>(input_row[d]);
        float       quantized_value;
        if (isnan(input_value)) {
            quantized_value = 0.0f;
        } else if (isinf(input_value)) {
            quantized_value = copysignf(kFp8Max, input_value);
        } else {
            quantized_value = fmaxf(-kFp8Max, fminf(kFp8Max, input_value * inv_scale));
        }
        kv_cache[cache_row + d] = static_cast<__nv_fp8_e4m3>(quantized_value);
    }
}

template<typename T, typename IndexT>
__global__ void gather_and_dequantize_fp8_kv_cache_kernel(const __nv_fp8_e4m3* __restrict__ kv_cache,
                                                          const float* __restrict__ kv_scales,
                                                          const IndexT* __restrict__ source_kernel_page_ids,
                                                          T* __restrict__ output,
                                                          int64_t     rows,
                                                          int64_t     num_heads,
                                                          int64_t     head_dim,
                                                          int64_t     physical_pages,
                                                          int64_t     physical_page_size,
                                                          int64_t     kernel_page_size,
                                                          int64_t     subdivision,
                                                          int64_t     cache_page_stride,
                                                          int64_t     scale_page_stride,
                                                          CacheLayout layout) {
    const int64_t total = rows * 2 * num_heads * kernel_page_size * head_dim;
    for (int64_t output_index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x; output_index < total;
         output_index += static_cast<int64_t>(blockDim.x) * gridDim.x) {
        int64_t       cursor = output_index;
        const int64_t d      = cursor % head_dim;
        cursor /= head_dim;
        const int64_t token = cursor % kernel_page_size;
        cursor /= kernel_page_size;
        const int64_t h = cursor % num_heads;
        cursor /= num_heads;
        const int64_t kv = cursor % 2;
        const int64_t r  = cursor / 2;

        const int64_t kernel_page   = static_cast<int64_t>(source_kernel_page_ids[r]);
        const int64_t physical_page = kernel_page / subdivision;
        const bool    valid_page    = kernel_page >= 0 && physical_page < physical_pages;
        CUDA_KERNEL_ASSERT_MSG(valid_page, "FP8 KV cache gather page id is out of bounds");
        if (!valid_page) {
            return;
        }

        const int64_t subpage       = kernel_page % subdivision;
        const int64_t storage_page  = layout == CacheLayout::PhysicalPage ? physical_page : kernel_page;
        const int64_t storage_token = layout == CacheLayout::PhysicalPage ? subpage * kernel_page_size + token : token;
        const int64_t storage_page_size = layout == CacheLayout::PhysicalPage ? physical_page_size : kernel_page_size;
        const int64_t cache_index       = storage_page * cache_page_stride
                                    + ((kv * num_heads + h) * storage_page_size + storage_token) * head_dim + d;
        const int64_t scale_index =
            storage_page * scale_page_stride + (kv * num_heads + h) * storage_page_size + storage_token;
        output[output_index] = static_cast<T>(static_cast<float>(kv_cache[cache_index]) * kv_scales[scale_index]);
    }
}

template<typename T, typename IndexT>
void launch_quantize_and_write(const at::Tensor&    k,
                               const at::Tensor&    v,
                               at::Tensor&          kv_cache,
                               at::Tensor&          kv_scales,
                               const at::Tensor&    physical_page_ids,
                               const at::Tensor&    token_offsets,
                               const CacheGeometry& geometry,
                               int64_t              physical_page_size,
                               int64_t              kernel_page_size,
                               int64_t              subdivision,
                               cudaStream_t         stream) {
    const int64_t rows = k.size(0) * 2 * k.size(1);
    if (rows == 0) {
        return;
    }
    quantize_and_write_fp8_kv_cache_kernel<T, IndexT>
        <<<rows, kThreads, 0, stream>>>(reinterpret_cast<const T*>(k.data_ptr()),
                                        reinterpret_cast<const T*>(v.data_ptr()),
                                        reinterpret_cast<__nv_fp8_e4m3*>(kv_cache.data_ptr()),
                                        kv_scales.data_ptr<float>(),
                                        physical_page_ids.data_ptr<IndexT>(),
                                        token_offsets.data_ptr<IndexT>(),
                                        k.size(0),
                                        k.size(1),
                                        k.size(2),
                                        geometry.physical_pages,
                                        physical_page_size,
                                        kernel_page_size,
                                        subdivision,
                                        geometry.cache_page_stride,
                                        geometry.scale_page_stride,
                                        geometry.layout);
}

template<typename T, typename IndexT>
void launch_gather(const at::Tensor&    kv_cache,
                   const at::Tensor&    kv_scales,
                   const at::Tensor&    source_kernel_page_ids,
                   at::Tensor&          output,
                   const CacheGeometry& geometry,
                   int64_t              physical_page_size,
                   int64_t              kernel_page_size,
                   int64_t              subdivision,
                   cudaStream_t         stream) {
    const int64_t total = output.numel();
    if (total == 0) {
        return;
    }
    const int64_t blocks = std::min<int64_t>((total + kThreads - 1) / kThreads, 4096);
    gather_and_dequantize_fp8_kv_cache_kernel<T, IndexT>
        <<<blocks, kThreads, 0, stream>>>(reinterpret_cast<const __nv_fp8_e4m3*>(kv_cache.data_ptr()),
                                          kv_scales.data_ptr<float>(),
                                          source_kernel_page_ids.data_ptr<IndexT>(),
                                          reinterpret_cast<T*>(output.data_ptr()),
                                          output.size(0),
                                          output.size(2),
                                          output.size(4),
                                          geometry.physical_pages,
                                          physical_page_size,
                                          kernel_page_size,
                                          subdivision,
                                          geometry.cache_page_stride,
                                          geometry.scale_page_stride,
                                          geometry.layout);
}

#endif  // ENABLE_FP8

}  // namespace

at::Tensor fused_rope_quantize_and_write_fp8_kv_cache(const at::Tensor&                qkv,
                                                      at::Tensor&                      kv_cache,
                                                      at::Tensor&                      kv_scales,
                                                      const at::Tensor&                batch_indices,
                                                      const at::Tensor&                positions,
                                                      const at::Tensor&                page_indptr,
                                                      const at::Tensor&                page_indices,
                                                      int64_t                          num_q_heads,
                                                      int64_t                          num_kv_heads,
                                                      int64_t                          kernel_page_size,
                                                      const RopeConfig&                rope_config,
                                                      const std::optional<at::Tensor>& cos_sin_cache) {
#ifndef ENABLE_FP8
    TORCH_CHECK(false, "FP8 support is not enabled in this CUDA build");
#else
    check_cuda_contiguous(qkv, "qkv");
    check_cuda_contiguous(batch_indices, "batch_indices");
    check_cuda_contiguous(positions, "positions");
    check_cuda_contiguous(page_indptr, "page_indptr");
    check_cuda_contiguous(page_indices, "page_indices");
    TORCH_CHECK(qkv.dim() == 2, "qkv must have shape [N, (QH + 2 * KVH) * D]");
    TORCH_CHECK(qkv.scalar_type() == at::ScalarType::Half || qkv.scalar_type() == at::ScalarType::BFloat16,
                "qkv must have dtype torch.float16 or torch.bfloat16");
    TORCH_CHECK(num_q_heads > 0 && num_kv_heads > 0 && num_q_heads >= num_kv_heads,
                "num_q_heads and num_kv_heads must be positive with num_q_heads >= num_kv_heads");
    TORCH_CHECK(num_q_heads <= 65535, "num_q_heads exceeds the CUDA grid y dimension limit");
    const int64_t packed_heads = num_q_heads + 2 * num_kv_heads;
    TORCH_CHECK(qkv.size(1) % packed_heads == 0, "qkv packed width must be divisible by QH + 2 * KVH");
    const int64_t head_dim = qkv.size(1) / packed_heads;
    TORCH_CHECK(head_dim > 0 && head_dim % 2 == 0 && head_dim <= 2 * kThreads,
                "head_dim must be positive, even, and at most ",
                2 * kThreads,
                ", got ",
                head_dim);
    TORCH_CHECK(kernel_page_size > 0, "kernel_page_size must be positive");
    TORCH_CHECK(batch_indices.dim() == 1 && batch_indices.numel() >= qkv.size(0),
                "batch_indices must contain at least N elements");
    TORCH_CHECK(positions.dim() == 1 && positions.numel() >= qkv.size(0), "positions must contain at least N elements");
    TORCH_CHECK(page_indptr.dim() == 1 && page_indptr.numel() >= 2, "page_indptr must contain at least two elements");
    TORCH_CHECK(page_indices.dim() == 1, "page_indices must be one-dimensional");
    TORCH_CHECK(batch_indices.scalar_type() == at::ScalarType::Int && positions.scalar_type() == at::ScalarType::Int
                    && page_indptr.scalar_type() == at::ScalarType::Int
                    && page_indices.scalar_type() == at::ScalarType::Int,
                "batch_indices, positions, page_indptr, and page_indices must have dtype torch.int32");
    TORCH_CHECK(rope_config.style != RopeStyle::Mrope, "fused dynamic FP8 decode does not support MRoPE");
    if (rope_config.style != RopeStyle::No) {
        TORCH_CHECK(rope_config.dim > 0 && rope_config.dim <= head_dim && rope_config.dim % 2 == 0,
                    "rope_config.dim must be positive, even, and no larger than head_dim");
    }

    check_same_device(kv_cache, qkv, "qkv");
    check_same_device(kv_cache, kv_scales, "kv_scales");
    check_same_device(kv_cache, batch_indices, "batch_indices");
    check_same_device(kv_cache, positions, "positions");
    check_same_device(kv_cache, page_indptr, "page_indptr");
    check_same_device(kv_cache, page_indices, "page_indices");
    const CacheGeometry geometry =
        validate_cache_geometry(kv_cache, kv_scales, num_kv_heads, head_dim, kernel_page_size, kernel_page_size, 1);

    const float2* cos_sin_ptr  = nullptr;
    int64_t       cos_sin_rows = 0;
    if (cos_sin_cache.has_value() && cos_sin_cache->defined() && cos_sin_cache->numel() != 0) {
        check_cuda_contiguous(*cos_sin_cache, "cos_sin_cache");
        check_same_device(kv_cache, *cos_sin_cache, "cos_sin_cache");
        TORCH_CHECK(cos_sin_cache->scalar_type() == at::ScalarType::Float,
                    "cos_sin_cache must have dtype torch.float32");
        TORCH_CHECK(cos_sin_cache->dim() == 2 && cos_sin_cache->size(1) == rope_config.dim,
                    "cos_sin_cache must have shape [max_positions, rope_config.dim]");
        TORCH_CHECK(rope_config.style == RopeStyle::Base || rope_config.style == RopeStyle::Yarn,
                    "cos_sin_cache is only supported for Base and Yarn RoPE");
        cos_sin_ptr  = reinterpret_cast<const float2*>(cos_sin_cache->data_ptr<float>());
        cos_sin_rows = cos_sin_cache->size(0);
    }

    const c10::cuda::CUDAGuard device_guard(kv_cache.device());
    at::Tensor                 q_output = at::empty({qkv.size(0), num_q_heads, head_dim}, qkv.options());
    const cudaStream_t         stream   = at::cuda::getCurrentCUDAStream(kv_cache.get_device()).stream();
#define LAUNCH_FUSED(input_type)                                                                                       \
    FT_ROPE_SWITCH(rope_config.style, ROPE_STYLE, [&] {                                                                \
        launch_fused_rope_quantize_and_write<input_type, ROPE_STYLE>(qkv,                                              \
                                                                     q_output,                                         \
                                                                     kv_cache,                                         \
                                                                     kv_scales,                                        \
                                                                     batch_indices,                                    \
                                                                     positions,                                        \
                                                                     page_indptr,                                      \
                                                                     page_indices,                                     \
                                                                     num_q_heads,                                      \
                                                                     num_kv_heads,                                     \
                                                                     head_dim,                                         \
                                                                     kernel_page_size,                                 \
                                                                     geometry,                                         \
                                                                     rope_config,                                      \
                                                                     cos_sin_ptr,                                      \
                                                                     cos_sin_rows,                                     \
                                                                     stream);                                          \
    })
    if (qkv.scalar_type() == at::ScalarType::Half) {
        LAUNCH_FUSED(half);
    } else {
#ifdef ENABLE_BF16
        LAUNCH_FUSED(__nv_bfloat16);
#else
        TORCH_CHECK(false, "BF16 support is not enabled in this CUDA build");
#endif
    }
#undef LAUNCH_FUSED
    check_cuda_error();
    return q_output;
#endif
}

void quantize_and_write_fp8_kv_cache(const at::Tensor& k,
                                     const at::Tensor& v,
                                     at::Tensor&       kv_cache,
                                     at::Tensor&       kv_scales,
                                     const at::Tensor& target_physical_page_ids,
                                     const at::Tensor& token_offsets,
                                     int64_t           physical_page_size,
                                     int64_t           kernel_page_size,
                                     int64_t           subdivision) {
#ifndef ENABLE_FP8
    TORCH_CHECK(false, "FP8 support is not enabled in this CUDA build");
#else
    check_cuda_contiguous(k, "k");
    check_cuda_contiguous(v, "v");
    check_cuda_contiguous(target_physical_page_ids, "target_physical_page_ids");
    check_cuda_contiguous(token_offsets, "token_offsets");
    TORCH_CHECK(k.dim() == 3, "k must have shape [N, H, D]");
    TORCH_CHECK(v.sizes() == k.sizes(), "v must have the same [N, H, D] shape as k");
    TORCH_CHECK(k.scalar_type() == at::ScalarType::Half || k.scalar_type() == at::ScalarType::BFloat16,
                "k and v must have dtype torch.float16 or torch.bfloat16");
    TORCH_CHECK(v.scalar_type() == k.scalar_type(), "v must have the same dtype as k");
    TORCH_CHECK(target_physical_page_ids.dim() == 1 && target_physical_page_ids.numel() == k.size(0),
                "target_physical_page_ids must have shape [N]");
    TORCH_CHECK(token_offsets.dim() == 1 && token_offsets.numel() == k.size(0), "token_offsets must have shape [N]");
    TORCH_CHECK(target_physical_page_ids.scalar_type() == at::ScalarType::Int
                    || target_physical_page_ids.scalar_type() == at::ScalarType::Long,
                "target_physical_page_ids must have dtype torch.int32 or torch.int64");
    TORCH_CHECK(token_offsets.scalar_type() == target_physical_page_ids.scalar_type(),
                "token_offsets must have the same integer dtype as target_physical_page_ids");

    check_same_device(kv_cache, k, "k");
    check_same_device(kv_cache, v, "v");
    check_same_device(kv_cache, kv_scales, "kv_scales");
    check_same_device(kv_cache, target_physical_page_ids, "target_physical_page_ids");
    check_same_device(kv_cache, token_offsets, "token_offsets");
    const CacheGeometry geometry = validate_cache_geometry(
        kv_cache, kv_scales, k.size(1), k.size(2), physical_page_size, kernel_page_size, subdivision);
    const c10::cuda::CUDAGuard device_guard(kv_cache.device());
#ifndef NDEBUG
    validate_unique_destinations_async(target_physical_page_ids, token_offsets, physical_page_size);
#endif
    const cudaStream_t stream = at::cuda::getCurrentCUDAStream(kv_cache.get_device()).stream();
#define LAUNCH_WRITE(input_type, index_type)                                                                           \
    launch_quantize_and_write<input_type, index_type>(k,                                                               \
                                                      v,                                                               \
                                                      kv_cache,                                                        \
                                                      kv_scales,                                                       \
                                                      target_physical_page_ids,                                        \
                                                      token_offsets,                                                   \
                                                      geometry,                                                        \
                                                      physical_page_size,                                              \
                                                      kernel_page_size,                                                \
                                                      subdivision,                                                     \
                                                      stream)
    if (target_physical_page_ids.scalar_type() == at::ScalarType::Int) {
        if (k.scalar_type() == at::ScalarType::Half) {
            LAUNCH_WRITE(__half, int32_t);
        } else {
            LAUNCH_WRITE(__nv_bfloat16, int32_t);
        }
    } else if (k.scalar_type() == at::ScalarType::Half) {
        LAUNCH_WRITE(__half, int64_t);
    } else {
        LAUNCH_WRITE(__nv_bfloat16, int64_t);
    }
#undef LAUNCH_WRITE
    check_cuda_error();
#endif
}

void gather_and_dequantize_fp8_kv_cache(const at::Tensor& kv_cache,
                                        const at::Tensor& kv_scales,
                                        const at::Tensor& source_kernel_page_ids,
                                        at::Tensor&       output,
                                        int64_t           physical_page_size,
                                        int64_t           kernel_page_size,
                                        int64_t           subdivision) {
#ifndef ENABLE_FP8
    TORCH_CHECK(false, "FP8 support is not enabled in this CUDA build");
#else
    check_cuda_contiguous(source_kernel_page_ids, "source_kernel_page_ids");
    check_cuda_contiguous(output, "output");
    TORCH_CHECK(source_kernel_page_ids.dim() == 1, "source_kernel_page_ids must have shape [R]");
    TORCH_CHECK(source_kernel_page_ids.scalar_type() == at::ScalarType::Int
                    || source_kernel_page_ids.scalar_type() == at::ScalarType::Long,
                "source_kernel_page_ids must have dtype torch.int32 or torch.int64");
    TORCH_CHECK(output.dim() == 5, "output must have shape [R, 2, H, kernel_page_size, D]");
    TORCH_CHECK(output.size(0) == source_kernel_page_ids.numel() && output.size(1) == 2
                    && output.size(3) == kernel_page_size,
                "output must have shape [R, 2, H, kernel_page_size, D]");
    TORCH_CHECK(output.scalar_type() == at::ScalarType::Half || output.scalar_type() == at::ScalarType::BFloat16,
                "output must have dtype torch.float16 or torch.bfloat16");

    check_same_device(kv_cache, kv_scales, "kv_scales");
    check_same_device(kv_cache, source_kernel_page_ids, "source_kernel_page_ids");
    check_same_device(kv_cache, output, "output");
    const CacheGeometry geometry = validate_cache_geometry(
        kv_cache, kv_scales, output.size(2), output.size(4), physical_page_size, kernel_page_size, subdivision);
    const c10::cuda::CUDAGuard device_guard(kv_cache.device());
    const cudaStream_t         stream = at::cuda::getCurrentCUDAStream(kv_cache.get_device()).stream();
#define LAUNCH_GATHER(output_type, index_type)                                                                         \
    launch_gather<output_type, index_type>(kv_cache,                                                                   \
                                           kv_scales,                                                                  \
                                           source_kernel_page_ids,                                                     \
                                           output,                                                                     \
                                           geometry,                                                                   \
                                           physical_page_size,                                                         \
                                           kernel_page_size,                                                           \
                                           subdivision,                                                                \
                                           stream)
    if (source_kernel_page_ids.scalar_type() == at::ScalarType::Int) {
        if (output.scalar_type() == at::ScalarType::Half) {
            LAUNCH_GATHER(__half, int32_t);
        } else {
            LAUNCH_GATHER(__nv_bfloat16, int32_t);
        }
    } else if (output.scalar_type() == at::ScalarType::Half) {
        LAUNCH_GATHER(__half, int64_t);
    } else {
        LAUNCH_GATHER(__nv_bfloat16, int64_t);
    }
#undef LAUNCH_GATHER
    check_cuda_error();
#endif
}

}  // namespace rtp_llm
