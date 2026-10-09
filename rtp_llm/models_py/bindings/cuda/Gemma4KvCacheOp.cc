#include "rtp_llm/models_py/bindings/cuda/Gemma4KvCacheOp.h"
#include "rtp_llm/models_py/bindings/cuda/kernels/gemma4_kv_cache.h"

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>

#include <limits>

namespace torch_ext {

std::tuple<at::Tensor, at::Tensor, at::Tensor> gemma4_gather_paged_kv_bf16(const at::Tensor& k_cache,
                                                                           const at::Tensor& v_cache,
                                                                           const at::Tensor& page_indices,
                                                                           int64_t           first_offset,
                                                                           int64_t           token_count,
                                                                           int64_t           page_size) {
    TORCH_CHECK(k_cache.is_cuda() && v_cache.is_cuda() && page_indices.is_cuda(),
                "gemma4_gather_paged_kv_bf16: inputs must be CUDA tensors");
    TORCH_CHECK(k_cache.scalar_type() == at::kBFloat16 && v_cache.scalar_type() == at::kBFloat16,
                "gemma4_gather_paged_kv_bf16: caches must be bfloat16");
    TORCH_CHECK(page_indices.scalar_type() == at::kInt, "gemma4_gather_paged_kv_bf16: page_indices must be int32");
    TORCH_CHECK(page_indices.is_contiguous(), "gemma4_gather_paged_kv_bf16: page_indices must be contiguous");
    TORCH_CHECK(k_cache.dim() == 4 && v_cache.sizes() == k_cache.sizes(),
                "gemma4_gather_paged_kv_bf16: caches must be matching [pages,heads,page,dim]");
    TORCH_CHECK(k_cache.size(2) == page_size && k_cache.size(3) % 8 == 0,
                "gemma4_gather_paged_kv_bf16: cache geometry mismatch");
    TORCH_CHECK(k_cache.stride(3) == 1 && v_cache.strides() == k_cache.strides(),
                "gemma4_gather_paged_kv_bf16: cache strides must match with contiguous head dimension");
    TORCH_CHECK(first_offset >= 0 && token_count >= 0, "gemma4_gather_paged_kv_bf16: offsets must be nonnegative");
    TORCH_CHECK(page_size > 0 && page_size <= std::numeric_limits<int32_t>::max() && first_offset < page_size,
                "gemma4_gather_paged_kv_bf16: invalid page geometry");
    TORCH_CHECK(k_cache.size(1) <= std::numeric_limits<int32_t>::max()
                    && k_cache.size(3) <= std::numeric_limits<int32_t>::max(),
                "gemma4_gather_paged_kv_bf16: dimensions exceed int32 limits");
    TORCH_CHECK(k_cache.get_device() == v_cache.get_device() && k_cache.get_device() == page_indices.get_device(),
                "gemma4_gather_paged_kv_bf16: inputs must share a device");

    const c10::cuda::CUDAGuard device_guard(k_cache.device());
    auto                       keys   = at::empty({token_count, k_cache.size(1), k_cache.size(3)}, k_cache.options());
    auto                       values = at::empty_like(keys);
    auto                       valid  = at::empty({token_count}, k_cache.options().dtype(at::kBool));
    rtp_llm::invokeGemma4GatherPagedKvBf16(
        reinterpret_cast<const __nv_bfloat16*>(k_cache.const_data_ptr<at::BFloat16>()),
        reinterpret_cast<const __nv_bfloat16*>(v_cache.const_data_ptr<at::BFloat16>()),
        page_indices.const_data_ptr<int32_t>(),
        reinterpret_cast<__nv_bfloat16*>(keys.mutable_data_ptr<at::BFloat16>()),
        reinterpret_cast<__nv_bfloat16*>(values.mutable_data_ptr<at::BFloat16>()),
        valid.mutable_data_ptr<bool>(),
        token_count,
        static_cast<int32_t>(first_offset),
        static_cast<int32_t>(k_cache.size(1)),
        static_cast<int32_t>(k_cache.size(3)),
        static_cast<int32_t>(page_size),
        k_cache.size(0),
        page_indices.numel(),
        k_cache.stride(0),
        k_cache.stride(1),
        k_cache.stride(2),
        at::cuda::getCurrentCUDAStream(k_cache.get_device()).stream());
    return {keys, values, valid};
}

void gemma4_append_swa_kv_cache_bf16(const at::Tensor& key,
                                     const at::Tensor& value,
                                     const at::Tensor& batch_indices,
                                     const at::Tensor& positions,
                                     const at::Tensor& k_cache,
                                     const at::Tensor& v_cache,
                                     const at::Tensor& page_indices,
                                     const at::Tensor& page_indptr,
                                     int64_t           page_size) {
    TORCH_CHECK(key.is_cuda() && value.is_cuda() && batch_indices.is_cuda() && positions.is_cuda() && k_cache.is_cuda()
                    && v_cache.is_cuda() && page_indices.is_cuda() && page_indptr.is_cuda(),
                "gemma4_append_swa_kv_cache_bf16: inputs must be CUDA tensors");
    TORCH_CHECK(key.scalar_type() == at::kBFloat16 && value.scalar_type() == at::kBFloat16
                    && k_cache.scalar_type() == at::kBFloat16 && v_cache.scalar_type() == at::kBFloat16,
                "gemma4_append_swa_kv_cache_bf16: K/V tensors must be bfloat16");
    TORCH_CHECK(batch_indices.scalar_type() == at::kInt && positions.scalar_type() == at::kInt
                    && page_indices.scalar_type() == at::kInt && page_indptr.scalar_type() == at::kInt,
                "gemma4_append_swa_kv_cache_bf16: metadata tensors must be int32");
    TORCH_CHECK(key.is_contiguous() && value.is_contiguous() && batch_indices.is_contiguous()
                    && positions.is_contiguous() && page_indices.is_contiguous() && page_indptr.is_contiguous(),
                "gemma4_append_swa_kv_cache_bf16: inputs and metadata must be contiguous");
    TORCH_CHECK(key.dim() == 3 && value.sizes() == key.sizes(),
                "gemma4_append_swa_kv_cache_bf16: key/value must be matching [tokens,heads,dim]");
    TORCH_CHECK(batch_indices.numel() == key.size(0) && positions.numel() == key.size(0),
                "gemma4_append_swa_kv_cache_bf16: metadata length must match tokens");
    TORCH_CHECK(k_cache.dim() == 4 && v_cache.sizes() == k_cache.sizes(),
                "gemma4_append_swa_kv_cache_bf16: caches must be matching [pages,heads,page,dim]");
    TORCH_CHECK(k_cache.size(1) == key.size(1) && k_cache.size(2) == page_size && k_cache.size(3) == key.size(2),
                "gemma4_append_swa_kv_cache_bf16: cache geometry mismatch");
    TORCH_CHECK(key.size(2) % 8 == 0 && key.size(1) <= std::numeric_limits<int32_t>::max()
                    && key.size(2) <= std::numeric_limits<int32_t>::max() && page_size > 0
                    && page_size <= std::numeric_limits<int32_t>::max(),
                "gemma4_append_swa_kv_cache_bf16: unsupported geometry");
    TORCH_CHECK(k_cache.stride(3) == 1 && v_cache.strides() == k_cache.strides(),
                "gemma4_append_swa_kv_cache_bf16: cache strides must match with contiguous head dimension");
    const int device = key.get_device();
    TORCH_CHECK(value.get_device() == device && batch_indices.get_device() == device && positions.get_device() == device
                    && k_cache.get_device() == device && v_cache.get_device() == device
                    && page_indices.get_device() == device && page_indptr.get_device() == device,
                "gemma4_append_swa_kv_cache_bf16: inputs must share a device");

    const c10::cuda::CUDAGuard device_guard(key.device());
    rtp_llm::invokeGemma4AppendSwaKvCacheBf16(
        reinterpret_cast<const __nv_bfloat16*>(key.const_data_ptr<at::BFloat16>()),
        reinterpret_cast<const __nv_bfloat16*>(value.const_data_ptr<at::BFloat16>()),
        batch_indices.const_data_ptr<int32_t>(),
        positions.const_data_ptr<int32_t>(),
        reinterpret_cast<__nv_bfloat16*>(k_cache.mutable_data_ptr<at::BFloat16>()),
        reinterpret_cast<__nv_bfloat16*>(v_cache.mutable_data_ptr<at::BFloat16>()),
        page_indices.const_data_ptr<int32_t>(),
        page_indptr.const_data_ptr<int32_t>(),
        key.size(0),
        static_cast<int32_t>(key.size(1)),
        static_cast<int32_t>(key.size(2)),
        static_cast<int32_t>(page_size),
        k_cache.size(0),
        page_indices.numel(),
        page_indptr.numel(),
        k_cache.stride(0),
        k_cache.stride(1),
        k_cache.stride(2),
        at::cuda::getCurrentCUDAStream(device).stream());
}

}  // namespace torch_ext
