#include "rtp_llm/models_py/bindings/cuda/ReuseKVCacheOp.h"
#include "rtp_llm/models_py/bindings/common/kernels/kv_cache_kernels.h"
#include "rtp_llm/models_py/bindings/core/torch_utils/TypeConvert.h"
#include "rtp_llm/models_py/bindings/common/Torch_ext.h"
#include <c10/cuda/CUDAGuard.h>
#include <cuda_runtime.h>

namespace rtp_llm {

void ReuseKVCacheIndexedBatched(torch::Tensor final_compressed_kv,
                                torch::Tensor final_k_pe,
                                torch::Tensor compressed_kv,
                                torch::Tensor k_pe,
                                torch::Tensor kv_cache_base,
                                torch::Tensor reuse_cache_page_indice,
                                torch::Tensor batch_reuse_info_vec,
                                torch::Tensor qo_indptr,
                                int           tokens_per_block) {

    TORCH_CHECK(final_compressed_kv.is_cuda(), "final_compressed_kv must be on CUDA");
    TORCH_CHECK(final_k_pe.is_cuda(), "final_k_pe must be on CUDA");
    TORCH_CHECK(compressed_kv.is_cuda(), "compressed_kv must be on CUDA");
    TORCH_CHECK(k_pe.is_cuda(), "k_pe must be on CUDA");
    TORCH_CHECK(kv_cache_base.is_cuda(), "kv_cache_base must be on CUDA");

    const int num_batches       = batch_reuse_info_vec.size(0);
    const int total_final_len   = final_compressed_kv.size(0);  // 从 final_compressed_kv 的第0维获取
    const int compressed_kv_dim = compressed_kv.size(1);
    const int k_pe_dim          = k_pe.size(1);
    const int kv_dim            = compressed_kv_dim + k_pe_dim;

    TORCH_CHECK(kv_cache_base.dim() == 3,
                "kv_cache_base must be [num_blocks, tokens_per_block, kv_dim]");
    TORCH_CHECK(kv_cache_base.size(1) == tokens_per_block,
                "kv_cache_base tokens per block mismatch: ",
                kv_cache_base.size(1),
                " != ",
                tokens_per_block);
    TORCH_CHECK(kv_cache_base.size(2) >= kv_dim,
                "kv_cache_base entry is too small: ",
                kv_cache_base.size(2),
                " < ",
                kv_dim);
    TORCH_CHECK(kv_cache_base.stride(2) == 1,
                "kv_cache_base innermost KV dimension must be contiguous");

    const int64_t kv_cache_block_stride = kv_cache_base.stride(0);
    const int64_t kv_cache_entry_stride = kv_cache_base.stride(1);

    StreamType stream = GET_CURRENT_STREAM();

    invokeReuseKVCacheIndexedBatched<__nv_bfloat16>(reinterpret_cast<__nv_bfloat16*>(final_compressed_kv.data_ptr()),
                                                    reinterpret_cast<__nv_bfloat16*>(final_k_pe.data_ptr()),
                                                    reinterpret_cast<const __nv_bfloat16*>(compressed_kv.data_ptr()),
                                                    reinterpret_cast<const __nv_bfloat16*>(k_pe.data_ptr()),
                                                    reinterpret_cast<const __nv_bfloat16*>(kv_cache_base.data_ptr()),
                                                    reuse_cache_page_indice.data_ptr<int32_t>(),
                                                    batch_reuse_info_vec.data_ptr<int32_t>(),
                                                    qo_indptr.data_ptr<int32_t>(),
                                                    num_batches,
                                                    total_final_len,  // 传入 total_final_len
                                                    compressed_kv_dim,
                                                    k_pe_dim,
                                                    tokens_per_block,
                                                    kv_cache_block_stride,
                                                    kv_cache_entry_stride,
                                                    stream);
}

void GatherMLALatentAndFillKPe(torch::Tensor final_compressed_kv,
                               torch::Tensor packed_kv,
                               torch::Tensor compressed_kv,
                               torch::Tensor k_pe,
                               torch::Tensor kv_cache_base,
                               torch::Tensor reuse_cache_page_indice,
                               torch::Tensor batch_reuse_info_vec,
                               torch::Tensor qo_indptr,
                               int           tokens_per_block,
                               int           packed_head_dim,
                               int           k_pe_offset) {
    TORCH_CHECK(final_compressed_kv.is_cuda(), "final_compressed_kv must be on CUDA");
    TORCH_CHECK(packed_kv.is_cuda(), "packed_kv must be on CUDA");
    TORCH_CHECK(compressed_kv.is_cuda(), "compressed_kv must be on CUDA");
    TORCH_CHECK(k_pe.is_cuda(), "k_pe must be on CUDA");
    TORCH_CHECK(kv_cache_base.is_cuda(), "kv_cache_base must be on CUDA");
    TORCH_CHECK(reuse_cache_page_indice.is_cuda(), "reuse_cache_page_indice must be on CUDA");
    TORCH_CHECK(batch_reuse_info_vec.is_cuda(), "batch_reuse_info_vec must be on CUDA");
    TORCH_CHECK(qo_indptr.is_cuda(), "qo_indptr must be on CUDA");

    const auto device = final_compressed_kv.device();
    TORCH_CHECK(packed_kv.device() == device && compressed_kv.device() == device && k_pe.device() == device
                    && kv_cache_base.device() == device && reuse_cache_page_indice.device() == device
                    && batch_reuse_info_vec.device() == device && qo_indptr.device() == device,
                "all gather MLA tensors must be on the same CUDA device");
    const c10::cuda::CUDAGuard device_guard(device);
    TORCH_CHECK(final_compressed_kv.scalar_type() == torch::kBFloat16 && packed_kv.scalar_type() == torch::kBFloat16
                    && compressed_kv.scalar_type() == torch::kBFloat16 && k_pe.scalar_type() == torch::kBFloat16
                    && kv_cache_base.scalar_type() == torch::kBFloat16,
                "gather_mla_latent_and_fill_k_pe requires BF16 data tensors");
    TORCH_CHECK(reuse_cache_page_indice.scalar_type() == torch::kInt32
                    && batch_reuse_info_vec.scalar_type() == torch::kInt32 && qo_indptr.scalar_type() == torch::kInt32,
                "gather_mla_latent_and_fill_k_pe requires int32 metadata");

    TORCH_CHECK(final_compressed_kv.dim() == 2 && final_compressed_kv.stride(1) == 1,
                "final_compressed_kv must be a 2D tensor with a contiguous inner dimension");
    TORCH_CHECK(packed_kv.dim() == 2 && packed_kv.stride(1) == 1,
                "packed_kv must be a 2D tensor with a contiguous inner dimension");
    TORCH_CHECK(compressed_kv.dim() == 2 && compressed_kv.stride(1) == 1,
                "compressed_kv must be a 2D tensor with a contiguous inner dimension");
    TORCH_CHECK(k_pe.dim() == 2 && k_pe.stride(1) == 1, "k_pe must be a 2D tensor with a contiguous inner dimension");
    TORCH_CHECK(kv_cache_base.dim() == 3 && kv_cache_base.stride(2) == 1,
                "kv_cache_base must be 3D with a contiguous inner dimension");
    TORCH_CHECK(reuse_cache_page_indice.is_contiguous() && reuse_cache_page_indice.dim() == 1,
                "reuse_cache_page_indice must be a contiguous vector");
    TORCH_CHECK(batch_reuse_info_vec.is_contiguous() && batch_reuse_info_vec.dim() == 2
                    && batch_reuse_info_vec.size(1) == 4,
                "batch_reuse_info_vec must be contiguous [batch, 4]");
    TORCH_CHECK(qo_indptr.is_contiguous() && qo_indptr.dim() == 1
                    && qo_indptr.size(0) == batch_reuse_info_vec.size(0) + 1,
                "qo_indptr must be a contiguous [batch + 1] vector");

    const int num_batches       = batch_reuse_info_vec.size(0);
    const int total_final_len   = final_compressed_kv.size(0);
    const int compressed_kv_dim = compressed_kv.size(1);
    const int k_pe_dim          = k_pe.size(1);
    TORCH_CHECK(num_batches > 0, "gather_mla_latent_and_fill_k_pe requires at least one batch");
    TORCH_CHECK(tokens_per_block > 0, "tokens_per_block must be positive");
    TORCH_CHECK(packed_head_dim > 0, "packed_head_dim must be positive");
    TORCH_CHECK(packed_kv.size(1) > 0 && packed_kv.size(1) % packed_head_dim == 0,
                "packed_kv must contain at least one complete packed head");
    TORCH_CHECK(k_pe_offset >= 0 && k_pe_offset + k_pe_dim <= packed_head_dim,
                "K_pe does not fit in the packed head gap");
    TORCH_CHECK(final_compressed_kv.size(1) == compressed_kv_dim,
                "gathered and current compressed KV dimensions disagree");
    TORCH_CHECK(packed_kv.size(0) == total_final_len, "packed_kv and gathered KV token counts disagree");
    TORCH_CHECK(k_pe.size(0) == compressed_kv.size(0), "current compressed KV and K_pe token counts disagree");
    TORCH_CHECK(kv_cache_base.size(1) == tokens_per_block,
                "kv_cache_base tokens per block mismatch: ",
                kv_cache_base.size(1),
                " != ",
                tokens_per_block);
    TORCH_CHECK(kv_cache_base.size(2) >= compressed_kv_dim + k_pe_dim,
                "kv_cache_base entry is too small for compressed KV and K_pe");

    StreamType stream = at::cuda::getCurrentCUDAStream(final_compressed_kv.get_device()).stream();
    invokeGatherMLALatentAndFillKPe<__nv_bfloat16>(reinterpret_cast<__nv_bfloat16*>(final_compressed_kv.data_ptr()),
                                                   reinterpret_cast<__nv_bfloat16*>(packed_kv.data_ptr()),
                                                   reinterpret_cast<const __nv_bfloat16*>(compressed_kv.data_ptr()),
                                                   reinterpret_cast<const __nv_bfloat16*>(k_pe.data_ptr()),
                                                   reinterpret_cast<const __nv_bfloat16*>(kv_cache_base.data_ptr()),
                                                   reuse_cache_page_indice.data_ptr<int32_t>(),
                                                   batch_reuse_info_vec.data_ptr<int32_t>(),
                                                   qo_indptr.data_ptr<int32_t>(),
                                                   num_batches,
                                                   total_final_len,
                                                   compressed_kv_dim,
                                                   k_pe_dim,
                                                   packed_kv.size(1) / packed_head_dim,
                                                   packed_head_dim,
                                                   k_pe_offset,
                                                   tokens_per_block,
                                                   final_compressed_kv.stride(0),
                                                   packed_kv.stride(0),
                                                   compressed_kv.stride(0),
                                                   k_pe.stride(0),
                                                   kv_cache_base.stride(0),
                                                   kv_cache_base.stride(1),
                                                   stream);
}

}  // namespace rtp_llm
