#include "rtp_llm/models_py/bindings/cuda/ReuseKVCacheOp.h"
#include "rtp_llm/cpp/kernels/kv_cache_kernels.h"
#include "rtp_llm/cpp/core/torch_utils/BufferTorchUtils.h"
#include "rtp_llm/models_py/bindings/common/Torch_ext.h"
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
                                                    kv_dim,
                                                    stream);
}

}  // namespace rtp_llm
