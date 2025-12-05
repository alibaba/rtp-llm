#pragma once

#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>

namespace rtp_llm {

void ReuseKVCacheIndexedBatched(torch::Tensor final_compressed_kv,
                                torch::Tensor final_k_pe,
                                torch::Tensor compressed_kv,
                                torch::Tensor k_pe,
                                torch::Tensor kv_cache_base,
                                torch::Tensor reuse_cache_page_indice,
                                torch::Tensor batch_reuse_info_vec,
                                torch::Tensor qo_indptr,
                                int           tokens_per_block);

}  // namespace rtp_llm
