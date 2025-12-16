#pragma once

#include <torch/extension.h>
#include <vector>
#include <memory>
#include <cuda_runtime.h>
#include "rtp_llm/models_py/bindings/OpDefs.h"
#include "rtp_llm/models_py/bindings/ParamsBase.h"

using namespace torch_ext;

namespace rtp_llm {

class FlashInferMlaAttnParams: public ParamsBase {
private:
    // Pre-allocated continuous buffers (HOST and DEVICE)
    torch::Tensor buf_h;  // Large continuous HOST buffer (pinned memory)
    torch::Tensor buf_d;  // Large continuous DEVICE buffer

    // Reserved sizes
    int max_batch_size_       = 0;
    int max_input_token_num_  = 0;
    int max_page_num_         = 0;
    int max_reuse_page_num_   = 0;
    int max_batch_reuse_info_ = 0;

    // Helper method to allocate many tensors in a continuous buffer
    static std::tuple<torch::Tensor, std::vector<torch::Tensor>>
    allocateManyBuffer(const std::vector<std::vector<int64_t>>& shapes, bool is_device);

    // Helper method to refresh buffer shapes and copy to device (single memcpy)
    void
    refreshBuffer(int batch_size, int input_token_num, int page_num, int reuse_page_num, int batch_reuse_info_size);

    // Internal method to fill params directly into HOST tensors
    void fillParamsInternal(torch::Tensor t_prefix_lengths,
                            torch::Tensor t_sequence_lengths,
                            torch::Tensor t_input_lengths,
                            torch::Tensor t_kv_cache_block_id_host,
                            int           batch_size,
                            int           seq_size_per_block,
                            int&          input_token_num,
                            int&          page_num,
                            int&          reuse_page_num,
                            int&          batch_reuse_info_size);

    // Ensure tensors are allocated with sufficient size
    void
    ensureTensorSize(int batch_size, int input_token_num, int page_num, int reuse_page_num, int batch_reuse_info_size);

public:
    void fillParams(torch::Tensor t_prefix_lengths,
                    torch::Tensor t_sequence_lengths,
                    torch::Tensor t_input_lengths,
                    torch::Tensor t_kv_cache_block_id_host,
                    int           seq_size_per_block);

    // Tensor views into buf_h and buf_d
    torch::Tensor batch_indice_h;
    torch::Tensor page_indice_h;
    torch::Tensor reuse_cache_page_indice_h;
    torch::Tensor decode_page_indptr_h;
    torch::Tensor prefill_page_indptr_h;
    torch::Tensor paged_kv_last_page_len_h;
    torch::Tensor qo_indptr_h;
    torch::Tensor kvlen_h;
    torch::Tensor positions_h;
    torch::Tensor batch_reuse_info_vec_h;

    torch::Tensor batch_indice_d;
    torch::Tensor page_indice_d;
    torch::Tensor reuse_cache_page_indice_d;
    torch::Tensor decode_page_indptr_d;
    torch::Tensor prefill_page_indptr_d;
    torch::Tensor paged_kv_last_page_len_d;
    torch::Tensor qo_indptr_d;
    torch::Tensor kvlen_d;
    torch::Tensor positions_d;
    torch::Tensor batch_reuse_info_vec_d;

    torch::Tensor batch_indice;
    torch::Tensor positions;
    torch::Tensor paged_kv_last_page_len;
    torch::Tensor kvlen;
    torch::Tensor page_indice;
    torch::Tensor reuse_cache_page_indice;
    torch::Tensor decode_page_indptr;
    torch::Tensor prefill_page_indptr;
    torch::Tensor qo_indptr;
    torch::Tensor batch_reuse_info_vec;
};
void registerPyFlashInferMlaParams(pybind11::module& m);

}  // namespace rtp_llm