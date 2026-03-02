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
    torch::Tensor buf_h;       // Large continuous HOST buffer (pinned memory)
    torch::Tensor buf_d;       // Large continuous DEVICE buffer
    torch::Tensor buf_h_i64_;  // HOST buffer for int64 tensors (slot_mapping)
    torch::Tensor buf_d_i64_;  // DEVICE buffer for int64 tensors (slot_mapping)

    // Reserved sizes
    int    max_batch_size_       = 0;
    int    max_input_token_num_  = 0;
    int    max_page_num_         = 0;
    int    max_reuse_page_num_   = 0;
    int    max_batch_reuse_info_ = 0;
    size_t max_i64_elements_     = 0;

    // Private slot_mapping tensors
    torch::Tensor slot_mapping_h_;
    torch::Tensor slot_mapping_d_;

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
    void ensureTensorSize(int  batch_size,
                          int  input_token_num,
                          int  page_num,
                          int  reuse_page_num,
                          int  batch_reuse_info_size,
                          bool is_cuda_graph = false,
                          bool is_capture    = false);

protected:
    static std::tuple<torch::Tensor, std::vector<torch::Tensor>> allocateManyBuffer(
        const std::vector<std::vector<int64_t>>& shapes, bool is_device, torch::ScalarType dtype = torch::kInt32);

public:
    void fillParams(torch::Tensor t_prefix_lengths,
                    torch::Tensor t_sequence_lengths,
                    torch::Tensor t_input_lengths,
                    torch::Tensor t_kv_cache_block_id_host,
                    int           seq_size_per_block,
                    bool          is_cuda_graph = false,
                    bool          is_capture    = false);

    // Tensor views into buf_h and buf_d
    torch::Tensor batch_indice_h;
    torch::Tensor page_indice_h;
    torch::Tensor reuse_cache_page_indice_h;
    torch::Tensor decode_page_indptr_h;
    torch::Tensor prefill_ragged_kv_len_indptr_h;
    torch::Tensor paged_kv_last_page_len_h;
    torch::Tensor qo_indptr_h;
    torch::Tensor kvlen_h;
    torch::Tensor positions_h;
    torch::Tensor batch_reuse_info_vec_h;

    torch::Tensor batch_indice_d;
    torch::Tensor page_indice_d;
    torch::Tensor reuse_cache_page_indice_d;
    torch::Tensor decode_page_indptr_d;
    torch::Tensor prefill_ragged_kv_len_indptr_d;
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
    torch::Tensor prefill_ragged_kv_len_indptr;
    torch::Tensor qo_indptr;
    torch::Tensor batch_reuse_info_vec;
    torch::Tensor slot_mapping;
};
void registerPyFlashInferMlaParams(pybind11::module& m);

}  // namespace rtp_llm