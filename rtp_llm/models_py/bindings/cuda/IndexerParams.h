#pragma once

#include <torch/extension.h>
#include <vector>

#include "rtp_llm/models_py/bindings/ParamsBase.h"
#include "rtp_llm/models_py/bindings/OpDefs.h"

namespace rtp_llm {

class IndexerParams: public ParamsBase {
private:
    // Pre-allocated continuous buffers (HOST and DEVICE)
    torch::Tensor buf_h_i32_;
    torch::Tensor buf_d_i32_;
    torch::Tensor buf_h_i64_;

    size_t max_i32_elements_ = 0;
    size_t max_i64_elements_ = 0;
    int    max_batch_size_   = 0;
    int    max_token_num_    = 0;
    int    max_seq_len_      = 0;

    torch::Tensor batch_indice_h_;
    torch::Tensor positions_h_;
    torch::Tensor expanded_seq_lens_h_;
    torch::Tensor topk_indices_offset_h_;
    torch::Tensor ks_h_;
    torch::Tensor ke_h_;
    torch::Tensor page_table_1_h_;

    torch::Tensor batch_indice_d_;
    torch::Tensor positions_d_;
    torch::Tensor expanded_seq_lens_d_;
    torch::Tensor topk_indices_offset_d_;
    torch::Tensor ks_d_;
    torch::Tensor ke_d_;
    torch::Tensor page_table_1_d_;

    // Helper method to allocate many tensors in a continuous buffer
    static std::tuple<torch::Tensor, std::vector<torch::Tensor>> allocateManyBuffer(
        const std::vector<std::vector<int64_t>>& shapes, bool is_device, torch::ScalarType dtype = torch::kInt32);

    void ensureTensorSize(int batch_size, int token_num, int max_seq_len);
    void fillParamsInternal(bool                 is_prefill,
                            const torch::Tensor& input_lengths_cpu,
                            const torch::Tensor& prefix_lengths_cpu,
                            const torch::Tensor& sequence_lengths_cpu,
                            int                  seq_size_per_block,
                            int64_t              total_tokens,
                            int64_t              max_seq_len,
                            torch::Tensor&       slot_mapping_h);
    void refreshBuffer(int batch_size, int token_num, int max_seq_len, bool is_prefill);

public:
    void fillParams(torch_ext::PyAttentionInputs attn_inputs, int seq_size_per_block);

    // Outputs
    torch::Tensor cu_q_seqlens;
    torch::Tensor cu_kv_seqlens;
    torch::Tensor seq_lens;
    torch::Tensor block_table;

    torch::Tensor expanded_seq_lens;
    torch::Tensor page_table_1;
    torch::Tensor topk_indices_offset;
    torch::Tensor positions_d;
    torch::Tensor slot_mapping;
    torch::Tensor ks;
    torch::Tensor ke;
    torch::Tensor schedule_metadata;

    int  batch_size = 0;
    bool is_prefill = false;
};

void registerPyIndexerParams(pybind11::module& m);

}  // namespace rtp_llm
