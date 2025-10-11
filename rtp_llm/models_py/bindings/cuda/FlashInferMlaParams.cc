#include "rtp_llm/models_py/bindings/cuda/FlashInferMlaParams.h"
#include "rtp_llm/cpp/utils/AssertUtils.h"
#include "rtp_llm/cpp/core/torch_utils/BufferTorchUtils.h"
using namespace torch_ext;

namespace rtp_llm {

FlashInferMlaParams FillFlashInferMlaParams(int page_size, 
                                           const PyAttentionInputs& attention_inputs, 
                                           const torch::Device& device) {
    FlashInferMlaParams params;

    auto      sequence_lengths_host = torchTensor2Buffer(attention_inputs.sequence_lengths);
    auto      input_lengths_host    = torchTensor2Buffer(attention_inputs.input_lengths);

    BufferPtr kv_cache_block_id_host;
    if (attention_inputs.kv_cache_block_id_host.size(0)) {
        kv_cache_block_id_host   = torchTensor2Buffer(attention_inputs.kv_cache_block_id_host);
    }

    BufferPtr prefix_lengths_host;
    if (attention_inputs.prefix_lengths.size(0)) {
        prefix_lengths_host   = torchTensor2Buffer(attention_inputs.prefix_lengths);
    }

    const int max_batch_blocks = kv_cache_block_id_host ? kv_cache_block_id_host->shape()[1] : -1;
    const int batch_size = input_lengths_host->shape()[0];

    int max_kv_len             = 0;
    int max_q_len              = 0;
    int accu_q_len             = 0;
    int offset                 = 0;
    int total_page_idx         = 0;

    std::vector<int> batch_indice;
    std::vector<int> positions;
    std::vector<int> paged_kv_last_page_len;
    std::vector<int> kvlen;
    std::vector<int> page_indice;
    std::vector<int> page_indptr = {0};
    std::vector<int> qo_indptr = {0};

    auto input_lengths          = input_lengths_host->data<int>();
    auto prefix_lengths         = prefix_lengths_host ? prefix_lengths_host->data<int>() : nullptr;
    auto sequence_lengths       = sequence_lengths_host ? sequence_lengths_host->data<int>() : nullptr;
    auto kv_cache_block_id      = kv_cache_block_id_host ? kv_cache_block_id_host->data<int>() : nullptr;

    for (int i = 0; i < batch_size; i++) {
        int seq_len = 0;
        if (prefix_lengths) {
            int input_length  = input_lengths[i];
            int prefix_length = prefix_lengths[i];

            for (int j = 0; j < input_length; j++) {
                batch_indice.push_back(i);
                positions.push_back(j + prefix_length);
                offset += 1;
            }
            seq_len   = input_length + prefix_length;
            max_q_len = max(max_q_len, input_length);
            accu_q_len += input_length;
        } else {
            batch_indice.push_back(i);
            positions.push_back(sequence_lengths[i]);
            seq_len         = sequence_lengths[i] + 1;
            accu_q_len += 1;
        }
        paged_kv_last_page_len.push_back((seq_len - 1) % page_size + 1);
        kvlen.push_back(seq_len);
        max_kv_len                = max(seq_len, max_kv_len);

        int page_num = (seq_len + page_size - 1) / page_size;

        if (kv_cache_block_id) {
            for (int j = 0; j < page_num; j++) {
                auto page_idx                 = kv_cache_block_id[i * max_batch_blocks + j];
                page_indice.push_back(page_idx);
            }
        }
        page_indptr.push_back(total_page_idx);
        qo_indptr.push_back(accu_q_len);
    }
    auto cuda_option = torch::dtype(torch::kInt).device(device).requires_grad(false);
    params.batch_indice = torch::tensor(batch_indice, cuda_option);
    params.page_indice = torch::tensor(page_indice, cuda_option);
    params.page_indptr = torch::tensor(page_indptr, cuda_option);
    params.paged_kv_last_page_len = torch::tensor(paged_kv_last_page_len, cuda_option);
    params.qo_indptr = torch::tensor(qo_indptr, cuda_option);
    params.kvlen = torch::tensor(kvlen, cuda_option);
    params.positions = torch::tensor(positions, cuda_option);
    return params;
}

} // namespace rtp_llm
