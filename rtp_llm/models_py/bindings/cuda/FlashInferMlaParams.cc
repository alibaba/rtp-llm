#include "rtp_llm/models_py/bindings/cuda/FlashInferMlaParams.h"
#include "rtp_llm/cpp/utils/AssertUtils.h"
#include "rtp_llm/cpp/core/torch_utils/BufferTorchUtils.h"
#include <cstdint>
using namespace torch_ext;

namespace rtp_llm {

MlaParams FlashInferMlaAttnParams::fillParams(torch::Tensor t_prefix_lengths,
                                              torch::Tensor t_sequence_lengths,
                                              torch::Tensor t_input_lengths,
                                              torch::Tensor t_kv_cache_block_id_host,
                                              int           seq_size_per_block) {
    MlaParams params;
    auto      sequence_lengths_host = torchTensor2Buffer(t_sequence_lengths);
    auto      input_lengths_host    = torchTensor2Buffer(t_input_lengths);

    BufferPtr kv_cache_block_id_host;
    if (t_kv_cache_block_id_host.size(0)) {
        kv_cache_block_id_host = torchTensor2Buffer(t_kv_cache_block_id_host);
    }

    BufferPtr prefix_lengths_host;
    if (t_prefix_lengths.size(0)) {
        prefix_lengths_host = torchTensor2Buffer(t_prefix_lengths);
    }

    const int max_batch_blocks = kv_cache_block_id_host ? kv_cache_block_id_host->shape()[1] : -1;
    const int batch_size       = input_lengths_host->shape()[0];

    int max_kv_len     = 0;
    int max_q_len      = 0;
    int accu_q_len     = 0;
    int accu_kv_len    = 0;
    int offset         = 0;
    int total_page_idx = 0;

    std::vector<int32_t>              batch_indice;
    std::vector<int32_t>              positions;
    std::vector<int32_t>              paged_kv_last_page_len;
    std::vector<int32_t>              kvlen;
    std::vector<int32_t>              page_indice;
    std::vector<int32_t>              reuse_cache_page_indice;
    std::vector<int32_t>              decode_page_indptr  = {0};
    std::vector<int32_t>              prefill_page_indptr = {0};
    std::vector<int32_t>              qo_indptr           = {0};
    std::vector<std::vector<int32_t>> batch_reuse_info_vec;
    int                               batch_start_idx = 0;

    auto input_lengths     = input_lengths_host->data<int>();
    auto prefix_lengths    = prefix_lengths_host ? prefix_lengths_host->data<int>() : nullptr;
    auto sequence_lengths  = sequence_lengths_host ? sequence_lengths_host->data<int>() : nullptr;
    auto kv_cache_block_id = kv_cache_block_id_host ? kv_cache_block_id_host->data<int>() : nullptr;

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
            accu_kv_len += seq_len;

            int page_num = (prefix_length + seq_size_per_block - 1) / seq_size_per_block;
            if (kv_cache_block_id) {
                for (int j = 0; j < page_num; j++) {
                    auto page_idx = kv_cache_block_id[i * max_batch_blocks + j];
                    reuse_cache_page_indice.push_back(page_idx);
                }
            }
            if (prefix_length) {
                batch_reuse_info_vec.push_back({i, prefix_length, batch_start_idx, page_num});
                batch_start_idx += page_num;
            } else {
                batch_reuse_info_vec.push_back({i, 0, 0, 0});
            }
        } else {
            batch_indice.push_back(i);
            positions.push_back(sequence_lengths[i]);
            seq_len = sequence_lengths[i] + 1;
            accu_q_len += 1;
            accu_kv_len += 1;
        }
        paged_kv_last_page_len.push_back((seq_len - 1) % seq_size_per_block + 1);
        kvlen.push_back(seq_len);
        max_kv_len = max(seq_len, max_kv_len);

        int page_num = (seq_len + seq_size_per_block - 1) / seq_size_per_block;

        if (kv_cache_block_id) {
            for (int j = 0; j < page_num; j++) {
                auto page_idx = kv_cache_block_id[i * max_batch_blocks + j];
                page_indice.push_back(page_idx);
                total_page_idx++;
            }
        }
        decode_page_indptr.push_back(total_page_idx);
        prefill_page_indptr.push_back(accu_kv_len);
        qo_indptr.push_back(accu_q_len);
    }
    auto cuda_option               = torch::dtype(torch::kInt).device(torch::DeviceType::CUDA).requires_grad(false);
    params.batch_indice            = torch::tensor(batch_indice, cuda_option);
    params.page_indice             = torch::tensor(page_indice, cuda_option);
    params.reuse_cache_page_indice = torch::tensor(reuse_cache_page_indice, cuda_option);
    params.decode_page_indptr      = torch::tensor(decode_page_indptr, cuda_option);
    params.prefill_page_indptr     = torch::tensor(prefill_page_indptr, cuda_option);
    params.paged_kv_last_page_len  = torch::tensor(paged_kv_last_page_len, cuda_option);
    params.qo_indptr               = torch::tensor(qo_indptr, cuda_option);
    params.kvlen                   = torch::tensor(kvlen, cuda_option);
    params.positions               = torch::tensor(positions, cuda_option);
    if (reuse_cache_page_indice.size() > 0) {
        std::vector<int32_t> flat;
        flat.reserve(batch_reuse_info_vec.size() * batch_reuse_info_vec[0].size());
        for (const auto& row : batch_reuse_info_vec) {
            flat.insert(flat.end(), row.begin(), row.end());
        }
        params.batch_reuse_info_vec =
            torch::tensor(flat, cuda_option)
                .view({(int64_t)batch_reuse_info_vec.size(), (int64_t)batch_reuse_info_vec[0].size()});
    }
    return params;
}

void registerPyFlashInferMlaParams(pybind11::module& m) {
    m.def(
        "fill_mla_params",
        [](torch::Tensor t_prefill_lengths,
           torch::Tensor t_sequence_lengths,
           torch::Tensor t_input_lengths,
           torch::Tensor t_kv_cache_block_id_host,
           int           seq_size_per_block) {
            auto params = std::make_shared<rtp_llm::FlashInferMlaAttnParams>();
            return params->fillParams(
                t_prefill_lengths, t_sequence_lengths, t_input_lengths, t_kv_cache_block_id_host, seq_size_per_block);
        },
        pybind11::arg("t_prefill_lengths"),
        pybind11::arg("t_sequence_lengths"),
        pybind11::arg("t_input_lengths"),
        pybind11::arg("t_kv_cache_block_id_host"),
        pybind11::arg("seq_size_per_block"));
}

}  // namespace rtp_llm
