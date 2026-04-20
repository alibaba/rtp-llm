#include "rtp_llm/cpp/models/context_parallel/RoundRobinProcessor.h"
#include "rtp_llm/cpp/core/ExecOps.h"
#include "rtp_llm/cpp/core/OpData.h"
#include "rtp_llm/cpp/utils/AssertUtils.h"
#include "rtp_llm/models_py/bindings/OpDefs.h"
#include <vector>

namespace rtp_llm {

bool RoundRobinProcessor::plan(const std::vector<int>& total_input_tokens,
                               std::vector<int>&       input_tokens,
                               std::vector<int>&       shuffle_indices,
                               int                     cp_rank,
                               int                     cp_size,
                               int                     cp_chunk_size,
                               int                     cp_padding_size) {
    const int input_token_size      = static_cast<int>(total_input_tokens.size());
    const int padded_seq_token_size = input_token_size + cp_padding_size;
    RTP_LLM_CHECK(cp_rank >= 0 && cp_rank < cp_size);
    RTP_LLM_CHECK(padded_seq_token_size % (page_size_ * cp_size) == 0);
    RTP_LLM_CHECK(cp_chunk_size == padded_seq_token_size / cp_size);

    // Page-level shuffle: rank gets blocks where block_idx % cp_size == cp_rank
    const int num_local_blocks = cp_chunk_size / page_size_;
    for (int j = 0; j < num_local_blocks; j++) {
        int global_block = cp_rank + j * cp_size;
        for (int t = 0; t < page_size_; t++) {
            shuffle_indices[j * page_size_ + t] = global_block * page_size_ + t;
        }
    }

    // Gather valid tokens with page-level stride
    const int* src = total_input_tokens.data();
    for (int j = 0; j < num_local_blocks; j++) {
        int global_block = cp_rank + j * cp_size;
        for (int t = 0; t < page_size_; t++) {
            int global_pos = global_block * page_size_ + t;
            if (global_pos < input_token_size) {
                input_tokens[j * page_size_ + t] = src[global_pos];
            } else {
                input_tokens[j * page_size_ + t] = 0;
            }
        }
    }
    return true;
}

torch::Tensor RoundRobinProcessor::generateQKVRestoreIndices(const torch::Tensor& prefill_cp_chunk_lengths,
                                                             int                  cp_size) {
    int num_prefill_streams = prefill_cp_chunk_lengths.size(0);
    int total_token_size    = torch::sum(prefill_cp_chunk_lengths).item<int>();

    // After all-gather the tensor is [rank0_chunk | rank1_chunk | ...].
    // We need to restore block-level interleaved order.
    // For each rank r, it holds local blocks [0, 1, 2, ...] which correspond to
    // global blocks [r, r+cp_size, r+2*cp_size, ...].
    // Within each block, tokens are contiguous (page_size tokens per block).

    torch::Tensor sorted_indices = torch::empty(
        {cp_size * total_token_size}, torch::TensorOptions(torch::kInt32).device(torch::kCPU).pinned_memory(true));
    int* indices_data = sorted_indices.data_ptr<int>();

    std::vector<int> chunk_lengths(num_prefill_streams);
    for (int s = 0; s < num_prefill_streams; s++) {
        chunk_lengths[s] = prefill_cp_chunk_lengths[s].item<int>();
    }

    for (int rank = 0; rank < cp_size; rank++) {
        int base   = rank * total_token_size;  // flat base for this rank
        int offset = 0;                        // cumulative chunk offset

        for (int s = 0; s < num_prefill_streams; s++) {
            int cl               = chunk_lengths[s];
            int num_local_blocks = cl / page_size_;

            for (int j = 0; j < num_local_blocks; j++) {
                int global_block = rank + j * cp_size;
                for (int t = 0; t < page_size_; t++) {
                    int dst_pos           = offset + global_block * page_size_ + t;
                    int src_pos           = base + offset / cp_size + j * page_size_ + t;
                    indices_data[dst_pos] = src_pos;
                }
            }
            offset += cl * cp_size;
        }
    }
    return sorted_indices;
}

torch::Tensor RoundRobinProcessor::generateQKVPaddingMask(const torch::Tensor& prefill_cp_chunk_lengths,
                                                          const torch::Tensor& prefill_cp_padding_lengths,
                                                          int                  cp_size) {
    int  num_prefill_streams = prefill_cp_chunk_lengths.size(0);
    auto padded_seq_lengths  = prefill_cp_chunk_lengths * cp_size;
    int  total_size          = torch::sum(padded_seq_lengths).item<int>();

    torch::Tensor padding_mask =
        torch::empty({total_size}, torch::TensorOptions(torch::kInt32).device(torch::kCPU).pinned_memory(true));
    int* mask_data = padding_mask.data_ptr<int>();

    int offset = 0;
    for (int i = 0; i < num_prefill_streams; i++) {
        int padded_length = padded_seq_lengths[i].item<int>();
        int padding_count = prefill_cp_padding_lengths[i].item<int>();
        int valid_count   = padded_length - padding_count;

        std::fill_n(mask_data + offset, valid_count, 1);
        if (padding_count > 0) {
            std::fill_n(mask_data + offset + valid_count, padding_count, 0);
        }
        offset += padded_length;
    }
    return padding_mask;
}

size_t RoundRobinProcessor::handleOutputs(torch::Tensor&                            hidden_states,
                                          const GptModelInputs&                     inputs,
                                          const torch_ext::PyContextParallelParams& cp_params) {
#if !USING_CUDA
    RTP_LLM_FAIL("Context parallel not supported on ROCm");
    return 0;
#else
    int prefill_cp_size = tp_size_;

    auto all_hidden_t =
        torch::empty({hidden_states.size(0) * prefill_cp_size, hidden_states.size(1)}, hidden_states.options());
    execAllGather({{all_hidden_t}, ParallelMode::TP, {hidden_states}, false});

    auto          prefill_qkv_restore_indice = cp_params.prefill_qkv_restore_indice;
    auto          prefill_qkv_padding_mask   = cp_params.prefill_qkv_padding_mask;
    torch::Tensor valid_indices              = torch::nonzero(prefill_qkv_padding_mask).squeeze(-1);
    int64_t       num_valid_tokens           = valid_indices.size(0);
    torch::Tensor combined_indices           = prefill_qkv_restore_indice.index_select(0, valid_indices);
    hidden_states                            = all_hidden_t.index_select(0, combined_indices);
    return num_valid_tokens;
#endif
}

}  // namespace rtp_llm
