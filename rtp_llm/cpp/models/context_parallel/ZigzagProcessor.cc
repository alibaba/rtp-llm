#include "rtp_llm/cpp/models/context_parallel/ZigzagProcessor.h"
#include "rtp_llm/cpp/devices/DeviceBase.h"
#include "rtp_llm/cpp/devices/OpData.h"
#include "rtp_llm/cpp/utils/AssertUtils.h"
#include "rtp_llm/models_py/bindings/OpDefs.h"
#include <numeric>
#include <vector>

using namespace std;

namespace rtp_llm {

bool ZigZagProcessor::plan(const std::vector<int>& total_input_tokens,
                           std::vector<int>&       input_tokens,
                           std::vector<int>&       shuffle_indices,
                           int                     cp_rank,
                           int                     cp_size,
                           int                     cp_chunk_size,
                           int                     cp_padding_size) {
    const int input_token_size      = static_cast<int>(total_input_tokens.size());
    const int padded_seq_token_size = input_token_size + cp_padding_size;
    RTP_LLM_CHECK(cp_rank >= 0 && cp_rank < cp_size);

    const int pair_size = padded_seq_token_size / (cp_size * 2);

    // Even pair (from start): indices are [cp_rank * pair_size, ...)
    const int even_source = cp_rank * pair_size;
    // Odd pair (from end): indices are [padded_seq_token_size - pair_size * (cp_rank + 1), ...)
    const int odd_source = padded_seq_token_size - pair_size * (cp_rank + 1);

    // Fill shuffle_indices
    std::iota(shuffle_indices.begin(), shuffle_indices.begin() + pair_size, even_source);
    std::iota(shuffle_indices.begin() + pair_size, shuffle_indices.begin() + pair_size * 2, odd_source);

    // Even pair: source indices [even_source, even_source + pair_size)
    if (even_source < input_token_size) {
        const int copy_size = std::min(pair_size, input_token_size - even_source);
        std::memcpy(input_tokens.data(), total_input_tokens.data() + even_source, copy_size * sizeof(int));
    }

    // Odd pair: source indices [odd_source, odd_source + pair_size)
    if (odd_source < input_token_size) {
        const int copy_size = std::min(pair_size, input_token_size - odd_source);
        std::memcpy(input_tokens.data() + pair_size, total_input_tokens.data() + odd_source, copy_size * sizeof(int));
    }
    return true;
}

torch::Tensor ZigZagProcessor::generateQKVRestoreIndices(const torch::Tensor& prefill_cp_chunk_lengths, int cp_size) {
    int           num_prefill_streams = prefill_cp_chunk_lengths.size(0);
    int           total_token_size    = torch::sum(prefill_cp_chunk_lengths).item<int>();
    torch::Tensor qkv_restore_indices =
        torch::empty({cp_size, total_token_size}, torch::TensorOptions(torch::kInt32).device(torch::kCPU));

    int* qkv_data = qkv_restore_indices.data_ptr<int>();

    // Optimized: Directly compute indices without generating full shuffle_indices each time
    int chunk_offset = 0;
    int seq_offset   = 0;
    for (int stream = 0; stream < num_prefill_streams; stream++) {
        int chunk_length    = prefill_cp_chunk_lengths[stream].item<int>();
        int prefill_qkv_len = chunk_length * cp_size;
        int pair_size       = chunk_length / 2;  // prefill_qkv_len / (cp_size * 2)

        // For each cp_rank, directly compute its indices without full shuffle generation
        for (int cp_rank = 0; cp_rank < cp_size; cp_rank++) {
            int* dst = qkv_data + cp_rank * total_token_size + chunk_offset;

            // Even pair (from start): indices are [cp_rank * pair_size, ...)
            const int even_source = cp_rank * pair_size + seq_offset;
            std::iota(dst, dst + pair_size, even_source);

            // Odd pair (from end): indices are [prefill_qkv_len - pair_size * (cp_rank + 1), ...)
            const int odd_source = prefill_qkv_len - pair_size * (cp_rank + 1) + seq_offset;
            std::iota(dst + pair_size, dst + pair_size * 2, odd_source);
        }
        chunk_offset += chunk_length;
        seq_offset += prefill_qkv_len;
    }
    torch::Tensor sorted_indices = torch::empty(
        {cp_size * total_token_size}, torch::TensorOptions(torch::kInt32).device(torch::kCPU).pinned_memory(true));
    int* indices_data = sorted_indices.data_ptr<int>();

    for (int flat_idx = 0; flat_idx < cp_size * total_token_size; flat_idx++) {
        int value           = qkv_data[flat_idx];
        indices_data[value] = flat_idx;
    }
    return sorted_indices;
}

torch::Tensor ZigZagProcessor::generateQKVPaddingMask(const torch::Tensor& prefill_cp_chunk_lengths,
                                                      const torch::Tensor& prefill_cp_padding_lengths,
                                                      int                  cp_size) {
    int num_prefill_streams = prefill_cp_chunk_lengths.size(0);

    // Calculate padded sequence lengths: chunk_length * cp_size
    auto padded_seq_lengths = prefill_cp_chunk_lengths * cp_size;

    // Calculate total mask size
    int total_size = torch::sum(padded_seq_lengths).item<int>();

    // Optimized: Initialize with 1s (valid tokens) first, then overwrite padding with 0s
    // This is faster than separate fill operations for large sequences
    torch::Tensor padding_mask =
        torch::empty({total_size}, torch::TensorOptions(torch::kInt32).device(torch::kCPU).pinned_memory(true));
    int* mask_data = padding_mask.data_ptr<int>();

    // Only fill padding regions (typically smaller than valid regions)
    int offset = 0;
    for (int i = 0; i < num_prefill_streams; i++) {
        int padded_length = padded_seq_lengths[i].item<int>();
        int padding_count = prefill_cp_padding_lengths[i].item<int>();
        int valid_count   = padded_length - padding_count;

        std::fill_n(mask_data + offset, valid_count, 1);

        if (padding_count > 0) {
            int valid_count = padded_length - padding_count;
            // Only overwrite padding tokens to 0
            std::fill_n(mask_data + offset + valid_count, padding_count, 0);
        }
        offset += padded_length;
    }
    return padding_mask;
}

size_t ZigZagProcessor::handleOutputs(DeviceBase*                               device,
                                      BufferPtr&                                hidden_states,
                                      const GptModelInputs&                     inputs,
                                      const torch_ext::PyContextParallelParams& cp_params) {
    int prefill_cp_size = device->getDeviceProperties().tp_size;

    BufferPtr all_hidden_states = device->allocateBuffer(
        {hidden_states->type(), {hidden_states->shape()[0] * prefill_cp_size, hidden_states->shape()[1]}},
        {"allgather_hidden_states"});
    device->allGather({{all_hidden_states}, ParallelMode::TP, {hidden_states}, false});

    auto          all_hidden_states_tensor   = Buffer2torchTensor(all_hidden_states, false);
    auto          prefill_qkv_restore_indice = cp_params.prefill_qkv_restore_indice;
    auto          prefill_qkv_padding_mask   = cp_params.prefill_qkv_padding_mask;
    torch::Tensor valid_indices              = torch::nonzero(prefill_qkv_padding_mask).squeeze(-1);
    int64_t       num_valid_tokens           = valid_indices.size(0);
    torch::Tensor combined_indices           = prefill_qkv_restore_indice.index_select(0, valid_indices);
    torch::Tensor valid_hidden_states        = all_hidden_states_tensor.index_select(0, combined_indices);
    hidden_states                            = torchTensor2Buffer(valid_hidden_states);
    return num_valid_tokens;
}

}  // namespace rtp_llm
