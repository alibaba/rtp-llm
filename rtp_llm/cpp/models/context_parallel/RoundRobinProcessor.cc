#include "rtp_llm/cpp/models/context_parallel/RoundRobinProcessor.h"
#include "rtp_llm/cpp/devices/DeviceBase.h"
#include "rtp_llm/cpp/devices/OpData.h"
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
    RTP_LLM_CHECK(padded_seq_token_size % cp_size == 0);
    RTP_LLM_CHECK(cp_chunk_size == padded_seq_token_size / cp_size);

    // shuffle_indices is an arithmetic sequence: cp_rank, cp_rank+cp_size, cp_rank+2*cp_size, ...
    for (int i = 0; i < cp_chunk_size; i++) {
        shuffle_indices[i] = cp_rank + i * cp_size;
    }

    // Number of valid (non-padding) tokens this rank receives.
    // Last valid global position for this rank: cp_rank + (valid_count-1)*cp_size < input_token_size
    // → valid_count = (input_token_size - cp_rank + cp_size - 1) / cp_size  (when cp_rank < input_token_size)
    int valid_count = (cp_rank < input_token_size)
                          ? (input_token_size - cp_rank + cp_size - 1) / cp_size
                          : 0;

    // Gather valid tokens with stride access
    const int* src = total_input_tokens.data();
    for (int i = 0; i < valid_count; i++) {
        input_tokens[i] = src[cp_rank + i * cp_size];
    }
    // Zero-fill padding portion
    if (valid_count < cp_chunk_size) {
        std::fill(input_tokens.begin() + valid_count, input_tokens.begin() + cp_chunk_size, 0);
    }
    return true;
}

torch::Tensor RoundRobinProcessor::generateQKVRestoreIndices(const torch::Tensor& prefill_cp_chunk_lengths,
                                                             int                  cp_size) {
    int num_prefill_streams = prefill_cp_chunk_lengths.size(0);
    int total_token_size    = torch::sum(prefill_cp_chunk_lengths).item<int>();

    // After all-gather the tensor is [rank0_chunk | rank1_chunk | ...].
    // restore[seq_offset + rank + j * cp_size] = rank * total_token_size + chunk_offset + j
    //
    // Optimized: fill cp_size interleaved arithmetic sequences simultaneously.
    // For each rank r, the write positions are seq_offset+r, seq_offset+r+cp_size, ...
    // and the values are base_r, base_r+1, base_r+2, ... (sequential).
    // This gives sequential reads per rank and stride-cp_size writes.
    // Since cp_size is small (2~8), we iterate over ranks in the outer loop
    // to keep the inner loop tight and branch-free.

    torch::Tensor sorted_indices = torch::empty(
        {cp_size * total_token_size}, torch::TensorOptions(torch::kInt32).device(torch::kCPU).pinned_memory(true));
    int* indices_data = sorted_indices.data_ptr<int>();

    std::vector<int> chunk_lengths(num_prefill_streams);
    for (int s = 0; s < num_prefill_streams; s++) {
        chunk_lengths[s] = prefill_cp_chunk_lengths[s].item<int>();
    }

    for (int rank = 0; rank < cp_size; rank++) {
        int base   = rank * total_token_size;  // flat base for this rank
        int dst    = rank;                      // first write position (global)
        int offset = 0;                         // cumulative chunk offset

        for (int s = 0; s < num_prefill_streams; s++) {
            int cl = chunk_lengths[s];
            // Write cl values: indices_data[dst], indices_data[dst+cp_size], ...
            // = base+offset, base+offset+1, base+offset+2, ...
            int val = base + offset;
            for (int j = 0; j < cl; j++) {
                indices_data[dst] = val;
                dst += cp_size;
                val++;
            }
            offset += cl;
        }
    }
    return sorted_indices;
}

torch::Tensor RoundRobinProcessor::generateQKVPaddingMask(const torch::Tensor& prefill_cp_chunk_lengths,
                                                          const torch::Tensor& prefill_cp_padding_lengths,
                                                          int                  cp_size) {
    int num_prefill_streams = prefill_cp_chunk_lengths.size(0);
    auto padded_seq_lengths = prefill_cp_chunk_lengths * cp_size;
    int  total_size         = torch::sum(padded_seq_lengths).item<int>();

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

size_t RoundRobinProcessor::handleOutputs(DeviceBase*                               device,
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
