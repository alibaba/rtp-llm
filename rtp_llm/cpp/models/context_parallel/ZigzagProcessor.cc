#include "rtp_llm/cpp/models/context_parallel/ZigzagProcessor.h"
#include "rtp_llm/models_py/bindings/core/ExecOps.h"
#include "rtp_llm/models_py/bindings/core/OpData.h"
#include "rtp_llm/cpp/utils/AssertUtils.h"
#include "rtp_llm/models_py/bindings/OpDefs.h"
#include <ATen/ops/searchsorted.h>
#include <algorithm>
#include <limits>
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

size_t ZigZagProcessor::handleOutputs(torch::Tensor&                            hidden_states,
                                      const GptModelInputs&                     inputs,
                                      const torch_ext::PyContextParallelParams& cp_params) {
#if !USING_CUDA
    RTP_LLM_FAIL("Context parallel not supported on ROCm");
    return 0;
#else
    constexpr int64_t kMaxRestoreScratchBytes = int64_t{1} << 30;

    RTP_LLM_CHECK_WITH_INFO(hidden_states.dim() == 2, "CP output must be 2-D, got dim=%ld", hidden_states.dim());
    RTP_LLM_CHECK_WITH_INFO(hidden_states.is_contiguous(), "CP output must be contiguous before all-gather");
    const int     prefill_cp_size = parallelism_config_.tp_size;
    const int64_t local_token_num = hidden_states.size(0);
    const int64_t hidden_size     = hidden_states.size(1);
    RTP_LLM_CHECK_WITH_INFO(prefill_cp_size > 0, "prefill CP size must be positive, got %d", prefill_cp_size);
    RTP_LLM_CHECK_WITH_INFO(local_token_num <= std::numeric_limits<int64_t>::max() / prefill_cp_size,
                            "CP padded token count overflows int64: local_tokens=%ld, cp_size=%d",
                            local_token_num,
                            prefill_cp_size);
    const int64_t padded_token_num = local_token_num * prefill_cp_size;

    const auto& restore_indices  = cp_params.prefill_qkv_restore_indice;
    const auto& padding_mask     = cp_params.prefill_qkv_padding_mask;
    auto        valid_indices    = torch::nonzero(padding_mask).squeeze(-1);
    const auto  num_valid_tokens = valid_indices.size(0);

    RTP_LLM_CHECK_WITH_INFO(restore_indices.numel() == padded_token_num,
                            "CP restore index count mismatch: indices=%ld, local_tokens=%ld, cp_size=%d",
                            restore_indices.numel(),
                            local_token_num,
                            prefill_cp_size);
    RTP_LLM_CHECK_WITH_INFO(padding_mask.numel() == restore_indices.numel(),
                            "CP padding mask count mismatch: mask=%ld, restore_indices=%ld",
                            padding_mask.numel(),
                            restore_indices.numel());

    if (num_valid_tokens == 0) {
        hidden_states = torch::empty({0, hidden_size}, hidden_states.options());
        return 0;
    }
    auto gather_to_output_indices = buildGatherToOutputIndices(restore_indices, padding_mask, num_valid_tokens);

    // Allocate the one required full-size result before communication, then
    // all-gather into a bounded scratch tensor. The previous implementation
    // kept a full rank-major gather alive while index_select allocated another
    // full restored tensor. For DSpARK at 1M tokens both are about 24 GiB.
    // Padding is bounded by CP alignment per request, so retaining its rows in
    // this destination costs little. Giving every gathered row a unique output
    // index also keeps index_copy_ deterministic; the valid compact prefix is
    // returned below and the padding tail is discarded.
    auto restored_padded = torch::empty({padded_token_num, hidden_size}, hidden_states.options());

    const auto element_size = static_cast<int64_t>(hidden_states.element_size());
    RTP_LLM_CHECK_WITH_INFO(hidden_size == 0
                                || hidden_size <= std::numeric_limits<int64_t>::max() / element_size / prefill_cp_size,
                            "CP all-gather row byte size overflows int64: hidden=%ld, element_size=%ld, cp_size=%d",
                            hidden_size,
                            element_size,
                            prefill_cp_size);
    const int64_t gathered_row_bytes = hidden_size * element_size * prefill_cp_size;
    const int64_t max_chunk_rows =
        gathered_row_bytes == 0 ? local_token_num : std::max<int64_t>(1, kMaxRestoreScratchBytes / gathered_row_bytes);
    const int64_t chunk_rows = std::min(local_token_num, max_chunk_rows);

    auto gather_scratch = torch::empty({chunk_rows * prefill_cp_size, hidden_size}, hidden_states.options());
    for (int64_t chunk_offset = 0; chunk_offset < local_token_num; chunk_offset += chunk_rows) {
        const int64_t current_chunk_rows = std::min(chunk_rows, local_token_num - chunk_offset);
        auto          send_chunk         = hidden_states.narrow(0, chunk_offset, current_chunk_rows);
        auto          recv_chunk         = gather_scratch.narrow(0, 0, current_chunk_rows * prefill_cp_size);
        execAllGather({{recv_chunk}, ParallelMode::TP, {send_chunk}, false});
        restoreGatheredChunk(
            restored_padded, recv_chunk, gather_to_output_indices, local_token_num, chunk_offset, prefill_cp_size);
    }
    hidden_states = restored_padded.narrow(0, 0, num_valid_tokens);
    return num_valid_tokens;
#endif
}

torch::Tensor ZigZagProcessor::buildGatherToOutputIndices(const torch::Tensor& restore_indices,
                                                          const torch::Tensor& padding_mask,
                                                          int64_t              num_valid_tokens) {
    RTP_LLM_CHECK_WITH_INFO(
        restore_indices.dim() == 1, "CP restore indices must be 1-D, got dim=%ld", restore_indices.dim());
    RTP_LLM_CHECK_WITH_INFO(padding_mask.dim() == 1, "CP padding mask must be 1-D, got dim=%ld", padding_mask.dim());
    RTP_LLM_CHECK_WITH_INFO(restore_indices.numel() == padding_mask.numel(),
                            "CP restore/mask size mismatch: %ld vs %ld",
                            restore_indices.numel(),
                            padding_mask.numel());

    auto valid_indices = torch::nonzero(padding_mask).squeeze(-1);
    RTP_LLM_CHECK_WITH_INFO(valid_indices.size(0) == num_valid_tokens,
                            "CP valid-token count mismatch: mask=%ld, expected=%ld",
                            valid_indices.size(0),
                            num_valid_tokens);
    auto long_options     = restore_indices.options().dtype(torch::kLong);
    auto gather_to_output = torch::empty({restore_indices.numel()}, long_options);

    auto valid_source_indices = restore_indices.index_select(0, valid_indices).to(torch::kLong);
    auto valid_output_indices = torch::arange(num_valid_tokens, long_options);
    gather_to_output.index_copy_(0, valid_source_indices, valid_output_indices);

    auto padding_indices        = torch::nonzero(padding_mask.logical_not()).squeeze(-1);
    auto padding_source_indices = restore_indices.index_select(0, padding_indices).to(torch::kLong);
    auto padding_output_indices = torch::arange(num_valid_tokens, restore_indices.numel(), long_options);
    RTP_LLM_CHECK_WITH_INFO(padding_source_indices.numel() == padding_output_indices.numel(),
                            "CP padding-source count mismatch: sources=%ld, outputs=%ld",
                            padding_source_indices.numel(),
                            padding_output_indices.numel());
    gather_to_output.index_copy_(0, padding_source_indices, padding_output_indices);
    return gather_to_output;
}

void ZigZagProcessor::restoreGatheredChunk(torch::Tensor&       restored_padded,
                                           const torch::Tensor& gathered_chunk,
                                           const torch::Tensor& gather_to_output_indices,
                                           int64_t              local_token_num,
                                           int64_t              chunk_offset,
                                           int                  cp_size) {
    RTP_LLM_CHECK_WITH_INFO(cp_size > 0, "CP size must be positive, got %d", cp_size);
    RTP_LLM_CHECK_WITH_INFO(
        gathered_chunk.dim() == 2, "CP gathered chunk must be 2-D, got dim=%ld", gathered_chunk.dim());
    RTP_LLM_CHECK_WITH_INFO(gathered_chunk.size(0) % cp_size == 0,
                            "CP gathered chunk rows %ld are not divisible by cp_size %d",
                            gathered_chunk.size(0),
                            cp_size);
    const int64_t chunk_rows = gathered_chunk.size(0) / cp_size;
    RTP_LLM_CHECK_WITH_INFO(chunk_offset >= 0 && chunk_offset + chunk_rows <= local_token_num,
                            "CP gathered chunk range [%ld, %ld) exceeds local token count %ld",
                            chunk_offset,
                            chunk_offset + chunk_rows,
                            local_token_num);
    RTP_LLM_CHECK_WITH_INFO(local_token_num <= std::numeric_limits<int64_t>::max() / cp_size,
                            "CP inverse restore size overflows int64: local_tokens=%ld, cp_size=%d",
                            local_token_num,
                            cp_size);
    RTP_LLM_CHECK_WITH_INFO(gather_to_output_indices.numel() == local_token_num * cp_size,
                            "CP inverse restore count mismatch: indices=%ld, local_tokens=%ld, cp_size=%d",
                            gather_to_output_indices.numel(),
                            local_token_num,
                            cp_size);
    RTP_LLM_CHECK_WITH_INFO(restored_padded.dim() == 2 && restored_padded.size(1) == gathered_chunk.size(1),
                            "CP restored/gathered hidden width mismatch: restored=%ld, gathered=%ld",
                            restored_padded.dim() == 2 ? restored_padded.size(1) : -1,
                            gathered_chunk.size(1));

    auto index_options       = gather_to_output_indices.options().dtype(torch::kLong);
    auto rank_offsets        = torch::arange(cp_size, index_options) * local_token_num;
    auto local_offsets       = torch::arange(chunk_offset, chunk_offset + chunk_rows, index_options);
    auto source_indices      = (rank_offsets.unsqueeze(1) + local_offsets.unsqueeze(0)).reshape({-1});
    auto destination_indices = gather_to_output_indices.index_select(0, source_indices);

    restored_padded.index_copy_(0, destination_indices, gathered_chunk);
}

torch::Tensor ZigZagProcessor::computeLocalLastHidden(const torch::Tensor&                      hidden_states,
                                                      const GptModelInputs&                     inputs,
                                                      const torch_ext::PyContextParallelParams& cp_params) {
    // chunk_len is this rank's local row count; the all-gather lays rank r at
    // offset r * chunk_len, so a chunk-concat flat index g decomposes as
    // g = owner_rank * chunk_len + local_off (see handleOutputs / generateQKVRestoreIndices).
    const int64_t chunk_len = hidden_states.size(0);

    // Equivalent to nonzero(padding_mask)[lm_output_indexes], but keeps the
    // output shape fixed and avoids the CUDA nonzero host sync.
    torch::Tensor lm_output_indexes = inputs.lm_output_indexes.to(torch::kLong);
    torch::Tensor valid_indices     = at::searchsorted(cp_params.prefill_qkv_padding_mask.cumsum(0),
                                                   lm_output_indexes + 1,
                                                   /*out_int32=*/false,
                                                   /*right=*/false);
    torch::Tensor sel = cp_params.prefill_qkv_restore_indice.index_select(0, valid_indices).to(torch::kLong);

    // sel is non-negative kLong; floor-division and the residual recover (rank, offset).
    torch::Tensor owner     = sel.div(chunk_len, c10::string_view("floor"));
    torch::Tensor local_off = sel - owner.mul(chunk_len);
    torch::Tensor mine      = (owner == static_cast<int64_t>(parallelism_config_.tp_rank));

    torch::Tensor local_selected = hidden_states.index_select(0, local_off);
    return torch::where(mine.unsqueeze(-1), local_selected, torch::zeros_like(local_selected));
}

void ZigZagProcessor::handleOutputsLastHidden(torch::Tensor&                            hidden_states,
                                              const GptModelInputs&                     inputs,
                                              const torch_ext::PyContextParallelParams& cp_params) {
#if !USING_CUDA
    RTP_LLM_FAIL("Context parallel not supported on ROCm");
#else
    // Each requested last-token row is owned by exactly one CP rank (zigzag is a
    // bijection on valid positions), so every other rank contributes an all-zero
    // row — an all-reduce-sum reconstructs the gathered rows exactly (no bf16
    // accumulation error) over a small [num_lm, hidden] buffer instead of the
    // full [seq, hidden] all-gather.
    torch::Tensor local_buf = computeLocalLastHidden(hidden_states, inputs, cp_params);
    auto          reduced   = execAllReduce({local_buf, ReduceOp::Sum, false, ParallelMode::TP});
    hidden_states           = reduced.buffer;
#endif
}

}  // namespace rtp_llm
