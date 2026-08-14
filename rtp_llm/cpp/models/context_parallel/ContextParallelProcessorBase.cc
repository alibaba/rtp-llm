#include "rtp_llm/cpp/models/context_parallel/ContextParallelProcessorBase.h"
#include "rtp_llm/models_py/bindings/core/ExecOps.h"
#include "rtp_llm/models_py/bindings/core/OpData.h"
#include "rtp_llm/cpp/utils/AssertUtils.h"
#include "rtp_llm/models_py/bindings/OpDefs.h"
#include <cstring>
#include <limits>

namespace rtp_llm {

void IContextParallelProcessor::handleInputs(GptModelInputs&                     model_input,
                                             torch_ext::PyContextParallelParams& cp_params) {
#if !USING_CUDA
    RTP_LLM_FAIL("Context parallel not supported on ROCm");
#else
    int prefill_cp_rank = parallelism_config_.tp_rank;
    int prefill_cp_size = parallelism_config_.tp_size;

    static const auto pinned_i32 = torch::TensorOptions(torch::kInt32).pinned_memory(true);

    const auto& total_input_tokens = model_input.combo_tokens;
    const auto& input_lengths      = model_input.input_lengths;
    const auto& sequence_lengths   = model_input.sequence_lengths;

    RTP_LLM_CHECK_WITH_INFO(prefill_cp_size > 0, "CP size must be positive, got %d", prefill_cp_size);
    RTP_LLM_CHECK_WITH_INFO(prefill_cp_rank >= 0 && prefill_cp_rank < prefill_cp_size,
                            "CP rank %d is outside [0, %d)",
                            prefill_cp_rank,
                            prefill_cp_size);
    RTP_LLM_CHECK_WITH_INFO(total_input_tokens.defined(), "CP combo tokens must be defined");
    RTP_LLM_CHECK_WITH_INFO(total_input_tokens.device().is_cpu(), "CP combo tokens must be on CPU");
    RTP_LLM_CHECK_WITH_INFO(total_input_tokens.scalar_type() == torch::kInt32, "CP combo tokens must be int32");
    RTP_LLM_CHECK_WITH_INFO(total_input_tokens.dim() == 1,
                            "CP combo tokens must be one-dimensional, got %ld dimensions",
                            total_input_tokens.dim());
    RTP_LLM_CHECK_WITH_INFO(input_lengths.defined(), "CP input lengths must be defined");
    RTP_LLM_CHECK_WITH_INFO(input_lengths.device().is_cpu(), "CP input lengths must be on CPU");
    RTP_LLM_CHECK_WITH_INFO(input_lengths.scalar_type() == torch::kInt32, "CP input lengths must be int32");
    RTP_LLM_CHECK_WITH_INFO(
        input_lengths.dim() == 1, "CP input lengths must be one-dimensional, got %ld dimensions", input_lengths.dim());
    RTP_LLM_CHECK_WITH_INFO(sequence_lengths.defined(), "CP sequence lengths must be defined");
    RTP_LLM_CHECK_WITH_INFO(sequence_lengths.device().is_cpu(), "CP sequence lengths must be on CPU");
    RTP_LLM_CHECK_WITH_INFO(sequence_lengths.dim() == 1,
                            "CP sequence lengths must be one-dimensional, got %ld dimensions",
                            sequence_lengths.dim());
    RTP_LLM_CHECK_WITH_INFO(input_lengths.size(0) >= sequence_lengths.size(0),
                            "CP input batch size %ld is smaller than decode batch size %ld",
                            input_lengths.size(0),
                            sequence_lengths.size(0));

    auto input_lengths_cpu_tensor = input_lengths.clone().pin_memory();
    auto cp_input_lengths         = input_lengths_cpu_tensor.clone().pin_memory();

    const int64_t original_token_count = total_input_tokens.size(0);

    size_t num_decode_stream  = sequence_lengths.size(0);
    size_t num_prefill_stream = input_lengths.size(0) - num_decode_stream;
    RTP_LLM_CHECK_WITH_INFO(num_decode_stream <= static_cast<size_t>(original_token_count),
                            "Decode batch size %zu exceeds CP token count %ld",
                            num_decode_stream,
                            original_token_count);

    auto prefill_cp_padding_lengths = torch::empty({(int64_t)num_prefill_stream}, pinned_i32);
    auto prefill_cp_chunk_lengths   = torch::empty({(int64_t)num_prefill_stream}, pinned_i32);
    int* padding_lengths            = prefill_cp_padding_lengths.data_ptr<int>();
    int* chunk_lengths              = prefill_cp_chunk_lengths.data_ptr<int>();

    size_t prefill_cp_split_tokens_size = 0;
    for (size_t p = 0; p < num_prefill_stream; ++p) {
        int num_prefill_token = input_lengths.data_ptr<int32_t>()[num_decode_stream + p];
        RTP_LLM_CHECK_WITH_INFO(num_prefill_token > 0,
                                "CP prefill stream %zu must contain at least one token, got %d",
                                p,
                                num_prefill_token);

        const size_t padded_seq_len_size = parallelism_config_.prefill_cp_config.padded_sequence_length(
            static_cast<size_t>(num_prefill_token), static_cast<size_t>(prefill_cp_size));
        RTP_LLM_CHECK_WITH_INFO(padded_seq_len_size <= static_cast<size_t>(std::numeric_limits<int>::max()),
                                "CP padded sequence length %zu exceeds int32 range",
                                padded_seq_len_size);
        const int padded_seq_len = static_cast<int>(padded_seq_len_size);
        int padding_size = padded_seq_len - num_prefill_token;
        int chunk_size   = padded_seq_len / prefill_cp_size;

        prefill_cp_split_tokens_size += chunk_size;
        padding_lengths[p] = padding_size;
        chunk_lengths[p]   = chunk_size;
    }

    auto cp_split_input_tokens =
        torch::empty({(int64_t)(num_decode_stream + prefill_cp_split_tokens_size)}, pinned_i32);
    auto prefill_shuffle_indices = torch::empty({(int64_t)prefill_cp_split_tokens_size}, pinned_i32);

    int* input_token_ptr             = cp_split_input_tokens.data_ptr<int>();
    int* input_length_ptr            = cp_input_lengths.data_ptr<int32_t>();
    int* prefill_shuffle_indices_ptr = prefill_shuffle_indices.data_ptr<int>();

    int input_token_idx       = 0;
    int prefill_shuffle_idx   = 0;
    int total_input_token_idx = 0;

    if (num_decode_stream > 0) {
        std::memcpy(input_token_ptr,
                    total_input_tokens.data_ptr<int32_t>() + total_input_token_idx,
                    num_decode_stream * sizeof(int));
        input_token_idx += num_decode_stream;
        total_input_token_idx += num_decode_stream;
    }

    for (size_t p = 0; p < num_prefill_stream; ++p) {
        int input_chunk_length   = prefill_cp_chunk_lengths.data_ptr<int>()[p];
        int input_padding_length = prefill_cp_padding_lengths.data_ptr<int>()[p];
        int input_length         = input_lengths.data_ptr<int32_t>()[num_decode_stream + p];
        RTP_LLM_CHECK_WITH_INFO(total_input_token_idx + input_length <= original_token_count,
                                "CP prefill stream %zu ends at token %d, beyond combo token count %ld",
                                p,
                                total_input_token_idx + input_length,
                                original_token_count);

        int*             src_tokens = total_input_tokens.data_ptr<int32_t>() + total_input_token_idx;
        std::vector<int> total_input_token_vec(src_tokens, src_tokens + input_length);
        std::vector<int> chunk_input_token(input_chunk_length, 0);
        std::vector<int> shuffle_index(input_chunk_length, -1);

        bool success = plan(total_input_token_vec,
                            chunk_input_token,
                            shuffle_index,
                            prefill_cp_rank,
                            prefill_cp_size,
                            input_chunk_length,
                            input_padding_length);
        RTP_LLM_CHECK_WITH_INFO(success, "Context parallel planning failed for prefill stream %zu", p);

        std::memcpy(input_token_ptr + input_token_idx, chunk_input_token.data(), input_chunk_length * sizeof(int));
        std::memcpy(
            prefill_shuffle_indices_ptr + prefill_shuffle_idx, shuffle_index.data(), input_chunk_length * sizeof(int));
        input_token_idx += input_chunk_length;
        prefill_shuffle_idx += input_chunk_length;
        total_input_token_idx += input_length;
        input_length_ptr[num_decode_stream + p] = input_chunk_length;
    }

    RTP_LLM_CHECK_WITH_INFO(total_input_token_idx == original_token_count,
                            "CP input lengths cover %d tokens, but combo tokens contain %ld",
                            total_input_token_idx,
                            original_token_count);

    model_input.combo_tokens  = cp_split_input_tokens;
    model_input.input_lengths = cp_input_lengths;
    auto cp_padding_lengths = prefill_cp_padding_lengths;
    auto cp_chunk_lengths   = prefill_cp_chunk_lengths;
    auto shuffle_indices    = prefill_shuffle_indices;

    auto qkv_restore_indice = generateQKVRestoreIndices(cp_chunk_lengths, prefill_cp_size);
    auto qkv_padding_mask   = generateQKVPaddingMask(cp_chunk_lengths, cp_padding_lengths, prefill_cp_size);

    cp_params.prefill_cp_padding_lengths       = cp_padding_lengths.cuda();
    cp_params.prefill_cp_chunk_lengths         = cp_chunk_lengths.cuda();
    cp_params.prefill_shuffle_indices          = shuffle_indices.cuda();
    cp_params.prefill_qkv_restore_indice       = qkv_restore_indice.cuda();
    cp_params.prefill_qkv_padding_mask         = qkv_padding_mask.cuda();
    cp_params.prefill_actual_input_lengths_cpu = input_lengths_cpu_tensor;
#endif
}

size_t IContextParallelProcessor::handleOutputs(torch::Tensor&                            hidden_states,
                                                const GptModelInputs&                     inputs,
                                                const torch_ext::PyContextParallelParams& cp_params) {
#if !USING_CUDA
    RTP_LLM_FAIL("Context parallel not supported on ROCm");
    return 0;
#else
    int prefill_cp_size = parallelism_config_.tp_size;

    auto all_hidden_t =
        torch::empty({hidden_states.size(0) * prefill_cp_size, hidden_states.size(1)}, hidden_states.options());
    execAllGather({{all_hidden_t}, ParallelMode::TP, {hidden_states}, false});

    int64_t num_valid_tokens = all_hidden_t.size(0);
    hidden_states            = all_hidden_t;
    return num_valid_tokens;
#endif
}

}  // namespace rtp_llm
