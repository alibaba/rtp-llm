#include "rtp_llm/cpp/models/context_parallel/ContextParallelProcessorBase.h"
#include "rtp_llm/models_py/bindings/core/ExecOps.h"
#include "rtp_llm/models_py/bindings/core/OpData.h"
#include "rtp_llm/cpp/utils/AssertUtils.h"
#include "rtp_llm/models_py/bindings/OpDefs.h"

namespace rtp_llm {

void IContextParallelProcessor::handleInputs(GptModelInputs&                     model_input,
                                             torch_ext::PyContextParallelParams& cp_params) {
#if !USING_CUDA
    RTP_LLM_FAIL("Context parallel not supported on ROCm");
#else
    int prefill_cp_rank = parallelism_config_.tp_rank;
    int prefill_cp_size = parallelism_config_.tp_size;
    int cp_align_size   = prefill_cp_size * 2;

    static const auto pinned_i32 = torch::TensorOptions(torch::kInt32).pinned_memory(true);

    auto& total_input_tokens       = model_input.combo_tokens;
    auto& total_hidden_states      = model_input.last_hidden_states;
    auto& input_lengths            = model_input.input_lengths;
    auto& sequence_lengths         = model_input.sequence_lengths;
    auto  input_lengths_cpu_tensor = input_lengths.clone().pin_memory();

    size_t num_decode_stream  = sequence_lengths.size(0);
    size_t num_prefill_stream = input_lengths.size(0) - num_decode_stream;

    auto prefill_cp_padding_lengths = torch::empty({(int64_t)num_prefill_stream}, pinned_i32);
    auto prefill_cp_chunk_lengths   = torch::empty({(int64_t)num_prefill_stream}, pinned_i32);
    int* padding_lengths            = prefill_cp_padding_lengths.data_ptr<int>();
    int* chunk_lengths              = prefill_cp_chunk_lengths.data_ptr<int>();

    size_t prefill_cp_split_tokens_size = 0;
    for (size_t p = 0; p < num_prefill_stream; ++p) {
        int num_prefill_token = input_lengths.data_ptr<int32_t>()[num_decode_stream + p];

        int padded_seq_len = ((num_prefill_token + cp_align_size - 1) / cp_align_size) * cp_align_size;
        int padding_size   = padded_seq_len - num_prefill_token;
        int chunk_size     = padded_seq_len / prefill_cp_size;

        prefill_cp_split_tokens_size += chunk_size;
        padding_lengths[p] = padding_size;
        chunk_lengths[p]   = chunk_size;
    }

    auto cp_split_input_tokens =
        torch::empty({(int64_t)(num_decode_stream + prefill_cp_split_tokens_size)}, pinned_i32);
    auto prefill_shuffle_indices = torch::empty({(int64_t)prefill_cp_split_tokens_size}, pinned_i32);

    const bool has_hidden_states = total_hidden_states.defined() && total_hidden_states.numel() > 0;
    bool       split_hidden_states = false;
    if (has_hidden_states) {
        RTP_LLM_CHECK_WITH_INFO(total_hidden_states.dim() == 2,
                                "CP MTP hidden states must be 2-D, got dim=%ld",
                                total_hidden_states.dim());
        const int64_t global_token_num = total_input_tokens.numel();
        const int64_t local_token_num  = cp_split_input_tokens.numel();
        if (total_hidden_states.size(0) == global_token_num) {
            split_hidden_states = true;
        } else {
            RTP_LLM_CHECK_WITH_INFO(total_hidden_states.size(0) == local_token_num,
                                    "CP MTP hidden states row count mismatch: rows=%ld, global_tokens=%ld, "
                                    "local_tokens=%ld",
                                    total_hidden_states.size(0),
                                    global_token_num,
                                    local_token_num);
        }
    }
    std::vector<int64_t> hidden_select_indices;
    std::vector<uint8_t> hidden_valid_mask;
    if (split_hidden_states) {
        hidden_select_indices.reserve(cp_split_input_tokens.numel());
        hidden_valid_mask.reserve(cp_split_input_tokens.numel());
    }

    int* input_token_ptr             = cp_split_input_tokens.data_ptr<int>();
    int* input_length_ptr            = input_lengths.data_ptr<int32_t>();
    int* prefill_shuffle_indices_ptr = prefill_shuffle_indices.data_ptr<int>();

    int input_token_idx       = 0;
    int total_input_token_idx = 0;

    if (num_decode_stream > 0) {
        std::memcpy(input_token_ptr,
                    total_input_tokens.data_ptr<int32_t>() + total_input_token_idx,
                    num_decode_stream * sizeof(int));
        if (split_hidden_states) {
            for (size_t i = 0; i < num_decode_stream; ++i) {
                hidden_select_indices.push_back(static_cast<int64_t>(i));
                hidden_valid_mask.push_back(1);
            }
        }
        input_token_idx += num_decode_stream;
        total_input_token_idx += num_decode_stream;
    }

    for (size_t p = 0; p < num_prefill_stream; ++p) {
        int input_chunk_length   = prefill_cp_chunk_lengths.data_ptr<int>()[p];
        int input_padding_length = prefill_cp_padding_lengths.data_ptr<int>()[p];
        int input_length         = input_lengths.data_ptr<int32_t>()[num_decode_stream + p];
        int hidden_src_offset    = total_input_token_idx;

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
        std::memcpy(prefill_shuffle_indices_ptr + input_token_idx - num_decode_stream,
                    shuffle_index.data(),
                    input_chunk_length * sizeof(int));
        if (split_hidden_states) {
            for (int i = 0; i < input_chunk_length; ++i) {
                const int src_idx = shuffle_index[i];
                if (src_idx >= 0 && src_idx < input_length) {
                    hidden_select_indices.push_back(static_cast<int64_t>(hidden_src_offset + src_idx));
                    hidden_valid_mask.push_back(1);
                } else {
                    hidden_select_indices.push_back(0);
                    hidden_valid_mask.push_back(0);
                }
            }
        }
        input_token_idx += input_chunk_length;
        total_input_token_idx += input_length;
        input_length_ptr[num_decode_stream + p] = input_chunk_length;
    }

    model_input.combo_tokens = cp_split_input_tokens;
    if (split_hidden_states) {
        auto select_indices = torch::from_blob(hidden_select_indices.data(),
                                               {(int64_t)hidden_select_indices.size()},
                                               torch::TensorOptions(torch::kInt64))
                                  .clone();
        auto valid_mask = torch::from_blob(
                              hidden_valid_mask.data(),
                              {(int64_t)hidden_valid_mask.size()},
                              torch::TensorOptions(torch::kUInt8))
                              .clone()
                              .to(torch::kBool);
        if (total_hidden_states.is_cuda()) {
            select_indices = select_indices.to(total_hidden_states.device(), true);
            valid_mask     = valid_mask.to(total_hidden_states.device(), true);
        }
        auto split_hidden = total_hidden_states.index_select(0, select_indices);
        split_hidden.masked_fill_(valid_mask.logical_not().unsqueeze(1), 0);
        model_input.last_hidden_states = split_hidden;
    }
    auto cp_padding_lengths  = prefill_cp_padding_lengths;
    auto cp_chunk_lengths    = prefill_cp_chunk_lengths;
    auto shuffle_indices     = prefill_shuffle_indices;

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
