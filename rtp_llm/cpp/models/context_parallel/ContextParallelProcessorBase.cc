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

    // TODO(async): CP planning is CPU-vector based today. Keep explicit host
    // mirrors here, then publish mutated model inputs back to CUDA.
    auto total_input_tokens =
        model_input.combo_tokens.is_cuda() ? model_input.combo_tokens.cpu().pin_memory() : model_input.combo_tokens;
    auto& total_hidden_states = model_input.last_hidden_states;
    auto  input_lengths =
        model_input.input_lengths.is_cuda() ? model_input.input_lengths.cpu().pin_memory() : model_input.input_lengths;
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

    const bool    has_hidden_states   = total_hidden_states.defined() && total_hidden_states.numel() > 0;
    bool          split_hidden_states = false;
    const int64_t global_token_num    = total_input_tokens.numel();
    if (has_hidden_states) {
        RTP_LLM_CHECK_WITH_INFO(
            total_hidden_states.dim() == 2, "CP MTP hidden states must be 2-D, got dim=%ld", total_hidden_states.dim());
        const int64_t local_token_num = cp_split_input_tokens.numel();
        if (model_input.last_hidden_states_layout == MtpHiddenStatesLayout::GLOBAL) {
            RTP_LLM_CHECK_WITH_INFO(total_hidden_states.size(0) == global_token_num,
                                    "global CP MTP hidden states row count mismatch: rows=%ld, global_tokens=%ld",
                                    total_hidden_states.size(0),
                                    global_token_num);
            split_hidden_states = true;
        } else if (model_input.last_hidden_states_layout == MtpHiddenStatesLayout::CP_LOCAL) {
            RTP_LLM_CHECK_WITH_INFO(total_hidden_states.size(0) == local_token_num,
                                    "local CP MTP hidden states row count mismatch: rows=%ld, local_tokens=%ld",
                                    total_hidden_states.size(0),
                                    local_token_num);
        } else {
            RTP_LLM_FAIL("CP MTP hidden states require an explicit GLOBAL or CP_LOCAL layout, rows=%ld, "
                         "global_tokens=%ld, local_tokens=%ld",
                         total_hidden_states.size(0),
                         global_token_num,
                         local_token_num);
        }
    } else {
        model_input.last_hidden_states_layout = MtpHiddenStatesLayout::NONE;
    }
    // Per-local-token remap: for each token this rank keeps after the CP split,
    // record its global source index + validity. Reused to CP-split every per-token
    // side input the same way as combo_tokens: MTP hidden states, text_tokens_mask
    // and combo_tokens_type_ids. Without splitting the mask/type_ids, the embedding
    // op would read a global-length mask misaligned with this rank's token chunk
    // (multimodal placeholder ids stay -1 but get unmasked -> out-of-bounds).
    const bool has_multimodal_input = hasMultimodalModelInputs(model_input);
    const bool has_explicit_position_ids =
        model_input.combo_position_ids.defined() && model_input.combo_position_ids.numel() > 0;
    // Multimodal features are token-aligned inputs too. Always build the
    // global-to-CP-local map when they are present, even if this model does not
    // expose an explicit position-id or token-type field.
    const bool need_token_remap = split_hidden_states || model_input.text_tokens_mask.defined()
                                  || model_input.combo_tokens_type_ids.defined() || has_multimodal_input;
    std::vector<int64_t> cp_select_indices;
    std::vector<uint8_t> cp_valid_mask;
    if (need_token_remap) {
        cp_select_indices.reserve(cp_split_input_tokens.numel());
        cp_valid_mask.reserve(cp_split_input_tokens.numel());
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
        if (need_token_remap) {
            for (size_t i = 0; i < num_decode_stream; ++i) {
                cp_select_indices.push_back(static_cast<int64_t>(i));
                cp_valid_mask.push_back(1);
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
        if (need_token_remap) {
            for (int i = 0; i < input_chunk_length; ++i) {
                const int src_idx = shuffle_index[i];
                if (src_idx >= 0 && src_idx < input_length) {
                    cp_select_indices.push_back(static_cast<int64_t>(hidden_src_offset + src_idx));
                    cp_valid_mask.push_back(1);
                } else {
                    cp_select_indices.push_back(0);
                    cp_valid_mask.push_back(0);
                }
            }
        }
        input_token_idx += input_chunk_length;
        total_input_token_idx += input_length;
        input_length_ptr[num_decode_stream + p] = input_chunk_length;
    }

    if (need_token_remap) {
        auto select_indices = torch::from_blob(cp_select_indices.data(),
                                               {(int64_t)cp_select_indices.size()},
                                               torch::TensorOptions(torch::kInt64))
                                  .clone();
        auto valid_mask =
            torch::from_blob(cp_valid_mask.data(), {(int64_t)cp_valid_mask.size()}, torch::TensorOptions(torch::kUInt8))
                .clone()
                .to(torch::kBool);

        if (split_hidden_states) {
            auto hidden_indices = select_indices;
            auto hidden_valid   = valid_mask;
            if (total_hidden_states.is_cuda()) {
                hidden_indices = hidden_indices.to(total_hidden_states.device(), true);
                hidden_valid   = hidden_valid.to(total_hidden_states.device(), true);
            }
            auto split_hidden = total_hidden_states.index_select(0, hidden_indices);
            split_hidden.masked_fill_(hidden_valid.logical_not().unsqueeze(1), 0);
            model_input.last_hidden_states = split_hidden;
        }

        // CP-split the per-token side inputs the same way as combo_tokens, so the
        // embedding op sees a mask / type_ids aligned with this rank's token chunk.
        // Padding positions (valid_mask == 0) -> 0: a 0 text_tokens_mask makes the
        // embedding kernel skip the word-table lookup for the junk padded id, and a
        // 0 token-type is the natural default.
        auto remap_token_field = [&](torch::Tensor& field) {
            if (!field.defined() || field.numel() == 0) {
                return;
            }
            auto src = field.is_cuda() ? field.cpu() : field;
            auto out = src.index_select(0, select_indices).contiguous();
            out.masked_fill_(valid_mask.logical_not(), 0);
            field = out.to(torch::kCUDA, /*non_blocking=*/true);
        };
        remap_token_field(model_input.text_tokens_mask);
        remap_token_field(model_input.combo_tokens_type_ids);

        // Position IDs are token-aligned too, but this extra remap is deliberately
        // limited to multimodal requests. Text-only CP keeps its existing path.
        if (has_multimodal_input && has_explicit_position_ids) {
            auto source = model_input.combo_position_ids.is_cuda() ? model_input.combo_position_ids.cpu() :
                                                                     model_input.combo_position_ids;
            source      = source.contiguous();
            RTP_LLM_CHECK_WITH_INFO(global_token_num > 0 && source.numel() % global_token_num == 0,
                                    "combo_position_ids numel (%ld) must be divisible by global token count (%ld)",
                                    source.numel(),
                                    global_token_num);
            const int64_t position_id_factor = source.numel() / global_token_num;
            auto output = source.reshape({global_token_num, position_id_factor}).index_select(0, select_indices);
            output.masked_fill_(valid_mask.logical_not().unsqueeze(1), 0);
            output                         = output.reshape({-1}).contiguous();
            model_input.combo_position_ids = output.is_pinned() ? output : output.pin_memory();
        }

        // CP-split multimodal features + locs. The injector overwrites local rows
        // [loc, loc+feature_rows) with feature rows; with CP, each global image
        // ends up at the local positions where cp_select_indices falls in the
        // image's global range. Zigzag CP gives each rank up to 2 contiguous local
        // runs per image (one from the even half, one from the odd half), so we
        // emit one (sliced_feature, local_loc) per run and keep the injector's
        // contiguous-narrow contract intact. This block is never entered for a
        // text-only request.
        const bool has_multimodal_features =
            model_input.multimodal_features.has_value() && !model_input.multimodal_features->empty();
        const bool has_multimodal_locs =
            model_input.mm_features_locs.defined() && model_input.mm_features_locs.numel() > 0;
        const bool has_mm_extra_input = model_input.mm_extra_input.has_value() && !model_input.mm_extra_input->empty();
        RTP_LLM_CHECK_WITH_INFO(!has_multimodal_features || has_multimodal_locs,
                                "multimodal_features require mm_features_locs");
        RTP_LLM_CHECK_WITH_INFO(!has_multimodal_locs || has_multimodal_features,
                                "mm_features_locs require multimodal_features");
        RTP_LLM_CHECK_WITH_INFO(!has_mm_extra_input || has_multimodal_features,
                                "mm_extra_input requires multimodal_features");
        if (has_multimodal_features) {
            auto& orig_features = model_input.multimodal_features.value();
            auto  orig_locs_cpu = model_input.mm_features_locs.is_cuda() ?
                                      model_input.mm_features_locs.cpu().contiguous() :
                                      model_input.mm_features_locs.contiguous();
            RTP_LLM_CHECK_WITH_INFO(orig_locs_cpu.dim() == 1 && orig_locs_cpu.scalar_type() == torch::kInt32,
                                    "mm_features_locs must be a 1-D int32 tensor");
            const auto orig_locs_acc = orig_locs_cpu.accessor<int32_t, 1>();
            const auto num_features  = orig_features.size();
            RTP_LLM_CHECK_WITH_INFO(static_cast<int64_t>(num_features) == orig_locs_cpu.size(0),
                                    "multimodal_features (%zu) and mm_features_locs (%ld) length mismatch",
                                    num_features,
                                    static_cast<int64_t>(orig_locs_cpu.size(0)));
            if (has_mm_extra_input) {
                RTP_LLM_CHECK_WITH_INFO(model_input.mm_extra_input->size() == num_features,
                                        "mm_extra_input (%zu) and multimodal_features (%zu) length mismatch",
                                        model_input.mm_extra_input->size(),
                                        num_features);
            }

            const int64_t              local_tokens = static_cast<int64_t>(cp_select_indices.size());
            std::vector<torch::Tensor> new_features;
            std::vector<torch::Tensor> new_extra_input;
            std::vector<torch::Tensor> deepstack_features(has_mm_extra_input ? num_features : 0);
            std::vector<int32_t>       new_locs;
            // Reserve worst-case 2 runs per image (zigzag's even + odd halves).
            new_features.reserve(num_features * 2);
            new_locs.reserve(num_features * 2);
            if (has_mm_extra_input) {
                new_extra_input.reserve(num_features * 2);
            }

            int64_t previous_feature_end = 0;
            auto    reshape_extra_input  = [](const torch::Tensor& extra_input, const torch::Tensor& feature) {
                RTP_LLM_CHECK_WITH_INFO(feature.dim() == 2,
                                        "multimodal feature must be 2-D when mm_extra_input is present");
                const int64_t feature_len = feature.size(0);
                const int64_t hidden_size = feature.size(1);
                RTP_LLM_CHECK_WITH_INFO(feature_len > 0 && hidden_size > 0,
                                        "multimodal feature tokens and hidden size must be positive");
                const int64_t feature_numel = feature_len * hidden_size;
                RTP_LLM_CHECK_WITH_INFO(extra_input.numel() % feature_numel == 0,
                                        "mm_extra_input numel (%ld) is not divisible by tokens * hidden (%ld)",
                                        extra_input.numel(),
                                        feature_numel);
                return extra_input.reshape({extra_input.numel() / feature_numel, feature_len, hidden_size});
            };
            auto slice_extra_input = [](const torch::Tensor& deepstack, int64_t token_start, int64_t token_end) {
                RTP_LLM_CHECK_WITH_INFO(deepstack.dim() == 3, "reshaped mm_extra_input must be 3-D");
                RTP_LLM_CHECK_WITH_INFO(token_start >= 0 && token_start <= token_end && token_end <= deepstack.size(1),
                                        "mm_extra_input token slice [%ld, %ld) is outside [0, %ld)",
                                        token_start,
                                        token_end,
                                        deepstack.size(1));
                return deepstack.slice(1, token_start, token_end).contiguous().reshape({-1});
            };

            for (size_t f = 0; f < num_features; ++f) {
                RTP_LLM_CHECK_WITH_INFO(orig_features[f].dim() == 2,
                                        "multimodal feature %zu must be 2-D, got dim=%ld",
                                        f,
                                        orig_features[f].dim());
                const int64_t g_start = orig_locs_acc[f];
                const int64_t g_len   = orig_features[f].size(0);
                const int64_t g_end   = g_start + g_len;
                RTP_LLM_CHECK_WITH_INFO(
                    g_len > 0 && orig_features[f].size(1) > 0, "multimodal feature %zu must have positive shape", f);
                RTP_LLM_CHECK_WITH_INFO(g_start >= 0 && g_end <= global_token_num,
                                        "multimodal feature %zu range [%ld,%ld) is outside global input [0,%ld)",
                                        f,
                                        g_start,
                                        g_end,
                                        global_token_num);
                RTP_LLM_CHECK_WITH_INFO(f == 0 || g_start >= previous_feature_end,
                                        "multimodal feature ranges must be sorted and non-overlapping: "
                                        "feature=%zu, start=%ld, previous_end=%ld",
                                        f,
                                        g_start,
                                        previous_feature_end);
                previous_feature_end = g_end;
                if (has_mm_extra_input) {
                    deepstack_features[f] =
                        reshape_extra_input(model_input.mm_extra_input.value()[f], orig_features[f]);
                }

                int64_t i = 0;
                while (i < local_tokens) {
                    const bool in_range = cp_valid_mask[i] != 0 && static_cast<int64_t>(cp_select_indices[i]) >= g_start
                                          && static_cast<int64_t>(cp_select_indices[i]) < g_end;
                    if (!in_range) {
                        ++i;
                        continue;
                    }
                    const int64_t run_local_start = i;
                    const int64_t run_feat_start  = static_cast<int64_t>(cp_select_indices[i]) - g_start;
                    int64_t       expected_feat   = run_feat_start;
                    int64_t       j               = i;
                    while (j < local_tokens && cp_valid_mask[j] != 0
                           && static_cast<int64_t>(cp_select_indices[j]) >= g_start
                           && static_cast<int64_t>(cp_select_indices[j]) < g_end
                           && (static_cast<int64_t>(cp_select_indices[j]) - g_start) == expected_feat) {
                        ++j;
                        ++expected_feat;
                    }
                    const int64_t run_len = j - run_local_start;
                    new_features.push_back(
                        orig_features[f].slice(0, run_feat_start, run_feat_start + run_len).contiguous());
                    if (has_mm_extra_input) {
                        new_extra_input.push_back(
                            slice_extra_input(deepstack_features[f], run_feat_start, run_feat_start + run_len));
                    }
                    new_locs.push_back(static_cast<int32_t>(run_local_start));
                    i = j;
                }
            }

            orig_features = std::move(new_features);
            if (has_mm_extra_input) {
                model_input.mm_extra_input = std::move(new_extra_input);
            }
            if (new_locs.empty()) {
                model_input.mm_features_locs = torch::empty({0}, pinned_i32);
            } else {
                auto locs_cpu = torch::from_blob(new_locs.data(),
                                                 {static_cast<int64_t>(new_locs.size())},
                                                 torch::TensorOptions(torch::kInt32))
                                    .clone();
                model_input.mm_features_locs = locs_cpu.to(torch::kCUDA, /*non_blocking=*/true);
            }
        }
    }

    if (has_hidden_states) {
        model_input.last_hidden_states_layout = MtpHiddenStatesLayout::CP_LOCAL;
    }

    model_input.combo_tokens  = cp_split_input_tokens.to(torch::kCUDA, /*non_blocking=*/true);
    model_input.input_lengths = input_lengths.to(torch::kCUDA, /*non_blocking=*/true);
    model_input.sequence_lengths =
        sequence_lengths.is_cuda() ? sequence_lengths : sequence_lengths.to(torch::kCUDA, /*non_blocking=*/true);

    auto cp_padding_lengths = prefill_cp_padding_lengths;
    auto cp_chunk_lengths   = prefill_cp_chunk_lengths;
    auto shuffle_indices    = prefill_shuffle_indices;

    auto qkv_restore_indice = generateQKVRestoreIndices(cp_chunk_lengths, prefill_cp_size);
    auto qkv_padding_mask   = generateQKVPaddingMask(cp_chunk_lengths, cp_padding_lengths, prefill_cp_size);

    cp_params.prefill_cp_padding_lengths       = cp_padding_lengths.to(torch::kCUDA, /*non_blocking=*/true);
    cp_params.prefill_cp_chunk_lengths         = cp_chunk_lengths.to(torch::kCUDA, /*non_blocking=*/true);
    cp_params.prefill_shuffle_indices          = shuffle_indices.to(torch::kCUDA, /*non_blocking=*/true);
    cp_params.prefill_qkv_restore_indice       = qkv_restore_indice.to(torch::kCUDA, /*non_blocking=*/true);
    cp_params.prefill_qkv_padding_mask         = qkv_padding_mask.to(torch::kCUDA, /*non_blocking=*/true);
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
