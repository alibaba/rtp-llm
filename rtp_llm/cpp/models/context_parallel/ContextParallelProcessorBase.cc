#include "rtp_llm/cpp/models/context_parallel/ContextParallelProcessorBase.h"
#include "rtp_llm/cpp/models/context_parallel/ZigzagTokenLayout.h"
#include "rtp_llm/cpp/multimodal_processor/MultimodalInputUtils.h"
#include "rtp_llm/models_py/bindings/core/ExecOps.h"
#include "rtp_llm/models_py/bindings/core/OpData.h"
#include "rtp_llm/cpp/utils/AssertUtils.h"
#include "rtp_llm/models_py/bindings/OpDefs.h"

namespace rtp_llm {

namespace {

#if USING_CUDA

struct CpTokenRemap {
    torch::Tensor select_indices;
    torch::Tensor valid_mask;
    int64_t       global_token_num;
    int64_t       local_token_num;
};

CpTokenRemap makeTokenRemap(const std::vector<int64_t>& source_indices,
                            const std::vector<uint8_t>& valid_bits,
                            int64_t                     global_token_num,
                            int64_t                     local_token_num) {
    RTP_LLM_CHECK_WITH_INFO(static_cast<int64_t>(source_indices.size()) == local_token_num,
                            "CP source index count (%zu) must equal local token count (%ld)",
                            source_indices.size(),
                            local_token_num);
    RTP_LLM_CHECK_WITH_INFO(valid_bits.size() == source_indices.size(),
                            "CP valid mask count must equal source index count");
    auto select_indices = torch::from_blob(const_cast<int64_t*>(source_indices.data()),
                                           {static_cast<int64_t>(source_indices.size())},
                                           torch::TensorOptions(torch::kInt64))
                              .clone();
    auto valid_mask = torch::from_blob(const_cast<uint8_t*>(valid_bits.data()),
                                       {static_cast<int64_t>(valid_bits.size())},
                                       torch::TensorOptions(torch::kUInt8))
                          .clone()
                          .to(torch::kBool);

    int64_t previous_source_idx = -1;
    for (size_t i = 0; i < source_indices.size(); ++i) {
        if (valid_bits[i] == 0) {
            continue;
        }
        RTP_LLM_CHECK_WITH_INFO(source_indices[i] > previous_source_idx,
                                "valid CP source indices must be strictly increasing: "
                                "local_idx=%zu, source_idx=%ld, previous_source_idx=%ld",
                                i,
                                source_indices[i],
                                previous_source_idx);
        previous_source_idx = source_indices[i];
    }
    return {std::move(select_indices), std::move(valid_mask), global_token_num, local_token_num};
}

void remapTokenField(torch::Tensor& field, const char* field_name, const CpTokenRemap& remap) {
    if (!field.defined() || field.numel() == 0) {
        return;
    }
    auto          source          = field.is_cuda() ? field.cpu() : field;
    const int64_t field_token_num = source.dim() > 0 ? source.size(0) : 0;
    RTP_LLM_CHECK_WITH_INFO(source.dim() == 1 && field_token_num == remap.global_token_num,
                            "%s must be a 1-D global per-token tensor: dim=%ld, tokens=%ld, "
                            "global_tokens=%ld, local_tokens=%ld",
                            field_name,
                            source.dim(),
                            field_token_num,
                            remap.global_token_num,
                            remap.local_token_num);
    auto output = source.index_select(0, remap.select_indices).contiguous();
    output.masked_fill_(remap.valid_mask.logical_not(), 0);
    field = std::move(output);
}

void remapPositionIds(GptModelInputs& model_input, const CpTokenRemap& remap) {
    auto& position_ids = model_input.combo_position_ids;
    if (!position_ids.defined() || position_ids.numel() == 0) {
        return;
    }
    RTP_LLM_CHECK_WITH_INFO(remap.global_token_num > 0 && position_ids.numel() % remap.global_token_num == 0,
                            "combo_position_ids numel (%ld) must be divisible by global token count (%ld)",
                            position_ids.numel(),
                            remap.global_token_num);
    const int64_t position_id_factor = position_ids.numel() / remap.global_token_num;
    auto          source             = position_ids.is_cuda() ? position_ids.cpu() : position_ids;
    auto output = source.reshape({remap.global_token_num, position_id_factor}).index_select(0, remap.select_indices);
    output.masked_fill_(remap.valid_mask.logical_not().unsqueeze(1), 0);
    output       = output.reshape({-1}).contiguous();
    position_ids = output.is_pinned() ? output : output.pin_memory();
}

void remapMultimodalInputs(GptModelInputs&             model_input,
                           const CpTokenRemap&         remap,
                           const torch::TensorOptions& pinned_i32) {
    auto& orig_features = model_input.multimodal_features.value();
    auto  orig_locs_cpu = model_input.mm_features_locs.is_cuda() ? model_input.mm_features_locs.cpu().contiguous() :
                                                                   model_input.mm_features_locs.contiguous();
    const auto orig_locs_acc = orig_locs_cpu.accessor<int32_t, 1>();
    const auto num_features  = orig_features.size();
    RTP_LLM_CHECK_WITH_INFO(static_cast<int64_t>(num_features) == orig_locs_cpu.size(0),
                            "multimodal_features (%zu) and mm_features_locs (%ld) length mismatch",
                            num_features,
                            static_cast<int64_t>(orig_locs_cpu.size(0)));

    const bool has_extra_input = model_input.mm_extra_input.has_value() && !model_input.mm_extra_input.value().empty();
    if (has_extra_input) {
        RTP_LLM_CHECK_WITH_INFO(model_input.mm_extra_input.value().size() == num_features,
                                "mm_extra_input (%zu) and multimodal_features (%zu) length mismatch",
                                model_input.mm_extra_input.value().size(),
                                num_features);
    }

    std::vector<torch::Tensor> new_features;
    std::vector<torch::Tensor> new_extra_input;
    std::vector<torch::Tensor> deepstack_features(has_extra_input ? num_features : 0);
    std::vector<int32_t>       new_locs;
    new_features.reserve(num_features * 2);
    new_locs.reserve(num_features * 2);
    if (has_extra_input) {
        new_extra_input.reserve(num_features * 2);
    }

    bool    has_previous_feature = false;
    int64_t previous_feature_end = 0;
    for (size_t feature_idx = 0; feature_idx < num_features; ++feature_idx) {
        RTP_LLM_CHECK_WITH_INFO(orig_features[feature_idx].dim() == 2,
                                "multimodal feature %zu must be 2-D, got dim=%ld",
                                feature_idx,
                                orig_features[feature_idx].dim());
        const int64_t feature_len = orig_features[feature_idx].size(0);
        const int64_t hidden_size = orig_features[feature_idx].size(1);
        RTP_LLM_CHECK_WITH_INFO(feature_len > 0 && hidden_size > 0,
                                "multimodal feature %zu tokens and hidden must be positive",
                                feature_idx);
        const int64_t feature_start = orig_locs_acc[feature_idx];
        RTP_LLM_CHECK_WITH_INFO(feature_start >= 0,
                                "multimodal feature %zu location must be non-negative, got %ld",
                                feature_idx,
                                feature_start);
        const int64_t feature_end = feature_start + feature_len;
        if (has_previous_feature) {
            RTP_LLM_CHECK_WITH_INFO(feature_start >= previous_feature_end,
                                    "multimodal feature ranges must be sorted and non-overlapping: "
                                    "feature=%zu, start=%ld, previous_end=%ld",
                                    feature_idx,
                                    feature_start,
                                    previous_feature_end);
        }
        has_previous_feature = true;
        previous_feature_end = feature_end;

        if (!has_extra_input) {
            continue;
        }
        const auto& extra_input         = model_input.mm_extra_input.value()[feature_idx];
        deepstack_features[feature_idx] = reshapeMultimodalExtraInput(extra_input, orig_features[feature_idx]);
    }

    const auto source_indices = remap.select_indices.accessor<int64_t, 1>();
    const auto valid_bits     = remap.valid_mask.accessor<bool, 1>();
    int64_t    local_idx      = 0;
    size_t     feature_idx    = 0;
    while (local_idx < remap.local_token_num && feature_idx < num_features) {
        if (!valid_bits[local_idx]) {
            ++local_idx;
            continue;
        }

        const int64_t source_idx    = source_indices[local_idx];
        const int64_t feature_start = orig_locs_acc[feature_idx];
        const int64_t feature_end   = feature_start + orig_features[feature_idx].size(0);
        if (source_idx >= feature_end) {
            ++feature_idx;
            continue;
        }
        if (source_idx < feature_start) {
            ++local_idx;
            continue;
        }

        const int64_t run_local_start   = local_idx;
        const int64_t run_feature_start = source_idx - feature_start;
        int64_t       expected_source   = source_idx;
        while (local_idx < remap.local_token_num && valid_bits[local_idx]
               && source_indices[local_idx] == expected_source && expected_source < feature_end) {
            ++local_idx;
            ++expected_source;
        }

        const int64_t run_len = local_idx - run_local_start;
        new_features.push_back(
            orig_features[feature_idx].slice(0, run_feature_start, run_feature_start + run_len).contiguous());
        if (has_extra_input) {
            new_extra_input.push_back(sliceDeepstackExtraInput(
                deepstack_features[feature_idx], run_feature_start, run_feature_start + run_len));
        }
        new_locs.push_back(static_cast<int32_t>(run_local_start));
    }

    orig_features = std::move(new_features);
    if (has_extra_input) {
        model_input.mm_extra_input = std::move(new_extra_input);
    }
    auto remapped_locs = torch::empty({static_cast<int64_t>(new_locs.size())}, pinned_i32);
    if (!new_locs.empty()) {
        std::memcpy(remapped_locs.data_ptr<int32_t>(), new_locs.data(), new_locs.size() * sizeof(int32_t));
    }
    model_input.mm_features_locs = std::move(remapped_locs);
}

void remapAlignedInputs(GptModelInputs&             model_input,
                        const std::vector<int64_t>& cp_select_indices,
                        const std::vector<uint8_t>& cp_valid_mask,
                        int64_t                     global_token_num,
                        int64_t                     local_token_num,
                        bool                        has_multimodal_input,
                        const torch::TensorOptions& pinned_i32) {
    auto remap = makeTokenRemap(cp_select_indices, cp_valid_mask, global_token_num, local_token_num);
    remapTokenField(model_input.text_tokens_mask, "text_tokens_mask", remap);
    remapTokenField(model_input.combo_tokens_type_ids, "combo_tokens_type_ids", remap);
    remapPositionIds(model_input, remap);
    if (has_multimodal_input) {
        remapMultimodalInputs(model_input, remap, pinned_i32);
    }
}

#endif

}  // namespace

void IContextParallelProcessor::handleInputs(GptModelInputs&                     model_input,
                                             torch_ext::PyContextParallelParams& cp_params) {
#if !USING_CUDA
    RTP_LLM_FAIL("Context parallel not supported on ROCm");
#else
    int        prefill_cp_rank = parallelism_config_.tp_rank;
    int        prefill_cp_size = parallelism_config_.tp_size;
    const bool has_input_embeddings =
        model_input.input_embeddings.has_value() && !model_input.input_embeddings.value().empty();
    const bool has_input_embedding_locs =
        model_input.input_embeddings_locs.defined() && model_input.input_embeddings_locs.numel() > 0;
    RTP_LLM_CHECK_WITH_INFO(!has_input_embeddings && !has_input_embedding_locs,
                            "Context parallel does not support input_embeddings");

    RTP_LLM_CHECK_WITH_INFO(!model_input.attention_mask.defined() || model_input.attention_mask.numel() == 0,
                            "Context parallel does not support an explicit attention_mask");

    const bool has_multimodal_input =
        model_input.multimodal_features.has_value() && !model_input.multimodal_features.value().empty();
    RTP_LLM_CHECK_WITH_INFO(!has_multimodal_input || model_input.mm_features_locs.defined(),
                            "mm_features_locs is required when multimodal_features is non-empty");

    static const auto pinned_i32 = torch::TensorOptions(torch::kInt32).pinned_memory(true);

    // TODO(async): CP planning is CPU-vector based today. Keep explicit host
    // mirrors here, then publish mutated model inputs back to CUDA.
    auto total_input_tokens =
        model_input.combo_tokens.is_cuda() ? model_input.combo_tokens.cpu().pin_memory() : model_input.combo_tokens;
    auto& total_hidden_states = model_input.last_hidden_states;
    auto  input_lengths =
        model_input.input_lengths.is_cuda() ? model_input.input_lengths.cpu().pin_memory() : model_input.input_lengths;
    auto& sequence_lengths = model_input.sequence_lengths;
    // Preserve global lengths before updating input_lengths in place for this CP rank.
    auto input_lengths_cpu_tensor = input_lengths.clone().pin_memory();

    size_t num_decode_stream  = sequence_lengths.size(0);
    size_t num_prefill_stream = input_lengths.size(0) - num_decode_stream;

    const bool has_prefix_lengths = model_input.prefix_lengths.defined() && model_input.prefix_lengths.numel() > 0;
    RTP_LLM_CHECK_WITH_INFO(!has_prefix_lengths || !model_input.prefix_lengths.is_cuda(),
                            "CP prefix_lengths must be a host tensor");
    RTP_LLM_CHECK_WITH_INFO(!has_prefix_lengths
                                || model_input.prefix_lengths.numel() == static_cast<int64_t>(num_prefill_stream),
                            "CP prefix_lengths must match the prefill stream count");
    const int32_t* prefix_lengths_ptr = has_prefix_lengths ? model_input.prefix_lengths.data_ptr<int32_t>() : nullptr;
    bool           has_prefix_reuse   = false;
    for (size_t p = 0; p < num_prefill_stream && has_prefix_lengths; ++p) {
        RTP_LLM_CHECK_WITH_INFO(prefix_lengths_ptr[p] >= 0, "CP prefix_lengths must be non-negative");
        has_prefix_reuse = has_prefix_reuse || prefix_lengths_ptr[p] > 0;
    }

    auto prefill_cp_padding_lengths = torch::empty({(int64_t)num_prefill_stream}, pinned_i32);
    auto prefill_cp_chunk_lengths   = torch::empty({(int64_t)num_prefill_stream}, pinned_i32);
    int* padding_lengths            = prefill_cp_padding_lengths.data_ptr<int>();
    int* chunk_lengths              = prefill_cp_chunk_lengths.data_ptr<int>();

    size_t prefill_cp_split_tokens_size = 0;
    for (size_t p = 0; p < num_prefill_stream; ++p) {
        int num_prefill_token = input_lengths.data_ptr<int32_t>()[num_decode_stream + p];

        const auto token_layout = makeZigzagTokenLayout(num_prefill_token, prefill_cp_size);

        prefill_cp_split_tokens_size += token_layout.token_count_per_rank;
        padding_lengths[p] = token_layout.padding_token_count;
        chunk_lengths[p]   = token_layout.token_count_per_rank;
    }

    auto cp_split_input_tokens =
        torch::empty({(int64_t)(num_decode_stream + prefill_cp_split_tokens_size)}, pinned_i32);
    auto          prefill_shuffle_indices = torch::empty({(int64_t)prefill_cp_split_tokens_size}, pinned_i32);
    const int64_t global_token_num        = total_input_tokens.numel();
    const int64_t local_token_num         = cp_split_input_tokens.numel();

    // Per-local-token remap: for each token this rank keeps after the CP split,
    // record its global source index + validity. Reused to CP-split every per-token
    // side input the same way as combo_tokens: text_tokens_mask and combo_tokens_type_ids.
    // Without splitting the mask/type_ids, the embedding
    // op would read a global-length mask misaligned with this rank's token chunk
    // (multimodal placeholder ids stay -1 but get unmasked -> out-of-bounds).
    const bool has_explicit_position_ids =
        model_input.combo_position_ids.defined() && model_input.combo_position_ids.numel() > 0;
    const bool need_token_remap = model_input.text_tokens_mask.defined() || model_input.combo_tokens_type_ids.defined()
                                  || has_explicit_position_ids || has_multimodal_input;
    const bool           need_source_map = need_token_remap || has_prefix_reuse;
    std::vector<int64_t> cp_select_indices;
    std::vector<uint8_t> cp_valid_mask;
    RTP_LLM_CHECK_WITH_INFO(!need_source_map || num_decode_stream == 0,
                            "Context parallel supports pure-prefill batches only when multimodal or prefix-reuse remap is required");
    if (need_source_map) {
        cp_select_indices.reserve(cp_split_input_tokens.numel());
        cp_valid_mask.reserve(cp_split_input_tokens.numel());
    }

    const bool has_hidden_states = total_hidden_states.defined() && total_hidden_states.numel() > 0;
    bool       should_split_hidden_states = false;
    if (has_hidden_states) {
        RTP_LLM_CHECK_WITH_INFO(
            total_hidden_states.dim() == 2, "CP MTP hidden states must be 2-D, got dim=%ld", total_hidden_states.dim());
        const int64_t hidden_token_num = total_hidden_states.size(0);
        const bool    matches_global   = hidden_token_num == global_token_num;
        const bool    matches_local    = hidden_token_num == local_token_num;
        RTP_LLM_CHECK_WITH_INFO(matches_global || matches_local,
                                "CP MTP hidden states row count mismatch: rows=%ld, global=%ld, local=%ld",
                                hidden_token_num,
                                global_token_num,
                                local_token_num);
        should_split_hidden_states = matches_global && (!matches_local || !prefer_local_hidden_states_);
    }
    std::vector<int64_t> hidden_select_indices;
    std::vector<uint8_t> hidden_valid_mask;
    if (should_split_hidden_states) {
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
        if (should_split_hidden_states) {
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
        int source_offset        = total_input_token_idx;

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
        if (need_source_map || should_split_hidden_states) {
            for (int i = 0; i < input_chunk_length; ++i) {
                const int  src_idx = shuffle_index[i];
                const bool valid   = src_idx >= 0 && src_idx < input_length;
                if (need_source_map) {
                    cp_select_indices.push_back(valid ? static_cast<int64_t>(source_offset + src_idx) : 0);
                    cp_valid_mask.push_back(valid ? 1 : 0);
                }
                if (should_split_hidden_states) {
                    hidden_select_indices.push_back(valid ? static_cast<int64_t>(source_offset + src_idx) : 0);
                    hidden_valid_mask.push_back(valid ? 1 : 0);
                }
            }
        }
        input_token_idx += input_chunk_length;
        total_input_token_idx += input_length;
        input_length_ptr[num_decode_stream + p] = input_chunk_length;
    }

    if (need_token_remap) {
        remapAlignedInputs(model_input,
                           cp_select_indices,
                           cp_valid_mask,
                           global_token_num,
                           local_token_num,
                           has_multimodal_input,
                           pinned_i32);
    }

    // shuffle indices are relative to the uncached input chunk. Materialize
    // absolute one-axis positions once when prefix reuse is active so every
    // transformer layer sees the same cache-aware positions without rebuilding
    // offsets in Python.
    if (!has_explicit_position_ids && has_prefix_reuse) {
        auto absolute_position_ids = torch::empty({local_token_num}, pinned_i32);
        std::memcpy(
            absolute_position_ids.data_ptr<int32_t>(), prefill_shuffle_indices_ptr, local_token_num * sizeof(int32_t));
        auto*  position_ids_ptr = absolute_position_ids.data_ptr<int32_t>();
        size_t local_offset     = 0;
        for (size_t p = 0; p < num_prefill_stream; ++p) {
            for (int i = 0; i < chunk_lengths[p]; ++i) {
                auto& position_id = position_ids_ptr[local_offset + i];
                if (cp_valid_mask[local_offset + i]) {
                    position_id += prefix_lengths_ptr[p];
                } else {
                    position_id = 0;
                }
            }
            local_offset += chunk_lengths[p];
        }
        model_input.combo_position_ids = std::move(absolute_position_ids);
    }

    if (should_split_hidden_states) {
        auto select_indices = torch::from_blob(hidden_select_indices.data(),
                                               {(int64_t)hidden_select_indices.size()},
                                               torch::TensorOptions(torch::kInt64))
                                  .clone();
        auto valid_mask = torch::from_blob(hidden_valid_mask.data(),
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
