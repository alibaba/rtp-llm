#include "rtp_llm/cpp/normal_engine/speculative/MtpBatchStreamProcessor.h"
#include "rtp_llm/cpp/core/ExecOps.h"
#include "rtp_llm/cpp/utils/TensorDebugUtils.h"
#include "rtp_llm/cpp/utils/StringUtil.h"
#include "rtp_llm/cpp/utils/AssertUtils.h"
#include <numeric>
#include <cstring>

namespace rtp_llm {

absl::Status MtpBatchStreamProcessor::dispatchPrefill(const StreamGroups& stream_groups,
                                                      const MergedOutput& prefill_output,
                                                      const MergedOutput& propose_output) const {
    RTP_LLM_LOG_DEBUG(__PRETTY_FUNCTION__);

    const size_t                      total_batch_size_out = stream_groups.totalSamplerBatchSizeOut();
    auto                              new_tokens_all = torch::empty({(int64_t)total_batch_size_out, 1}, torch::kInt32);
    std::vector<StreamSpecUpdateInfo> spec_update_infos;

    preparePrefillSpecUpdateInfo(stream_groups, prefill_output, propose_output, new_tokens_all, spec_update_infos);

    // we set propose token in extra loop to avoid cuda sync
    updateProposeTokens(stream_groups, propose_output, spec_update_infos);

    // update streams
    stream_groups.updateStreams(spec_update_infos);

    RTP_LLM_LOG_DEBUG("dispatch prefill done");
    return absl::OkStatus();
}

absl::Status MtpBatchStreamProcessor::dispatchDecode(const StreamGroups&                          stream_groups,
                                                     const speculative::SpeculativeSamplerOutput& spec_decode_output,
                                                     const MergedOutput& draft_prefill_output) const {
    RTP_LLM_LOG_DEBUG(__PRETTY_FUNCTION__);

    std::vector<StreamSpecUpdateInfo> spec_update_infos;

    prepareDecodeSpecUpdateInfo(stream_groups, spec_decode_output, draft_prefill_output, spec_update_infos);

    // to avoid cuda sync, we need to set propose token in extra loop
    updateProposeTokens(stream_groups, draft_prefill_output, spec_update_infos);

    stream_groups.updateStreams(spec_update_infos);

    RTP_LLM_LOG_DEBUG("dispatch decode done");
    return absl::OkStatus();
}

absl::StatusOr<GptModelInputs>
MtpBatchStreamProcessor::gatherDecodeModelInput(const StreamGroups& stream_groups) const {
    auto model_input = NormalBatchStreamProcessor::gatherModelInput(stream_groups);

    RTP_LLM_CHECK(model_input.ok());

    if (propose_step_ == 1) {
        return model_input;
    }

    gatherHiddenStates(stream_groups, model_input.value());

    return model_input;
}

absl::StatusOr<SamplerInputs> MtpBatchStreamProcessor::gatherSpecSamplerInput(
    const StreamGroups& stream_groups, const GptModelInputs& model_inputs, const GptModelOutputs& model_output) const {
    (void)model_inputs;
    RTP_LLM_CHECK(!stream_groups.empty());
    auto all_streams      = stream_groups.allStreams();
    bool return_all_probs = stream_groups.needReturnAllProbs();

    for (auto& stream : all_streams) {
        RTP_LLM_CHECK_WITH_INFO(stream->maxBatchSize() == 1, "stream tile num must be 1 in ScoreExecutor");
    }

    size_t score_len        = propose_step_ + 1;
    size_t total_batch_size = stream_groups.size() * score_len;

    SamplerInputs sampler_inputs =
        allocateSamplerInputs(stream_groups, total_batch_size, total_batch_size, propose_step_);
    fillSamplerCommonInputs(sampler_inputs, all_streams, true, propose_step_);

    int batch_idx = 0;
    for (auto& stream : all_streams) {
        auto complete_token_ids = stream->completeTokenIds();
        auto seq_len            = stream->seqLength();
        auto current_batch_size = score_len;

        for (int i = 0; i < current_batch_size; ++i) {
            memcpy(sampler_inputs.token_ids.data_ptr<int32_t>() + ((batch_idx) * (sampler_inputs.step + 1)),
                   complete_token_ids.data_ptr<int32_t>(),
                   seq_len * sizeof(int));
            batch_idx += 1;
        }
        RTP_LLM_LOG_DEBUG("stream [%ld], sampler inputs token ids = [%s]",
                          stream->streamId(),
                          tensorDebugStringWithData<int32_t>(sampler_inputs.token_ids).c_str());
    }

    auto vocab_size = (size_t)model_output.logits.size(1);
    if (return_all_probs) {
        sampler_inputs.all_probs = torch::zeros({(int64_t)total_batch_size, (int64_t)vocab_size},
                                                torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA));
    }

    sampler_inputs.logits = model_output.logits.clone();

    RTP_LLM_LOG_DEBUG("sampler inputs logits [%s]",
                      tensorDebugStringWithData<float>(sampler_inputs.logits.cpu(), 10).c_str());

    RTP_LLM_LOG_DEBUG("gatherSamplerInput done");
    return std::move(sampler_inputs);
}

void MtpBatchStreamProcessor::updateProposeTokens(const StreamGroups&                stream_groups,
                                                  const MergedOutput&                draft_prefill_output,
                                                  std::vector<StreamSpecUpdateInfo>& spec_update_infos) const {
    const auto propose_token_ids_h = draft_prefill_output.sampler_output.token_ids.cpu().clone();

    int token_stride  = propose_token_ids_h.size(1);
    int batch_idx_in  = 0;
    int batch_idx_out = 0;
    int stream_idx    = 0;

    for (auto& stream : stream_groups.allStreams()) {
        auto cur_batch_size   = stream->currentBatchSize();
        auto next_batch_size  = stream->nextBatchSize();
        auto sp_output_buffer = stream->getSPOutputBuffer();

        int propose_token = -1;
        if (propose_token_ids_h.scalar_type() == torch::kLong) {
            propose_token = propose_token_ids_h.data_ptr<int64_t>()[batch_idx_out * token_stride + token_stride - 1];
        } else {
            propose_token = propose_token_ids_h.data_ptr<int32_t>()[batch_idx_out * token_stride + token_stride - 1];
        }

        spec_update_infos[stream_idx].draft_token = propose_token;
        batch_idx_in += cur_batch_size;
        batch_idx_out += next_batch_size;

        stream_idx++;
    }
}

void MtpBatchStreamProcessor::prepareDecodeDraftModelInput(const StreamGroups& stream_groups,
                                                           GptModelInputs&     model_input) {
    size_t batch_size = stream_groups.size();
    int    batch_idx  = 0;

    auto combo_tokens      = torch::empty({(int64_t)batch_size}, torch::kInt32).pin_memory();
    auto lm_output_indexes = torch::empty({(int64_t)batch_size}, torch::kInt32).pin_memory();

    for (const auto& stream : stream_groups.allStreams()) {
        int propose_token                            = stream->getSPOutputBuffer()->tokens.data_ptr<int>()[1];
        combo_tokens.data_ptr<int>()[batch_idx]      = propose_token;
        lm_output_indexes.data_ptr<int>()[batch_idx] = batch_idx;
        batch_idx++;
    }

    model_input.combo_tokens      = std::move(combo_tokens);
    model_input.lm_output_indexes = std::move(lm_output_indexes);
}

void MtpBatchStreamProcessor::prepareOneStepSpecDecodeModelInput(const StreamGroups& stream_groups,
                                                                 GptModelInputs&     model_input) {
    size_t batch_size = stream_groups.size();

    // prepare target model input buffer
    auto target_prefix_lengths = model_input.sequence_lengths.cpu().clone().pin_memory();

    // allocate target_combo_tokens shape [batch_size, propose_step_ + 1]
    auto target_combo_tokens =
        torch::empty({(int64_t)(stream_groups.size() * (propose_step_ + 1))}, torch::kInt32).pin_memory();

    // copy propose tokens to target_combo_tokens
    int batch_idx = 0;

    for (const auto& stream : stream_groups.allStreams()) {
        auto sp_output_buffer = stream->getSPOutputBuffer();
        int* propose_tokens   = sp_output_buffer->tokens.data_ptr<int>();

        // print vector string
        RTP_LLM_LOG_DEBUG("propose_tokens = [%s]", tensorDebugStringWithData<int>(sp_output_buffer->tokens).c_str());

        memcpy(target_combo_tokens.data_ptr<int>() + batch_idx * (propose_step_ + 1), propose_tokens, sizeof(int) * 2);

        batch_idx++;
    }

    // update model_input
    model_input.combo_tokens       = std::move(target_combo_tokens);
    model_input.prefix_lengths     = target_prefix_lengths;
    model_input.sequence_lengths   = torch::empty({0}, torch::kInt32).pin_memory();
    model_input.last_hidden_states = torch::Tensor();

    for (int i = 0; i < model_input.input_lengths.size(0); i++) {
        model_input.input_lengths.data_ptr<int>()[i] = propose_step_ + 1;
    }

    // set lm_output_indexes
    auto lm_output_indexes = torch::empty({(int64_t)(batch_size * (propose_step_ + 1))}, torch::kInt32).pin_memory();
    for (int i = 0; i < batch_size * (propose_step_ + 1); i++) {
        lm_output_indexes.data_ptr<int>()[i] = i;
    }
    model_input.lm_output_indexes = std::move(lm_output_indexes);
}

void MtpBatchStreamProcessor::updateDecodeDraftModelInput(GptModelInputs&        model_input,
                                                          const GptModelOutputs& model_output,
                                                          const torch::Tensor&   draft_token_ids) {
    int batch_size                 = model_input.combo_tokens.size(0);
    model_input.last_hidden_states = model_output.all_hidden_states;

    // here combo_tokens is a device buffer
    model_input.combo_tokens = draft_token_ids.reshape({batch_size});

    model_input.sequence_lengths = model_input.sequence_lengths.cpu().clone().pin_memory();
    for (int i = 0; i < batch_size; i++) {
        model_input.sequence_lengths.data_ptr<int>()[i]++;
    }
}

void MtpBatchStreamProcessor::updatePrefillPostDraftModelInput(GptModelInputs&        model_input,
                                                               const GptModelOutputs& model_output,
                                                               const SamplerOutput&   sampler_output) {
    model_input.last_hidden_states = model_output.all_hidden_states;
    const auto& new_all_token_ids  = sampler_output.token_ids;

    // set model_input.combo_tokens
    const size_t batch_size   = new_all_token_ids.size(0);
    const size_t token_stride = new_all_token_ids.size(1);
    // token_ids may be a CUDA tensor; move to CPU for data_ptr access.
    const torch::Tensor new_all_token_ids_cpu =
        new_all_token_ids.is_cuda() ? new_all_token_ids.cpu() : new_all_token_ids;

    int* input_lengths = model_input.input_lengths.data_ptr<int>();
    int* combo_tokens  = model_input.combo_tokens.data_ptr<int>();

    int offset = 0;
    for (int i = 0; i < batch_size; i++) {
        // should shift one token for combo_tokens
        int input_length = input_lengths[i];
        memcpy(combo_tokens + offset, combo_tokens + offset + 1, (input_length - 1) * sizeof(int));

        // set new token id
        int new_token_id = new_all_token_ids_cpu.data_ptr<int>()[i * token_stride + token_stride - 1];
        combo_tokens[offset + input_length - 1] = new_token_id;

        offset += input_length;
    }
}

void MtpBatchStreamProcessor::updateDecodePostDraftModelInput(
    GptModelInputs&                              model_input,
    const GptModelOutputs&                       model_output,
    const speculative::SpeculativeSamplerOutput& speculative_sampler_output,
    const size_t                                 batch_size,
    torch::Tensor&                               hidden_states_d_t,
    size_t&                                      total_accept_len) {
    auto& accept_lens = speculative_sampler_output.accept_len;
    total_accept_len  = std::accumulate(accept_lens.begin(), accept_lens.end(), 0);

    model_input.combo_tokens = torch::empty({(int64_t)total_accept_len}, torch::kInt32).pin_memory();

    int  token_offset      = 0;
    auto lm_output_indexes = torch::empty({(int64_t)batch_size}, torch::kInt32).pin_memory();

    std::vector<torch::Tensor> hidden_states_list;
    for (int i = 0; i < batch_size; i++) {
        RTP_LLM_CHECK_WITH_INFO(accept_lens[i] == (size_t)speculative_sampler_output.accept_tokens[i].numel(),
                                "accept_lens[%d] = %d, speculative_sampler_output.accept_tokens[%d].numel() = %d",
                                i,
                                accept_lens[i],
                                i,
                                speculative_sampler_output.accept_tokens[i].numel());

        memcpy(model_input.combo_tokens.data_ptr<int>() + token_offset,
               speculative_sampler_output.accept_tokens[i].data_ptr<int>(),
               accept_lens[i] * sizeof(int));

        auto hidden_slice = model_output.all_hidden_states.narrow(0, i * (propose_step_ + 1), accept_lens[i]);
        hidden_states_list.push_back(hidden_slice);

        model_input.input_lengths.data_ptr<int>()[i] = accept_lens[i];
        token_offset += accept_lens[i];
        lm_output_indexes.data_ptr<int>()[i] = token_offset - 1;
    }

    hidden_states_d_t              = torch::cat(hidden_states_list).contiguous();
    model_input.last_hidden_states = hidden_states_d_t;
    model_input.lm_output_indexes  = std::move(lm_output_indexes);
}

void MtpBatchStreamProcessor::updateOneStepDraftSamplerOutput(const StreamGroups& stream_groups,
                                                              SamplerOutput&      draft_sampler_output,
                                                              torch::Tensor&      draft_token_probs_d_t) {
    const size_t batch_size      = stream_groups.size();
    auto         draft_token_ids = torch::empty({(int64_t)batch_size, (int64_t)propose_step_}, torch::kInt32);

    std::vector<torch::Tensor> draft_token_probs_list;
    int                        batch_idx = 0;

    for (const auto& stream : stream_groups.allStreams()) {
        auto sp_output_buffer                                      = stream->getSPOutputBuffer();
        auto propose_tokens                                        = sp_output_buffer->tokens.data_ptr<int>();
        draft_token_ids.data_ptr<int>()[batch_idx * propose_step_] = propose_tokens[1];
        draft_token_probs_list.push_back(sp_output_buffer->all_probs);
        batch_idx++;
    }

    draft_token_probs_d_t          = torch::stack(draft_token_probs_list, 0).contiguous();
    draft_sampler_output.all_probs = draft_token_probs_d_t;
    draft_sampler_output.token_ids = std::move(draft_token_ids);
}

void MtpBatchStreamProcessor::updateMultiStepDraftSamplerOutput(const StreamGroups&         stream_groups,
                                                                SamplerOutput&              draft_sampler_output,
                                                                torch::Tensor&              draft_token_ids_d_t,
                                                                torch::Tensor&              spec_token_ids_d_t,
                                                                torch::Tensor&              draft_token_probs_d_t,
                                                                std::vector<torch::Tensor>& draft_token_probs_list) {
    std::vector<torch::Tensor> prev_draft_token_probs_list;
    for (const auto& stream : stream_groups.allStreams()) {
        auto sp_output_buffer = stream->getSPOutputBuffer();
        prev_draft_token_probs_list.push_back(sp_output_buffer->all_probs);
    }

    auto pre_draft_token_probs = torch::stack(prev_draft_token_probs_list, 0).contiguous();
    draft_token_probs_list.insert(draft_token_probs_list.begin(), pre_draft_token_probs);

    draft_token_probs_d_t          = torch::cat(draft_token_probs_list, 1).contiguous();
    draft_sampler_output.all_probs = draft_token_probs_d_t;

    // draft_token_ids_d_t = draft_token_ids_d_t[:, 1:]
    spec_token_ids_d_t             = draft_token_ids_d_t.slice(1, 1).contiguous();
    draft_sampler_output.token_ids = spec_token_ids_d_t;
}

void MtpBatchStreamProcessor::preparePrefillSpecUpdateInfo(const StreamGroups&                stream_groups,
                                                           const MergedOutput&                prefill_output,
                                                           const MergedOutput&                propose_output,
                                                           const torch::Tensor&               new_tokens_all,
                                                           std::vector<StreamSpecUpdateInfo>& spec_update_infos) const {
    const auto& sampler_output       = prefill_output.sampler_output;
    const auto& draft_sampler_output = propose_output.sampler_output;
    const auto& draft_model_output   = propose_output.model_output;

    const auto& new_all_token_ids         = sampler_output.token_ids;
    const auto& propose_new_all_token_ids = draft_sampler_output.token_ids;

    RTP_LLM_LOG_DEBUG("new_all_token_ids = [%s]", tensorDebugStringWithData<int32_t>(new_all_token_ids).c_str());
    RTP_LLM_LOG_DEBUG("propose_new_all_token_ids = [%s]",
                      tensorDebugStringWithData<int64_t>(propose_new_all_token_ids).c_str());

    const size_t total_batch_size_out = stream_groups.totalSamplerBatchSizeOut();
    RTP_LLM_CHECK(total_batch_size_out == (size_t)new_all_token_ids.size(0));
    const size_t token_stride = new_all_token_ids.size(1);

    // token_ids and success may be CUDA tensors; move to CPU once before iterating.
    const torch::Tensor new_all_token_ids_cpu =
        new_all_token_ids.is_cuda() ? new_all_token_ids.cpu() : new_all_token_ids;
    const torch::Tensor success_cpu = sampler_output.success.defined() ? sampler_output.success.cpu() : torch::Tensor();

    int batch_idx_in  = 0;
    int batch_idx_out = 0;
    int token_offset  = 0;

    for (auto& stream : stream_groups.allStreams()) {
        auto cur_batch_size  = stream->currentBatchSize();
        auto next_batch_size = stream->nextBatchSize();
        auto token_size      = stream->currentExecuteTokenSize();

        // normal stream info
        auto new_tokens = new_tokens_all.narrow(0, batch_idx_out, next_batch_size);
        for (size_t i = 0; i < next_batch_size; ++i) {
            new_tokens.data_ptr<int32_t>()[i] =
                new_all_token_ids_cpu.data_ptr<int32_t>()[(batch_idx_out + i) * token_stride + token_stride - 1];
        }

        for (int i = 0; i < cur_batch_size; ++i) {
            if (success_cpu.defined() && !(success_cpu.data_ptr<bool>()[batch_idx_in + i])) {
                stream->reportError(ErrorCode::UNKNOWN_ERROR, "sampler generate token id failed");
            }
        }

        // speculative decoding info
        torch::Tensor propose_all_probs =
            draft_sampler_output.all_probs.narrow(0, batch_idx_out, next_batch_size).to(torch::kCUDA).clone();

        torch::Tensor last_hidden_states;
        if (propose_step_ > 1) {
            last_hidden_states = draft_model_output.all_hidden_states.narrow(0, token_offset + token_size - 1, 1);
        }

        spec_update_infos.push_back({new_tokens, 1, -1, std::move(last_hidden_states), std::move(propose_all_probs)});

        batch_idx_in += cur_batch_size;
        batch_idx_out += next_batch_size;
        token_offset += token_size;
    }
}

void MtpBatchStreamProcessor::prepareDecodeSpecUpdateInfo(
    const StreamGroups&                          stream_groups,
    const speculative::SpeculativeSamplerOutput& spec_decode_output,
    const MergedOutput&                          draft_prefill_output,
    std::vector<StreamSpecUpdateInfo>&           spec_update_infos) const {
    const auto& accept_len    = spec_decode_output.accept_len;
    const auto& accept_tokens = spec_decode_output.accept_tokens;

    const auto& draft_model_output   = draft_prefill_output.model_output;
    const auto& draft_sampler_output = draft_prefill_output.sampler_output;

    int batch_idx_in  = 0;
    int batch_idx_out = 0;
    int token_offset  = 0;

    for (auto& stream : stream_groups.allStreams()) {
        auto cur_batch_size  = stream->currentBatchSize();
        auto next_batch_size = stream->nextBatchSize();

        // speculative decoding info
        torch::Tensor propose_all_probs =
            draft_sampler_output.all_probs.narrow(0, batch_idx_out, next_batch_size).to(torch::kCUDA).clone();

        torch::Tensor last_hidden_states;
        if (propose_step_ > 1) {
            auto slice_t =
                draft_model_output.all_hidden_states.narrow(0, token_offset + accept_len[batch_idx_out] - 1, 1);
            last_hidden_states = slice_t;
        }

        spec_update_infos.push_back({accept_tokens[batch_idx_out],
                                     accept_len[batch_idx_out],
                                     -1,
                                     std::move(last_hidden_states),
                                     std::move(propose_all_probs)});

        token_offset += accept_len[batch_idx_out];
        batch_idx_in += cur_batch_size;
        batch_idx_out += next_batch_size;
    }
}

void MtpBatchStreamProcessor::gatherHiddenStates(const StreamGroups& stream_groups, GptModelInputs& model_input) const {
    auto            all_streams = stream_groups.allStreams();
    c10::ScalarType dtype       = c10::ScalarType::Undefined;
    size_t          hidden_size = 0;

    size_t all_hidden_tokens_num = 0;
    for (auto& stream : all_streams) {
        auto& hidden_states = stream->getSPOutputBuffer()->hidden_states;
        RTP_LLM_CHECK(hidden_states.defined());
        RTP_LLM_CHECK(hidden_states.dim() == 2);
        if (dtype == c10::ScalarType::Undefined) {
            dtype = hidden_states.scalar_type();
        } else {
            RTP_LLM_CHECK(dtype == hidden_states.scalar_type());
        }
        if (hidden_size == 0) {
            hidden_size = hidden_states.size(1);
        } else {
            RTP_LLM_CHECK(hidden_size == (size_t)hidden_states.size(1));
        }
        all_hidden_tokens_num += hidden_states.size(0);
    }

    // copy hidden
    torch::Tensor all_hidden_states;
    if (all_streams.size() == 0) {
        model_input.last_hidden_states = torch::Tensor();
        return;
    } else if (all_streams.size() == 1) {
        all_hidden_states = all_streams.front()->getSPOutputBuffer()->hidden_states;
    } else if (all_streams.size() < 8) {
        all_hidden_states = torch::empty({(int64_t)all_hidden_tokens_num, (int64_t)hidden_size},
                                         torch::TensorOptions().dtype(dtype).device(torch::kCUDA));
        size_t index      = 0;
        for (auto& stream : all_streams) {
            auto  sp_output_buffer = stream->getSPOutputBuffer();
            auto& hidden_states    = sp_output_buffer->hidden_states;
            auto  hidden_num       = hidden_states.size(0);
            all_hidden_states.narrow(0, index, hidden_num).copy_(hidden_states);
            index += hidden_num;
        }
    } else {
        all_hidden_states = torch::empty({(int64_t)all_hidden_tokens_num, (int64_t)hidden_size},
                                         torch::TensorOptions().dtype(dtype).device(torch::kCUDA));

        MultiMergeCopyParams params;
        params.dst_ptr         = all_hidden_states.data_ptr();
        size_t accu_dst_offset = 0;
        for (auto& stream : all_streams) {
            auto   sp_output_buffer = stream->getSPOutputBuffer();
            auto&  hidden_states    = sp_output_buffer->hidden_states;
            size_t hidden_copy_size = hidden_states.nbytes();
            params.src_ptrs.push_back(hidden_states.data_ptr());
            params.copy_size.push_back(hidden_copy_size);
            params.dst_offsets.push_back(accu_dst_offset);
            accu_dst_offset += hidden_copy_size;
        }

        if (accu_dst_offset > 0) {
            execMultiMergeCopy(params);
        }
    }

    model_input.last_hidden_states = all_hidden_states;
}

}  // namespace rtp_llm