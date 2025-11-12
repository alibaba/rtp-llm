#include "rtp_llm/cpp/normal_engine/speculative/MtpBatchStreamProcessor.h"
#include "rtp_llm/cpp/core/torch_utils/BufferTorchUtils.h"
#include "rtp_llm/cpp/utils/StringUtil.h"
#include "rtp_llm/cpp/utils/AssertUtils.h"
#include <numeric>
#include <cstring>

namespace rtp_llm {

absl::Status MtpBatchStreamProcessor::dispatchPrefill(const StreamGroups& stream_groups,
                                                      const MergedOutput& prefill_output,
                                                      const MergedOutput& propose_output) const {
    RTP_LLM_LOG_DEBUG(__PRETTY_FUNCTION__);
    const auto& sampler_output       = prefill_output.sampler_output;
    const auto& draft_sampler_output = propose_output.sampler_output;

    const auto& new_all_token_ids         = sampler_output.token_ids;
    const auto& propose_new_all_token_ids = draft_sampler_output.token_ids;

    RTP_LLM_LOG_DEBUG("new_all_token_ids = [%s]", new_all_token_ids->debugStringWithData<int32_t>().c_str());
    RTP_LLM_LOG_DEBUG("propose_new_all_token_ids = [%s]",
                      propose_new_all_token_ids->debugStringWithData<int64_t>().c_str());

    const size_t total_batch_size_out = stream_groups.totalSamplerBatchSizeOut();
    RTP_LLM_CHECK(total_batch_size_out == new_all_token_ids->shape()[0]);
    int  batch_idx_in     = 0;
    int  batch_idx_out    = 0;
    int  token_offset     = 0;
    bool return_all_probs = stream_groups.needReturnAllProbs();
    auto new_tokens_all   = CACHED_HOST_BUF(TYPE_INT32, {(size_t)total_batch_size_out, (size_t)1});

    // TODO(yinzhi): consider PD-separation for mtp
    for (auto& stream : stream_groups.allStreams()) {
        auto cur_batch_size  = stream->currentBatchSize();
        auto next_batch_size = stream->nextBatchSize();
        auto token_size      = stream->currentExecuteTokenSize();

        dispatchSingleStream(
            stream, prefill_output, batch_idx_in, batch_idx_out, token_offset, return_all_probs, new_tokens_all);

        dispatchProposePrefillSingleStream(stream,
                                           propose_output,
                                           batch_idx_in,
                                           batch_idx_out,
                                           token_offset,
                                           token_size,
                                           return_all_probs,
                                           new_tokens_all);

        stream->setSpDecodeStream();
        stream->setScoreLen(propose_step_ + 1);

        batch_idx_in += cur_batch_size;
        batch_idx_out += next_batch_size;
        token_offset += token_size;
    }

    // to avoid cuda sync, we need to set propose token in extra loop
    setProposeTokensForAllStreams(stream_groups, propose_output, new_tokens_all);

    RTP_LLM_LOG_DEBUG("dispatch done");
    return absl::OkStatus();
}

absl::Status MtpBatchStreamProcessor::dispatchDecode(const StreamGroups&                          stream_groups,
                                                     const speculative::SpeculativeSamplerOutput& spec_decode_output,
                                                     const MergedOutput& draft_prefill_output) const {
    RTP_LLM_LOG_DEBUG(__PRETTY_FUNCTION__);
    const auto& accept_len    = spec_decode_output.accept_len;
    const auto& accept_tokens = spec_decode_output.accept_tokens;

    const size_t total_batch_size_out = stream_groups.totalSamplerBatchSizeOut();

    int  batch_idx_in     = 0;
    int  batch_idx_out    = 0;
    int  token_offset     = 0;
    bool return_all_probs = stream_groups.needReturnAllProbs();
    auto new_tokens_all   = CACHED_HOST_BUF(TYPE_INT32, {(size_t)total_batch_size_out, (size_t)1});

    for (auto& stream : stream_groups.allStreams()) {
        auto cur_batch_size  = stream->currentBatchSize();
        auto next_batch_size = stream->nextBatchSize();

        new_tokens_all->data<int32_t>()[batch_idx_out] =
            accept_tokens[batch_idx_out]->data<int32_t>()[accept_len[batch_idx_out] - 1];

        // TODO(yinzhi): mtp not support extra sample control for now

        stream->update({accept_tokens[batch_idx_out],
                        accept_len[batch_idx_out],
                        nullptr,
                        nullptr,
                        nullptr,
                        nullptr,
                        nullptr,
                        nullptr,
                        nullptr,
                        nullptr});

        dispatchProposePrefillSingleStream(stream,
                                           draft_prefill_output,
                                           batch_idx_in,
                                           batch_idx_out,
                                           token_offset,
                                           accept_len[batch_idx_out],
                                           return_all_probs,
                                           new_tokens_all);

        token_offset += accept_len[batch_idx_out];
        batch_idx_in += cur_batch_size;
        batch_idx_out += next_batch_size;
    }

    // to avoid cuda sync, we need to set propose token in extra loop
    setProposeTokensForAllStreams(stream_groups, draft_prefill_output, new_tokens_all);

    return absl::OkStatus();
}

void MtpBatchStreamProcessor::dispatchProposePrefillSingleStream(GenerateStreamPtr         stream,
                                                                 const MergedOutput&       propose_output,
                                                                 int                       batch_idx_in,
                                                                 int                       batch_idx_out,
                                                                 int                       token_offset,
                                                                 int                       accept_len,
                                                                 bool                      return_all_probs,
                                                                 const rtp_llm::BufferPtr& new_tokens_all) const {
    const auto& draft_model_output   = propose_output.model_output;
    const auto& draft_sampler_output = propose_output.sampler_output;

    BufferPtr propose_all_probs = nullptr;
    if (return_all_probs) {
        auto next_batch_size = stream->nextBatchSize();
        propose_all_probs    = draft_sampler_output.all_probs->slice(batch_idx_out, next_batch_size, false);
        propose_all_probs->updateParent(draft_sampler_output.all_probs);
    }

    // only update last hidden states for mtp method
    BufferPtr last_hidden_states = nullptr;

    RTP_LLM_LOG_DEBUG("batch_idx_in: %d, batch_idx_out: %d, token_offset: %d, token_size: %d",
                      batch_idx_in,
                      batch_idx_out,
                      token_offset,
                      accept_len);

    if (propose_step_ > 1) {
        last_hidden_states = draft_model_output.all_hidden_states->slice(token_offset + accept_len - 1, 1, false);
        last_hidden_states->updateParent(draft_model_output.all_hidden_states);
    }

    // update speculative info
    stream->setLastHiddenStates(last_hidden_states);
    auto sp_output_buffer = stream->getSPOutputBuffer();

    if (propose_all_probs) {
        // lazy allocate buffer
        if (!sp_output_buffer->all_probs) {
            size_t vocab_size           = propose_all_probs->shape()[1];
            sp_output_buffer->all_probs = device_->allocateBuffer(
                {rtp_llm::DataType::TYPE_FP32, {1, vocab_size}, rtp_llm::AllocationType::DEVICE}, {"mtp_all_probs"});
        }
        device_->copy({sp_output_buffer->all_probs->view(0, 1), *propose_all_probs});
    }
}

absl::StatusOr<GptModelInputs>
MtpBatchStreamProcessor::gatherDecodeModelInput(const StreamGroups& stream_groups) const {
    auto model_input = NormalBatchStreamProcessor::gatherModelInput(stream_groups);

    RTP_LLM_CHECK(model_input.ok());

    if (propose_step_ == 1) {
        return model_input;
    }

    auto              all_streams = stream_groups.allStreams();
    rtp_llm::DataType type        = rtp_llm::DataType::TYPE_INVALID;
    size_t            hidden_size = 0;

    size_t all_hidden_tokens_num = 0;
    for (auto& stream : all_streams) {
        auto hidden_states = stream->getLastHiddenStates();
        RTP_LLM_CHECK(hidden_states != nullptr);
        RTP_LLM_CHECK(hidden_states->dim() == 2);
        if (type == rtp_llm::DataType::TYPE_INVALID) {
            type = hidden_states->type();
        } else {
            // check all hidden states has same type
            RTP_LLM_CHECK(type == hidden_states->type());
        }
        if (hidden_size == 0) {
            hidden_size = hidden_states->shape()[1];
        } else {
            // check all hidden states has same shape[1]
            RTP_LLM_CHECK(hidden_size == hidden_states->shape()[1]);
        }
        all_hidden_tokens_num += hidden_states->shape()[0];
    }

    // copy hidden
    BufferPtr all_hidden_states = nullptr;
    if (all_streams.size() == 0) {
        all_hidden_states = device_->allocateBuffer({type, {0, hidden_size}, rtp_llm::AllocationType::DEVICE}, {});
    } else if (all_streams.size() == 1) {
        all_hidden_states = all_streams.front()->getLastHiddenStates();
    } else if (all_streams.size() < 8) {
        all_hidden_states =
            device_->allocateBuffer({type, {all_hidden_tokens_num, hidden_size}, rtp_llm::AllocationType::DEVICE}, {});
        size_t index = 0;
        for (auto& stream : all_streams) {
            auto hidden_states = stream->getLastHiddenStates();
            auto hidden_num    = hidden_states->shape()[0];
            device_->copy({all_hidden_states->view(index, hidden_num), *hidden_states});
            index += hidden_num;
        }
    } else {
        all_hidden_states =
            device_->allocateBuffer({type, {all_hidden_tokens_num, hidden_size}, rtp_llm::AllocationType::DEVICE}, {});

        MultiMergeCopyParams params;
        params.dst_ptr         = all_hidden_states->data();
        size_t accu_dst_offset = 0;
        for (auto& stream : all_streams) {
            BufferPtr hidden_states    = stream->getLastHiddenStates();
            size_t    hidden_copy_size = hidden_states->sizeBytes();
            params.src_ptrs.push_back(hidden_states->data());
            params.copy_size.push_back(hidden_copy_size);
            params.dst_offsets.push_back(accu_dst_offset);
            accu_dst_offset += hidden_copy_size;
        }

        if (accu_dst_offset > 0) {
            device_->multiMergeCopy(params);
        }
    }

    model_input.value().last_hidden_states = all_hidden_states;
    return model_input;
}

absl::StatusOr<SamplerInputs> MtpBatchStreamProcessor::gatherSpecSamplerInput(
    const StreamGroups& stream_groups, const GptModelInputs& model_inputs, const GptModelOutputs& model_output) const {
    RTP_LLM_CHECK(!stream_groups.empty());
    auto all_streams      = stream_groups.allStreams();
    bool return_all_probs = stream_groups.needReturnAllProbs();

    for (auto& stream : all_streams) {
        RTP_LLM_CHECK_WITH_INFO(stream->maxBatchSize() == 1, "stream tile num must be 1 in ScoreExecutor");
    }

    size_t score_len        = propose_step_ + 1;
    size_t total_batch_size = stream_groups.size() * score_len;

    SamplerInputs sampler_inputs = allocateSamplerInputs(
        stream_groups, total_batch_size, total_batch_size, model_inputs.sequence_lengths, propose_step_);
    setCommonSamplerInputs(sampler_inputs, all_streams, true, propose_step_);

    int batch_idx = 0;
    for (auto& stream : all_streams) {
        const auto& complete_token_ids = stream->completeTokenIds();
        auto        seq_len            = stream->seqLength();
        auto        current_batch_size = score_len;

        for (int i = 0; i < current_batch_size; ++i) {
            memcpy(sampler_inputs.token_ids->dataWithOffset<int32_t>((batch_idx) * (sampler_inputs.step + 1)),
                   complete_token_ids->dataWithOffset<int32_t>(0),
                   (seq_len + 1) * sizeof(int));
            batch_idx += 1;
        }

        RTP_LLM_LOG_DEBUG("stream [%ld], complete token ids = [%s]",
                          stream->streamId(),
                          complete_token_ids->debugStringWithData<int32_t>(sampler_inputs.step).c_str());
        RTP_LLM_LOG_DEBUG("stream [%ld], sampler inputs token ids = [%s]",
                          stream->streamId(),
                          sampler_inputs.token_ids->debugStringWithData<int32_t>().c_str());
    }

    auto vocab_size       = model_output.logits->shape()[1];
    sampler_inputs.logits = device_->allocateBuffer(
        {model_output.logits->type(), {total_batch_size, vocab_size}, rtp_llm::AllocationType::DEVICE}, {});
    if (return_all_probs) {
        sampler_inputs.all_probs = device_->allocateBuffer(
            {rtp_llm::DataType::TYPE_FP32, {total_batch_size, vocab_size}, rtp_llm::AllocationType::DEVICE}, {});
        device_->bufMemset(*sampler_inputs.all_probs, 0);
    }

    device_->copy({*sampler_inputs.logits, *model_output.logits});

    RTP_LLM_LOG_DEBUG("sampler inputs logits [%s]",
                      device_->clone({*sampler_inputs.logits, rtp_llm::AllocationType::HOST})
                          ->debugStringWithData<float>(10)
                          .c_str());

    RTP_LLM_LOG_DEBUG("gatherSamplerInput done");
    return std::move(sampler_inputs);
}

void MtpBatchStreamProcessor::setProposeTokensForAllStreams(const StreamGroups&      stream_groups,
                                                            const MergedOutput&      draft_prefill_output,
                                                            const rtp_llm::BufferPtr new_tokens_all) const {
    const auto propose_token_ids_h =
        device_->clone({*draft_prefill_output.sampler_output.token_ids, AllocationType::HOST});

    int  token_stride  = propose_token_ids_h->shape()[1];
    int  batch_idx_in  = 0;
    int  batch_idx_out = 0;
    auto dtype         = propose_token_ids_h->type();

    for (auto& stream : stream_groups.allStreams()) {
        auto cur_batch_size   = stream->currentBatchSize();
        auto next_batch_size  = stream->nextBatchSize();
        auto sp_output_buffer = stream->getSPOutputBuffer();

        int propose_token = -1;
        if (dtype == DataType::TYPE_INT64) {
            propose_token = propose_token_ids_h->data<int64_t>()[batch_idx_out * token_stride + token_stride - 1];
        } else {
            propose_token = propose_token_ids_h->data<int32_t>()[batch_idx_out * token_stride + token_stride - 1];
        }

        int target_token = new_tokens_all->data<int32_t>()[batch_idx_out];

        *(sp_output_buffer->tokens->dataWithOffset<int>(0)) = propose_token;

        std::vector<int> propose_tokens = {target_token, propose_token};
        stream->setProposeToken(propose_tokens);

        batch_idx_in += cur_batch_size;
        batch_idx_out += next_batch_size;
    }
}

void MtpBatchStreamProcessor::prepareDecodeDraftModelInput(const StreamGroups& stream_groups,
                                                           GptModelInputs&     model_input) {
    size_t batch_size = stream_groups.size();
    int    batch_idx  = 0;

    auto combo_tokens = device_->allocateBuffer({DataType::TYPE_INT32, {batch_size}, AllocationType::HOST});
    auto lm_output_indexes =
        device_->allocateBuffer({rtp_llm::DataType::TYPE_INT32, {batch_size}, rtp_llm::AllocationType::HOST}, {});

    for (const auto& stream : stream_groups.allStreams()) {
        int propose_token                         = stream->getProposeToken()[1];
        combo_tokens->data<int>()[batch_idx]      = propose_token;
        lm_output_indexes->data<int>()[batch_idx] = batch_idx;
        batch_idx++;
    }

    model_input.combo_tokens      = combo_tokens;
    model_input.lm_output_indexes = lm_output_indexes;
}

void MtpBatchStreamProcessor::prepareOneStepSpecDecodeModelInput(const StreamGroups& stream_groups,
                                                                 GptModelInputs&     model_input) {
    size_t batch_size = stream_groups.size();

    BufferPtr draft_token_probs = device_->allocateBuffer(
        {DataType::TYPE_FP32, {(size_t)batch_size, (size_t)propose_step_, vocab_size_}, AllocationType::DEVICE});

    device_->bufMemset(*draft_token_probs, 0);

    // prepare target model input buffer
    auto target_prefix_lengths = device_->clone({*model_input.sequence_lengths, AllocationType::HOST});

    // allocate target_combo_tokens shape [batch_size, propose_step_ + 1]
    auto target_combo_tokens = device_->allocateBuffer(
        {DataType::TYPE_INT32, {(size_t)stream_groups.size() * (propose_step_ + 1)}, AllocationType::HOST});

    // copy propose tokens to target_combo_tokens
    int batch_idx = 0;

    for (const auto& stream : stream_groups.allStreams()) {
        auto& propose_tokens   = stream->getProposeToken();
        auto  sp_output_buffer = stream->getSPOutputBuffer();
        // print vector string
        RTP_LLM_LOG_DEBUG("propose_tokens = [%s]", vectorToString(propose_tokens).c_str());

        memcpy(target_combo_tokens->dataWithOffset<int>(batch_idx * (propose_step_ + 1)),
               propose_tokens.data(),
               sizeof(int) * propose_tokens.size());

        batch_idx++;
    }

    // update model_input
    model_input.combo_tokens       = target_combo_tokens;
    model_input.prefix_lengths     = target_prefix_lengths;
    model_input.sequence_lengths   = device_->allocateBuffer({DataType::TYPE_INT32, {0}, AllocationType::HOST});
    model_input.last_hidden_states = nullptr;

    for (int i = 0; i < model_input.input_lengths->shape()[0]; i++) {
        model_input.input_lengths->data<int>()[i] = propose_step_ + 1;
    }

    // set lm_output_indexes
    auto lm_output_indexes = device_->allocateBuffer(
        {rtp_llm::DataType::TYPE_INT32, {batch_size * (propose_step_ + 1)}, rtp_llm::AllocationType::HOST}, {});
    for (int i = 0; i < batch_size * (propose_step_ + 1); i++) {
        lm_output_indexes->data<int>()[i] = i;
    }
    model_input.lm_output_indexes = lm_output_indexes;
}

void MtpBatchStreamProcessor::updateDecodeDraftModelInput(GptModelInputs&        model_input,
                                                          const GptModelOutputs& model_output,
                                                          const torch::Tensor&   draft_token_ids) {
    int batch_size                 = model_input.combo_tokens->shape()[0];
    model_input.last_hidden_states = model_output.all_hidden_states;
    model_input.combo_tokens       = torchTensor2Buffer(draft_token_ids.reshape({batch_size}));

    for (int i = 0; i < batch_size; i++) {
        model_input.sequence_lengths->data<int>()[i]++;
    }
}

void MtpBatchStreamProcessor::updatePrefillPostDraftModelInput(GptModelInputs&        model_input,
                                                               const GptModelOutputs& model_output,
                                                               const SamplerOutput&   sampler_output) {
    model_input.last_hidden_states = model_output.all_hidden_states;
    const auto& new_all_token_ids  = sampler_output.token_ids;

    // set model_input.combo_tokens
    const size_t batch_size   = new_all_token_ids->shape()[0];
    const size_t token_stride = new_all_token_ids->shape()[1];

    int* input_lengths = (int*)model_input.input_lengths->data();
    int* combo_tokens  = (int*)model_input.combo_tokens->data();

    int offset = 0;
    for (int i = 0; i < batch_size; i++) {
        // should shift one token for combo_tokens
        int input_length = input_lengths[i];
        memcpy(combo_tokens + offset, combo_tokens + offset + 1, (input_length - 1) * sizeof(int));

        // set new token id
        int new_token_id                        = new_all_token_ids->data<int>()[i * token_stride + token_stride - 1];
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

    auto last_hidden_states = device_->allocateBuffer({model_output.all_hidden_states->type(),
                                                       {total_accept_len, model_output.all_hidden_states->shape()[1]},
                                                       AllocationType::DEVICE});

    model_input.combo_tokens =
        device_->allocateBuffer({DataType::TYPE_INT32, {total_accept_len}, AllocationType::HOST});

    int  token_offset = 0;
    auto lm_output_indexes =
        device_->allocateBuffer({rtp_llm::DataType::TYPE_INT32, {batch_size}, rtp_llm::AllocationType::HOST});

    std::vector<torch::Tensor> hidden_states_list;
    for (int i = 0; i < batch_size; i++) {
        RTP_LLM_CHECK_WITH_INFO(accept_lens[i] == speculative_sampler_output.accept_tokens[i]->size(),
                                "accept_lens[%d] = %d, speculative_sampler_output.accept_tokens[%d]->size() = %d",
                                i,
                                accept_lens[i],
                                i,
                                speculative_sampler_output.accept_tokens[i]->size());

        memcpy(model_input.combo_tokens->dataWithOffset<int>(token_offset),
               speculative_sampler_output.accept_tokens[i]->data<int>(),
               accept_lens[i] * sizeof(int));

        auto hidden_slice = model_output.all_hidden_states->view(i * (propose_step_ + 1), accept_lens[i]);
        hidden_states_list.push_back(Buffer2torchTensor(hidden_slice, false));

        model_input.input_lengths->data<int>()[i] = accept_lens[i];
        token_offset += accept_lens[i];
        lm_output_indexes->data<int>()[i] = token_offset - 1;
    }

    hidden_states_d_t              = torch::cat(hidden_states_list).contiguous();
    model_input.last_hidden_states = torchTensor2Buffer(hidden_states_d_t);
    model_input.lm_output_indexes  = lm_output_indexes;
}

void MtpBatchStreamProcessor::updateOneStepDraftSamplerOutput(const StreamGroups& stream_groups,
                                                              SamplerOutput&      draft_sampler_output,
                                                              torch::Tensor&      draft_token_probs_d_t) {
    const size_t batch_size = stream_groups.size();
    BufferPtr    draft_token_ids =
        device_->allocateBuffer({DataType::TYPE_INT32, {batch_size, (size_t)propose_step_}, AllocationType::HOST});

    std::vector<torch::Tensor> draft_token_probs_list;
    int                        batch_idx = 0;

    for (const auto& stream : stream_groups.allStreams()) {
        auto sp_output_buffer                                   = stream->getSPOutputBuffer();
        auto propose_tokens                                     = stream->getProposeToken();
        draft_token_ids->data<int>()[batch_idx * propose_step_] = propose_tokens[1];
        draft_token_probs_list.push_back(Buffer2torchTensor(sp_output_buffer->all_probs, false));
        batch_idx++;
    }

    draft_token_probs_d_t          = torch::stack(draft_token_probs_list, 0).contiguous();
    draft_sampler_output.all_probs = torchTensor2Buffer(draft_token_probs_d_t);
    draft_sampler_output.token_ids = draft_token_ids;
}

void MtpBatchStreamProcessor::updateMultiStepDraftSamplerOutput(const StreamGroups&         stream_groups,
                                                                SamplerOutput&              draft_sampler_output,
                                                                torch::Tensor&              draft_token_ids_d_t,
                                                                torch::Tensor&              draft_token_probs_d_t,
                                                                std::vector<torch::Tensor>& draft_token_probs_list) {
    std::vector<torch::Tensor> prev_draft_token_probs_list;
    for (const auto& stream : stream_groups.allStreams()) {
        auto sp_output_buffer = stream->getSPOutputBuffer();
        prev_draft_token_probs_list.push_back(Buffer2torchTensor(sp_output_buffer->all_probs, false));
    }

    auto pre_draft_token_probs = torch::stack(prev_draft_token_probs_list, 0).contiguous();
    draft_token_probs_list.insert(draft_token_probs_list.begin(), pre_draft_token_probs);

    draft_token_probs_d_t          = torch::cat(draft_token_probs_list, 1).contiguous();
    draft_sampler_output.all_probs = torchTensor2Buffer(draft_token_probs_d_t);

    // draft_token_ids_d_t = draft_token_ids_d_t[:, 1:]
    draft_token_ids_d_t            = draft_token_ids_d_t.slice(1, 1).contiguous();
    draft_sampler_output.token_ids = torchTensor2Buffer(draft_token_ids_d_t);
}
}  // namespace rtp_llm