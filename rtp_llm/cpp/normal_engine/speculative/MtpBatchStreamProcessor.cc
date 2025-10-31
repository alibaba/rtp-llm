#include "rtp_llm/cpp/normal_engine/speculative/MtpBatchStreamProcessor.h"

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

        dispatchProposePrefillSingleStream(
            stream, propose_output, batch_idx_in, batch_idx_out, token_offset, return_all_probs, new_tokens_all);

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
        auto token_size      = stream->currentExecuteTokenSize();

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

        dispatchProposePrefillSingleStream(
            stream, draft_prefill_output, batch_idx_in, batch_idx_out, token_offset, return_all_probs, new_tokens_all);

        batch_idx_in += cur_batch_size;
        batch_idx_out += next_batch_size;
        token_offset += token_size;
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
    auto      token_size         = stream->currentExecuteTokenSize();

    RTP_LLM_LOG_DEBUG("batch_idx_in: %d, batch_idx_out: %d, token_offset: %d, token_size: %d",
                      batch_idx_in,
                      batch_idx_out,
                      token_offset,
                      token_size);

    last_hidden_states = draft_model_output.all_hidden_states->slice(token_offset + token_size - 1, 1, false);
    last_hidden_states->updateParent(draft_model_output.all_hidden_states);

    // update speculative info
    stream->setLastHiddenStates(last_hidden_states);
    auto   sp_output_buffer = stream->getSPOutputBuffer();
    size_t propose_step     = sp_output_buffer->propose_step;

    if (propose_all_probs) {
        // lazy allocate buffer
        if (!sp_output_buffer->all_probs) {
            size_t vocab_size           = propose_all_probs->shape()[1];
            sp_output_buffer->all_probs = device_->allocateBuffer(
                {rtp_llm::DataType::TYPE_FP32, {propose_step, vocab_size}, rtp_llm::AllocationType::DEVICE},
                {"mtp_all_probs"});
        }
        device_->copy({sp_output_buffer->all_probs->view(0, 1), *propose_all_probs});
    }
}

absl::StatusOr<GptModelInputs>
MtpBatchStreamProcessor::gatherDecodeModelInput(const StreamGroups& stream_groups) const {
    auto model_input = NormalBatchStreamProcessor::gatherModelInput(stream_groups);
    RTP_LLM_CHECK(model_input.ok());
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
        all_hidden_tokens_num += stream->currentExecuteTokenSize();
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

    SamplerInputs sampler_inputs =
        allocateSamplerInputs(stream_groups, total_batch_size, total_batch_size, model_inputs.sequence_lengths);
    setCommonSamplerInputs(sampler_inputs, all_streams, true);

    int batch_idx = 0;
    for (auto& stream : all_streams) {
        const auto& complete_token_ids = stream->completeTokenIds();
        auto        seq_len            = stream->seqLength();
        auto        current_batch_size = score_len;

        for (int i = 0; i < current_batch_size; ++i) {
            memcpy(sampler_inputs.token_ids->dataWithOffset<int32_t>((batch_idx) * (sampler_inputs.step + 1)),
                   complete_token_ids->dataWithOffset<int32_t>(0),
                   (seq_len - current_batch_size + i + 1) * sizeof(int));
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

    int token_stride  = propose_token_ids_h->shape()[1];
    int batch_idx_in  = 0;
    int batch_idx_out = 0;
    for (auto& stream : stream_groups.allStreams()) {
        auto cur_batch_size  = stream->currentBatchSize();
        auto next_batch_size = stream->nextBatchSize();

        auto sp_output_buffer = stream->getSPOutputBuffer();
        int  propose_token    = propose_token_ids_h->data<int64_t>()[batch_idx_out * token_stride + token_stride - 1];
        int  target_token     = new_tokens_all->data<int32_t>()[batch_idx_out];

        *(sp_output_buffer->tokens->dataWithOffset<int>(0)) = propose_token;

        std::vector<int> propose_tokens = {target_token, propose_token};
        stream->setProposeToken(propose_tokens);

        batch_idx_in += cur_batch_size;
        batch_idx_out += next_batch_size;
    }
}
}  // namespace rtp_llm