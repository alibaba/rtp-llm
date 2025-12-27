#include "rtp_llm/cpp/speculative_engine/speculative_sampler/SpeculativeSampler.h"
#include "rtp_llm/cpp/devices/utils/DebugUtils.h"
#include "rtp_llm/cpp/utils/AssertUtils.h"
#include "rtp_llm/cpp/utils/StatusUtil.h"

namespace rtp_llm {

absl::StatusOr<SpeculativeSamplerOutput> SpeculativeSampler::sample(const std::list<GenerateStreamPtr>& streams) const {
    bool                     contain_topp        = false;
    bool                     force_stream_sample = device_->initParams().sp_config.force_stream_sample;
    torch::Device            target_device       = device_->getTorchDevice();
    SpeculativeSamplerOutput sample_output;

    for (const GenerateStreamPtr& stream : streams) {
        contain_topp |= !stream->generateConfig()->top1();
    }
    if (!contain_topp || force_stream_sample || target_device != torch::Device(torch::kCUDA)) {
        streamSample(sample_output, streams);
    } else {
        batchSample(sample_output, streams);
    }
    return sample_output;
}

void SpeculativeSampler::updateSampleStream(SpeculativeExecutorStreamOutputPtr& propose_stream_output,
                                            SpeculativeExecutorStreamOutputPtr& score_stream_output,
                                            size_t                              propose_step,
                                            size_t                              accept_len,
                                            BufferPtr&                          accept_tokens,
                                            const GenerateStreamPtr&            stream) const {

    std::shared_ptr<GenerateConfig>& stream_config = stream->generateConfig();

    RTP_LLM_LOG_DEBUG("stream [%ld], topk = [%d], topp = [%f], propose_token_num = [%d], accept_token_num = [%d]",
                      stream->streamId(),
                      stream_config->top_k,
                      stream_config->top_p,
                      propose_step,
                      accept_len);

    if (propose_stream_output->tokens) {
        printBufferData(*propose_stream_output->tokens, "propose tokens");
    }

    if (score_stream_output->tokens) {
        printBufferData(*score_stream_output->tokens, "verify tokens");
    }

    if (accept_tokens) {
        printBufferData(*accept_tokens, "accept_tokens");
    }

    rtp_llm::BufferPtr logits        = nullptr;
    rtp_llm::BufferPtr hidden_states = nullptr;
    rtp_llm::BufferPtr loss          = nullptr;
    rtp_llm::BufferPtr softmax_probs = nullptr;

    // TODO(xyz): optimize deepclone
    if (stream->generateConfig()->return_logits) {
        logits = device_->clone(
            {score_stream_output->logits->view(0, accept_len), rtp_llm::AllocationType::HOST, {"return_logits"}});
    }

    if (stream->needReturnHiddenStates()) {
        if (stream->getContainProposeToken() || (stream->getProposeStream() != nullptr && stream->spIterCount() > 0)) {
            hidden_states = score_stream_output->hidden_states->slice(0, accept_len, false);
            hidden_states->updateParent(score_stream_output->hidden_states);
        } else {
            hidden_states = score_stream_output->hidden_states;
        }
        RTP_LLM_LOG_DEBUG("sample hidden states: %s", hidden_states->debugStringMeta().c_str());
    }
    if (score_stream_output->loss) {
        loss = device_->clone({*score_stream_output->loss, rtp_llm::AllocationType::HOST, {"return_loss"}});
    }

    if (score_stream_output->softmax_probs) {
        softmax_probs = device_->allocateBuffer(
            {rtp_llm::DataType::TYPE_FP32, {1, accept_len}, rtp_llm::AllocationType::HOST}, {"return_softmax_probs"});
        device_->copy(
            {(*softmax_probs)[0].view(0, accept_len), (*score_stream_output->softmax_probs)[0].view(0, accept_len)});
    }

    stream->step();
    stream->spStep();

    if (stream->isPerfTest()) {
        device_->bufMemset(*accept_tokens, 0);
    }

    StreamUpdateInfo update_info{std::move(accept_tokens),
                                 (int)accept_len,
                                 hidden_states,
                                 std::move(logits),
                                 std::move(softmax_probs),
                                 nullptr,
                                 nullptr,
                                 std::move(loss),
                                 nullptr,
                                 hidden_states};

    stream->update(update_info);
    stream->setReuseLength(stream->seqLength() - 1);
    stream->setAccepedBounsToken(accept_len > propose_step);
    stream->incSpEditSearchIndex(accept_len - 1);
    stream->setSpEditRun(false);
}

void SpeculativeSampler::streamSample(SpeculativeSamplerOutput&           sample_output,
                                      const std::list<GenerateStreamPtr>& streams) const {
    RTP_LLM_LOG_DEBUG("stream rejection sample");
    for (const GenerateStreamPtr& stream : streams) {
        if (stream->finishedWithoutLock() || stream->stoppedWithoutLock()) {
            continue;
        }
        // size_t stream_id = stream->streamId();
        SpeculativeExecutorStreamOutputPtr propose_stream_output = stream->getProposeStream()->getSPOutputBuffer();
        SpeculativeExecutorStreamOutputPtr score_stream_output   = stream->getScoreStream()->getSPOutputBuffer();
        if (propose_stream_output == nullptr || score_stream_output == nullptr) {
            continue;
        }

        size_t                           propose_step  = propose_stream_output->propose_step;
        std::shared_ptr<GenerateConfig>& stream_config = stream->generateConfig();
        size_t                           accept_len    = 0;

        if (propose_step == 0) {
            accept_len = 1;
        } else if (stream_config->top1() || propose_stream_output->all_probs == nullptr) {
            accept_len =
                top1Sample(propose_step, propose_stream_output, score_stream_output, stream->forceSpAccept()).value();
        } else {
            // TODO(xyz): catch exception for specified stream
            auto status =
                stochasticSample(propose_step, propose_stream_output, score_stream_output, stream->forceSpAccept());
            if (status.ok()) {
                accept_len = status.value();
            } else {
                stream->setStopWithoutLock(
                    ErrorCode::OUT_OF_VOCAB_RANGE,
                    "Multinomial sum deviates too much from 1.0, there maybe exist nan in model output");
                continue;
            }
        }

        if (!stream->isFakeStream() && propose_step != 0) {
            sample_output.propose_token_num += propose_step;
            sample_output.accept_token_num += accept_len;
            sample_output.stream_num++;
        }

        rtp_llm::BufferPtr accept_tokens = device_->allocateBuffer(
            {rtp_llm::DataType::TYPE_INT32, {1, accept_len}, rtp_llm::AllocationType::HOST}, {"accept_tokens"});

        memcpy(accept_tokens->data(), score_stream_output->tokens->data(), sizeof(int32_t) * accept_len);

        updateSampleStream(propose_stream_output, score_stream_output, propose_step, accept_len, accept_tokens, stream);
    }
}

void SpeculativeSampler::batchSample(SpeculativeSamplerOutput&           sample_output,
                                     const std::list<GenerateStreamPtr>& streams) const {
    RTP_LLM_LOG_DEBUG("batch rejection sample");
    std::list<GenerateStreamPtr> sample_streams;

    std::list<GenerateStreamPtr> top1_sample_streams;
    for (const GenerateStreamPtr& stream : streams) {
        if (stream->finishedWithoutLock() || stream->stoppedWithoutLock()) {
            continue;
        }

        SpeculativeExecutorStreamOutputPtr propose_stream_output = stream->getProposeStream()->getSPOutputBuffer();
        SpeculativeExecutorStreamOutputPtr score_stream_output   = stream->getScoreStream()->getSPOutputBuffer();

        if (propose_stream_output == nullptr || score_stream_output == nullptr) {
            continue;
        }
        if (propose_stream_output->propose_step == 0) {
            size_t accept_len = 1;

            rtp_llm::BufferPtr accept_tokens = device_->allocateBuffer(
                {rtp_llm::DataType::TYPE_INT32, {1, accept_len}, rtp_llm::AllocationType::HOST}, {"accept_tokens"});

            memcpy(accept_tokens->data(), score_stream_output->tokens->data(), sizeof(int32_t) * accept_len);

            updateSampleStream(propose_stream_output, score_stream_output, 0, accept_len, accept_tokens, stream);
            continue;
        }

        if (stream->generateConfig()->top1() == true
            || (propose_stream_output->propose_step > 0 && !propose_stream_output->all_probs)) {
            top1_sample_streams.push_back(stream);
        } else {
            sample_streams.push_back(stream);
        }
    }

    streamSample(sample_output, top1_sample_streams);

    if (sample_streams.empty()) {
        return;
    }

    // batch sample
    size_t num_speculate_tokens      = 0;
    size_t batch_size                = sample_streams.size();
    bool   fallback_to_stream_sample = false;

    std::vector<torch::Tensor> draft_probs_buffer_list;
    std::vector<torch::Tensor> draft_token_ids_buffer_list;
    std::vector<torch::Tensor> target_probs_buffer_list;
    for (const GenerateStreamPtr& stream : sample_streams) {
        SpeculativeExecutorStreamOutputPtr propose_stream_output = stream->getProposeStream()->getSPOutputBuffer();
        SpeculativeExecutorStreamOutputPtr score_stream_output   = stream->getScoreStream()->getSPOutputBuffer();
        size_t                             propose_step          = propose_stream_output->propose_step;

        if (num_speculate_tokens == 0) {
            num_speculate_tokens = propose_step;
        } else if (num_speculate_tokens != propose_step) {
            RTP_LLM_LOG_DEBUG("fallback to propose step since there is no same propose step %d %d",
                              num_speculate_tokens,
                              propose_step);
            fallback_to_stream_sample = true;
            break;
        }

        draft_probs_buffer_list.push_back(Buffer2torchTensor(propose_stream_output->all_probs, false));
        draft_token_ids_buffer_list.push_back(Buffer2torchTensor(propose_stream_output->tokens, false));
        target_probs_buffer_list.push_back(Buffer2torchTensor(score_stream_output->all_probs, false));
    }

    if (fallback_to_stream_sample) {
        return streamSample(sample_output, sample_streams);
    }

    torch::Device host_device   = torch::Device(torch::kCPU);
    torch::Device target_device = device_->getTorchDevice();

    torch::Tensor draft_probs_d     = torch::stack(draft_probs_buffer_list, 0).contiguous();
    torch::Tensor draft_token_ids_d = torch::cat(draft_token_ids_buffer_list, 0).contiguous().to(target_device);

    torch::Tensor target_probs_d     = torch::stack(target_probs_buffer_list, 0).contiguous();
    torch::Tensor uniform_samples_d  = torch::rand({(long)batch_size, (long)num_speculate_tokens + 1},
                                                  torch::TensorOptions().device(target_device).dtype(torch::kFloat));
    torch::Tensor output_token_ids_d = torch::zeros({(long)batch_size, (long)num_speculate_tokens + 1},
                                                    torch::TensorOptions().device(target_device).dtype(torch::kInt32));
    torch::Tensor output_accepted_token_num_d =
        torch::zeros({(long)batch_size}, torch::TensorOptions().device(target_device).dtype(torch::kInt32));
    torch::Tensor output_emitted_token_num_d =
        torch::zeros({(long)batch_size}, torch::TensorOptions().device(target_device).dtype(torch::kInt32));

    size_t propose_vocab_size = draft_probs_d.size(2);
    size_t score_vocab_size   = target_probs_d.size(2);

    RTP_LLM_LOG_DEBUG("propose_vocab_size = %d, score_vocab_size = %d", propose_vocab_size, score_vocab_size);

    if (propose_vocab_size > score_vocab_size) {
        draft_probs_d = draft_probs_d.narrow(2, 0, score_vocab_size);
    } else if (propose_vocab_size < score_vocab_size) {
        long padding_size = score_vocab_size - propose_vocab_size;
        auto zeros_padding =
            torch::zeros({(long)batch_size, draft_probs_d.size(1), padding_size}, draft_probs_d.options());
        draft_probs_d = torch::cat({draft_probs_d, zeros_padding}, 2);
    }

    device_->chainSpeculativeSampling({draft_probs_d,
                                       draft_token_ids_d,
                                       uniform_samples_d,
                                       target_probs_d,
                                       output_token_ids_d,
                                       output_accepted_token_num_d,
                                       output_emitted_token_num_d});

    torch::Tensor output_token_ids_h         = output_token_ids_d.to(host_device).contiguous();
    torch::Tensor output_emitted_token_num_h = output_emitted_token_num_d.to(host_device).contiguous();

    size_t i = 0;
    for (const GenerateStreamPtr& stream : sample_streams) {
        SpeculativeExecutorStreamOutputPtr propose_stream_output = stream->getProposeStream()->getSPOutputBuffer();
        SpeculativeExecutorStreamOutputPtr score_stream_output   = stream->getScoreStream()->getSPOutputBuffer();
        size_t                             propose_step          = num_speculate_tokens;
        size_t                             accept_len            = output_emitted_token_num_h[i].item<int32_t>();

        if (!stream->isFakeStream()) {
            sample_output.propose_token_num += propose_step;
            sample_output.accept_token_num += accept_len;
            sample_output.stream_num++;
        }

        if (stream->forceSpAccept()) {
            accept_len = propose_step + 1;
        }

        rtp_llm::BufferPtr accept_tokens = device_->allocateBuffer(
            {rtp_llm::DataType::TYPE_INT32, {1, accept_len}, rtp_llm::AllocationType::HOST}, {"accept_tokens"});

        memcpy(accept_tokens->data(), output_token_ids_h[i].data_ptr<int32_t>(), sizeof(int32_t) * accept_len);

        if (accept_len == num_speculate_tokens + 1) {
            *accept_tokens->dataWithOffset<int32_t>(accept_len - 1) =
                *score_stream_output->tokens->dataWithOffset<int32_t>(accept_len - 1);
        }

        updateSampleStream(propose_stream_output, score_stream_output, propose_step, accept_len, accept_tokens, stream);
        i++;
    }
}

absl::StatusOr<size_t> SpeculativeSampler::top1Sample(size_t                                    propose_step,
                                                      const SpeculativeExecutorStreamOutputPtr& propose_stream_output,
                                                      const SpeculativeExecutorStreamOutputPtr& score_stream_output,
                                                      bool                                      force_accept) const {
    size_t accept_len = 0;
    while (accept_len < propose_step) {
        if (force_accept) {
            int32_t propose_token_id = *propose_stream_output->tokens->dataWithOffset<int32_t>(accept_len);
            *score_stream_output->tokens->dataWithOffset<int32_t>(accept_len) = propose_token_id;
        } else if ((*propose_stream_output->tokens->dataWithOffset<int32_t>(accept_len))
                   != (*score_stream_output->tokens->dataWithOffset<int32_t>(accept_len))) {
            break;
        }
        accept_len++;
    }
    return accept_len + 1;
}

absl::StatusOr<size_t>
SpeculativeSampler::stochasticSample(size_t                                    propose_step,
                                     const SpeculativeExecutorStreamOutputPtr& propose_stream_output,
                                     const SpeculativeExecutorStreamOutputPtr& score_stream_output,
                                     bool                                      force_accept) const {

    torch::Tensor score_all_probs    = Buffer2torchTensor(score_stream_output->all_probs, false);
    size_t        score_vocab_size   = score_all_probs.size(1);
    torch::Tensor propose_all_probs  = Buffer2torchTensor(propose_stream_output->all_probs, false);
    size_t        propose_vocab_size = propose_all_probs.size(1);

    if (propose_vocab_size > score_vocab_size) {
        propose_all_probs = propose_all_probs.narrow(1, 0, score_vocab_size);
    } else if (propose_vocab_size < score_vocab_size) {
        long padding_size  = score_vocab_size - propose_vocab_size;
        auto zeros_padding = torch::zeros({propose_all_probs.size(0), padding_size}, propose_all_probs.options());
        propose_all_probs  = torch::cat({propose_all_probs, zeros_padding}, 1);
    }

    torch::Device host_device   = torch::Device(torch::kCPU);
    torch::Device target_device = device_->getTorchDevice();

    torch::Tensor randoms     = torch::rand({(long)propose_step}, torch::Device(host_device)).to(torch::kFloat);
    size_t        accept_len  = 0;
    torch::Tensor row_indices = torch::arange((long)propose_step, torch::Device(target_device)).to(torch::kInt32);
    torch::Tensor col_indices =
        torch::from_blob(propose_stream_output->tokens->dataWithOffset<int32_t>(0), {(long)propose_step}, torch::kInt32)
            .to(torch::Device(target_device));

    torch::Tensor score_probs   = score_all_probs.index({row_indices, col_indices}).to(torch::Device(host_device));
    torch::Tensor propose_probs = propose_all_probs.index({row_indices, col_indices}).to(torch::Device(host_device));
    propose_probs = propose_probs.maximum(torch::full_like(propose_probs, 1e-7).to(torch::Device(host_device)));
    torch::Tensor div_probs = score_probs.div(propose_probs);

    while (accept_len < propose_step) {
        int32_t propose_token_id = *propose_stream_output->tokens->dataWithOffset<int32_t>(accept_len);
        if (randoms[accept_len].greater(div_probs[accept_len]).item<bool>() && !force_accept) {
            auto new_p = score_all_probs[accept_len]
                             .subtract(propose_all_probs[accept_len])
                             .maximum(torch::zeros_like(score_all_probs[accept_len], torch::Device(target_device)));
            auto        norm_p    = new_p.div(new_p.sum(0));
            const float threshold = 0.01f;
            auto        check_sum = norm_p.sum(0).item<float>();
            if (std::isnan(check_sum) || std::isinf(check_sum) || (std::fabs(check_sum - 1.0f) > threshold)) {
                return absl::StatusOr<float>(absl::InvalidArgumentError(
                    "Multinomial sum deviates too much from 1.0, there maybe exist nan in model output"));
            }

            auto new_token_tensor                                             = norm_p.multinomial(1);
            *score_stream_output->tokens->dataWithOffset<int32_t>(accept_len) = new_token_tensor.item<int32_t>();
            break;
        }
        *score_stream_output->tokens->dataWithOffset<int32_t>(accept_len) = propose_token_id;
        accept_len++;
    }
    return accept_len + 1;
}

};  // namespace rtp_llm