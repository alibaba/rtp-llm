#include "rtp_llm/cpp/speculative_engine/speculative_sampler/SpeculativeSampler.h"
#include "rtp_llm/cpp/devices/utils/DebugUtils.h"
#include "rtp_llm/cpp/utils/AssertUtils.h"
#include "rtp_llm/cpp/utils/StatusUtil.h"


namespace rtp_llm {


absl::StatusOr<SpeculativeSamplerOutput> SpeculativeSampler::sample(const std::list<GenerateStreamPtr>& streams) const {
    SpeculativeSamplerOutput sampler_output;
    // TODO(xyz): optimize the RejectionSampler with batch processing interface
    for (const GenerateStreamPtr& stream : streams) {
        if (stream->finishedWithoutLock() || stream->stoppedWithoutLock()) {
            continue;
        }
        // size_t stream_id = stream->streamId();
        SpeculativeExecutorStreamOutputPtr propose_stream_output = stream->getProposeStream()->getSPOutputBuffer();
        SpeculativeExecutorStreamOutputPtr scorer_stream_output  = stream->getScoreStream()->getSPOutputBuffer();
        if (propose_stream_output == nullptr || scorer_stream_output == nullptr) {
            continue;
        }

        size_t propose_step = propose_stream_output->propose_step;

        sampler_output.propose_token_num += propose_step;

        std::shared_ptr<GenerateConfig>&          stream_config         = stream->generateConfig();
        size_t                                    accepted_len          = 0;
        if (propose_step == 0) {
            accepted_len = 1;
        } else if (stream_config->top1()) {
            CHECK_AND_ASSIGN(accepted_len, top1Sample(propose_step, propose_stream_output, scorer_stream_output));
        } else {
            // TODO(xyz): catch exception for specified stream
            auto status = stochasticSample(propose_step, propose_stream_output, scorer_stream_output);
            if (status.ok()) {
                accepted_len = status.value();
            } else {
                stream->setStopWithoutLock(ErrorCode::OUT_OF_VOCAB_RANGE, "Multinomial sum deviates too much from 1.0, there maybe exist nan in model output");
                continue;
            }
        }

        RTP_LLM_LOG_DEBUG("stream [%d], topk = [%d], topp = [%f], propose_token_num = [%d], accept_token_num = [%d]",
                        stream->streamId(),
                        stream_config->top_k,
                        stream_config->top_p,
                        propose_step,
                        accepted_len);

        printBufferData(*propose_stream_output->tokens, "propose tokens");
        printBufferData(*scorer_stream_output->tokens, "verify tokens");

        sampler_output.accept_token_num += accepted_len;

        rtp_llm::BufferPtr accepted_tokens = 
            device_->allocateBuffer({rtp_llm::DataType::TYPE_INT32, {1, accepted_len}, rtp_llm::AllocationType::HOST}, {"accepted_tokens"});
        
        memcpy(accepted_tokens->data(), scorer_stream_output->tokens->data(), sizeof(int32_t) * accepted_len);

        rtp_llm::BufferPtr logits        = nullptr;
        rtp_llm::BufferPtr hidden_states = nullptr;
        rtp_llm::BufferPtr loss          = nullptr;
        rtp_llm::BufferPtr softmax_probs = nullptr;

        // TODO(xyz): optimize deepclone
        if (stream->generateConfig()->return_logits) {
            logits = device_->clone(
                {scorer_stream_output->logits->view(0, accepted_len), rtp_llm::AllocationType::HOST, {"return_logits"}});
        }

        if (stream->needReturnHiddenStates()) {
            if (stream->getContainProposeToken() || (stream->getProposeStream() != nullptr && stream->getProposeStream()->getMtpTokenIndex() > 0)) {
                hidden_states = scorer_stream_output->hidden_states->slice(0, accepted_len, false);
                hidden_states->updateParent(scorer_stream_output->hidden_states);
            } else {
                hidden_states = scorer_stream_output->hidden_states;
            }
            RTP_LLM_LOG_DEBUG("sample hidden states: %s", hidden_states->debugStringMeta().c_str());

        }
        if (scorer_stream_output->loss) {
            loss = device_->clone({*scorer_stream_output->loss, rtp_llm::AllocationType::HOST, {"return_loss"}});
        }

        if (scorer_stream_output->softmax_probs) {
            softmax_probs =
                device_->allocateBuffer({rtp_llm::DataType::TYPE_FP32, {1, accepted_len}, rtp_llm::AllocationType::HOST}, {"return_softmax_probs"});
            device_->copy(
                {(*softmax_probs)[0].view(0, accepted_len), (*scorer_stream_output->softmax_probs)[0].view(0, accepted_len)});
        }


        stream->step();
        StreamUpdateInfo update_info{std::move(accepted_tokens), (int)accepted_len, hidden_states, std::move(logits), std::move(softmax_probs), nullptr, nullptr, std::move(loss), hidden_states};

        stream->update(update_info);
        stream->setReuseLength(stream->seqLength() - 1);
        stream->setFallbackPrefixLength(stream->reuseLength());
        stream->setAccepedBounsToken(accepted_len > propose_step);
        stream->incSpEditSearchIndex(accepted_len - 1);
        stream->setSpEditRun(false);
    }
    RTP_LLM_LOG_DEBUG("speculative sample done");
    return sampler_output;
}

absl::StatusOr<size_t> SpeculativeSampler::top1Sample(size_t                                    propose_step,
                                    const SpeculativeExecutorStreamOutputPtr& propose_stream_output,
                                    const SpeculativeExecutorStreamOutputPtr& scorer_stream_output) const {
    size_t accepted_len = 0;
    while (accepted_len < propose_step) {
        if ((*propose_stream_output->tokens->dataWithOffset<int32_t>(accepted_len))
            != (*scorer_stream_output->tokens->dataWithOffset<int32_t>(accepted_len))) {
            break;
        }
        accepted_len++;
    }
    return accepted_len + 1;
}

absl::StatusOr<size_t> SpeculativeSampler::stochasticSample(size_t                                    propose_step,
                                            const SpeculativeExecutorStreamOutputPtr& propose_stream_output,
                                            const SpeculativeExecutorStreamOutputPtr& scorer_stream_output) const {
    torch::Tensor score_all_probs   = Buffer2torchTensor(scorer_stream_output->all_probs, false);
    size_t score_vocab_size   = score_all_probs.size(1);

    if (!propose_stream_output->all_probs) {
        auto all_probs = device_->allocateBuffer(
            {rtp_llm::DataType::TYPE_FP32, {propose_step, score_vocab_size}, rtp_llm::AllocationType::HOST}, {""});
        device_->bufMemset(*all_probs, 0);
        for (size_t i = 0; i < propose_step; i++) {
            *(all_probs->view(i, 1).dataWithOffset<float>(*propose_stream_output->tokens->dataWithOffset<int32_t>(i))) = 1.0;
        }
        propose_stream_output->all_probs = device_->clone({*all_probs, rtp_llm::AllocationType::DEVICE, {"all_probs"}});
    }


    torch::Tensor propose_all_probs = Buffer2torchTensor(propose_stream_output->all_probs, false);

    size_t propose_vocab_size = propose_all_probs.size(1);

    if (propose_vocab_size > score_vocab_size) {
        propose_all_probs = propose_all_probs.narrow(1, 0, score_vocab_size);
    } else if (propose_vocab_size < score_vocab_size) {
        long padding_size  = score_vocab_size - propose_vocab_size;
        auto zeros_padding = torch::zeros({propose_all_probs.size(0), padding_size}, propose_all_probs.options());
        propose_all_probs  = torch::cat({propose_all_probs, zeros_padding}, 1);
    }

    torch::Device host_device = torch::Device(torch::kCPU);
    torch::Device target_device = device_->getTorchDevice();

    torch::Tensor randoms      = torch::rand({(long)propose_step}, torch::Device(host_device)).to(torch::kFloat);
    size_t        accepted_len = 0;
    torch::Tensor row_indices  = torch::arange((long)propose_step, torch::Device(target_device)).to(torch::kInt32);
    torch::Tensor col_indices  = torch::from_blob(propose_stream_output->tokens->dataWithOffset<int32_t>(0),
                                                    {(long)propose_step},
                                                    torch::kInt32)
                                    .to(torch::Device(target_device));

    torch::Tensor score_probs   = score_all_probs.index({row_indices, col_indices}).to(torch::Device(host_device));
    torch::Tensor propose_probs = propose_all_probs.index({row_indices, col_indices}).to(torch::Device(host_device));
    propose_probs = propose_probs.maximum(torch::full_like(propose_probs, 1e-7).to(torch::Device(host_device)));
    torch::Tensor div_probs     = score_probs.div(propose_probs);

    while (accepted_len < propose_step) {
        int32_t propose_token_id = *propose_stream_output->tokens->dataWithOffset<int32_t>(accepted_len);
        if (randoms[accepted_len].greater(div_probs[accepted_len]).item<bool>()) {
            auto new_p = score_all_probs[accepted_len]
                                .subtract(propose_all_probs[accepted_len])
                                .maximum(torch::zeros_like(score_all_probs[accepted_len], torch::Device(target_device)));
            auto norm_p                                                          = new_p.div(new_p.sum(0));
            const float threshold = 0.01f;
            auto check_sum = norm_p.sum(0).item<float>();
            if (std::isnan(check_sum) || std::isinf(check_sum) || (std::fabs(check_sum - 1.0f) > threshold)) {
                return absl::StatusOr<float>(absl::InvalidArgumentError("Multinomial sum deviates too much from 1.0, there maybe exist nan in model output"));
            }

            auto new_token_tensor                                                = norm_p.multinomial(1);
            *scorer_stream_output->tokens->dataWithOffset<int32_t>(accepted_len) = new_token_tensor.item<int32_t>();
            break;
        }
        *scorer_stream_output->tokens->dataWithOffset<int32_t>(accepted_len) = propose_token_id;
        accepted_len++;
    }
    return accepted_len + 1;
}

};