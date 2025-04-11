#include "maga_transformer/cpp/speculative_engine/speculative_sampler/RejectionSampler.h"
#include "maga_transformer/cpp/utils/StatusUtil.h"
#include "src/fastertransformer/core/Buffer.h"
#include "maga_transformer/cpp/utils/Logger.h"
#include <ATen/ops/zeros_like.h>
#include <c10/core/Device.h>
#include <c10/core/DeviceType.h>
#include <c10/core/ScalarType.h>
#include <cstddef>
#include <torch/csrc/autograd/generated/variable_factories.h>
#include <torch/types.h>

namespace ft = fastertransformer;
namespace rtp_llm {

absl::StatusOr<SpeculativeSamplerOutput> RejectionSampler::sample(const std::list<GenerateStreamPtr>& streams,
                                                                  const ProposeOutput&                proposer_output,
                                                                  const ScoreOutput& scorer_output) const {
    FT_LOG_DEBUG(__PRETTY_FUNCTION__);
    SpeculativeSamplerOutput sampler_output;
    FT_CHECK(proposer_output.outputs.size() == scorer_output.outputs.size());
    // TODO(xyz): optimize the RejectionSampler with batch processing interface
    for (const GenerateStreamPtr& stream : streams) {
        size_t stream_id = stream->streamId();
        const SpeculativeExecutorStreamOutputPtr& propose_stream_output = proposer_output.outputs.at(stream_id);
        const SpeculativeExecutorStreamOutputPtr& scorer_stream_output  = scorer_output.outputs.at(stream_id);
        size_t propose_step = propose_stream_output->propose_step;
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
                sampler_output.outputs.emplace_back(
                    propose_step, 0, nullptr, nullptr, nullptr, nullptr, false);
                continue;
            }
        }

        FT_LOG_DEBUG("stream [%d], topk = [%d], topp = [%f], propose_token_num = [%d], accept_token_num = [%d]",
                     stream->streamId(),
                     stream_config->top_k,
                     stream_config->top_p,
                     propose_step,
                     accepted_len);

        ft::BufferPtr accepted_tokens =
            device_->allocateBuffer({ft::DataType::TYPE_INT32, {1, accepted_len}, ft::AllocationType::HOST}, {"accepted_tokens"});
        device_->copy(
            {(*accepted_tokens)[0].view(0, accepted_len), (*scorer_stream_output->tokens)[0].view(0, accepted_len)});

        ft::BufferPtr logits        = nullptr;
        ft::BufferPtr hidden_states = nullptr;
        ft::BufferPtr loss          = nullptr;

        // TODO(xyz): optimize deepclone
        if (stream->generateConfig()->return_logits) {
            logits = device_->clone(
                {scorer_stream_output->logits->view(0, accepted_len), ft::AllocationType::HOST, {"return_logits"}});
        }

        if (stream->needReturnHiddenStates()) {
            hidden_states = device_->clone({scorer_stream_output->hidden_states->view(0, accepted_len),
                ft::AllocationType::DEVICE,
                {"return_hidden_states"}});
            FT_LOG_DEBUG("sample hidden states: %s", hidden_states->debugStringMeta().c_str());

        }
        if (scorer_stream_output->loss) {
            loss = device_->clone({*scorer_stream_output->loss, ft::AllocationType::HOST, {"return_loss"}});
        }
        sampler_output.outputs.emplace_back(
            propose_step, accepted_len, std::move(accepted_tokens), std::move(logits), std::move(hidden_states), std::move(loss), accepted_len > propose_step);
    }
    FT_LOG_DEBUG("speculative sample done");
    return sampler_output;
}

absl::StatusOr<size_t> RejectionSampler::top1Sample(size_t                                    propose_step,
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

absl::StatusOr<size_t> RejectionSampler::stochasticSample(size_t                                    propose_step,
                                          const SpeculativeExecutorStreamOutputPtr& propose_stream_output,
                                          const SpeculativeExecutorStreamOutputPtr& scorer_stream_output) const {
    torch::Tensor score_all_probs   = Buffer2torchTensor(scorer_stream_output->all_probs, false);
    size_t score_vocab_size   = score_all_probs.size(1);

    if (!propose_stream_output->all_probs) {
        auto all_probs = device_->allocateBuffer(
            {ft::DataType::TYPE_FP32, {propose_step, score_vocab_size}, ft::AllocationType::HOST}, {""});
        device_->bufMemset(*all_probs, 0);
        for (size_t i = 0; i < propose_step; i++) {
            *(all_probs->view(i, 1).dataWithOffset<float>(*propose_stream_output->tokens->dataWithOffset<int32_t>(i))) = 1.0;
        }
        propose_stream_output->all_probs = device_->clone({*all_probs, ft::AllocationType::DEVICE, {"all_probs"}});
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

};  // namespace rtp_llm