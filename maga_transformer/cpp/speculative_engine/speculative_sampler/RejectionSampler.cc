#include "maga_transformer/cpp/speculative_engine/speculative_sampler/RejectionSampler.h"
#include "src/fastertransformer/core/Buffer.h"
#include "src/fastertransformer/utils/logger.h"
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
    size_t stream_index = 0;
    size_t propose_step = proposer_output.propose_step;
    // TODO(xyz): optimize the RejectionSampler with batch processing interface
    for (const GenerateStreamPtr& stream : streams) {
        const SpeculativeExecutorStreamOutputPtr& propose_stream_output = proposer_output.outputs[stream_index];
        const SpeculativeExecutorStreamOutputPtr& scorer_stream_output  = scorer_output.outputs[stream_index];
        std::shared_ptr<GenerateConfig>&          stream_config         = stream->generateConfig();
        size_t                                    accepted_len          = 0;
        if (stream_config->top1()) {
            accepted_len = top1Sample(propose_step, propose_stream_output, scorer_stream_output);
        } else {
            accepted_len = stochasticSample(propose_step, propose_stream_output, scorer_stream_output);
        }
        FT_LOG_DEBUG("stream [%d], topk = [%d], topp = [%f], propose_tokens = [%d], accept_tokens = [%d]",
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

        // TODO(xyz): optimize deepclone
        if (stream->generateConfig()->return_logits) {
            logits = device_->clone(
                {scorer_stream_output->logits->view(0, accepted_len), ft::AllocationType::HOST, {"return_logits"}});
        }
        if (stream->generateConfig()->return_hidden_states) {
            hidden_states = device_->clone({scorer_stream_output->hidden_states->view(0, accepted_len),
                                            ft::AllocationType::HOST,
                                            {"return_hidden_states"}});
        }
        sampler_output.outputs.emplace_back(
            propose_step, accepted_len, std::move(accepted_tokens), std::move(logits), std::move(hidden_states), accepted_len > propose_step);
        stream_index++;
    }
    FT_LOG_DEBUG("speculative sample done");
    return sampler_output;
}

size_t RejectionSampler::top1Sample(size_t                                    propose_step,
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

size_t RejectionSampler::stochasticSample(size_t                                    propose_step,
                                          const SpeculativeExecutorStreamOutputPtr& propose_stream_output,
                                          const SpeculativeExecutorStreamOutputPtr& scorer_stream_output) const {
    torch::Tensor propose_all_probs = Buffer2torchTensor(propose_stream_output->all_probs, false);
    torch::Tensor score_all_probs   = Buffer2torchTensor(scorer_stream_output->all_probs, false);

    size_t propose_vocab_size = propose_all_probs.size(1);
    size_t score_vocab_size   = score_all_probs.size(1);

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
    torch::Tensor col_indices  = torch::from_blob(propose_stream_output->tokens->dataWithOffset<int32_t>(accepted_len),
                                                 {(long)propose_step},
                                                 torch::kInt32)
                                    .to(torch::Device(target_device));
    torch::Tensor score_probs   = score_all_probs.index({row_indices, col_indices}).to(torch::Device(host_device));
    torch::Tensor propose_probs = propose_all_probs.index({row_indices, col_indices}).to(torch::Device(host_device));
    torch::Tensor div_probs     = score_probs.div(propose_probs);
    while (accepted_len < propose_step) {
        int32_t propose_token_id = *propose_stream_output->tokens->dataWithOffset<int32_t>(accepted_len);
        if (randoms[accepted_len].greater(div_probs[accepted_len]).item<bool>()) {
            auto new_p = score_all_probs[accepted_len]
                             .subtract(propose_all_probs[accepted_len])
                             .maximum(torch::zeros_like(score_all_probs[accepted_len], torch::Device(target_device)));
            auto norm_p                                                          = new_p.div(new_p.sum(0));
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