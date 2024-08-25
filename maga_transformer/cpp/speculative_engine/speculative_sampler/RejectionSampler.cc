#include "maga_transformer/cpp/speculative_engine/speculative_sampler/RejectionSampler.h"

namespace ft = fastertransformer;
namespace rtp_llm {

absl::StatusOr<SpeculativeSamplerOutput> RejectionSampler::sample(const std::list<GenerateStreamPtr>& streams,
                                                const ProposeOutput& proposer_output,
                                                const ScoreOutput&   scorer_output) const {
    FT_LOG_DEBUG(__PRETTY_FUNCTION__);
    SpeculativeSamplerOutput sampler_output;
    FT_CHECK(proposer_output.outputs.size() == scorer_output.outputs.size());
    size_t stream_index = 0;
    size_t propose_step = proposer_output.propose_step;
    for (const GenerateStreamPtr& stream: streams) {
        const SpeculativeExecutorStreamOutputPtr& propose_stream_output = proposer_output.outputs[stream_index];
        const SpeculativeExecutorStreamOutputPtr& scorer_stream_output = scorer_output.outputs[stream_index];
        std::shared_ptr<GenerateConfig>& stream_config = stream->generateConfig();
        size_t accepted_len = 0;
        if (stream_config->top_k == 1) {
            accepted_len = top1Sample(propose_step, propose_stream_output, scorer_stream_output);
        } else {
            FT_FAIL("RejectionSampler only support top_k == 1");
        }
        FT_LOG_DEBUG("stream [%d], propose_tokens = [%d], accept_tokens = [%d]", stream->streamId(), propose_step, accepted_len);
        ft::BufferPtr accepted_tokens = device_->allocateBuffer({ft::DataType::TYPE_INT32, {1, accepted_len}, ft::AllocationType::HOST});
        device_->copy({(*accepted_tokens)[0].view(0, accepted_len), (*scorer_stream_output->tokens)[0].view(0, accepted_len)});
        sampler_output.outputs.emplace_back(propose_step, accepted_len, std::move(accepted_tokens));
        stream_index++;
    }
    FT_LOG_DEBUG("speculative sample done");
    return sampler_output;
}

size_t RejectionSampler::top1Sample(size_t propose_step, const SpeculativeExecutorStreamOutputPtr& propose_stream_output, const SpeculativeExecutorStreamOutputPtr& scorer_stream_output) const {
    size_t accepted_len = 0;
    while (accepted_len < propose_step) {
        if ((*propose_stream_output->tokens->dataWithOffset<int32_t>(accepted_len)) != (*scorer_stream_output->tokens->dataWithOffset<int32_t>(accepted_len))) {
            break;
        }
        accepted_len++;
    }
    return std::min(propose_step, accepted_len + 1);
}

};