#include "rtp_llm/cpp/speculative_engine/score_executor/ScoreExecutor.h"
#include "rtp_llm/cpp/utils/StatusUtil.h"
#include "rtp_llm/cpp/speculative_engine/score_executor/ScoreOutput.h"
#include "rtp_llm/cpp/speculative_engine/score_executor/ScoreStream.h"

namespace rtp_llm {

absl::StatusOr<ScoreOutput> ScoreExecutor::score(const std::list<GenerateStreamPtr>& streams,
                                        const ProposeOutput&                proposer_output) {
    RTP_LLM_LOG_DEBUG(__PRETTY_FUNCTION__);

    std::list<GenerateStreamPtr> score_streams;

    ScoreOutput score_output;
    for (const GenerateStreamPtr& stream : streams) {
        size_t stream_id = stream->streamId();
        SpeculativeExecutorStreamOutputPtr stream_propose_output = proposer_output.outputs.at(stream_id);
        size_t propose_step = stream_propose_output->propose_step;
        score_output.outputs[stream_id] = std::make_shared<SpeculativeExecutorStreamOutput>();
        score_output.outputs[stream_id]->propose_step = propose_step;
        score_streams.emplace_back(std::make_shared<ScoreStream>(
            *stream, propose_step, stream_propose_output->tokens == nullptr ? nullptr : &stream_propose_output->tokens, &score_output));
    }

    for (auto& stream: score_streams) {
        RTP_LLM_LOG_DEBUG("before score stream[%d]: %s", stream->streamId(), stream->debugString().c_str());
    }

    RETURN_IF_STATUS_ERROR(score_normal_executor_.process(score_streams));

    for (auto& stream: score_streams) {
        RTP_LLM_LOG_DEBUG("post score stream[%d]: %s", stream->streamId(), stream->debugString().c_str());
    }

    RTP_LLM_LOG_DEBUG("score done");
    return score_output;
}


absl::StatusOr<ScoreOutput> ScoreExecutor::mtpScore(const std::list<GenerateStreamPtr>& streams,
                                                  const ProposeOutput&                proposer_output,
                                                  const std::list<GenerateStreamPtr>& prefill_streams) {
    RTP_LLM_LOG_DEBUG(__PRETTY_FUNCTION__);

    std::list<GenerateStreamPtr> score_streams;

    ScoreOutput score_output;
    for (const GenerateStreamPtr& stream : streams) {
        size_t stream_id = stream->streamId();
        SpeculativeExecutorStreamOutputPtr stream_propose_output = proposer_output.outputs.at(stream_id);
        size_t propose_step = stream_propose_output->propose_step;
        score_output.outputs[stream_id] = std::make_shared<SpeculativeExecutorStreamOutput>();
        score_output.outputs[stream_id]->propose_step = propose_step;
        score_streams.emplace_back(std::make_shared<ScoreStream>(
            *stream, propose_step, stream_propose_output->tokens == nullptr ? nullptr : &stream_propose_output->tokens, &score_output));
    }

    ScoreOutput fake_score_output;
    for (const GenerateStreamPtr& stream : prefill_streams) {
        size_t stream_id = stream->streamId();
        fake_score_output.outputs[stream_id] = std::make_shared<SpeculativeExecutorStreamOutput>();
        fake_score_output.outputs[stream_id]->propose_step = 1;
        score_streams.emplace_back(std::make_shared<ScoreStream>(
            *stream, 0, nullptr, &fake_score_output));
    }

    for (auto& stream: score_streams) {
        RTP_LLM_LOG_DEBUG("before score stream[%d]: %s", stream->streamId(), stream->debugString().c_str());
    }

    RETURN_IF_STATUS_ERROR(score_normal_executor_.process(score_streams));

    for (auto& stream: score_streams) {
        RTP_LLM_LOG_DEBUG("post score stream[%d]: %s", stream->streamId(), stream->debugString().c_str());
    }

    RTP_LLM_LOG_DEBUG("score done");
    return score_output;
}

bool ScoreExecutor::updateEplbConfig(const EplbConfig& config) {
    score_normal_executor_.updateEplbConfig(config);
    normal_executor_.updateEplbConfig(config);
    return true;
}
} // namespace rtp_llm