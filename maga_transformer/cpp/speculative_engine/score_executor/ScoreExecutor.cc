#include "maga_transformer/cpp/speculative_engine/score_executor/ScoreExecutor.h"
#include "maga_transformer/cpp/utils/StatusUtil.h"
#include "maga_transformer/cpp/speculative_engine/score_executor/ScoreOutput.h"
#include "maga_transformer/cpp/speculative_engine/score_executor/ScoreStream.h"

namespace rtp_llm {

absl::StatusOr<ScoreOutput> ScoreExecutor::score(const std::list<GenerateStreamPtr>& streams,
                                        const ProposeOutput&                proposer_output) {
    FT_LOG_DEBUG(__PRETTY_FUNCTION__);

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
        FT_LOG_DEBUG("before score stream[%d]: %s", stream->streamId(), stream->debugString().c_str());
    }

    RETURN_IF_STATUS_ERROR(score_normal_executor_.process(score_streams));

    for (auto& stream: score_streams) {
        FT_LOG_DEBUG("post score stream[%d]: %s", stream->streamId(), stream->debugString().c_str());
    }

    FT_LOG_DEBUG("score done");
    return score_output;
}


} // namespace rtp_llm