#include "maga_transformer/cpp/speculative_engine/score_executor/ScoreExecutor.h"
#include "maga_transformer/cpp/common/status_util.h"
#include "maga_transformer/cpp/speculative_engine/score_executor/ScoreOutput.h"
#include "maga_transformer/cpp/speculative_engine/score_executor/ScoreStream.h"

namespace rtp_llm {

absl::StatusOr<ScoreOutput> ScoreExecutor::score(const std::list<GenerateStreamPtr>& streams,
                                        const ProposeOutput&                proposer_output) {
    FT_LOG_DEBUG(__PRETTY_FUNCTION__);

    size_t stream_index = 0;
    std::list<GenerateStreamPtr> score_streams;

    ScoreOutput score_output(proposer_output.propose_step, streams.size());
    for (const GenerateStreamPtr& stream : streams) {
        score_streams.emplace_back(std::make_shared<ScoreStream>(*stream, proposer_output.outputs[stream_index], score_output.outputs[stream_index], proposer_output.propose_step));
        stream_index++;
    }

    for (auto& stream: score_streams) {
        FT_LOG_DEBUG("before score stream[%d]: %s", stream->streamId(), stream->debugString().c_str());
    }

    RETURN_IF_STATUS_ERROR(normal_executor_.process(score_streams));

    for (auto& stream: score_streams) {
        FT_LOG_DEBUG("post score stream[%d]: %s", stream->streamId(), stream->debugString().c_str());
    }

    FT_LOG_DEBUG("score done");
    return score_output;
}


} // namespace rtp_llm