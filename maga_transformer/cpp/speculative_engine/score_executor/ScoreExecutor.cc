#include "maga_transformer/cpp/speculative_engine/score_executor/ScoreExecutor.h"
#include "maga_transformer/cpp/utils/StatusUtil.h"
#include "maga_transformer/cpp/speculative_engine/score_executor/ScoreOutput.h"
#include "maga_transformer/cpp/speculative_engine/score_executor/ScoreStream.h"

namespace rtp_llm {

absl::StatusOr<ScoreOutput> ScoreExecutor::score(const std::list<GenerateStreamPtr>& streams,
                                        const ProposeOutput&                proposer_output) {
    FT_LOG_DEBUG(__PRETTY_FUNCTION__);

    size_t stream_index = 0;
    std::list<GenerateStreamPtr> score_streams;

    ScoreOutput score_output(streams.size());
    for (const GenerateStreamPtr& stream : streams) {
        size_t propose_step = proposer_output.outputs[stream_index]->propose_step;
        score_output.outputs[stream_index]->propose_step = propose_step;
        score_streams.emplace_back(std::make_shared<ScoreStream>(
            *stream, proposer_output.outputs[stream_index], score_output.outputs[stream_index], propose_step));
        stream_index++;
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