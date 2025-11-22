#include "rtp_llm/cpp/speculative_engine/score_executor/ScoreExecutor.h"
#include "rtp_llm/cpp/speculative_engine/score_executor/ScoreStream.h"
#include "rtp_llm/cpp/utils/StatusUtil.h"

namespace rtp_llm {

absl::Status ScoreExecutor::score(const std::list<GenerateStreamPtr>& streams, bool skip_check) {
    std::list<GenerateStreamPtr> score_streams;

    RTP_LLM_LOG_DEBUG("score begin");

    for (const GenerateStreamPtr& stream : streams) {
        RTP_LLM_LOG_DEBUG("before create score stream [%ld]: %s", stream->streamId(), stream->debugString().c_str());
    }

    for (const GenerateStreamPtr& stream : streams) {
        if (!skip_check) {
            if (stream->stoppedWithoutLock() || stream->finishedWithoutLock()) {
                continue;
            }
        }

        GenerateStreamPtr score_stream;
        if (stream->containScorestream()) {
            score_stream = stream->getScoreStream();
            dynamic_cast<ScoreStream*>(score_stream.get())->updateStream(*stream);
        } else {
            score_stream = std::make_shared<ScoreStream>(*stream);
            stream->setScoreStream(score_stream);
        }
        score_streams.push_back(score_stream);
    }

    for (const GenerateStreamPtr& stream : score_streams) {
        RTP_LLM_LOG_DEBUG("before score stream [%ld]: %s", stream->streamId(), stream->debugString().c_str());
    }

    RETURN_IF_STATUS_ERROR(score_normal_executor_.process(score_streams));

    for (auto& stream : score_streams) {
        RTP_LLM_LOG_DEBUG("post score stream [%ld]: %s", stream->streamId(), stream->debugString().c_str());
    }

    RTP_LLM_LOG_DEBUG("score done");
    return absl::OkStatus();
}

bool ScoreExecutor::updateEplbConfig(const EPLBConfig& config) {
    score_normal_executor_.updateEplbConfig(config);
    normal_executor_.updateEplbConfig(config);
    return true;
}

}  // namespace rtp_llm
