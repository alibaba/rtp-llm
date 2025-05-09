#include "maga_transformer/cpp/speculative_engine/propose_executor/MTPExecutor.h"
#include "maga_transformer/cpp/utils/StatusUtil.h"
#include "maga_transformer/cpp/stream/GenerateStream.h"
#include "maga_transformer/cpp/stream/StreamGroups.h"
#include "maga_transformer/cpp/speculative_engine/propose_executor/ProposeOutput.h"
#include "maga_transformer/cpp/speculative_engine/propose_executor/MTPStream.h"

namespace rtp_llm {

absl::StatusOr<ProposeOutput> MTPExecutor::propose(const std::list<GenerateStreamPtr>& streams) {
    RTP_LLM_LOG_DEBUG(__PRETTY_FUNCTION__);
    std::list<GenerateStreamPtr> propose_streams;
    ProposeOutput propose_output;

    for (auto& stream: streams) {
        RTP_LLM_LOG_DEBUG("before create mtp stream[%d]: %s", stream->streamId(), stream->debugString().c_str());
    }

    for (auto& stream : streams) {
        size_t stream_id = stream->streamId();
        propose_output.outputs[stream_id] = std::make_shared<SpeculativeExecutorStreamOutput>();
        propose_output.outputs[stream_id]->propose_step = propose_step_;
        propose_streams.emplace_back(std::make_shared<MTPStream>(
            *stream, std::make_shared<ProposeOutput>(propose_output), propose_step_));
    }

    for (auto& stream: propose_streams) {
        RTP_LLM_LOG_DEBUG("before propose stream[%d]: %s", stream->streamId(), stream->debugString().c_str());
    }


    for (size_t i = 0; i < propose_step_; i++) {
        RETURN_IF_STATUS_ERROR(mtp_executors_[i]->process(propose_streams));
        for (auto& stream : propose_streams) {
            auto mtp_stream = std::static_pointer_cast<MTPStream>(stream);
            mtp_stream->shiftRightOneToken();
            mtp_stream->updatePrefixLen();
        }
    }

    for (auto& stream: propose_streams) {
        RTP_LLM_LOG_DEBUG("after propose stream[%d]: %s", stream->streamId(), stream->debugString().c_str());
    }

    RTP_LLM_LOG_DEBUG("propose done");

    return propose_output;
}

bool MTPExecutor::updateEplbConfig(const EplbConfig& config) {
    for (auto& executor : mtp_executors_) {
        if (executor) {
            executor->updateEplbConfig(config);
        }
    }
    for (auto& executor : normal_mtp_executors_) {
        if (executor) {
            executor->updateEplbConfig(config);
        }
    }
    return true;
}

}  // namespace rtp_llm