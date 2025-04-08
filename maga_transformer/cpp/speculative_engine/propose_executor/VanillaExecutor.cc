#include "maga_transformer/cpp/speculative_engine/propose_executor/VanillaExecutor.h"
#include "maga_transformer/cpp/utils/StatusUtil.h"
#include "maga_transformer/cpp/stream/GenerateStream.h"
#include "maga_transformer/cpp/stream/StreamGroups.h"
#include "maga_transformer/cpp/speculative_engine/propose_executor/ProposeOutput.h"
#include "maga_transformer/cpp/speculative_engine/propose_executor/VanillaStream.h"

namespace rtp_llm {

absl::StatusOr<ProposeOutput> VanillaExecutor::propose(const std::list<GenerateStreamPtr>& streams) {
    FT_LOG_DEBUG(__PRETTY_FUNCTION__);
    std::list<GenerateStreamPtr> propose_streams;
    ProposeOutput propose_output;

    for (auto& stream : streams) {
        if (stream->needFinish() || stream->stoppedWithoutLock()) {
            continue;
        }
        size_t stream_id = stream->streamId();
        propose_output.outputs[stream_id] = std::make_shared<SpeculativeExecutorStreamOutput>();
        propose_output.outputs[stream_id]->propose_step = propose_step_;
        propose_streams.emplace_back(std::make_shared<VanillaStream>(*stream, &propose_output, propose_step_));
    }

    for (auto& stream: propose_streams) {
        FT_LOG_DEBUG("before propose stream[%d]: %s", stream->streamId(), stream->debugString().c_str());
    }

    for (size_t i = 0; i < propose_step_; i++) {
        bool stop_propose = propose_streams.empty();
        tpSyncStopFinishedStream(stop_propose);
        if (stop_propose) {
            FT_LOG_DEBUG("early stop propose");
            break;
        }
        RETURN_IF_STATUS_ERROR(normal_executor_.process(propose_streams));
        propose_streams.erase(std::remove_if(propose_streams.begin(),
                                             propose_streams.end(),
                                             [&](auto stream) {
                                                return std::dynamic_pointer_cast<VanillaStream>(stream)->checkFinish();
                                             }), propose_streams.end());

    }

    for (auto& stream: propose_streams) {
        FT_LOG_DEBUG("after propose stream[%d]: %s", stream->streamId(), stream->debugString().c_str());
    }

    FT_LOG_DEBUG("propose done");

    return propose_output;
}

}  // namespace rtp_llm