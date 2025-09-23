#include "rtp_llm/cpp/speculative_engine/propose_executor/VanillaExecutor.h"
#include "rtp_llm/cpp/utils/StatusUtil.h"
#include "rtp_llm/cpp/engine_base/stream/GenerateStream.h"
#include "rtp_llm/cpp/engine_base/stream/StreamGroups.h"
#include "rtp_llm/cpp/speculative_engine/propose_executor/ProposeStream.h"

namespace rtp_llm {

absl::Status VanillaExecutor::propose(const std::list<GenerateStreamPtr>& streams, bool skip_check) {
    std::list<GenerateStreamPtr> propose_streams;

    for (auto& stream : streams) {
        RTP_LLM_LOG_DEBUG("before create vanilla stream [%ld]: %s", stream->streamId(), stream->debugString().c_str());
    }

    for (auto& stream : streams) {
        if (!skip_check) {
            if (stream->stoppedWithoutLock() || stream->finishedWithoutLock()) {
                continue;
            }
        }

        GenerateStreamPtr propose_stream;
        if (stream->containProposeStream()) {
            propose_stream = stream->getProposeStream();
            dynamic_cast<ProposeStream*>(propose_stream.get())->updateStream(*stream, propose_step_);
        } else {
            propose_stream = std::make_shared<ProposeStream>(*stream, propose_step_);
            stream->setProposeStream(propose_stream);
        }
        propose_streams.push_back(propose_stream);
    }

    for (auto& stream : propose_streams) {
        RTP_LLM_LOG_DEBUG("before propose stream [%ld]: %s", stream->streamId(), stream->debugString().c_str());
    }

    for (size_t i = 0; i < propose_step_; i++) {
        // remove stopped/finished stream
        propose_streams.remove_if([](const GenerateStreamPtr& stream) {
            return stream->stoppedWithoutLock() || stream->finishedWithoutLock();
        });
        RETURN_IF_STATUS_ERROR(normal_executor_.process(propose_streams));
    }

    for (auto& stream : propose_streams) {
        RTP_LLM_LOG_DEBUG("after propose stream [%ld]: %s", stream->streamId(), stream->debugString().c_str());
    }

    RTP_LLM_LOG_DEBUG("propose done");

    return absl::OkStatus();
}

}  // namespace rtp_llm