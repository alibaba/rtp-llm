#include "rtp_llm/cpp/speculative_engine/propose_executor/VanillaExecutor.h"
#include "rtp_llm/cpp/utils/StatusUtil.h"
#include "rtp_llm/cpp/stream/GenerateStream.h"
#include "rtp_llm/cpp/stream/StreamGroups.h"
#include "rtp_llm/cpp/speculative_engine/propose_executor/ProposeStream.h"

namespace rtp_llm {

absl::Status VanillaExecutor::propose(const std::list<GenerateStreamPtr>& streams, bool skip_check) {
    std::list<GenerateStreamPtr> propose_streams;

    for (auto& stream: streams) {
        RTP_LLM_LOG_DEBUG("before create vanilla stream [%d]: %s", stream->streamId(), stream->debugString().c_str());
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

    for (auto& stream: propose_streams) {
        RTP_LLM_LOG_DEBUG("before propose stream [%d]: %s", stream->streamId(), stream->debugString().c_str());
    }

    for (size_t i = 0; i < propose_step_; i++) {
        RETURN_IF_STATUS_ERROR(normal_executor_.process(propose_streams));
    }

    for (auto& stream: propose_streams) {
        RTP_LLM_LOG_DEBUG("after propose stream [%d]: %s", stream->streamId(), stream->debugString().c_str());
    }

    RTP_LLM_LOG_DEBUG("propose done");

    return absl::OkStatus();
}

}  // namespace rtp_llm