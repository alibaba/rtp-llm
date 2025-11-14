#include "rtp_llm/cpp/speculative_engine/propose_executor/MTPExecutor.h"
#include "rtp_llm/cpp/utils/StatusUtil.h"
#include "rtp_llm/cpp/engine_base/stream/GenerateStream.h"
#include "rtp_llm/cpp/engine_base/stream/StreamGroups.h"
#include "rtp_llm/cpp/speculative_engine/propose_executor/MTPStream.h"

namespace rtp_llm {

absl::Status MTPExecutor::propose(const std::list<GenerateStreamPtr>& streams, bool skip_check) {
    std::list<GenerateStreamPtr> propose_streams;

    for (auto& stream : streams) {
        RTP_LLM_LOG_DEBUG("before create mtp stream [%ld]: %s", stream->streamId(), stream->debugString().c_str());
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
            dynamic_cast<MTPStream*>(propose_stream.get())->updateStream(*stream, propose_step_);
        } else {
            propose_stream = std::make_shared<MTPStream>(*stream, propose_step_);
            stream->setProposeStream(propose_stream);
        }
        propose_streams.push_back(propose_stream);
    }

    for (auto& stream : propose_streams) {
        RTP_LLM_LOG_DEBUG("before propose stream [%ld]: %s", stream->streamId(), stream->debugString().c_str());
    }

    for (size_t i = 0; i < propose_step_; i++) {
        if (i > 0) {
            // remove stopped/finished stream
            propose_streams.remove_if([](const GenerateStreamPtr& stream) {
                return stream->stoppedWithoutLock() || stream->finishedWithoutLock();
            });

            for (auto& stream : propose_streams) {
                auto mtp_stream = std::static_pointer_cast<MTPStream>(stream);
                mtp_stream->shiftRightOneToken(*stream);
            }
        }
        RETURN_IF_STATUS_ERROR(mtp_executors_[i]->process(propose_streams));
    }

    for (auto& stream : propose_streams) {
        RTP_LLM_LOG_DEBUG("after propose stream [%ld]: %s", stream->streamId(), stream->debugString().c_str());
    }

    RTP_LLM_LOG_DEBUG("propose done");

    return absl::OkStatus();
}

bool MTPExecutor::updateEplbConfig(const EPLBConfig& config) {
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