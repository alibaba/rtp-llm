#include "maga_transformer/cpp/speculative_engine/propose_executor/VanillaExecutor.h"
#include "maga_transformer/cpp/common/status_util.h"
#include "maga_transformer/cpp/stream/GenerateStream.h"
#include "maga_transformer/cpp/stream/StreamGroups.h"
#include "maga_transformer/cpp/speculative_engine/propose_executor/ProposeOutput.h"
#include "maga_transformer/cpp/speculative_engine/propose_executor/VanillaStream.h"

namespace rtp_llm {

absl::StatusOr<ProposeOutput> VanillaExecutor::propose(const std::list<GenerateStreamPtr>& streams) {
    FT_LOG_DEBUG(__PRETTY_FUNCTION__);
    std::list<GenerateStreamPtr> propose_streams;
    ProposeOutput propose_output(propose_step_, streams.size());

    size_t stream_index = 0;
    for (auto& stream : streams) {
        propose_streams.emplace_back(std::make_shared<VanillaStream>(*stream, propose_output.outputs[stream_index], propose_step_));
        stream_index++;
    }

    for (auto& stream: propose_streams) {
        FT_LOG_DEBUG("before propose stream[%d]: %s", stream->streamId(), stream->debugString().c_str());
    }

    for (size_t i = 0; i < propose_step_; i++) {
        RETURN_IF_STATUS_ERROR(normal_executor_.process(propose_streams));
    }

    for (auto& stream: propose_streams) {
        FT_LOG_DEBUG("before propose stream[%d]: %s", stream->streamId(), stream->debugString().c_str());
    }

    FT_LOG_DEBUG("propose done");

    return propose_output;
}

}  // namespace rtp_llm