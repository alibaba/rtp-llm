#include "rtp_llm/cpp/speculative_engine/SpeculativeScheduler.h"
#include "rtp_llm/cpp/utils/StatusUtil.h"

namespace rtp_llm {

absl::StatusOr<std::list<GenerateStreamPtr>> SpeculativeScheduler::schedule(size_t reserve_step) {
    if (!pending_sp_run_streams_.empty()) {
        std::list<GenerateStreamPtr> moved_pending_sp_run_streams;
        moved_pending_sp_run_streams.splice(moved_pending_sp_run_streams.end(), pending_sp_run_streams_);
        return moved_pending_sp_run_streams;
    }
    CHECK_AND_RETURN_REF(streams, FIFOScheduler::schedule(reserve_step));
    if (streams.empty()) {
        return streams;
    }
    std::list<GenerateStreamPtr> normal_run_streams;
    for (auto& stream : streams) {
        if (stream->disableSpRun()) {
            normal_run_streams.emplace_back(stream);
        } else {
            pending_sp_run_streams_.emplace_back(stream);
        }
    }
    return normal_run_streams;
}

}  // namespace rtp_llm
