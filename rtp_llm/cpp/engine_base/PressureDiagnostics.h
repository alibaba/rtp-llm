#pragma once
#include "rtp_llm/cpp/cache/block_tree_cache/BlockTreeCache.h"
#include "rtp_llm/cpp/utils/ErrorCode.h"
#include "rtp_llm/cpp/normal_engine/PressureStepRecorder.h"
#include <chrono>
namespace rtp_llm {
enum class PressureControlAction {
    ENABLE     = 1,
    MARK_PHASE = 2,
    DISABLE    = 3
};
struct PressureSessionControl {
    std::string           session_id;
    PressureControlAction action;
    std::string           phase;
    size_t                recorder_capacity{0};
};
// This dedicated status distinguishes unavailable capability from collection failure.
inline constexpr ErrorCode kPressureDiagnosticsUnsupported = static_cast<ErrorCode>(55101);
inline constexpr uint32_t  kMaxPressureStepRecords         = 1024;
inline constexpr int       kPressureRpcTimeoutMs           = 1000;
inline std::chrono::system_clock::time_point
resolvePressureControlDeadline(std::chrono::system_clock::time_point deadline,
                               std::chrono::system_clock::time_point now = std::chrono::system_clock::now()) {
    return deadline == std::chrono::system_clock::time_point::max() ?
               now + std::chrono::milliseconds(kPressureRpcTimeoutMs) :
               deadline;
}
inline ErrorResult<CachePressureSnapshot> unsupportedCachePressure() {
    return {kPressureDiagnosticsUnsupported, "cache pressure diagnostics unsupported"};
}
}  // namespace rtp_llm
