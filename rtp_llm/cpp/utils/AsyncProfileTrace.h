#pragma once

#include <string>
#include <sstream>
#include <vector>
#include "rtp_llm/cpp/utils/AsyncProfilingScope.h"

namespace rtp_llm {

namespace detail {
// Diagnostic serialization must throw on stream failure so the native export
// cannot be replaced by a silently truncated JSON fragment.
std::ostringstream asyncProfileOutputStream();
AsyncProfileKind               exportedAsyncProfileKind(const AsyncProfileEvent& event);
int64_t                        asyncProfileBaseTimeNs(const std::string& trace);
std::vector<AsyncProfileEvent> resolveAsyncProfileLinks(const std::vector<AsyncProfileEvent>& events);
}  // namespace detail

// Append only RTP-LLM async records, preserving every byte of the Kineto trace.
// Throws on an unsupported layout/time base. The caller retains the native file.
std::string mergeAsyncProfileTrace(const std::string& native_trace, const std::vector<AsyncProfileEvent>& events);

// Perfetto-native logical tracks. The bundle contains the untouched CPU/GPU
// JSON plus this protobuf stream; Perfetto imports both into one time domain.
std::string asyncProfileWorkflowTrace(const std::string& native_trace, const std::vector<AsyncProfileEvent>& events);
void        writeAsyncProfileBundle(const std::string&                    path,
                                    const std::string&                    native_trace,
                                    const std::vector<AsyncProfileEvent>& events);

// Atomically replace the native export only after a complete successful merge.
void appendAsyncProfileTrace(const std::string& path, const std::vector<AsyncProfileEvent>& events);

}  // namespace rtp_llm
