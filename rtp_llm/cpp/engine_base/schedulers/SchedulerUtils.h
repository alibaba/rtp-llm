#pragma once

#include "rtp_llm/cpp/engine_base/stream/GenerateStream.h"

namespace rtp_llm {

inline constexpr char kMixedForceBatchGroupError[] =
    "force batch group must not mix max_new_tokens == 0 and max_new_tokens != 0 streams";
inline constexpr char kMixedExecutionModeBatchError[] =
    "model execution batch must not mix max_new_tokens == 0 and max_new_tokens != 0 streams";
inline constexpr char kDecodeRolePrefillOnlyError[] = "DECODE role does not support max_new_tokens == 0";

inline bool isPrefillOnly(const GenerateStreamPtr& stream) {
    return stream->generateConfig()->isPrefillOnly();
}

template<typename StreamRange>
bool hasMixedExecutionModes(const StreamRange& streams) {
    bool has_execution_mode = false;
    bool prefill_only       = false;
    for (const auto& stream : streams) {
        const bool stream_prefill_only = isPrefillOnly(stream);
        if (!has_execution_mode) {
            has_execution_mode = true;
            prefill_only       = stream_prefill_only;
        } else if (prefill_only != stream_prefill_only) {
            return true;
        }
    }
    return false;
}

}  // namespace rtp_llm
