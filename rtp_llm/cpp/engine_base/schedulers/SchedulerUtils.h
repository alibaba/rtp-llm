#pragma once

#include "rtp_llm/cpp/engine_base/stream/GenerateStream.h"

namespace rtp_llm {

inline constexpr char kMixedForceBatchGroupError[] =
    "force batch group must not mix prefill-only, generation, or hidden-state capture execution modes";
inline constexpr char kMixedExecutionModeBatchError[] =
    "model execution batch must not mix max_new_tokens == 0 and max_new_tokens != 0 streams";
inline constexpr char kMixedCaptureModeBatchError[] =
    "hidden-state capture requires an isolated batch of first full-context streams";
inline constexpr char kInvalidCaptureContextError[] =
    "hidden-state capture requires the first full prompt without cache reuse, chunks, or verification";
inline constexpr char kDecodeRolePrefillOnlyError[] = "DECODE role does not support max_new_tokens == 0";

inline bool isPrefillOnly(const GenerateStreamPtr& stream) {
    return stream->generateConfig()->isPrefillOnly();
}

struct StreamExecutionMode {
    bool prefill_only          = false;
    bool capture_hidden_states = false;

    bool operator!=(const StreamExecutionMode& other) const {
        return prefill_only != other.prefill_only || capture_hidden_states != other.capture_hidden_states;
    }
};

inline StreamExecutionMode streamExecutionMode(const GenerateStreamPtr& stream) {
    return {isPrefillOnly(stream), false};
}

template<typename StreamRange>
bool hasMixedExecutionModes(const StreamRange& streams) {
    bool                has_execution_mode = false;
    StreamExecutionMode execution_mode;
    for (const auto& stream : streams) {
        const auto mode = streamExecutionMode(stream);
        if (!has_execution_mode) {
            has_execution_mode = true;
            execution_mode     = mode;
        } else if (execution_mode != mode) {
            return true;
        }
    }
    return false;
}

}  // namespace rtp_llm
