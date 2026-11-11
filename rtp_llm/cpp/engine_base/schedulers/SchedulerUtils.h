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

struct StreamExecutionMode {
    bool prefill_only = false;

    bool operator!=(const StreamExecutionMode& other) const {
        return prefill_only != other.prefill_only;
    }
};

inline StreamExecutionMode streamExecutionMode(const GenerateStreamPtr& stream) {
    return {isPrefillOnly(stream)};
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
