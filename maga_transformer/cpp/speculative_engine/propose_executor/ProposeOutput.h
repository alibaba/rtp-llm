#pragma once

#include "maga_transformer/cpp/speculative_engine/SpeculativeStreamOutput.h"
namespace rtp_llm {

struct ProposeOutput {
public:

    std::string debugString() const {
        std::stringstream debug_string;
        debug_string << "ProposeOutput { ";
        debug_string << "outputs: [";
        for (auto& [stream_id, output] : outputs) {
            debug_string << output->debugString() << ", ";
        }
        debug_string << "}";
        return debug_string.str();
    }

    bool hasNoPropose() const {
        for (auto& [stream_id, output] : outputs) {
            if (output->tokens && output->tokens->size() > 0) {
                return false;
            }
        }
        return true;
    }

public:
    std::unordered_map<size_t, SpeculativeExecutorStreamOutputPtr> outputs;  // stream_id -> SpeculativeExecutorStreamOutputPtr
};

}  // namespace rtp_llm