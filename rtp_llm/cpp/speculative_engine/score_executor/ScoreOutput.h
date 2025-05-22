#pragma once
#include <cstddef>
#include <memory>

#include "rtp_llm/cpp/speculative_engine/SpeculativeStreamOutput.h"

namespace rtp_llm {

struct ScoreOutput {
public:
    std::string debugString() const {
        std::stringstream debug_string;
        debug_string << "ScoreOutput { ";
        debug_string << "outputs: [";
        for (auto& [stream_id, output] : outputs) {
            debug_string << output->debugString() << ", ";
        }
        debug_string << "}";
        return debug_string.str();
    }

public:
    std::unordered_map<size_t, SpeculativeExecutorStreamOutputPtr> outputs;  // stream_id -> SpeculativeExecutorStreamOutputPtr
};

}  // namespace rtp_llm