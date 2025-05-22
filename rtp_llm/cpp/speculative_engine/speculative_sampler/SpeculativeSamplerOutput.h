#pragma once
#include <vector>

#include "rtp_llm/cpp/core/Buffer.h"
#include "rtp_llm/cpp/speculative_engine/SpeculativeStreamOutput.h"

namespace rtp_llm {

struct SpeculativeSamplerOutput {
public:
    std::string debugString() const {
        std::stringstream debug_string;
        debug_string << "SpeculativeSamplerOutput { ";
        debug_string << "outputs: [";
        for (auto& output : outputs) {
            debug_string << output.debugString() << ", ";
        }
        debug_string << "}";
        return debug_string.str();
    }

public:
    std::vector<SpeculativeSamplerStreamOutput> outputs;  // output for each stream
};

}  // namespace rtp_llm