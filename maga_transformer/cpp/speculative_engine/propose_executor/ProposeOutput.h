#pragma once

#include "maga_transformer/cpp/speculative_engine/SpeculativeStreamOutput.h"
namespace rtp_llm {

struct ProposeOutput {
public:
     ProposeOutput(size_t stream_num): outputs(stream_num) {
        for (size_t i = 0; i < stream_num; i++) {
            outputs[i] = std::make_shared<SpeculativeExecutorStreamOutput>();
        }
    }

    std::string debugString() const {
        std::stringstream debug_string;
        debug_string << "ProposeOutput { ";
        debug_string << "outputs: [";
        for (auto& output : outputs) {
            debug_string << output->debugString() << ", ";
        }
        debug_string << "}";
        return debug_string.str();
    }

    bool hasNoPropose() const {
        for (auto& output : outputs) {
            if (output->tokens && output->tokens->size() > 0) {
                return false;
            }
        }
        return true;
    }

public:
    std::vector<SpeculativeExecutorStreamOutputPtr> outputs;  // outputs for each stream
};

}  // namespace rtp_llm