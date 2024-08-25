#pragma once

#include "maga_transformer/cpp/speculative_engine/SpeculativeStreamOutput.h"
namespace rtp_llm {

struct ProposeOutput {
public:
    ProposeOutput(size_t propose_step, size_t stream_num): propose_step(propose_step), outputs(stream_num) {
        for (size_t i = 0; i < stream_num; i++) {
            outputs[i] = std::make_shared<SpeculativeExecutorStreamOutput>();
        }
    }

    std::string debugString() const {
        std::stringstream debug_string;
        debug_string << "ProposeOutput { "
                     << "propose_step: " << propose_step;
        debug_string << ", outputs: [";
        for (auto& output : outputs) {
            debug_string << output->debugString() << ", ";
        }
        debug_string << "}";
        return debug_string.str();
    }

public:
    size_t                                          propose_step;
    std::vector<SpeculativeExecutorStreamOutputPtr> outputs;  // outputs for each stream
};

}  // namespace rtp_llm