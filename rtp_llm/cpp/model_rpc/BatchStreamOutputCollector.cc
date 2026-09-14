#include "rtp_llm/cpp/model_rpc/BatchStreamOutputCollector.h"
#include <utility>

namespace rtp_llm {

ErrorInfo BatchStreamOutputCollector::add(GenerateOutputs output) {
    if (output.generate_outputs.empty()) {
        return ErrorInfo(ErrorCode::UNKNOWN_ERROR, "batch item produced an empty output");
    }
    if (empty()) {
        tokens_.resize(output.generate_outputs.size());
        softmax_probs_.resize(tokens_.size());
    } else if (output.generate_outputs.size() != tokens_.size() || output.request_id != output_.request_id) {
        return ErrorInfo(ErrorCode::UNKNOWN_ERROR, "batch item output count or request ID changed");
    }
    for (size_t i = 0; i < tokens_.size(); ++i) {
        auto& next = output.generate_outputs[i];
        if (next.output_ids.defined()) {
            tokens_[i].push_back(next.output_ids);
        }
        if (next.aux_info.softmax_probs.has_value()) {
            softmax_probs_[i].push_back(*next.aux_info.softmax_probs);
        }
        if (!output_.generate_outputs.empty()) {
            const auto& previous = output_.generate_outputs[i];
            if (!next.loss.has_value())
                next.loss = previous.loss;
            if (!next.logits.has_value())
                next.logits = previous.logits;
            if (!next.hidden_states.has_value())
                next.hidden_states = previous.hidden_states;
            if (!next.all_hidden_states.has_value())
                next.all_hidden_states = previous.all_hidden_states;
            if (!next.prompt_logits.has_value())
                next.prompt_logits = previous.prompt_logits;
            if (!next.aux_info.all_probs.has_value())
                next.aux_info.all_probs = previous.aux_info.all_probs;
            if (!next.aux_info.cum_log_probs.has_value())
                next.aux_info.cum_log_probs = previous.aux_info.cum_log_probs;
        }
    }
    output_ = std::move(output);
    return ErrorInfo::OkStatus();
}

GenerateOutputs BatchStreamOutputCollector::finish() {
    for (size_t i = 0; i < tokens_.size(); ++i) {
        auto& output = output_.generate_outputs[i];
        if (!tokens_[i].empty()) {
            output.output_ids               = torch::cat(tokens_[i], -1);
            output.aux_info.step_output_len = output.output_ids.size(-1);
        }
        if (!softmax_probs_[i].empty()) {
            output.aux_info.softmax_probs = torch::cat(softmax_probs_[i], -1);
        }
    }
    return std::move(output_);
}

}  // namespace rtp_llm
