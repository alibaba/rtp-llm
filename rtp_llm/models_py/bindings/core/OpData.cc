#include "rtp_llm/models_py/bindings/core/OpData.h"
#include "rtp_llm/cpp/utils/TensorDebugUtils.h"

#include <optional>
#include <functional>
#include <algorithm>

namespace rtp_llm {

std::string combineStrings(const std::vector<std::string>& vec) {
    std::string result = "\" ";
    for (const auto& s : vec) {
        result += s + ", ";
    }
    result += "\"";
    return result;
}

std::string GptModelInputs::debugString(bool force) const {
    if (!Logger::getEngineLogger().isDebugMode() && !force) {
        return "";
    }
    auto              tb = [](const torch::Tensor& t) -> std::string { return tensorDebugString(t); };
    std::stringstream debug_string;
    debug_string << "GptModelInputs { "
                 << "trace_ids: " << combineStrings(trace_ids) << ", combo_tokens: " << tb(combo_tokens)
                 << ", input_lengths: " << tb(input_lengths) << ", sequence_lengths: " << tb(sequence_lengths)
                 << ", prefix_lengths: " << tb(prefix_lengths);
    if (sequence_lengths_plus_1.defined()) {
        debug_string << ", sequence_lengths_plus_1: " << tb(sequence_lengths_plus_1);
    }
    if (combo_position_ids.defined()) {
        debug_string << ", combo_position_ids: " << tb(combo_position_ids);
    }
    if (last_hidden_states.defined()) {
        debug_string << ", last_hidden_states: " << tb(last_hidden_states);
    }
    if (kv_cache_kernel_block_id.defined()) {
        debug_string << ", kv_cache_kernel_block_id: " << tb(kv_cache_kernel_block_id);
    }
    if (kv_cache_block_id.defined()) {
        debug_string << ", kv_cache_block_id: " << tb(kv_cache_block_id);
    }
    if (attention_mask.defined()) {
        debug_string << ", attention_mask: " << tb(attention_mask);
    }
    if (request_id.defined()) {
        debug_string << ", request_id: " << tb(request_id);
    }
    if (request_pd_separation.defined()) {
        debug_string << ", request_pd_separation: " << tb(request_pd_separation);
    }
    if (cache_keys.defined()) {
        debug_string << ", cache_keys: " << tb(cache_keys);
    }
    debug_string << ", kv_block_stride_bytes: " << kv_block_stride_bytes;
    debug_string << ", pd_separation: " << pd_separation;
    debug_string << "}";
    return debug_string.str();
}

}  // namespace rtp_llm
