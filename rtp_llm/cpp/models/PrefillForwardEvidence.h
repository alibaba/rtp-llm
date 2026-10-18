#pragma once

#include "rtp_llm/models_py/bindings/core/OpData.h"
#include <cstdint>
#include <optional>
#include <sstream>
#include <string>
#include <vector>

namespace rtp_llm {

struct PrefillForwardEvidence {
    std::vector<int64_t> request_ids;
    int64_t              physical_tokens    = 0;
    int64_t              local_tokens       = 0;
    int64_t              tp_rank            = 0;
    int64_t              context_rows       = 0;
    int64_t              request_id_rows    = -1;
    int                  request_id_dtype   = -1;
    bool                 request_ids_on_cpu = false;
    std::string          error;
};

// Called only after a successful forward return. GptModelInputs::request_id
// contains context stream IDs, not decode rows or SP padding. Never copy a
// device tensor for observation, and do not expose cache-store internals to Python.
inline std::optional<PrefillForwardEvidence> collectPrefillForwardEvidence(const GptModelInputs& inputs,
                                                                           int64_t               physical_tokens,
                                                                           int64_t               model_id,
                                                                           int64_t               tp_rank,
                                                                           int64_t               tp_size,
                                                                           bool                  sequence_parallel) {
    // model_id == 0 identifies the target, as in the wrapper's graph-role policy.
    if (model_id != 0 || inputs.warmup || inputs.skip_run || inputs.is_fake_stream || inputs.is_target_verify
        || inputs.is_mtp_draft_update || !inputs.input_lengths.defined() || !inputs.sequence_lengths.defined()
        || inputs.sequence_lengths.numel() != 0 || inputs.input_lengths.numel() == 0) {
        return std::nullopt;
    }
    PrefillForwardEvidence event;
    event.physical_tokens = physical_tokens;
    event.tp_rank         = tp_rank;
    const int64_t batch   = inputs.input_lengths.numel();
    event.context_rows    = batch;
    if (inputs.request_id.defined()) {
        event.request_id_rows    = inputs.request_id.numel();
        event.request_id_dtype   = static_cast<int>(inputs.request_id.scalar_type());
        event.request_ids_on_cpu = inputs.request_id.device().is_cpu();
    }
    if (!inputs.prefix_lengths.defined() || inputs.prefix_lengths.numel() != batch) {
        event.error = "prefill_context_count_mismatch";
    } else if (!inputs.request_id.defined() || !inputs.request_id.device().is_cpu()
               || inputs.request_id.scalar_type() != torch::kInt64 || inputs.request_id.dim() != 1
               || inputs.request_id.numel() != batch) {
        event.error = "prefill_request_ids_require_cpu_int64_context_rows";
    } else if (tp_size <= 0 || tp_rank < 0 || tp_rank >= tp_size || physical_tokens <= 0
               || (sequence_parallel && physical_tokens % tp_size != 0)) {
        event.error = "prefill_physical_token_geometry_mismatch";
    } else {
        const auto ids = inputs.request_id.accessor<int64_t, 1>();
        event.request_ids.reserve(static_cast<size_t>(batch));
        for (int64_t index = 0; index < batch; ++index) {
            event.request_ids.push_back(ids[index]);
        }
        event.local_tokens = sequence_parallel ? physical_tokens / tp_size : physical_tokens;
    }
    return event;
}

inline std::string prefillForwardEvidenceJson(const PrefillForwardEvidence& event, int64_t time_ns) {
    std::ostringstream out;
    if (!event.error.empty()) {
        out << "{\"event\":\"evidence_error\",\"observation_event\":\"target_prefill_forward\","
               "\"reason\":\""
            << event.error << "\",\"tp_rank\":" << event.tp_rank << ",\"context_rows\":" << event.context_rows
            << ",\"request_id_rows\":" << event.request_id_rows << ",\"request_id_dtype\":" << event.request_id_dtype
            << ",\"request_ids_on_cpu\":" << (event.request_ids_on_cpu ? "true" : "false");
    } else {
        out << "{\"event\":\"target_prefill_forward\",\"kind\":\"target_prefill_forward\",\"request_ids\":[";
        for (size_t index = 0; index < event.request_ids.size(); ++index) {
            if (index != 0) {
                out << ',';
            }
            out << event.request_ids[index];
        }
        out << "],\"actual_batch\":" << event.request_ids.size() << ",\"physical_tokens\":" << event.physical_tokens
            << ",\"local_tokens\":" << event.local_tokens << ",\"tp_rank\":" << event.tp_rank
            << ",\"boundary\":\"forward_return\",\"gpu_synchronized\":false";
    }
    out << ",\"time_ns\":" << time_ns << '}';
    return out.str();
}

}  // namespace rtp_llm
